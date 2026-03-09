"""
Dynamic Batching for ASR Training.

Provides a dataset wrapper that creates variable-size batches based on
padded VRAM cost (num_samples × max_sample_duration) rather than a fixed
sample count. This avoids OOM by ensuring each batch stays within GPU
memory limits.

Key insight: VRAM cost is proportional to batch_size × longest_sample
(because all samples are padded to the max length in the batch), NOT to
the sum of durations.

Also provides a VRAM probing function that runs synthetic forward+backward
passes to find the memory cost per padded-second, then computes a safe
budget. Never risks OOM — critical for shared-memory GPUs where OOM
freezes the system.
"""

import torch
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.utils import logging


class DynamicBatchingDataset(IterableDataset):
    """
    Wraps an inner streaming dataset and yields pre-collated batches
    where the padded cost (num_samples × max_duration_in_batch) does not
    exceed max_padded_budget_sec.

    The DataLoader should use batch_size=None when using this wrapper
    (each __iter__ yield IS a complete collated batch).
    """

    def __init__(self, inner_dataset, max_padded_budget_sec: float, sample_rate: int = 16000):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.max_padded_budget_sec = max_padded_budget_sec
        self.sample_rate = sample_rate

    def __iter__(self):
        batch = []
        max_dur_in_batch = 0.0

        for sample in self.inner_dataset:
            # sample is (audio_tensor, audio_len, tokens_tensor, tokens_len)
            audio_len = sample[1].item() if hasattr(sample[1], 'item') else sample[1]
            duration = audio_len / self.sample_rate

            # Skip samples that alone exceed the budget
            if duration > self.max_padded_budget_sec:
                continue

            # Calculate what the padded cost would be if we add this sample
            new_max_dur = max(max_dur_in_batch, duration)
            new_count = len(batch) + 1
            new_padded_cost = new_count * new_max_dur

            # If adding this sample would exceed budget, yield current batch
            if batch and new_padded_cost > self.max_padded_budget_sec:
                yield _speech_collate_fn(batch, pad_id=0)
                batch = []
                max_dur_in_batch = 0.0
                # Recalculate for fresh batch
                new_max_dur = duration

            batch.append(sample)
            max_dur_in_batch = max(max_dur_in_batch, duration)

        # Yield remaining samples
        if batch:
            yield _speech_collate_fn(batch, pad_id=0)

    def __len__(self):
        return len(self.inner_dataset)


def _run_probe_step(model, num_samples, per_sample_len, token_count, vocab_size):
    """Run a single forward+backward probe and return peak VRAM in GB."""
    torch.cuda.reset_peak_memory_stats()

    audio = torch.randn(num_samples, per_sample_len, device='cuda')
    audio_len = torch.full((num_samples,), per_sample_len, dtype=torch.long, device='cuda')
    tokens = torch.randint(0, vocab_size, (num_samples, token_count), device='cuda')
    tokens_len = torch.full((num_samples,), token_count, dtype=torch.long, device='cuda')
    batch = (audio, audio_len, tokens, tokens_len)

    result = model.training_step(batch, 0)
    loss = result['loss'] if isinstance(result, dict) else result
    loss.backward()

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    del audio, audio_len, tokens, tokens_len, batch, result, loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return peak_gb


def probe_max_batch_duration(model, target_vram_gb: float, sample_rate: int = 16000):
    """
    Probe GPU memory to find the maximum padded batch budget that fits
    within target_vram_gb.

    VRAM for a batch = f(batch_size × max_sample_length) because all
    samples are padded to the longest one. We probe with uniform-length
    batches (where total_padded = batch_size × sample_length) at two
    different sizes to measure the linear relationship, then extrapolate.

    Uses two small safe probes — never risks OOM. Critical for
    shared-memory GPUs (e.g., GB10) where OOM freezes the system.

    Args:
        model: ASR model (on CUDA)
        target_vram_gb: Target VRAM usage in GB
        sample_rate: Audio sample rate

    Returns:
        max_padded_budget_sec: Safe maximum padded budget in seconds
            (batch_size × max_sample_duration must stay under this)
    """
    import os

    freeze_encoder_epochs = int(os.environ.get('FREEZE_ENCODER', '0'))
    vocab_size = model.tokenizer.vocab_size

    logging.info("=" * 60)
    logging.info("VRAM PROBE — Finding max padded batch budget")
    logging.info(f"  Target VRAM:   {target_vram_gb:.1f} GB")
    logging.info(f"  Sample rate:   {sample_rate}")
    logging.info("=" * 60)

    # Freeze encoder if it will be frozen during early training
    encoder_frozen = False
    if freeze_encoder_epochs != 0 and hasattr(model, 'encoder'):
        logging.info("  Encoder frozen: YES (matching training phase 1)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True

    # Monkey-patch log methods (training_step calls self.log outside trainer)
    _orig_log = model.log
    _orig_log_dict = model.log_dict
    model.log = lambda *a, **kw: None
    model.log_dict = lambda *a, **kw: None

    model.train()

    # Measure baseline VRAM (model + optimizer, no batch)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logging.info(f"  Base VRAM (model+optimizer): {base_gb:.2f} GB")
    logging.info(f"  Available for batches: {target_vram_gb - base_gb:.2f} GB")

    # Two probes with different padded costs (batch_size × sample_length)
    # Probe 1: 4 samples × 5s = 20 padded-seconds
    # Probe 2: 4 samples × 10s = 40 padded-seconds
    # Both are small enough to never OOM
    probes = [
        (4, 5.0),   # (num_samples, per_sample_sec)
        (4, 10.0),
    ]
    padded_costs = []  # padded-seconds
    peak_vrams = []    # GB

    for num_samples, per_sample_sec in probes:
        per_sample_len = int(per_sample_sec * sample_rate)
        token_count = max(1, int(per_sample_sec * 10))
        padded_sec = num_samples * per_sample_sec

        peak_gb = _run_probe_step(model, num_samples, per_sample_len, token_count, vocab_size)
        padded_costs.append(padded_sec)
        peak_vrams.append(peak_gb)
        logging.info(f"  Probe: {num_samples} × {per_sample_sec:.0f}s = {padded_sec:.0f} padded-sec -> {peak_gb:.2f} GB")

    # Linear extrapolation: peak_vram = base_cost + rate * padded_seconds
    pc1, pc2 = padded_costs
    pv1, pv2 = peak_vrams
    rate_gb_per_padded_sec = (pv2 - pv1) / (pc2 - pc1)
    base_cost = pv1 - rate_gb_per_padded_sec * pc1

    logging.info(f"  Memory model: {base_cost:.2f} GB fixed + {rate_gb_per_padded_sec:.4f} GB/padded-sec")

    max_duration = 0.0
    if rate_gb_per_padded_sec <= 0:
        safe_duration = pc2
        logging.warning(f"  Rate <= 0, falling back to {safe_duration:.0f} padded-sec")
    else:
        # Extrapolate: target_vram = base_cost + rate * max_padded
        max_padded = (target_vram_gb - base_cost) / rate_gb_per_padded_sec
        max_duration = max_padded
        # Apply 95% safety margin
        safe_duration = max_padded * 0.95
        # Clamp to reasonable range
        safe_duration = max(10.0, min(safe_duration, 5000.0))

    # Restore model state
    model.log = _orig_log
    model.log_dict = _orig_log_dict
    if encoder_frozen and hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = True
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # Show what this means in practice
    max_duration_cfg = 45.0  # typical max_duration from config
    typical_batch = int(safe_duration / max_duration_cfg)

    logging.info("=" * 60)
    logging.info(f"VRAM PROBE COMPLETE")
    logging.info(f"  Extrapolated max:    {max_duration:.0f} padded-sec" if rate_gb_per_padded_sec > 0 else "  Extrapolation: N/A")
    logging.info(f"  Safe budget:         {safe_duration:.0f} padded-sec (95% margin)")
    logging.info(f"  Target VRAM:         {target_vram_gb:.1f} GB")
    logging.info(f"  Example: {typical_batch} × {max_duration_cfg:.0f}s = {typical_batch * max_duration_cfg:.0f} padded-sec (worst case)")
    logging.info(f"  Example: {int(safe_duration / 10)} × 10s = {int(safe_duration / 10) * 10} padded-sec (avg case)")
    logging.info("=" * 60)

    return safe_duration
