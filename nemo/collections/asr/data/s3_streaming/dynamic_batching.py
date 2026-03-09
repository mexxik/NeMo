"""
Dynamic Batching for ASR Training.

Provides a dataset wrapper that creates variable-size batches based on
total audio duration rather than a fixed sample count. This avoids OOM
by ensuring the total duration per batch stays within GPU memory limits.

Also provides a VRAM probing function that runs synthetic forward+backward
passes to find the maximum total batch duration that fits in target VRAM.
"""

import torch
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.utils import logging


class DynamicBatchingDataset(IterableDataset):
    """
    Wraps an inner streaming dataset and yields pre-collated batches
    where total audio duration does not exceed max_batch_duration_sec.

    The DataLoader should use batch_size=None when using this wrapper
    (each __iter__ yield IS a complete collated batch).
    """

    def __init__(self, inner_dataset, max_batch_duration_sec: float, sample_rate: int = 16000):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.max_batch_duration_sec = max_batch_duration_sec
        self.sample_rate = sample_rate

    def __iter__(self):
        batch = []
        total_duration = 0.0

        for sample in self.inner_dataset:
            # sample is (audio_tensor, audio_len, tokens_tensor, tokens_len)
            audio_len = sample[1].item() if hasattr(sample[1], 'item') else sample[1]
            duration = audio_len / self.sample_rate

            # Skip samples that alone exceed the budget
            if duration > self.max_batch_duration_sec:
                continue

            # If adding this sample would exceed budget, yield current batch
            if batch and (total_duration + duration) > self.max_batch_duration_sec:
                yield _speech_collate_fn(batch, pad_id=0)
                batch = []
                total_duration = 0.0

            batch.append(sample)
            total_duration += duration

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
    Probe GPU memory to find the maximum total batch duration that fits
    within target_vram_gb.

    Uses two small safe probes to measure VRAM-per-second of audio, then
    extrapolates to find the maximum duration. Never risks OOM — critical
    for shared-memory GPUs (e.g., GB10) where OOM freezes the system
    instead of throwing a catchable exception.

    Args:
        model: ASR model (on CUDA)
        target_vram_gb: Target VRAM usage in GB
        sample_rate: Audio sample rate

    Returns:
        max_batch_duration_sec: Safe maximum total batch duration in seconds
    """
    import os

    freeze_encoder_epochs = int(os.environ.get('FREEZE_ENCODER', '0'))
    vocab_size = model.tokenizer.vocab_size

    logging.info("=" * 60)
    logging.info("VRAM PROBE — Finding max batch duration")
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

    # Two small safe probes to measure VRAM scaling
    # Use small durations that are guaranteed to fit
    num_samples = 10
    probe_durations = [30.0, 60.0]  # seconds total batch duration
    probe_peaks = []

    for dur in probe_durations:
        per_sample_sec = dur / num_samples
        per_sample_len = int(per_sample_sec * sample_rate)
        token_count = max(1, int(per_sample_sec * 10))

        peak_gb = _run_probe_step(model, num_samples, per_sample_len, token_count, vocab_size)
        probe_peaks.append(peak_gb)
        logging.info(f"  Probe: {dur:.0f}s -> {peak_gb:.2f} GB")

    # Linear extrapolation: peak = base_cost + rate * duration
    # Solve from two points: rate = (peak2 - peak1) / (dur2 - dur1)
    d1, d2 = probe_durations
    p1, p2 = probe_peaks
    rate_gb_per_sec = (p2 - p1) / (d2 - d1)
    base_cost = p1 - rate_gb_per_sec * d1  # Fixed cost (model, optimizer, activations overhead)

    logging.info(f"  Memory model: {base_cost:.2f} GB fixed + {rate_gb_per_sec:.4f} GB/sec")

    if rate_gb_per_sec <= 0:
        # Edge case: rate is zero or negative (very small model or measurement noise)
        # Fall back to the larger probe duration as our safe max
        safe_duration = d2
        logging.warning(f"  Rate <= 0, falling back to {safe_duration:.0f}s")
    else:
        # Extrapolate: target_vram = base_cost + rate * max_duration
        max_duration = (target_vram_gb - base_cost) / rate_gb_per_sec
        # Apply 85% safety margin (accounts for variable sample lengths, fragmentation)
        safe_duration = max_duration * 0.85
        # Clamp to reasonable range
        safe_duration = max(10.0, min(safe_duration, 1200.0))

    # Restore model state
    model.log = _orig_log
    model.log_dict = _orig_log_dict
    if encoder_frozen and hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = True
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    logging.info("=" * 60)
    logging.info(f"VRAM PROBE COMPLETE")
    logging.info(f"  Extrapolated max: {max_duration:.0f}s" if rate_gb_per_sec > 0 else "  Extrapolation: N/A")
    logging.info(f"  Safe duration:    {safe_duration:.0f}s (85% margin)")
    logging.info(f"  Target VRAM:      {target_vram_gb:.1f} GB")
    logging.info("=" * 60)

    return safe_duration
