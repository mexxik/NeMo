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
passes at the actual worst-case sample length (max_duration from config)
to measure per-sample VRAM cost. Never risks OOM — critical for
shared-memory GPUs where OOM freezes the system.
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

    Uses length bucketing: accumulates samples into a sort buffer, sorts
    by duration, then greedily packs similar-length samples into batches.
    This minimizes padding waste and maximizes GPU utilization.

    On unified-memory GPUs (e.g. GB10), all batches are padded to a fixed
    max_audio_len so the CUDA allocator sees identical tensor shapes every
    step and reuses memory instead of reserving ever-larger blocks.

    The DataLoader should use batch_size=None when using this wrapper
    (each __iter__ yield IS a complete collated batch).
    """

    def __init__(self, inner_dataset, max_padded_budget_sec: float,
                 sample_rate: int = 16000, sort_buffer_size: int = 500,
                 max_duration_sec: float = None):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.max_padded_budget_sec = max_padded_budget_sec
        self.sample_rate = sample_rate
        self.sort_buffer_size = sort_buffer_size

    def _pack_batches(self, samples):
        """
        Pack sorted samples into batches greedily.
        Samples must be sorted by duration (ascending).
        Yields collated batch tuples.
        """
        batch = []
        max_dur_in_batch = 0.0

        for sample, duration in samples:
            new_max_dur = max(max_dur_in_batch, duration)
            new_count = len(batch) + 1
            new_padded_cost = new_count * new_max_dur

            if batch and new_padded_cost > self.max_padded_budget_sec:
                yield _speech_collate_fn(batch, pad_id=0)
                batch = []
                max_dur_in_batch = 0.0

            batch.append(sample)
            max_dur_in_batch = max(max_dur_in_batch, duration)

        if batch:
            yield self._pad_to_fixed(_speech_collate_fn(batch, pad_id=0))

    def __iter__(self):
        sort_buffer = []  # list of (sample, duration)

        for sample in self.inner_dataset:
            audio_len = sample[1].item() if hasattr(sample[1], 'item') else sample[1]
            duration = audio_len / self.sample_rate

            # Skip samples that alone exceed the budget
            if duration > self.max_padded_budget_sec:
                continue

            sort_buffer.append((sample, duration))

            # When buffer is full, sort by duration and pack into batches
            if len(sort_buffer) >= self.sort_buffer_size:
                sort_buffer.sort(key=lambda x: x[1])
                yield from self._pack_batches(sort_buffer)
                sort_buffer = []

        # Drain remaining buffer
        if sort_buffer:
            sort_buffer.sort(key=lambda x: x[1])
            yield from self._pack_batches(sort_buffer)

    def __len__(self):
        # Return total samples (not batches). The SampleProgressBar callback
        # increments by actual batch size each step, so the progress bar
        # shows samples_processed / total_samples accurately.
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


def probe_max_batch_duration(model, target_vram_gb: float, max_sample_duration: float = 45.0, sample_rate: int = 16000):
    """
    Probe GPU memory to find the maximum padded batch budget that fits
    within target_vram_gb.

    Probes at the ACTUAL worst-case sample length (max_sample_duration)
    with batch_size=1 and batch_size=2. This directly measures VRAM cost
    at the real padding length, so extrapolation is accurate.

    Uses `torch.cuda.mem_get_info()` to verify safety before each probe.
    Never risks OOM — critical for shared-memory GPUs (e.g., GB10).

    Args:
        model: ASR model (on CUDA)
        target_vram_gb: Target VRAM usage in GB
        max_sample_duration: Max sample duration from config (seconds)
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
    logging.info(f"  Target VRAM:       {target_vram_gb:.1f} GB")
    logging.info(f"  Max sample dur:    {max_sample_duration:.1f}s")
    logging.info(f"  Sample rate:       {sample_rate}")
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
    torch.cuda.empty_cache()

    # Check total GPU memory
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_gb = total_bytes / (1024 ** 3)
    free_gb = free_bytes / (1024 ** 3)
    used_gb = total_gb - free_gb
    logging.info(f"  GPU total:         {total_gb:.2f} GB")
    logging.info(f"  GPU used:          {used_gb:.2f} GB")
    logging.info(f"  GPU free:          {free_gb:.2f} GB")

    # Probe at actual worst-case sample length with batch_size=1 and 2
    # This measures real per-sample VRAM cost at the padding length
    # that will actually occur in training
    per_sample_len = int(max_sample_duration * sample_rate)
    token_count = max(1, int(max_sample_duration * 10))  # ~10 BPE tokens/sec

    # Safety check: estimate if even batch_size=1 fits
    # A rough estimate: 1 sample of 45s at 16kHz = 720K floats = ~2.7 MB raw
    # But activations + gradients can be 100-1000x that. We check free VRAM.
    min_needed_gb = 3.0  # need at least 3 GB free for a single sample probe
    if free_gb < min_needed_gb:
        logging.error(f"  Only {free_gb:.1f} GB free, need at least {min_needed_gb:.1f} GB for probing")
        # Fall back to a very conservative budget
        safe_budget = max_sample_duration * 2  # just 2 samples at max duration
        logging.warning(f"  Falling back to conservative budget: {safe_budget:.0f} padded-sec")
        model.log = _orig_log
        model.log_dict = _orig_log_dict
        if encoder_frozen and hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = True
        return safe_budget

    # Probe 1: batch_size=1 at max_sample_duration
    peak_1 = _run_probe_step(model, 1, per_sample_len, token_count, vocab_size)
    logging.info(f"  Probe: 1 × {max_sample_duration:.0f}s -> {peak_1:.2f} GB")

    # Check if batch_size=2 is safe before probing
    # Estimate: peak_2 ≈ peak_1 + (peak_1 - base_used) where base_used is without batch
    torch.cuda.empty_cache()
    base_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    per_sample_cost_est = peak_1 - base_used_gb
    estimated_peak_2 = peak_1 + per_sample_cost_est

    if estimated_peak_2 > total_gb * 0.85:
        # batch_size=2 would be risky, extrapolate from single probe
        logging.info(f"  Skipping batch_size=2 probe (estimated {estimated_peak_2:.1f} GB, too close to limit)")
        per_sample_vram = per_sample_cost_est
    else:
        # Probe 2: batch_size=2 at max_sample_duration
        peak_2 = _run_probe_step(model, 2, per_sample_len, token_count, vocab_size)
        logging.info(f"  Probe: 2 × {max_sample_duration:.0f}s -> {peak_2:.2f} GB")
        per_sample_vram = peak_2 - peak_1  # marginal cost of one more sample

    logging.info(f"  Per-sample VRAM at {max_sample_duration:.0f}s: {per_sample_vram:.2f} GB")
    logging.info(f"  Base VRAM (model+opt): {base_used_gb:.2f} GB")

    # Calculate max batch size that fits in target VRAM
    available_gb = target_vram_gb - base_used_gb
    max_batch_size = max(1, int(available_gb / per_sample_vram))

    # Padded budget = max_batch_size × max_sample_duration
    # Apply 95% safety margin
    raw_budget = max_batch_size * max_sample_duration
    safe_budget = raw_budget * 0.95

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
    logging.info(f"  Max batch size at {max_sample_duration:.0f}s: {max_batch_size}")
    logging.info(f"  Raw budget:          {raw_budget:.0f} padded-sec")
    logging.info(f"  Safe budget:         {safe_budget:.0f} padded-sec (95% margin)")
    logging.info(f"  Target VRAM:         {target_vram_gb:.1f} GB")
    logging.info(f"  Worst case: {max_batch_size} × {max_sample_duration:.0f}s samples")
    logging.info(f"  Typical:    {int(safe_budget / 10)} × 10s samples")
    logging.info("=" * 60)

    return safe_budget
