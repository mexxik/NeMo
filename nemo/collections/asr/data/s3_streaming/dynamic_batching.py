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

    The DataLoader should use batch_size=None when using this wrapper
    (each __iter__ yield IS a complete collated batch).
    """

    def __init__(self, inner_dataset, max_padded_budget_sec: float,
                 sample_rate: int = 16000, sort_buffer_size: int = 500,
                 max_batch_size: int = 64, max_sample_duration: float = None,
                 rnnt_budget: float = 0, emit_clean_flag: bool = False,
                 joint_budget: float = 0):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.max_padded_budget_sec = max_padded_budget_sec
        self.sample_rate = sample_rate
        self.sort_buffer_size = sort_buffer_size
        self.max_batch_size = max_batch_size
        self.max_sample_duration = max_sample_duration
        # Selective backprop: only append the per-batch cleanliness bool to the
        # collated tuple when the feature is active. When off, batches stay the
        # standard 4-tuple so NeMo's training_step unpacking is unchanged.
        self.emit_clean_flag = emit_clean_flag
        # RNNT memory budget: caps B × max_dur² to prevent OOM.
        # RNNT joint memory is O(B × T × U × V) where T,U ∝ duration,
        # so real VRAM cost ≈ O(B × dur²). When set > 0, dynamically limits
        # batch size based on the longest sample in the batch.
        # max_padded_budget_sec (linear) still controls total audio seconds.
        # Whichever limit hits first wins.
        self.rnnt_budget = rnnt_budget
        # Joint memory budget: caps B × max_dur × max_target_tokens, i.e. the
        # real O(B × T × U) shape of the joint tensor.
        #
        # rnnt_budget above uses dur² as a stand-in for T × U, which only holds
        # when target length tracks duration. It does not for character-style
        # schemes: `phoneme_word` emits one token per IPA character plus a tag
        # per word, and MergeBuffer then concatenates 2-6 utterances, so U ran
        # to ~1000 on 15-20s merges while dur² predicted far less. That combination
        # asked for a 48.9 GiB gradient buffer and OOMed 85% into an epoch — only
        # once the short→long curriculum reached its longest window.
        #
        # Measured in token-seconds (max_dur × max_tokens × B). 0 disables.
        self.joint_budget = joint_budget

    def _bump_emitted(self, weight):
        """Advance the inner dataset's consumer-side bar counter at batch emit.

        `weight` is the total raw pulls represented by the batch (sum of each
        emitted sample's per-yield weight, which captures merges and dropped
        samples). Counting here — after the sort buffer — instead of at pull
        time keeps the progress bar within ~one prefetch queue of the training
        loop instead of a whole sort-buffer + merge-buffer ahead of it.
        """
        rc = getattr(self.inner_dataset, '_raw_emitted', None)
        if rc is not None and weight:
            with rc.get_lock():
                rc.value += weight

    def _collate(self, batch, clean):
        """Collate a batch and append the batch-level cleanliness flag.

        Selective backprop: batches are homogeneous (all-clean or all-noisy),
        so a single bool describes the batch. The training loop pops this 5th
        element and uses it to gate decoder/joint gradient updates. The flag is
        a plain Python bool so Lightning's batch-to-device transfer leaves it
        untouched. When selective backprop is off, returns the standard 4-tuple
        unchanged.
        """
        collated = _speech_collate_fn(batch, pad_id=0)
        if self.emit_clean_flag:
            return (*collated, clean)
        return collated

    def _pack_batches(self, samples):
        """
        Pack sorted samples into batches greedily.
        Samples must be sorted by (clean, duration) ascending so each batch is
        homogeneous in cleanliness; a batch is also flushed whenever the flag
        flips. `samples` items are (sample, duration, weight, clean).
        Yields collated batch tuples with a trailing cleanliness bool.
        """
        batch = []
        batch_w = 0
        max_dur_in_batch = 0.0
        max_tok_in_batch = 0
        batch_clean = True

        for sample, duration, weight, clean in samples:
            new_max_dur = max(max_dur_in_batch, duration)
            new_max_tok = max(max_tok_in_batch, self._target_len(sample))
            new_count = len(batch) + 1

            # Flush on budget overflow OR a cleanliness change (keeps batches
            # homogeneous for selective backprop).
            if batch and (
                self._batch_full(new_count, new_max_dur, new_max_tok)
                or clean != batch_clean
            ):
                self._bump_emitted(batch_w)
                yield self._collate(batch, batch_clean)
                batch = []
                batch_w = 0
                max_dur_in_batch = 0.0
                max_tok_in_batch = 0

            batch.append(sample)
            batch_w += weight
            batch_clean = clean
            max_dur_in_batch = max(max_dur_in_batch, duration)
            max_tok_in_batch = max(max_tok_in_batch, self._target_len(sample))

        if batch:
            self._bump_emitted(batch_w)
            yield self._collate(batch, batch_clean)

    def __iter__(self):
        if self.sort_buffer_size <= 1:
            # Greedy mode: pack samples directly as they arrive (no sorting).
            # Used with curriculum bucket-stride where the interleaver already
            # produces mixed-duration samples in a deliberate order.
            yield from self._iter_greedy()
        else:
            # Sort-buffer mode: accumulate, sort by duration, then pack.
            yield from self._iter_sorted()

    @staticmethod
    def _target_len(sample):
        """Target token count for a collate 4-tuple (audio, audio_len, tokens, tokens_len)."""
        try:
            n = sample[3]
        except (IndexError, TypeError):
            return 0
        return int(n.item()) if hasattr(n, 'item') else int(n)

    def _batch_full(self, count, max_dur, max_tokens=0):
        """Check if adding one more sample would exceed any limit."""
        # Linear budget: total padded seconds
        if count * max_dur > self.max_padded_budget_sec:
            return True
        # RNNT memory budget: B × max_dur²
        if self.rnnt_budget > 0 and count * max_dur * max_dur > self.rnnt_budget:
            return True
        # Joint memory budget: B × max_dur × max_target_tokens (∝ B × T × U).
        # Uses the measured target length, so it holds for label schemes whose
        # targets are not proportional to duration.
        if (
            self.joint_budget > 0
            and max_tokens > 0
            and count * max_dur * max_tokens > self.joint_budget
        ):
            return True
        # Hard batch size cap
        if count > self.max_batch_size:
            return True
        return False

    def _iter_greedy(self):
        """Greedy packing: accumulate samples and emit when budget exceeded."""
        batch = []
        batch_w = 0
        max_dur_in_batch = 0.0
        max_tok_in_batch = 0
        batch_clean = True
        carry_w = 0  # weight of dropped samples, folded into the next kept one

        for sample in self.inner_dataset:
            weight = getattr(self.inner_dataset, '_cur_yield_weight', 1)
            clean = getattr(self.inner_dataset, '_cur_yield_clean', True)
            audio_len = sample[1].item() if hasattr(sample[1], 'item') else sample[1]
            duration = audio_len / self.sample_rate

            if self.max_sample_duration is not None and duration > self.max_sample_duration:
                carry_w += weight
                continue

            new_max_dur = max(max_dur_in_batch, duration)
            new_max_tok = max(max_tok_in_batch, self._target_len(sample))
            new_count = len(batch) + 1

            # Flush on budget overflow OR a cleanliness change (keeps batches
            # homogeneous for selective backprop).
            if batch and (
                self._batch_full(new_count, new_max_dur, new_max_tok)
                or clean != batch_clean
            ):
                self._bump_emitted(batch_w)
                yield self._collate(batch, batch_clean)
                batch = []
                batch_w = 0
                max_dur_in_batch = 0.0
                max_tok_in_batch = 0

            batch.append(sample)
            batch_w += weight + carry_w
            batch_clean = clean
            carry_w = 0
            max_dur_in_batch = max(max_dur_in_batch, duration)
            max_tok_in_batch = max(max_tok_in_batch, self._target_len(sample))

        if batch:
            self._bump_emitted(batch_w)
            yield self._collate(batch, batch_clean)
        if carry_w:
            self._bump_emitted(carry_w)

    def _iter_sorted(self):
        """Sort-buffer mode: accumulate, sort by (clean, duration), then pack.

        Sorting on cleanliness first groups all clean samples together and all
        noisy samples together, so duration-based packing stays efficient
        within each group and only the single clean→noisy boundary forces an
        extra (possibly short) batch per buffer flush.
        """
        sort_buffer = []
        carry_w = 0  # weight of dropped samples, folded into the next kept one

        for sample in self.inner_dataset:
            weight = getattr(self.inner_dataset, '_cur_yield_weight', 1)
            clean = getattr(self.inner_dataset, '_cur_yield_clean', True)
            audio_len = sample[1].item() if hasattr(sample[1], 'item') else sample[1]
            duration = audio_len / self.sample_rate

            if self.max_sample_duration is not None and duration > self.max_sample_duration:
                carry_w += weight
                continue
            if duration > self.max_padded_budget_sec:
                carry_w += weight
                continue

            sort_buffer.append((sample, duration, weight + carry_w, clean))
            carry_w = 0

            if len(sort_buffer) >= self.sort_buffer_size:
                sort_buffer.sort(key=lambda x: (x[3], x[1]))
                yield from self._pack_batches(sort_buffer)
                sort_buffer = []

        if sort_buffer:
            sort_buffer.sort(key=lambda x: (x[3], x[1]))
            yield from self._pack_batches(sort_buffer)
        if carry_w:
            self._bump_emitted(carry_w)

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
    per_sample_len = int(max_sample_duration * sample_rate)
    token_count = max(1, int(max_sample_duration * 10))  # ~10 BPE tokens/sec

    # Safety check: estimate if even batch_size=1 fits
    min_needed_gb = 3.0
    if free_gb < min_needed_gb:
        logging.error(f"  Only {free_gb:.1f} GB free, need at least {min_needed_gb:.1f} GB for probing")
        safe_budget = max_sample_duration * 2
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
    torch.cuda.empty_cache()
    base_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    per_sample_cost_est = peak_1 - base_used_gb
    estimated_peak_2 = peak_1 + per_sample_cost_est

    if estimated_peak_2 > total_gb * 0.85:
        logging.info(f"  Skipping batch_size=2 probe (estimated {estimated_peak_2:.1f} GB, too close to limit)")
        per_sample_vram = per_sample_cost_est
    else:
        # Probe 2: batch_size=2 at max_sample_duration
        peak_2 = _run_probe_step(model, 2, per_sample_len, token_count, vocab_size)
        logging.info(f"  Probe: 2 × {max_sample_duration:.0f}s -> {peak_2:.2f} GB")
        per_sample_vram = peak_2 - peak_1

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
