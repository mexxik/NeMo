"""
Streaming WebDataset-based audio merging dataset for ASR training.

This module provides a FAST dataset that merges multiple audio samples on-the-fly
using WebDataset's streaming approach instead of random tar file access.

Key optimization: Instead of randomly accessing tar files for each manifest entry,
we stream all tar files sequentially and pair samples from shuffle buffers.

Usage:
    model:
      train_ds:
        is_tarred: true
        tarred_audio_filepaths: /path/to/audio_*.tar
        manifest_filepath: /path/to/manifest.json
        merge_audio: true
        merge_streams: 2  # Number of audio streams to merge
        merge_silence_min_ms: 200
        merge_silence_max_ms: 1000
"""

import io
import random
import struct
from collections import deque
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.data.utils import expand_sharded_filepaths
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging
from nemo.utils.distributed import webdataset_split_by_workers

__all__ = [
    'StreamingMergedTarredAudioToBPEDataset',
]


def fast_wav_to_float32(wav_bytes: bytes) -> Optional[np.ndarray]:
    """
    Fast WAV to float32 extraction without librosa/soundfile.

    Assumes 16-bit PCM WAV (standard for NeMo datasets).

    Args:
        wav_bytes: Raw WAV file bytes

    Returns:
        Float32 numpy array normalized to [-1, 1], or None if parsing failed
    """
    try:
        if len(wav_bytes) < 44:
            return None

        # Verify RIFF header
        if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
            return None

        # Find 'data' chunk
        data_offset = wav_bytes.find(b'data')
        if data_offset == -1:
            return None

        # Read data size and extract PCM
        data_size = struct.unpack('<I', wav_bytes[data_offset + 4:data_offset + 8])[0]
        pcm_start = data_offset + 8
        pcm_bytes = wav_bytes[pcm_start:pcm_start + data_size]

        # Convert int16 to float32 in one step
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    except Exception:
        return None


def concatenate_audio_arrays(
    audio_arrays: List[np.ndarray],
    sample_rate: int = 16000,
    silence_min_ms: int = 200,
    silence_max_ms: int = 1000,
) -> np.ndarray:
    """
    Fast concatenation of audio arrays with random silence gaps.

    Uses single allocation for efficiency.
    """
    if len(audio_arrays) == 0:
        return np.array([], dtype=np.float32)

    if len(audio_arrays) == 1:
        return audio_arrays[0]

    # Pre-calculate total length
    total_samples = sum(arr.shape[0] for arr in audio_arrays)
    silence_lengths = []
    for _ in range(len(audio_arrays) - 1):
        silence_ms = random.randint(silence_min_ms, silence_max_ms)
        silence_samples = int(sample_rate * silence_ms / 1000)
        silence_lengths.append(silence_samples)
        total_samples += silence_samples

    # Single allocation
    result = np.zeros(total_samples, dtype=np.float32)

    # Copy data
    offset = 0
    for i, arr in enumerate(audio_arrays):
        result[offset:offset + arr.shape[0]] = arr
        offset += arr.shape[0]
        if i < len(silence_lengths):
            offset += silence_lengths[i]

    return result


class ShuffleBuffer:
    """
    A shuffle buffer that accumulates samples and yields them in random order.

    This enables randomization while maintaining streaming efficiency.
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer: List = []
        self.exhausted = False

    def add(self, item):
        """Add an item to the buffer."""
        self.buffer.append(item)

    def is_ready(self) -> bool:
        """Check if buffer has enough items to start yielding."""
        return len(self.buffer) >= self.buffer_size or self.exhausted

    def mark_exhausted(self):
        """Mark the source as exhausted (no more items coming)."""
        self.exhausted = True

    def get_random(self):
        """Get and remove a random item from the buffer."""
        if not self.buffer:
            return None
        idx = random.randint(0, len(self.buffer) - 1)
        # Swap with last and pop for O(1) removal
        self.buffer[idx], self.buffer[-1] = self.buffer[-1], self.buffer[idx]
        return self.buffer.pop()

    def __len__(self):
        return len(self.buffer)


class StreamingMergedTarredAudioToBPEDataset(IterableDataset):
    """
    A FAST streaming dataset that merges audio samples from tarred datasets.

    Key difference from MergedTarredAudioToBPEDataset:
    - Uses WebDataset to stream tar files sequentially (fast)
    - Pairs samples from shuffle buffers instead of random tar access (fast)
    - No pre-computed merge manifest needed

    How it works:
    1. Stream all tar files sequentially using WebDataset
    2. Accumulate samples in a shuffle buffer
    3. When buffer is full, randomly pick N samples to merge
    4. Concatenate audio and text, yield merged sample

    Args:
        audio_tar_filepaths: Glob pattern or list of tar file paths
        manifest_filepath: Path to manifest (for text lookup)
        tokenizer: Tokenizer for encoding text
        sample_rate: Audio sample rate
        merge_count: Number of audio samples to merge (default: 2)
        merge_buffer_size: Size of shuffle buffer for random pairing
        merge_silence_min_ms: Minimum silence gap between merged samples
        merge_silence_max_ms: Maximum silence gap between merged samples
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 2048,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_merged_duration: Optional[float] = 30.0,
        trim: bool = False,
        use_start_end_token: bool = True,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
        # Merge-specific parameters
        merge_count: int = 2,
        merge_buffer_size: int = 2000,
        merge_silence_min_ms: int = 200,
        merge_silence_max_ms: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.int_values = int_values
        self.augmentor = augmentor
        self.shuffle_n = shuffle_n
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_merged_duration = max_merged_duration
        self.trim = trim
        self.use_start_end_token = use_start_end_token
        self.global_rank = global_rank
        self.world_size = world_size if world_size > 0 else 1
        self.return_sample_id = return_sample_id

        # Merge parameters
        self.merge_count = merge_count
        self.merge_buffer_size = merge_buffer_size
        self.merge_silence_min_ms = merge_silence_min_ms
        self.merge_silence_max_ms = merge_silence_max_ms

        # BOS/EOS tokens
        if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id is not None:
            self.bos_id = tokenizer.bos_id if use_start_end_token else None
        else:
            self.bos_id = None
        if hasattr(tokenizer, 'eos_id') and tokenizer.eos_id is not None:
            self.eos_id = tokenizer.eos_id if use_start_end_token else None
        else:
            self.eos_id = None

        # Load manifest for text lookup (file_id -> text mapping)
        self.text_mapping = self._load_manifest_text(manifest_filepath)

        # Expand tar file paths
        self.audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

        # Featurizer for fallback audio processing
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        logging.info(
            f"StreamingMergedTarredAudioToBPEDataset: {len(self.audio_tar_filepaths)} tar files, "
            f"merge_count={merge_count}, buffer_size={merge_buffer_size}, "
            f"silence={merge_silence_min_ms}-{merge_silence_max_ms}ms"
        )

    def _load_manifest_text(self, manifest_filepath: str) -> Dict[str, str]:
        """Load manifest and create file_id -> text mapping."""
        import json
        import os

        text_mapping = {}
        with open(manifest_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_path = entry.get('audio_filepath', '')
                # Handle pipe-separated paths (take first for lookup)
                if '|' in audio_path:
                    audio_path = audio_path.split('|')[0]
                file_id = os.path.splitext(os.path.basename(audio_path))[0]
                text_mapping[file_id] = entry.get('text', '')

        return text_mapping

    def _extract_audio_from_sample(self, sample: dict) -> Optional[dict]:
        """Extract audio bytes and key from a WebDataset sample."""
        # Standard webdataset format
        for ext in ['wav', 'mp3', 'flac', 'opus', 'ogg']:
            if ext in sample:
                return {'audio': sample[ext], 'key': sample.get('__key__', '')}

        # Custom format: key ends with audio extension
        for key, value in sample.items():
            if key.startswith('__'):
                continue
            key_lower = key.lower()
            if any(key_lower.endswith(f'.{ext}') for ext in ['wav', 'mp3', 'flac', 'opus', 'ogg']):
                return {'audio': value, 'key': key}

        return None

    def _process_audio_bytes(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Convert audio bytes to numpy array."""
        # Fast path: direct WAV parsing
        audio = fast_wav_to_float32(audio_bytes)
        if audio is not None:
            return audio

        # Fallback: use featurizer
        try:
            audio_stream = io.BytesIO(audio_bytes)
            features = self.featurizer.process(audio_stream, offset=0, duration=None, trim=self.trim)
            audio_stream.close()
            return features.numpy()
        except Exception:
            return None

    def _create_webdataset_pipeline(self) -> wds.DataPipeline:
        """Create WebDataset pipeline for streaming tar files."""
        return wds.DataPipeline(
            wds.SimpleShardList(urls=self.audio_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(self.shuffle_n),
            wds.tarfile_to_samples(),
        )

    def __iter__(self) -> Iterator:
        """
        Iterate over merged samples.

        Strategy:
        1. Stream samples from WebDataset into a shuffle buffer
        2. When buffer has enough samples, randomly pick merge_count samples
        3. Merge audio and text, yield the merged sample
        """
        pipeline = self._create_webdataset_pipeline()
        buffer = ShuffleBuffer(buffer_size=self.merge_buffer_size)

        sample_idx = 0

        # Stream samples into buffer
        for raw_sample in pipeline:
            extracted = self._extract_audio_from_sample(raw_sample)
            if extracted is None:
                continue

            audio_bytes = extracted['audio']
            file_key = extracted['key']

            # Get file_id for text lookup
            import os
            file_id = os.path.splitext(os.path.basename(file_key))[0]

            # Process audio
            audio = self._process_audio_bytes(audio_bytes)
            if audio is None:
                continue

            # Apply duration filters
            duration = len(audio) / self.sample_rate
            if self.min_duration is not None and duration < self.min_duration:
                continue
            if self.max_duration is not None and duration > self.max_duration:
                continue

            # Get text
            text = self.text_mapping.get(file_id, '')

            # Add to buffer
            buffer.add({
                'audio': audio,
                'text': text,
                'duration': duration,
                'file_id': file_id,
            })

            # When buffer is ready, yield merged samples
            while len(buffer) >= self.merge_count and buffer.is_ready():
                merged_sample = self._create_merged_sample(buffer, sample_idx)
                if merged_sample is not None:
                    yield merged_sample
                    sample_idx += 1

        # Mark buffer exhausted and drain remaining samples
        buffer.mark_exhausted()
        while len(buffer) >= self.merge_count:
            merged_sample = self._create_merged_sample(buffer, sample_idx)
            if merged_sample is not None:
                yield merged_sample
                sample_idx += 1

        # Handle remaining samples (less than merge_count)
        # Option 1: yield them as single samples
        # Option 2: discard them
        # We'll yield them as smaller merged batches
        while len(buffer) > 0:
            count = min(len(buffer), self.merge_count)
            samples = [buffer.get_random() for _ in range(count)]
            samples = [s for s in samples if s is not None]
            if samples:
                merged = self._merge_samples(samples, sample_idx)
                if merged is not None:
                    yield merged
                    sample_idx += 1

    def _create_merged_sample(self, buffer: ShuffleBuffer, sample_idx: int):
        """Create a merged sample from buffer."""
        samples = []
        total_duration = 0

        # Pick samples that fit within max_merged_duration
        attempts = 0
        while len(samples) < self.merge_count and attempts < self.merge_count * 3:
            attempts += 1
            sample = buffer.get_random()
            if sample is None:
                break

            # Check if adding this sample would exceed max duration
            if self.max_merged_duration is not None:
                if total_duration + sample['duration'] > self.max_merged_duration:
                    # Put it back and try another
                    buffer.add(sample)
                    continue

            samples.append(sample)
            total_duration += sample['duration']

        if len(samples) < self.merge_count:
            # Not enough samples, put them back
            for s in samples:
                buffer.add(s)
            return None

        return self._merge_samples(samples, sample_idx)

    def _merge_samples(self, samples: List[dict], sample_idx: int):
        """Merge multiple samples into one training sample."""
        if not samples:
            return None

        # Concatenate audio
        audio_arrays = [s['audio'] for s in samples]
        merged_audio = concatenate_audio_arrays(
            audio_arrays,
            sample_rate=self.sample_rate,
            silence_min_ms=self.merge_silence_min_ms,
            silence_max_ms=self.merge_silence_max_ms,
        )

        # Concatenate text with space separator
        texts = [s['text'] for s in samples]
        merged_text = ' '.join(texts)

        # Audio features
        f = torch.from_numpy(merged_audio)
        fl = torch.tensor(merged_audio.shape[0]).long()

        # Tokenize text
        t = self.tokenizer.text_to_ids(merged_text)

        if self.bos_id is not None:
            t = [self.bos_id] + t
        if self.eos_id is not None:
            t = t + [self.eos_id]

        tl = len(t)

        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), sample_idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return _speech_collate_fn(batch, pad_id=0)

    def __len__(self):
        """Approximate length (actual may vary due to merging)."""
        return len(self.text_mapping) // self.merge_count

    @property
    def output_types(self):
        """Define output types for NeMo."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }
