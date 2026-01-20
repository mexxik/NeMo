"""
On-the-fly audio merging dataset for ASR training.

This module provides dataset classes that merge multiple audio samples on-the-fly
during training, avoiding the need to pre-process and store merged audio files.

The merge manifest format uses pipe-separated paths:
{
    "audio_filepath": "audio_1.wav|audio_2.wav|audio_3.wav",
    "shard_id": "0|0|1",
    "tar_dir": "/path/to/tar1|/path/to/tar1|/path/to/tar2",
    "text": "First sentence. Second sentence. Third sentence.",
    "duration": 8.5,
    "durations": "2.1|3.2|3.2",
    "merge_count": 3,
    "lang": "en"
}

Usage:
    # In your training config yaml:
    model:
      train_ds:
        manifest_filepath: /path/to/merged_manifest.json
        is_tarred: true
        merge_audio: true
        merge_silence_min_ms: 200
        merge_silence_max_ms: 1000
"""

import io
import json
import os
import random
import struct
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

__all__ = [
    'MergedTarredAudioToBPEDataset',
]


class TarFileCache:
    """
    Cache for open tarfile handles and their member indices.

    Key optimizations:
    - Caches open file handles to avoid repeated open/close
    - Caches member name -> TarInfo mapping for O(1) lookups instead of O(n)
    - LRU eviction to bound memory usage
    """

    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        self._cache: Dict[str, tarfile.TarFile] = {}
        self._index_cache: Dict[str, Dict[str, tarfile.TarInfo]] = {}
        self._access_order: List[str] = []

    def get(self, tar_path: str) -> Optional[tarfile.TarFile]:
        """Get a tarfile handle, opening if needed."""
        if tar_path in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(tar_path)
            self._access_order.append(tar_path)
            return self._cache[tar_path]

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_open:
            oldest = self._access_order.pop(0)
            try:
                self._cache[oldest].close()
            except Exception:
                pass
            del self._cache[oldest]
            if oldest in self._index_cache:
                del self._index_cache[oldest]

        # Open new tarfile
        try:
            tf = tarfile.open(tar_path, 'r')
            self._cache[tar_path] = tf
            self._access_order.append(tar_path)

            # Build index for O(1) member lookups
            self._index_cache[tar_path] = {m.name: m for m in tf.getmembers()}

            return tf
        except Exception as e:
            logging.warning(f"Failed to open tarfile {tar_path}: {e}")
            return None

    def get_member(self, tar_path: str, member_name: str) -> Optional[tarfile.TarInfo]:
        """Get a TarInfo by name with O(1) lookup from cached index."""
        if tar_path not in self._index_cache:
            # Ensure tar is opened and indexed
            if self.get(tar_path) is None:
                return None

        return self._index_cache.get(tar_path, {}).get(member_name)

    def close_all(self):
        """Close all cached tarfile handles."""
        for tf in self._cache.values():
            try:
                tf.close()
            except Exception:
                pass
        self._cache.clear()
        self._index_cache.clear()
        self._access_order.clear()


def extract_audio_from_tar(
    tar_cache: TarFileCache,
    tar_dir: str,
    shard_id: int,
    audio_filename: str,
) -> Optional[bytes]:
    """
    Extract audio bytes from a tarfile.

    Args:
        tar_cache: TarFileCache instance for caching open tarfiles
        tar_dir: Directory containing tar files
        shard_id: Shard ID (tar file index)
        audio_filename: Filename within the tar

    Returns:
        Audio bytes or None if extraction failed
    """
    tar_path = os.path.join(tar_dir, f"audio_{shard_id}.tar")

    tf = tar_cache.get(tar_path)
    if tf is None:
        return None

    try:
        # Use cached index for O(1) lookup instead of O(n) getmember()
        member = tar_cache.get_member(tar_path, audio_filename)
        if member is None:
            logging.warning(f"Audio file {audio_filename} not found in {tar_path}")
            return None

        file_obj = tf.extractfile(member)
        if file_obj:
            return file_obj.read()
    except Exception as e:
        logging.warning(f"Error extracting {audio_filename} from {tar_path}: {e}")

    return None


def fast_wav_to_float32(wav_bytes: bytes) -> Optional[np.ndarray]:
    """
    Fast WAV to float32 conversion without librosa/soundfile.

    Assumes 16-bit PCM WAV at 16kHz (skips resampling).
    This is ~10x faster than librosa/soundfile for simple WAV files.

    Args:
        wav_bytes: Raw WAV file bytes

    Returns:
        Float32 numpy array normalized to [-1, 1], or None if parsing failed
    """
    try:
        # WAV header: first 44 bytes for standard PCM
        # RIFF header (12 bytes) + fmt chunk (24 bytes) + data header (8 bytes)
        if len(wav_bytes) < 44:
            return None

        # Verify RIFF header
        if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
            return None

        # Find 'data' chunk (may not be at offset 36 if there are extra chunks)
        data_offset = wav_bytes.find(b'data')
        if data_offset == -1:
            return None

        # Read data chunk size (4 bytes after 'data')
        data_size = struct.unpack('<I', wav_bytes[data_offset + 4:data_offset + 8])[0]
        pcm_start = data_offset + 8

        # Extract PCM data
        pcm_bytes = wav_bytes[pcm_start:pcm_start + data_size]

        # Convert 16-bit PCM to float32
        # Using numpy for fast vectorized conversion
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    except Exception:
        return None


def concatenate_audio_tensors(
    audio_tensors: List[torch.Tensor],
    sample_rate: int = 16000,
    silence_min_ms: int = 200,
    silence_max_ms: int = 1000,
) -> torch.Tensor:
    """
    Concatenate audio tensors with random silence gaps.

    Args:
        audio_tensors: List of 1D audio tensors
        sample_rate: Sample rate in Hz
        silence_min_ms: Minimum silence duration in milliseconds
        silence_max_ms: Maximum silence duration in milliseconds

    Returns:
        Concatenated audio tensor
    """
    if len(audio_tensors) == 0:
        return torch.tensor([])

    if len(audio_tensors) == 1:
        return audio_tensors[0]

    chunks = []
    for i, audio in enumerate(audio_tensors):
        chunks.append(audio)

        # Add silence between samples (not after last)
        if i < len(audio_tensors) - 1:
            silence_ms = random.randint(silence_min_ms, silence_max_ms)
            if silence_ms > 0:
                silence_samples = int(sample_rate * silence_ms / 1000)
                silence = torch.zeros(silence_samples, dtype=audio.dtype)
                chunks.append(silence)

    return torch.cat(chunks)


class MergedTarredAudioToBPEDataset(IterableDataset):
    """
    A dataset that supports on-the-fly audio merging from tarred audio files.

    This dataset reads a manifest where each entry contains pipe-separated paths
    pointing to multiple audio files that should be merged during training.
    The audio files are extracted from tar archives and concatenated with
    random silence gaps.

    Unlike the standard TarredAudioToBPEDataset which iterates over tar files,
    this dataset iterates over manifest entries and extracts audio on-demand.

    Manifest format for merged samples:
    {
        "audio_filepath": "audio_1.wav|audio_2.wav",
        "shard_id": "0|0",
        "tar_dir": "/path/to/tar|/path/to/tar",
        "text": "First sentence. Second sentence.",
        "duration": 5.3,
        "merge_count": 2,
        "lang": "en"
    }

    Args:
        manifest_filepath: Path to the merged manifest JSON file
        tokenizer: Tokenizer for encoding text
        sample_rate: Audio sample rate
        merge_silence_min_ms: Minimum silence gap between merged samples (ms)
        merge_silence_max_ms: Maximum silence gap between merged samples (ms)
        shuffle: Whether to shuffle manifest entries
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        global_rank: Current process rank for distributed training
        world_size: Total number of processes for distributed training
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],  # Kept for API compatibility, not used
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trim: bool = False,
        use_start_end_token: bool = True,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
        # Merge-specific parameters
        merge_audio: bool = True,
        merge_silence_min_ms: int = 200,
        merge_silence_max_ms: int = 1000,
    ):
        self.manifest_filepath = manifest_filepath
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.int_values = int_values
        self.augmentor = augmentor
        self.shuffle_n = shuffle_n
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.trim = trim
        self.use_start_end_token = use_start_end_token
        self.global_rank = global_rank
        self.world_size = world_size if world_size > 0 else 1
        self.return_sample_id = return_sample_id
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

        # Load and filter manifest
        self.manifest_entries = self._load_manifest()

        # Per-worker tar cache (initialized lazily)
        self._tar_cache = None

        # Create featurizer for audio processing
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values)

        logging.info(
            f"MergedTarredAudioToBPEDataset: {len(self.manifest_entries)} entries, "
            f"silence={merge_silence_min_ms}-{merge_silence_max_ms}ms"
        )

    def _load_manifest(self) -> List[Dict]:
        """Load and filter manifest entries."""
        entries = []
        with open(self.manifest_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                duration = entry.get('duration', 0)

                # Apply duration filters
                if self.min_duration is not None and duration < self.min_duration:
                    continue
                if self.max_duration is not None and duration > self.max_duration:
                    continue

                entries.append(entry)

        return entries

    def _get_tar_cache(self) -> TarFileCache:
        """Get or create worker-local tar cache."""
        if self._tar_cache is None:
            self._tar_cache = TarFileCache(max_open=64)
        return self._tar_cache

    def __len__(self):
        return len(self.manifest_entries)

    def __iter__(self):
        """Iterate over manifest entries, yielding processed samples."""
        # Get worker info for multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Calculate total number of workers across all ranks
        total_workers = self.world_size * num_workers
        global_worker_id = self.global_rank * num_workers + worker_id

        # Shard entries across workers
        entries = self.manifest_entries[global_worker_id::total_workers]

        # Optional shuffling
        if self.shuffle_n > 0:
            random.shuffle(entries)

        tar_cache = self._get_tar_cache()

        for idx, entry in enumerate(entries):
            sample = self._process_entry(entry, tar_cache, idx)
            if sample is not None:
                yield sample

    def _process_entry(self, entry: Dict, tar_cache: TarFileCache, idx: int):
        """Process a manifest entry and return a training sample."""
        # Parse pipe-separated fields
        audio_filepaths = entry.get('audio_filepath', '').split('|')
        shard_ids_str = entry.get('shard_id', '')
        tar_dirs_str = entry.get('tar_dir', '')

        if '|' in shard_ids_str:
            shard_ids = [int(s) for s in shard_ids_str.split('|')]
        else:
            shard_ids = [int(shard_ids_str)] if shard_ids_str else [0] * len(audio_filepaths)

        if '|' in tar_dirs_str:
            tar_dirs = tar_dirs_str.split('|')
        else:
            tar_dirs = [tar_dirs_str] * len(audio_filepaths)

        # Extract and process each audio file
        audio_arrays = []
        for audio_path, shard_id, tar_dir in zip(audio_filepaths, shard_ids, tar_dirs):
            if not tar_dir or not audio_path:
                continue

            audio_bytes = extract_audio_from_tar(tar_cache, tar_dir, shard_id, audio_path)
            if audio_bytes is None:
                logging.debug(f"Failed to extract {audio_path} from {tar_dir}/audio_{shard_id}.tar")
                continue

            # Fast path: direct WAV parsing (assumes 16kHz 16-bit PCM)
            samples = fast_wav_to_float32(audio_bytes)
            if samples is not None:
                audio_arrays.append(samples)
            else:
                # Fallback to featurizer for non-standard WAV files
                audio_stream = io.BytesIO(audio_bytes)
                try:
                    features = self.featurizer.process(
                        audio_stream,
                        offset=0,
                        duration=None,
                        trim=self.trim,
                        orig_sr=None,
                    )
                    audio_arrays.append(features.numpy())
                except Exception as e:
                    logging.debug(f"Failed to process {audio_path}: {e}")
                finally:
                    audio_stream.close()

        if len(audio_arrays) == 0:
            logging.warning(f"No audio extracted for entry {idx}, skipping")
            return None

        # Fast concatenation with random silence (numpy, then convert to tensor once)
        merged_audio = self._fast_concatenate(audio_arrays)

        # Audio features
        f = torch.from_numpy(merged_audio)
        fl = torch.tensor(merged_audio.shape[0]).long()

        # Tokenize text
        text = entry.get('text', '')
        t = self.tokenizer.text_to_ids(text)

        if self.bos_id is not None:
            t = [self.bos_id] + t
        if self.eos_id is not None:
            t = t + [self.eos_id]

        tl = len(t)

        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def _fast_concatenate(self, audio_arrays: List[np.ndarray]) -> np.ndarray:
        """
        Fast numpy-based audio concatenation with random silence.

        Args:
            audio_arrays: List of float32 numpy arrays

        Returns:
            Concatenated float32 numpy array
        """
        if len(audio_arrays) == 0:
            return np.array([], dtype=np.float32)

        if len(audio_arrays) == 1:
            return audio_arrays[0]

        # Pre-calculate total length for single allocation
        total_samples = sum(arr.shape[0] for arr in audio_arrays)

        # Add silence lengths
        silence_lengths = []
        for _ in range(len(audio_arrays) - 1):
            silence_ms = random.randint(self.merge_silence_min_ms, self.merge_silence_max_ms)
            silence_samples = int(self.sample_rate * silence_ms / 1000)
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
                # Silence is already zeros, just advance offset
                offset += silence_lengths[i]

        return result

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return _speech_collate_fn(batch, pad_id=0)

    def __del__(self):
        """Clean up tar cache on deletion."""
        if self._tar_cache is not None:
            self._tar_cache.close_all()

    @property
    def output_types(self):
        """Define output types for NeMo."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }
