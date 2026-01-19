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
import os
import random
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import (
    TarredAudioToBPEDataset,
    _TarredAudioToTextDataset,
    _speech_collate_fn,
)
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

__all__ = [
    'MergedTarredAudioToBPEDataset',
]


class TarFileCache:
    """
    Cache for open tarfile handles to avoid repeated open/close overhead.
    Thread-local to support multiple DataLoader workers.
    """

    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        self._cache: Dict[str, tarfile.TarFile] = {}
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

        # Open new tarfile
        try:
            tf = tarfile.open(tar_path, 'r')
            self._cache[tar_path] = tf
            self._access_order.append(tar_path)
            return tf
        except Exception as e:
            logging.warning(f"Failed to open tarfile {tar_path}: {e}")
            return None

    def close_all(self):
        """Close all cached tarfile handles."""
        for tf in self._cache.values():
            try:
                tf.close()
            except Exception:
                pass
        self._cache.clear()
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
        member = tf.getmember(audio_filename)
        file_obj = tf.extractfile(member)
        if file_obj:
            return file_obj.read()
    except KeyError:
        logging.warning(f"Audio file {audio_filename} not found in {tar_path}")
    except Exception as e:
        logging.warning(f"Error extracting {audio_filename} from {tar_path}: {e}")

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


class MergedTarredAudioToBPEDataset(TarredAudioToBPEDataset):
    """
    A TarredAudioToBPEDataset that supports on-the-fly audio merging.

    When merge_audio=True, the manifest should contain pipe-separated paths
    that will be merged during training. This avoids pre-processing overhead
    and allows different silence gaps each epoch (data augmentation).

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
        merge_audio: Whether to enable on-the-fly merging
        merge_silence_min_ms: Minimum silence gap between merged samples (ms)
        merge_silence_max_ms: Maximum silence gap between merged samples (ms)
        ... (all other args from TarredAudioToBPEDataset)
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
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
        merge_audio: bool = False,
        merge_silence_min_ms: int = 200,
        merge_silence_max_ms: int = 1000,
    ):
        self.merge_audio = merge_audio
        self.merge_silence_min_ms = merge_silence_min_ms
        self.merge_silence_max_ms = merge_silence_max_ms

        # Per-worker tar cache (initialized lazily)
        self._tar_cache = None

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            use_start_end_token=use_start_end_token,
            shard_strategy=shard_strategy,
            shard_manifests=shard_manifests,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
            manifest_parse_func=manifest_parse_func,
        )

        if self.merge_audio:
            logging.info(
                f"On-the-fly audio merging enabled: "
                f"silence={merge_silence_min_ms}-{merge_silence_max_ms}ms"
            )

    def _get_tar_cache(self) -> TarFileCache:
        """Get or create worker-local tar cache."""
        if self._tar_cache is None:
            self._tar_cache = TarFileCache(max_open=32)
        return self._tar_cache

    def _build_sample(self, tup):
        """
        Build training sample, with optional on-the-fly merging.

        If merge_audio is enabled and the manifest entry has pipe-separated paths,
        this will extract multiple audio files and concatenate them with silence.
        """
        audio_bytes, audio_filename, offset_id = tup

        # Get manifest entry
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]

        # Check if this is a merged sample
        audio_filepath = getattr(manifest_entry, 'audio_filepath', audio_filename)

        if self.merge_audio and '|' in str(audio_filepath):
            # This is a merged sample - extract and concatenate multiple audio files
            return self._build_merged_sample(manifest_entry, manifest_idx)
        else:
            # Regular sample - use parent implementation
            return super()._build_sample(tup)

    def _build_merged_sample(self, manifest_entry, manifest_idx):
        """
        Build a merged sample by extracting and concatenating multiple audio files.
        """
        # Parse pipe-separated fields
        audio_filepaths = str(manifest_entry.audio_filepath).split('|')

        # Get shard_ids and tar_dirs if available
        shard_ids_str = getattr(manifest_entry, 'shard_id', None)
        tar_dirs_str = getattr(manifest_entry, 'tar_dir', None)

        if shard_ids_str and '|' in str(shard_ids_str):
            shard_ids = [int(s) for s in str(shard_ids_str).split('|')]
        else:
            shard_ids = [0] * len(audio_filepaths)

        if tar_dirs_str and '|' in str(tar_dirs_str):
            tar_dirs = str(tar_dirs_str).split('|')
        else:
            # Fallback: try to infer from audio_tar_filepaths
            tar_dirs = [os.path.dirname(self.audio_tar_filepaths[0])] * len(audio_filepaths)

        tar_cache = self._get_tar_cache()

        # Extract and process each audio file
        audio_tensors = []
        for audio_path, shard_id, tar_dir in zip(audio_filepaths, shard_ids, tar_dirs):
            audio_bytes = extract_audio_from_tar(tar_cache, tar_dir, shard_id, audio_path)
            if audio_bytes is None:
                logging.warning(f"Failed to extract {audio_path}, skipping merge")
                continue

            # Process audio bytes through featurizer
            audio_stream = io.BytesIO(audio_bytes)
            try:
                features = self.featurizer.process(
                    audio_stream,
                    offset=0,
                    duration=None,
                    trim=self.trim,
                    orig_sr=None,
                )
                audio_tensors.append(features)
            except Exception as e:
                logging.warning(f"Failed to process {audio_path}: {e}")
            finally:
                audio_stream.close()

        if len(audio_tensors) == 0:
            logging.warning("No audio extracted for merged sample, returning None")
            return None

        # Concatenate with random silence
        merged_audio = concatenate_audio_tensors(
            audio_tensors,
            sample_rate=self.featurizer.sample_rate,
            silence_min_ms=self.merge_silence_min_ms,
            silence_max_ms=self.merge_silence_max_ms,
        )

        # Audio features
        f = merged_audio
        fl = torch.tensor(merged_audio.shape[0]).long()

        # Text features (already combined in manifest)
        t = manifest_entry.text_tokens
        tl = len(manifest_entry.text_tokens)

        self.manifest_processor.process_text_by_sample(sample=manifest_entry)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), manifest_idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __del__(self):
        """Clean up tar cache on deletion."""
        if self._tar_cache is not None:
            self._tar_cache.close_all()
