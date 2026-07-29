"""
Disk TAR streaming for ASR datasets.

Streams TAR files from local disk storage.
Provides the same interface as S3TarStream for unified usage.
"""

import json
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from nemo.utils import logging

from .s3_tar_stream import fast_wav_to_float32


# Optional per-tar timing diagnostic. Enable with DATASET_PROFILE=1.
# When on, every tar stream emits log lines with the breakdown of where
# per-member time goes: tar iter, manifest lookup, extractfile, WAV decode.
# Lines are emitted periodically (every DATASET_PROFILE_EVERY members,
# default 2000) AND at tar completion. Lets us see whether the wall is
# SQLite/manifest queries, tar header parsing, or audio decode without
# waiting minutes for a 3GB tar to drain.
_DATASET_PROFILE = os.environ.get('DATASET_PROFILE', '0') == '1'
try:
    _DATASET_PROFILE_EVERY = int(os.environ.get('DATASET_PROFILE_EVERY', '2000'))
except ValueError:
    _DATASET_PROFILE_EVERY = 2000


class DiskTarStream:
    """
    Streams samples from TAR files stored on local disk.

    Provides the same interface as S3TarStream for unified usage.
    Supports both dict and SQLiteManifestProvider for manifest lookups.
    """

    def __init__(
        self,
        tar_path: str,
        manifest_entries: Union[Dict[str, dict], "SQLiteManifestProvider"],
    ):
        """
        Initialize disk TAR streamer.

        Args:
            tar_path: Path to TAR file on disk
            manifest_entries: Dict or SQLiteManifestProvider for manifest lookups.
                              Must support `in` operator and `[]` access.
        """
        self.tar_path = tar_path
        self.manifest_entries = manifest_entries
        self._exhausted = False

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over samples in the TAR file.

        Yields:
            Dict with keys: audio (np.array), text (str), duration (float), lang (str), filename (str)
        """
        self._exhausted = False

        if not os.path.exists(self.tar_path):
            logging.error(f"TAR file not found: {self.tar_path}")
            self._exhausted = True
            return

        # Timing buckets — only populated when DATASET_PROFILE=1.
        members_seen = 0
        members_matched = 0
        t_iter = 0.0
        t_lookup = 0.0
        t_extract = 0.0
        t_decode = 0.0
        t_start = time.perf_counter() if _DATASET_PROFILE else 0.0
        last_emit_seen = 0

        def _emit_profile(tag: str):
            if not _DATASET_PROFILE or members_seen == 0:
                return
            tot = time.perf_counter() - t_start
            other = max(0.0, tot - (t_iter + t_lookup + t_extract + t_decode))
            try:
                wid = -1
                import torch.utils.data as _tud
                wi = _tud.get_worker_info()
                if wi is not None:
                    wid = wi.id
            except Exception:
                wid = -1
            logging.info(
                f"[TAR_PROF wid={wid} {tag}] {os.path.basename(self.tar_path)} "
                f"seen={members_seen} matched={members_matched} "
                f"iter={t_iter*1000:.0f}ms "
                f"lookup={t_lookup*1000:.0f}ms ({t_lookup/max(members_seen,1)*1e6:.0f}us/member) "
                f"extract={t_extract*1000:.0f}ms "
                f"decode={t_decode*1000:.0f}ms "
                f"other={other*1000:.0f}ms total={tot*1000:.0f}ms"
            )

        try:
            with tarfile.open(self.tar_path, mode='r:*') as tar:
                tar_iter = iter(tar)
                while True:
                    if _DATASET_PROFILE:
                        _t0 = time.perf_counter()
                    try:
                        member = next(tar_iter)
                    except StopIteration:
                        if _DATASET_PROFILE:
                            t_iter += time.perf_counter() - _t0
                        break
                    except Exception:
                        if _DATASET_PROFILE:
                            t_iter += time.perf_counter() - _t0
                        break
                    if _DATASET_PROFILE:
                        t_iter += time.perf_counter() - _t0
                    if not member.isfile():
                        continue
                    members_seen += 1

                    filename = member.name

                    if _DATASET_PROFILE:
                        _t0 = time.perf_counter()
                    in_manifest = filename in self.manifest_entries
                    if _DATASET_PROFILE:
                        t_lookup += time.perf_counter() - _t0
                    if not in_manifest:
                        continue
                    members_matched += 1

                    # Extract audio bytes
                    try:
                        if _DATASET_PROFILE:
                            _t0 = time.perf_counter()
                        audio_file = tar.extractfile(member)
                        if audio_file is None:
                            continue
                        audio_bytes = audio_file.read()
                        if _DATASET_PROFILE:
                            t_extract += time.perf_counter() - _t0
                    except Exception as e:
                        logging.warning(f"Failed to extract {filename}: {e}")
                        continue

                    if _DATASET_PROFILE:
                        _t0 = time.perf_counter()
                    audio = fast_wav_to_float32(audio_bytes)
                    if _DATASET_PROFILE:
                        t_decode += time.perf_counter() - _t0
                    if audio is None:
                        logging.debug(f"Failed to parse WAV: {filename}")
                        continue

                    if _DATASET_PROFILE:
                        _t0 = time.perf_counter()
                    entry = self.manifest_entries[filename]
                    if _DATASET_PROFILE:
                        t_lookup += time.perf_counter() - _t0

                    yield {
                        'audio': audio,
                        'text': entry.get('text', ''),
                        'duration': entry.get('duration', len(audio) / 16000),
                        'lang': entry.get('lang', 'unknown'),
                        'phonemes': entry.get('phonemes'),
                        'filename': filename,
                    }

                    if _DATASET_PROFILE and members_seen - last_emit_seen >= _DATASET_PROFILE_EVERY:
                        _emit_profile("partial")
                        last_emit_seen = members_seen

        except Exception as e:
            logging.error(f"Error streaming TAR from {self.tar_path}: {e}")

        _emit_profile("done")

        self._exhausted = True

    @property
    def exhausted(self) -> bool:
        """Check if stream is exhausted."""
        return self._exhausted


class DiskManifestLoader:
    """
    Loads manifest files from local disk.

    Provides the same interface as S3ManifestLoader for unified usage.
    """

    def __init__(self):
        """Initialize disk manifest loader."""
        pass

    def load_manifest(
        self,
        manifest_path: str,
    ) -> Dict[str, dict]:
        """
        Load manifest from disk and return filename -> entry mapping.

        Args:
            manifest_path: Path to manifest file

        Returns:
            Dict mapping audio filename to manifest entry
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        logging.info(f"Loading manifest from {manifest_path}")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            content = f.read()

        entries = self._parse_manifest_content(content)
        logging.info(f"Loaded {len(entries)} manifest entries from disk")
        return entries

    def _parse_manifest_content(self, content: str) -> Dict[str, dict]:
        """Parse manifest content into filename -> entry mapping."""
        entries = {}
        for line in content.strip().split('\n'):
            if not line:
                continue
            try:
                entry = json.loads(line)
                filename = entry.get('audio_filepath', '')
                if filename:
                    entries[filename] = entry
            except json.JSONDecodeError:
                continue
        return entries

    def count_manifest_entries(self, manifest_path: str) -> int:
        """
        Count entries in a manifest file.

        Args:
            manifest_path: Path to manifest file

        Returns:
            Number of entries (non-empty lines) in the manifest
        """
        if not os.path.exists(manifest_path):
            logging.warning(f"Manifest not found: {manifest_path}")
            return 0

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                content = f.read()
            count = sum(1 for line in content.strip().split('\n') if line.strip())
            return count
        except Exception as e:
            logging.warning(f"Failed to count entries in {manifest_path}: {e}")
            return 0

    def list_tar_files(self, source_dir: str, sqlite_cache=None, source: str = None) -> List[str]:
        """
        List TAR files in a directory.

        Args:
            source_dir: Directory containing TAR files
            sqlite_cache: Optional SQLiteManifestCache for caching TAR file lists
            source: Source name for cache key (required if sqlite_cache is provided)

        Returns:
            List of TAR file paths sorted by name
        """
        # Try to get from SQLite cache first
        if sqlite_cache is not None and source is not None:
            cached_tars = sqlite_cache.get_tar_files(source)
            if cached_tars is not None:
                return cached_tars

        if not os.path.isdir(source_dir):
            logging.warning(f"Directory not found: {source_dir}")
            return []

        tar_files = []
        for filename in os.listdir(source_dir):
            if filename.endswith('.tar'):
                tar_files.append(os.path.join(source_dir, filename))

        tar_files = sorted(tar_files)

        # Cache to SQLite if available
        if sqlite_cache is not None and source is not None and tar_files:
            try:
                sqlite_cache.add_tar_files(source, tar_files)
            except Exception as e:
                logging.warning(f"[{source}] Failed to cache TAR files: {e}")

        return tar_files
