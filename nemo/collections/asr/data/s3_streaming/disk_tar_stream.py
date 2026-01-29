"""
Disk TAR streaming for ASR datasets.

Streams TAR files from local disk storage.
Provides the same interface as S3TarStream for unified usage.
"""

import json
import os
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from nemo.utils import logging

from .s3_tar_stream import fast_wav_to_float32


class DiskTarStream:
    """
    Streams samples from TAR files stored on local disk.

    Provides the same interface as S3TarStream for unified usage.
    """

    def __init__(
        self,
        tar_path: str,
        manifest_entries: Dict[str, dict],
    ):
        """
        Initialize disk TAR streamer.

        Args:
            tar_path: Path to TAR file on disk
            manifest_entries: Dict mapping audio filename to manifest entry
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

        try:
            with tarfile.open(self.tar_path, mode='r:*') as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    filename = member.name

                    # Check if this file is in our manifest
                    if filename not in self.manifest_entries:
                        continue

                    # Extract audio bytes
                    try:
                        audio_file = tar.extractfile(member)
                        if audio_file is None:
                            continue
                        audio_bytes = audio_file.read()
                    except Exception as e:
                        logging.warning(f"Failed to extract {filename}: {e}")
                        continue

                    # Parse WAV to float32
                    audio = fast_wav_to_float32(audio_bytes)
                    if audio is None:
                        logging.debug(f"Failed to parse WAV: {filename}")
                        continue

                    # Get manifest entry
                    entry = self.manifest_entries[filename]

                    yield {
                        'audio': audio,
                        'text': entry.get('text', ''),
                        'duration': entry.get('duration', len(audio) / 16000),
                        'lang': entry.get('lang', 'unknown'),
                        'filename': filename,
                    }

        except Exception as e:
            logging.error(f"Error streaming TAR from {self.tar_path}: {e}")

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

    def list_tar_files(self, source_dir: str) -> List[str]:
        """
        List TAR files in a directory.

        Args:
            source_dir: Directory containing TAR files

        Returns:
            List of TAR file paths sorted by name
        """
        if not os.path.isdir(source_dir):
            logging.warning(f"Directory not found: {source_dir}")
            return []

        tar_files = []
        for filename in os.listdir(source_dir):
            if filename.endswith('.tar'):
                tar_files.append(os.path.join(source_dir, filename))

        return sorted(tar_files)
