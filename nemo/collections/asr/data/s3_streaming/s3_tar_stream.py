"""
S3 TAR streaming for ASR datasets.

Streams TAR files directly from S3 without downloading the entire file.
Parses TAR members on-the-fly and extracts audio samples.
"""

import io
import json
import struct
import tarfile
import time
from typing import Dict, Iterator, Optional, Union

import numpy as np

from nemo.utils import logging

try:
    import boto3
    from botocore.config import Config as BotoConfig
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


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


class S3TarStream:
    """
    Streams samples from TAR files stored on S3.

    Features:
    - Streams TAR content without downloading entire file
    - Parses TAR headers on-the-fly
    - Matches audio files to manifest entries
    - Handles network retries
    - Supports both dict and SQLiteManifestProvider for manifest lookups
    """

    def __init__(
        self,
        s3_bucket: str,
        tar_key: str,
        manifest_entries: Union[Dict[str, dict], "SQLiteManifestProvider"],
        s3_client=None,
        aws_region: str = "us-east-1",
        max_retries: int = 3,
    ):
        """
        Initialize S3 TAR streamer.

        Args:
            s3_bucket: S3 bucket name
            tar_key: Key (path) to TAR file in bucket
            manifest_entries: Dict or SQLiteManifestProvider for manifest lookups.
                              Must support `in` operator and `[]` access.
            s3_client: Optional pre-configured boto3 S3 client
            aws_region: AWS region (if creating new client)
            max_retries: Max retries for S3 operations
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 streaming. Install with: pip install boto3")

        self.s3_bucket = s3_bucket
        self.tar_key = tar_key
        self.manifest_entries = manifest_entries
        self.max_retries = max_retries

        if s3_client is not None:
            self.s3_client = s3_client
        else:
            boto_config = BotoConfig(
                region_name=aws_region,
                retries={'max_attempts': max_retries, 'mode': 'adaptive'},
                connect_timeout=10,
                read_timeout=30,
            )
            self.s3_client = boto3.client('s3', config=boto_config)

        self._exhausted = False

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over samples in the TAR file.

        Yields:
            Dict with keys: audio (np.array), text (str), duration (float), lang (str), filename (str)
        """
        self._exhausted = False
        processed_files = set()  # Track files we've already yielded

        for attempt in range(self.max_retries):
            try:
                # Get the TAR file as a streaming response
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.tar_key)
                body = response['Body']

                # Wrap in a file-like object for tarfile
                tar_stream = S3StreamWrapper(body)

                # Open as tarfile in streaming mode
                total_files_in_tar = 0
                matched_files_in_tar = 0
                with tarfile.open(fileobj=tar_stream, mode='r|*') as tar:
                    for member in tar:
                        if not member.isfile():
                            continue

                        total_files_in_tar += 1
                        filename = member.name

                        # Skip already processed files (from previous retry)
                        if filename in processed_files:
                            continue

                        # Check if this file is in our manifest
                        if filename not in self.manifest_entries:
                            continue

                        matched_files_in_tar += 1

                        # Extract audio bytes
                        try:
                            audio_file = tar.extractfile(member)
                            if audio_file is None:
                                continue
                            audio_bytes = audio_file.read()
                        except Exception as e:
                            # Connection error during extraction - raise to trigger retry
                            if 'IncompleteRead' in str(e) or 'Connection' in str(e):
                                raise
                            logging.warning(f"Failed to extract {filename}: {e}")
                            continue

                        # Parse WAV to float32
                        audio = fast_wav_to_float32(audio_bytes)
                        if audio is None:
                            continue

                        # Get manifest entry
                        entry = self.manifest_entries[filename]

                        # Mark as processed before yielding
                        processed_files.add(filename)

                        yield {
                            'audio': audio,
                            'text': entry.get('text', ''),
                            'duration': entry.get('duration', len(audio) / 16000),
                            'lang': entry.get('lang', 'unknown'),
                            'filename': filename,
                        }

                # Successfully completed - exit retry loop
                break

            except Exception as e:
                error_str = str(e)
                is_connection_error = 'IncompleteRead' in error_str or 'Connection' in error_str

                if is_connection_error and attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    sleep_time = 2 ** attempt
                    logging.warning(
                        f"Connection error on {self.tar_key}, retry {attempt + 1}/{self.max_retries} "
                        f"in {sleep_time}s (processed {len(processed_files)} files so far)"
                    )
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Error streaming TAR from s3://{self.s3_bucket}/{self.tar_key}: {e}")
                    break

        self._exhausted = True

    @property
    def exhausted(self) -> bool:
        """Check if stream is exhausted."""
        return self._exhausted


class S3StreamWrapper:
    """
    Wraps boto3 StreamingBody to provide file-like interface for tarfile.

    tarfile requires read() and close() methods.
    """

    def __init__(self, streaming_body):
        self.body = streaming_body
        self._buffer = b''

    def read(self, size=-1):
        """Read bytes from the stream."""
        if size == -1:
            # Read all remaining
            return self._buffer + self.body.read()

        # Read from buffer first
        if len(self._buffer) >= size:
            result = self._buffer[:size]
            self._buffer = self._buffer[size:]
            return result

        # Need more data
        needed = size - len(self._buffer)
        new_data = self.body.read(needed)
        result = self._buffer + new_data
        self._buffer = b''
        return result

    def close(self):
        """Close the stream."""
        self.body.close()


class S3ManifestLoader:
    """
    Loads manifest files from S3 with local caching.
    """

    # Class-level cache directory
    _default_cache_dir = None

    def __init__(self, s3_client=None, aws_region: str = "us-east-1", cache_dir: str = None):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 streaming. Install with: pip install boto3")

        if s3_client is not None:
            self.s3_client = s3_client
        else:
            self.s3_client = boto3.client('s3', region_name=aws_region)

        # Set cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        elif S3ManifestLoader._default_cache_dir:
            self.cache_dir = S3ManifestLoader._default_cache_dir
        else:
            # Default to ~/.cache/nemo_s3_manifests
            import os
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nemo_s3_manifests")

        # Create cache directory if needed
        import os
        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def set_default_cache_dir(cls, cache_dir: str):
        """Set default cache directory for all instances."""
        cls._default_cache_dir = cache_dir

    def _get_cache_path(self, s3_bucket: str, manifest_key: str) -> str:
        """Get local cache path for a manifest."""
        import os
        import hashlib
        # Create a safe filename from bucket and key
        cache_key = f"{s3_bucket}/{manifest_key}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        safe_name = manifest_key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{cache_hash}_{safe_name}")

    def load_manifest(
        self,
        s3_bucket: str,
        manifest_key: str,
        use_cache: bool = True,
    ) -> Dict[str, dict]:
        """
        Load manifest from S3 (or local cache) and return filename -> entry mapping.

        Args:
            s3_bucket: S3 bucket name
            manifest_key: Key (path) to manifest file
            use_cache: If True, use local cache when available

        Returns:
            Dict mapping audio filename to manifest entry
        """
        import os

        cache_path = self._get_cache_path(s3_bucket, manifest_key)

        # Try to load from cache first
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                entries = self._parse_manifest_content(content)
                logging.debug(f"Loaded {len(entries)} manifest entries from cache: {cache_path}")
                return entries
            except Exception as e:
                logging.warning(f"Failed to load cached manifest, fetching from S3: {e}")

        # Download from S3
        logging.info(f"Downloading manifest from s3://{s3_bucket}/{manifest_key}")
        response = self.s3_client.get_object(Bucket=s3_bucket, Key=manifest_key)
        content = response['Body'].read().decode('utf-8')

        # Save to cache
        if use_cache:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.debug(f"Cached manifest to: {cache_path}")
            except Exception as e:
                logging.warning(f"Failed to cache manifest: {e}")

        entries = self._parse_manifest_content(content)
        logging.info(f"Loaded {len(entries)} manifest entries from S3")
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

    def count_manifest_entries(
        self,
        s3_bucket: str,
        manifest_key: str,
        use_cache: bool = True,
    ) -> int:
        """
        Count entries in a manifest file.

        Uses local cache when available to avoid re-downloading.

        Args:
            s3_bucket: S3 bucket name
            manifest_key: Key (path) to manifest file
            use_cache: If True, use local cache when available

        Returns:
            Number of entries (non-empty lines) in the manifest
        """
        import os

        cache_path = self._get_cache_path(s3_bucket, manifest_key)

        try:
            # Try cache first
            if use_cache and os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Download from S3
                response = self.s3_client.get_object(Bucket=s3_bucket, Key=manifest_key)
                content = response['Body'].read().decode('utf-8')

                # Save to cache
                if use_cache:
                    try:
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    except Exception:
                        pass  # Ignore cache write errors

            # Count non-empty lines
            count = sum(1 for line in content.strip().split('\n') if line.strip())
            return count
        except Exception as e:
            logging.warning(f"Failed to count entries in s3://{s3_bucket}/{manifest_key}: {e}")
            return 0

    def list_tar_files(self, s3_bucket: str, prefix: str, sqlite_cache=None, source: str = None) -> list:
        """
        List TAR files in an S3 prefix.

        Args:
            s3_bucket: S3 bucket name
            prefix: S3 prefix (e.g., "asr-data/common_en_train/")
            sqlite_cache: Optional SQLiteManifestCache for caching TAR file lists
            source: Source name for cache key (required if sqlite_cache is provided)

        Returns:
            List of TAR file keys sorted by name
        """
        # Try to get from SQLite cache first
        if sqlite_cache is not None and source is not None:
            cached_tars = sqlite_cache.get_tar_files(source)
            if cached_tars is not None:
                logging.debug(f"[{source}] Using cached TAR file list ({len(cached_tars)} files)")
                return cached_tars

        # List from S3
        tar_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.tar'):
                    tar_files.append(key)

        tar_files = sorted(tar_files)

        # Cache to SQLite if available
        if sqlite_cache is not None and source is not None and tar_files:
            try:
                sqlite_cache.add_tar_files(source, tar_files)
                logging.debug(f"[{source}] Cached {len(tar_files)} TAR files to SQLite")
            except Exception as e:
                logging.warning(f"[{source}] Failed to cache TAR files: {e}")

        return tar_files
