"""
S3 TAR streaming for ASR datasets.

Streams TAR files directly from S3 without downloading the entire file.
Parses TAR members on-the-fly and extracts audio samples.
"""

import io
import json
import struct
import tarfile
from typing import Dict, Iterator, Optional

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
    """

    def __init__(
        self,
        s3_bucket: str,
        tar_key: str,
        manifest_entries: Dict[str, dict],
        s3_client=None,
        aws_region: str = "us-east-1",
        max_retries: int = 3,
    ):
        """
        Initialize S3 TAR streamer.

        Args:
            s3_bucket: S3 bucket name
            tar_key: Key (path) to TAR file in bucket
            manifest_entries: Dict mapping audio filename to manifest entry
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
                retries={'max_attempts': max_retries, 'mode': 'adaptive'}
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

        try:
            # Get the TAR file as a streaming response
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.tar_key)
            body = response['Body']

            # Wrap in a file-like object for tarfile
            tar_stream = S3StreamWrapper(body)

            # Open as tarfile in streaming mode
            with tarfile.open(fileobj=tar_stream, mode='r|*') as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    filename = member.name

                    # Check if this file is in our manifest
                    if filename not in self.manifest_entries:
                        # Skip files not in manifest (filtered out)
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
            logging.error(f"Error streaming TAR from s3://{self.s3_bucket}/{self.tar_key}: {e}")

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
    Loads manifest files from S3.
    """

    def __init__(self, s3_client=None, aws_region: str = "us-east-1"):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 streaming. Install with: pip install boto3")

        if s3_client is not None:
            self.s3_client = s3_client
        else:
            self.s3_client = boto3.client('s3', region_name=aws_region)

    def load_manifest(
        self,
        s3_bucket: str,
        manifest_key: str,
    ) -> Dict[str, dict]:
        """
        Load manifest from S3 and return filename -> entry mapping.

        Args:
            s3_bucket: S3 bucket name
            manifest_key: Key (path) to manifest file

        Returns:
            Dict mapping audio filename to manifest entry
        """
        logging.info(f"Loading manifest from s3://{s3_bucket}/{manifest_key}")

        response = self.s3_client.get_object(Bucket=s3_bucket, Key=manifest_key)
        content = response['Body'].read().decode('utf-8')

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

        logging.info(f"Loaded {len(entries)} manifest entries")
        return entries

    def count_manifest_entries(
        self,
        s3_bucket: str,
        manifest_key: str,
    ) -> int:
        """
        Count entries in a manifest file without full parsing.

        This is faster than load_manifest() when you only need the count.

        Args:
            s3_bucket: S3 bucket name
            manifest_key: Key (path) to manifest file

        Returns:
            Number of entries (non-empty lines) in the manifest
        """
        try:
            response = self.s3_client.get_object(Bucket=s3_bucket, Key=manifest_key)
            content = response['Body'].read().decode('utf-8')

            # Count non-empty lines
            count = sum(1 for line in content.strip().split('\n') if line.strip())
            return count
        except Exception as e:
            logging.warning(f"Failed to count entries in s3://{s3_bucket}/{manifest_key}: {e}")
            return 0

    def list_tar_files(self, s3_bucket: str, prefix: str) -> list:
        """
        List TAR files in an S3 prefix.

        Args:
            s3_bucket: S3 bucket name
            prefix: S3 prefix (e.g., "asr-data/common_en_train/")

        Returns:
            List of TAR file keys sorted by name
        """
        tar_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.tar'):
                    tar_files.append(key)

        return sorted(tar_files)
