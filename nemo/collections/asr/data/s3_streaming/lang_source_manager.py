"""
Language source manager for streaming dataset.

Manages multiple TAR sources for a single language with automatic rotation.
Supports both S3 and local disk storage.
"""

import os
from typing import Dict, Iterator, List, Optional

from nemo.utils import logging

from .disk_tar_stream import DiskManifestLoader, DiskTarStream
from .filters import FilterConfig, SampleFilter
from .prefetch_buffer import PrefetchBuffer
from .s3_tar_stream import S3ManifestLoader, S3TarStream
from .token_augmenter import TokenAugmenter


class LanguageSourceManager:
    """
    Manages streaming from multiple TAR sources for a single language.

    Features:
    - Multiple TAR files per source (audio_0.tar, audio_1.tar, ...)
    - Multiple sources per language (e.g., common_en + yodas_en)
    - Automatic rotation when a TAR/source is exhausted
    - Applies filters and token augmentation
    - Epoch tracking
    - Optional prefetching for reduced I/O latency
    - Supports both S3 and local disk storage
    """

    def __init__(
        self,
        lang: str,
        sources: List[str],
        sample_filter: SampleFilter,
        token_augmenter: TokenAugmenter,
        # Storage config - either S3 or disk
        storage_type: str = "s3",  # "s3" or "disk"
        # S3 config
        s3_bucket: str = None,
        s3_prefix: str = None,
        s3_client=None,
        # Disk config
        data_root: str = None,
        # Common
        prefetch_buffer_size: int = 100,
    ):
        """
        Initialize language source manager.

        Args:
            lang: Language code (e.g., "en", "uk", "zh")
            sources: List of source names (e.g., ["common_en_train", "yodas_en_en000"])
            sample_filter: Filter for duration/char rate
            token_augmenter: Augmenter for adding <eou> tokens
            storage_type: "s3" or "disk"
            s3_bucket: S3 bucket name (required for S3)
            s3_prefix: S3 prefix (e.g., "asr-data/")
            s3_client: Boto3 S3 client (required for S3)
            data_root: Root directory for data (required for disk)
            prefetch_buffer_size: Number of samples to prefetch (0 to disable)
        """
        self.lang = lang
        self.sources = sources
        self.sample_filter = sample_filter
        self.token_augmenter = token_augmenter
        self.prefetch_buffer_size = prefetch_buffer_size
        self.storage_type = storage_type

        # Storage-specific setup
        if storage_type == "s3":
            if s3_bucket is None or s3_client is None:
                raise ValueError("s3_bucket and s3_client required for S3 storage")
            self.s3_bucket = s3_bucket
            self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix else ''
            self.s3_client = s3_client
            self.manifest_loader = S3ManifestLoader(s3_client=s3_client)
        elif storage_type == "disk":
            if data_root is None:
                raise ValueError("data_root required for disk storage")
            self.data_root = data_root
            self.manifest_loader = DiskManifestLoader()
        else:
            raise ValueError(f"Unknown storage_type: {storage_type}. Use 's3' or 'disk'")

        # State tracking
        self._current_source_idx = 0
        self._current_tar_idx = 0
        self._current_stream: Optional[Iterator] = None
        self._tar_files_by_source: Dict[str, List[str]] = {}
        self._manifests_by_source: Dict[str, Dict[str, dict]] = {}
        self._epoch = 0
        self._exhausted = False
        self._initialized = False

        # Prefetch buffer (created on first use)
        self._prefetch_buffer: Optional[PrefetchBuffer] = None
        self._prefetch_iter: Optional[Iterator] = None

    def _initialize(self):
        """Load manifests and discover TAR files for all sources."""
        if self._initialized:
            return

        logging.info(f"[{self.lang}] Initializing {len(self.sources)} sources ({self.storage_type})...")

        for source in self.sources:
            if self.storage_type == "s3":
                self._init_s3_source(source)
            else:
                self._init_disk_source(source)

        self._initialized = True

        # Start first stream
        self._advance_to_next_stream()

    def _init_s3_source(self, source: str):
        """Initialize a source from S3."""
        source_prefix = f"{self.s3_prefix}{source}/"

        # List TAR files
        tar_files = self.manifest_loader.list_tar_files(self.s3_bucket, source_prefix)
        if not tar_files:
            logging.warning(f"[{self.lang}] No TAR files found in {source_prefix}")
            return

        self._tar_files_by_source[source] = tar_files
        logging.info(f"[{self.lang}] {source}: {len(tar_files)} TAR files")

        # Load manifest
        manifest_key = f"{source_prefix}tarred_audio_manifest.json"
        try:
            manifest = self.manifest_loader.load_manifest(self.s3_bucket, manifest_key)
            self._manifests_by_source[source] = manifest
            logging.info(f"[{self.lang}] {source}: {len(manifest)} manifest entries")
        except Exception as e:
            logging.error(f"[{self.lang}] Failed to load manifest for {source}: {e}")

    def _init_disk_source(self, source: str):
        """Initialize a source from local disk."""
        source_dir = os.path.join(self.data_root, source)

        if not os.path.isdir(source_dir):
            logging.warning(f"[{self.lang}] Source directory not found: {source_dir}")
            return

        # List TAR files
        tar_files = self.manifest_loader.list_tar_files(source_dir)
        if not tar_files:
            logging.warning(f"[{self.lang}] No TAR files found in {source_dir}")
            return

        self._tar_files_by_source[source] = tar_files
        logging.info(f"[{self.lang}] {source}: {len(tar_files)} TAR files")

        # Load manifest
        manifest_path = os.path.join(source_dir, "tarred_audio_manifest.json")
        try:
            manifest = self.manifest_loader.load_manifest(manifest_path)
            self._manifests_by_source[source] = manifest
            logging.info(f"[{self.lang}] {source}: {len(manifest)} manifest entries")
        except Exception as e:
            logging.error(f"[{self.lang}] Failed to load manifest for {source}: {e}")

    def _advance_to_next_stream(self) -> bool:
        """
        Advance to the next TAR file stream.

        Returns:
            True if successfully advanced, False if all sources exhausted
        """
        while self._current_source_idx < len(self.sources):
            source = self.sources[self._current_source_idx]

            if source not in self._tar_files_by_source:
                self._current_source_idx += 1
                self._current_tar_idx = 0
                continue

            tar_files = self._tar_files_by_source[source]

            if self._current_tar_idx < len(tar_files):
                tar_path = tar_files[self._current_tar_idx]
                manifest = self._manifests_by_source.get(source, {})

                logging.debug(f"[{self.lang}] Opening stream: {tar_path}")

                if self.storage_type == "s3":
                    stream = S3TarStream(
                        s3_bucket=self.s3_bucket,
                        tar_key=tar_path,
                        manifest_entries=manifest,
                        s3_client=self.s3_client,
                    )
                else:
                    stream = DiskTarStream(
                        tar_path=tar_path,
                        manifest_entries=manifest,
                    )

                self._current_stream = iter(stream)
                self._current_tar_idx += 1
                return True

            # Move to next source
            self._current_source_idx += 1
            self._current_tar_idx = 0

        # All sources exhausted for this epoch
        return False

    def _generate_samples(self) -> Iterator[dict]:
        """
        Generator that yields samples from all sources.

        This is used as the source factory for the prefetch buffer.
        """
        if not self._initialized:
            self._initialize()

        while True:
            if self._current_stream is None:
                if not self._advance_to_next_stream():
                    # All sources exhausted
                    self._exhausted = True
                    return

            try:
                sample = next(self._current_stream)

                # Apply filter
                if not self.sample_filter(sample):
                    continue

                # Apply token augmentation
                sample = self.token_augmenter(sample)

                yield sample

            except StopIteration:
                # Current stream exhausted, try next
                self._current_stream = None
                continue

    def get_next_sample(self) -> Optional[dict]:
        """
        Get next valid sample from this language's sources.

        Uses prefetch buffer if enabled for reduced I/O latency.

        Returns:
            Sample dict or None if all sources exhausted
        """
        # Use prefetch buffer if enabled
        if self.prefetch_buffer_size > 0:
            # Initialize prefetch buffer on first call
            if self._prefetch_buffer is None:
                self._prefetch_buffer = PrefetchBuffer(
                    source_factory=self._generate_samples,
                    buffer_size=self.prefetch_buffer_size,
                    name=f"lang-{self.lang}",
                )
                self._prefetch_iter = iter(self._prefetch_buffer)

            try:
                return next(self._prefetch_iter)
            except StopIteration:
                self._exhausted = True
                return None

        # Non-prefetch path (original behavior)
        if not self._initialized:
            self._initialize()

        while True:
            if self._current_stream is None:
                if not self._advance_to_next_stream():
                    # All sources exhausted
                    self._exhausted = True
                    return None

            try:
                sample = next(self._current_stream)

                # Apply filter
                if not self.sample_filter(sample):
                    continue

                # Apply token augmentation
                sample = self.token_augmenter(sample)

                return sample

            except StopIteration:
                # Current stream exhausted, try next
                self._current_stream = None
                continue

    def reset(self):
        """Reset to beginning of all sources for new epoch."""
        # Stop prefetch buffer if running
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
            self._prefetch_iter = None

        self._current_source_idx = 0
        self._current_tar_idx = 0
        self._current_stream = None
        self._exhausted = False
        self._epoch += 1
        logging.info(f"[{self.lang}] Reset for epoch {self._epoch}")

    @property
    def exhausted(self) -> bool:
        """Check if all sources are exhausted."""
        return self._exhausted

    @property
    def epoch(self) -> int:
        """Current epoch number."""
        return self._epoch

    @property
    def total_samples(self) -> int:
        """Get total number of manifest entries across all sources."""
        if not self._initialized:
            self._initialize()
        return sum(len(m) for m in self._manifests_by_source.values())

    def stop(self):
        """Stop prefetch buffer and cleanup resources."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
            self._prefetch_iter = None

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
