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
from .sqlite_manifest import SQLiteManifestCache, SQLiteManifestProvider
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
        s3_endpoint_url: str = None,
        # Disk config
        data_root: str = None,
        # Common
        prefetch_buffer_size: int = 0,  # Disabled for true sequential streaming
        # SQLite manifest cache (memory optimization)
        sqlite_cache_path: str = None,
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
            s3_endpoint_url: S3 endpoint URL (for R2, MinIO, etc.)
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
            if s3_bucket is None:
                raise ValueError("s3_bucket required for S3 storage")
            self.s3_bucket = s3_bucket
            self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix else ''
            self.s3_endpoint_url = s3_endpoint_url
            # S3 client and manifest loader created lazily in _initialize()
            # (boto3 clients are not fork-safe, so we create fresh ones in workers)
            self.s3_client = None
            self.manifest_loader = None
        elif storage_type == "disk":
            if data_root is None:
                raise ValueError("data_root required for disk storage")
            self.data_root = data_root
            self.manifest_loader = DiskManifestLoader()
        else:
            raise ValueError(f"Unknown storage_type: {storage_type}. Use 's3' or 'disk'")

        # SQLite manifest cache (shared across workers, saves RAM)
        self.sqlite_cache_path = sqlite_cache_path
        self._sqlite_cache: Optional[SQLiteManifestCache] = None
        self._manifest_provider: Optional[SQLiteManifestProvider] = None

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
        """Initialize base infrastructure (S3 client, SQLite cache) - sources are lazy."""
        if self._initialized:
            return

        # Check if we're in a DataLoader worker (reduce log verbosity)
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        in_worker = worker_info is not None

        # Worker sharding info
        self._worker_id = worker_info.id if worker_info else 0
        self._num_workers = worker_info.num_workers if worker_info else 1

        if not in_worker:
            logging.info(f"[{self.lang}] Initializing ({self.storage_type}), {len(self.sources)} sources available...")

        # Create S3 client if needed (fresh client per worker, boto3 is not fork-safe)
        if self.storage_type == "s3" and self.s3_client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
                boto_config = BotoConfig(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    signature_version='s3v4',
                    connect_timeout=10,
                    read_timeout=30,
                )
                client_kwargs = {'config': boto_config}
                if self.s3_endpoint_url:
                    client_kwargs['endpoint_url'] = self.s3_endpoint_url
                self.s3_client = boto3.client('s3', **client_kwargs)
                self.manifest_loader = S3ManifestLoader(s3_client=self.s3_client)
            except Exception as e:
                logging.error(f"[{self.lang}] Failed to create S3 client: {e}")
                raise

        # Initialize SQLite cache if available (saves RAM with multiple workers)
        if self.sqlite_cache_path and os.path.exists(self.sqlite_cache_path):
            try:
                self._sqlite_cache = SQLiteManifestCache(self.sqlite_cache_path, read_only=True)
                self._manifest_provider = SQLiteManifestProvider(self._sqlite_cache)
                if not in_worker:
                    logging.info(f"[{self.lang}] Using SQLite manifest cache: {self.sqlite_cache_path}")
            except Exception as e:
                logging.warning(f"[{self.lang}] Failed to open SQLite cache, falling back to dict: {e}")
                self._sqlite_cache = None
                self._manifest_provider = None

        # Track which sources have been initialized
        self._initialized_sources: set = set()

        self._initialized = True
        # NOTE: Sources are initialized lazily in _advance_to_next_stream()
        # This allows training to start immediately without loading all manifests

    def _init_s3_source(self, source: str, verbose: bool = True):
        """Initialize a source from S3."""
        source_prefix = f"{self.s3_prefix}{source}/"

        # List TAR files (use SQLite cache to avoid repeated S3 list calls)
        tar_files = self.manifest_loader.list_tar_files(
            self.s3_bucket, source_prefix,
            sqlite_cache=self._sqlite_cache,
            source=source
        )
        if not tar_files:
            logging.warning(f"[{self.lang}] No TAR files found in {source_prefix}")
            return

        # Shard TAR files across workers
        if self._num_workers > 1:
            tar_files = [t for i, t in enumerate(tar_files) if i % self._num_workers == self._worker_id]
            if verbose:
                logging.info(f"[{self.lang}] {source}: Worker {self._worker_id}/{self._num_workers}: {len(tar_files)} TAR files (sharded)")
        else:
            if verbose:
                logging.info(f"[{self.lang}] {source}: {len(tar_files)} TAR files")

        self._tar_files_by_source[source] = tar_files

        # Skip manifest loading if using SQLite cache
        if self._manifest_provider is not None:
            logging.debug(f"[{self.lang}] {source}: using SQLite manifest cache")
            return

        # Load manifest (fallback when SQLite not available)
        manifest_key = f"{source_prefix}tarred_audio_manifest.json"
        try:
            manifest = self.manifest_loader.load_manifest(self.s3_bucket, manifest_key)
            self._manifests_by_source[source] = manifest
            if verbose:
                logging.info(f"[{self.lang}] {source}: {len(manifest)} manifest entries")
        except Exception as e:
            logging.error(f"[{self.lang}] Failed to load manifest for {source}: {e}")

    def _init_disk_source(self, source: str, verbose: bool = True):
        """Initialize a source from local disk."""
        source_dir = os.path.join(self.data_root, source)

        if not os.path.isdir(source_dir):
            logging.warning(f"[{self.lang}] Source directory not found: {source_dir}")
            return

        # List TAR files (use SQLite cache)
        tar_files = self.manifest_loader.list_tar_files(
            source_dir,
            sqlite_cache=self._sqlite_cache,
            source=source
        )
        if not tar_files:
            logging.warning(f"[{self.lang}] No TAR files found in {source_dir}")
            return

        # Shard TAR files across workers
        if self._num_workers > 1:
            tar_files = [t for i, t in enumerate(tar_files) if i % self._num_workers == self._worker_id]
            if verbose:
                logging.info(f"[{self.lang}] {source}: Worker {self._worker_id}/{self._num_workers}: {len(tar_files)} TAR files (sharded)")
        else:
            if verbose:
                logging.info(f"[{self.lang}] {source}: {len(tar_files)} TAR files")

        self._tar_files_by_source[source] = tar_files

        # Skip manifest loading if using SQLite cache
        if self._manifest_provider is not None:
            logging.debug(f"[{self.lang}] {source}: using SQLite manifest cache")
            return

        # Load manifest (fallback when SQLite not available)
        manifest_path = os.path.join(source_dir, "tarred_audio_manifest.json")
        try:
            manifest = self.manifest_loader.load_manifest(manifest_path)
            self._manifests_by_source[source] = manifest
            if verbose:
                logging.info(f"[{self.lang}] {source}: {len(manifest)} manifest entries")
        except Exception as e:
            logging.error(f"[{self.lang}] Failed to load manifest for {source}: {e}")

    def _advance_to_next_stream(self) -> bool:
        """
        Advance to the next TAR file stream.

        Returns:
            True if successfully advanced, False if all sources exhausted
        """
        # Check if we're in a DataLoader worker (reduce log verbosity)
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        in_worker = worker_info is not None

        while self._current_source_idx < len(self.sources):
            source = self.sources[self._current_source_idx]

            # Lazy init: initialize source only when we need it
            if source not in self._initialized_sources:
                if self.storage_type == "s3":
                    self._init_s3_source(source, verbose=not in_worker)
                else:
                    self._init_disk_source(source, verbose=not in_worker)
                self._initialized_sources.add(source)

            if source not in self._tar_files_by_source:
                self._current_source_idx += 1
                self._current_tar_idx = 0
                continue

            tar_files = self._tar_files_by_source[source]

            if self._current_tar_idx < len(tar_files):
                tar_path = tar_files[self._current_tar_idx]

                # Use SQLite provider if available, otherwise fall back to dict
                if self._manifest_provider is not None:
                    manifest_entries = self._manifest_provider
                else:
                    manifest_entries = self._manifests_by_source.get(source, {})

                logging.debug(f"[{self.lang}] Opening stream: {tar_path}")

                if self.storage_type == "s3":
                    stream = S3TarStream(
                        s3_bucket=self.s3_bucket,
                        tar_key=tar_path,
                        manifest_entries=manifest_entries,
                        s3_client=self.s3_client,
                    )
                else:
                    stream = DiskTarStream(
                        tar_path=tar_path,
                        manifest_entries=manifest_entries,
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

        # Track empty TARs to avoid downloading all TARs when manifest doesn't match
        empty_tar_count = 0
        max_empty_tars = 2  # Give up after 2 consecutive empty TARs
        samples_from_current_tar = 0  # Track samples from current TAR stream

        while True:
            if self._current_stream is None:
                if not self._advance_to_next_stream():
                    # All sources exhausted
                    self._exhausted = True
                    return None
                samples_from_current_tar = 0  # Reset only when opening new TAR

            try:
                sample = next(self._current_stream)
                samples_from_current_tar += 1

                # Apply filter
                if not self.sample_filter(sample):
                    continue

                # Apply token augmentation
                sample = self.token_augmenter(sample)

                empty_tar_count = 0  # Reset on success
                return sample

            except StopIteration:
                # Current stream exhausted
                if samples_from_current_tar == 0:
                    empty_tar_count += 1
                    if empty_tar_count >= max_empty_tars:
                        logging.warning(f"[{self.lang}] {empty_tar_count} consecutive empty TARs, giving up")
                        self._exhausted = True
                        return None
                else:
                    empty_tar_count = 0  # TAR had samples, reset
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
        # When using SQLite, _manifests_by_source is empty
        # The count is handled by the dataset class which built the cache
        if self._sqlite_cache is not None:
            return self._sqlite_cache.count_entries()
        return sum(len(m) for m in self._manifests_by_source.values())

    def stop(self):
        """Stop prefetch buffer and cleanup resources."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
            self._prefetch_iter = None
        if self._sqlite_cache is not None:
            self._sqlite_cache.close()
            self._sqlite_cache = None
            self._manifest_provider = None

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class SingleSourceManager:
    """
    Manages streaming from a single TAR source.

    Simplified version of LanguageSourceManager for source-level round-robin.
    Each instance handles exactly one dataset source.
    """

    def __init__(
        self,
        lang: str,
        source: str,
        sample_filter: SampleFilter,
        token_augmenter: TokenAugmenter,
        # Storage config
        storage_type: str = "s3",
        # S3 config
        s3_bucket: str = None,
        s3_prefix: str = None,
        s3_endpoint_url: str = None,
        # Disk config
        data_root: str = None,
        # Common
        prefetch_buffer_size: int = 0,  # Disabled for true sequential streaming
        # SQLite manifest cache
        sqlite_cache_path: str = None,
    ):
        """
        Initialize single source manager.

        Args:
            lang: Language code (e.g., "en", "uk")
            source: Source name (e.g., "common_en_train")
            sample_filter: Filter for duration/char rate
            token_augmenter: Augmenter for adding <eou> tokens
            storage_type: "s3" or "disk"
            s3_bucket: S3 bucket name (required for S3)
            s3_prefix: S3 prefix
            s3_endpoint_url: S3 endpoint URL (for R2, MinIO, etc.)
            data_root: Root directory for data (required for disk)
            prefetch_buffer_size: Number of samples to prefetch
            sqlite_cache_path: Path to SQLite manifest cache
        """
        self.lang = lang
        self.source = source
        self.sample_filter = sample_filter
        self.token_augmenter = token_augmenter
        self.prefetch_buffer_size = prefetch_buffer_size
        self.storage_type = storage_type

        # Storage-specific setup
        if storage_type == "s3":
            if s3_bucket is None:
                raise ValueError("s3_bucket required for S3 storage")
            self.s3_bucket = s3_bucket
            self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix else ''
            self.s3_endpoint_url = s3_endpoint_url
            # S3 client and manifest loader created lazily in _initialize()
            # (boto3 clients are not fork-safe, so we create fresh ones in workers)
            self.s3_client = None
            self.manifest_loader = None
        elif storage_type == "disk":
            if data_root is None:
                raise ValueError("data_root required for disk storage")
            self.data_root = data_root
            self.manifest_loader = DiskManifestLoader()
        else:
            raise ValueError(f"Unknown storage_type: {storage_type}")

        # SQLite manifest cache
        self.sqlite_cache_path = sqlite_cache_path
        self._sqlite_cache: Optional[SQLiteManifestCache] = None
        self._manifest_provider: Optional[SQLiteManifestProvider] = None

        # State tracking
        self._current_tar_idx = 0
        self._current_stream: Optional[Iterator] = None
        self._tar_files: List[str] = []
        self._manifest: Dict[str, dict] = {}
        self._epoch = 0
        self._exhausted = False
        self._initialized = False
        self._samples_yielded = 0  # Track samples for diagnostics

        # Prefetch buffer
        self._prefetch_buffer: Optional[PrefetchBuffer] = None
        self._prefetch_iter: Optional[Iterator] = None

    def _initialize(self):
        """Load manifest and discover TAR files for this source."""
        if self._initialized:
            return

        # Check if we're in a DataLoader worker (reduce log verbosity)
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        in_worker = worker_info is not None

        # Worker sharding info
        self._worker_id = worker_info.id if worker_info else 0
        self._num_workers = worker_info.num_workers if worker_info else 1

        if not in_worker:
            logging.info(f"[{self.lang}:{self.source}] Initializing ({self.storage_type})...")

        # Create S3 client if needed (fresh client per worker, boto3 is not fork-safe)
        if self.storage_type == "s3" and self.s3_client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
                boto_config = BotoConfig(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    signature_version='s3v4',
                    connect_timeout=10,
                    read_timeout=30,
                )
                client_kwargs = {'config': boto_config}
                if self.s3_endpoint_url:
                    client_kwargs['endpoint_url'] = self.s3_endpoint_url
                self.s3_client = boto3.client('s3', **client_kwargs)
                self.manifest_loader = S3ManifestLoader(s3_client=self.s3_client)
            except Exception as e:
                logging.error(f"[{self.lang}:{self.source}] Failed to create S3 client: {e}")
                raise

        # Initialize SQLite cache if available
        if self.sqlite_cache_path and os.path.exists(self.sqlite_cache_path):
            try:
                self._sqlite_cache = SQLiteManifestCache(self.sqlite_cache_path, read_only=True)
                self._manifest_provider = SQLiteManifestProvider(self._sqlite_cache)
                logging.debug(f"[{self.lang}:{self.source}] Using SQLite manifest cache")
            except Exception as e:
                logging.warning(f"[{self.lang}:{self.source}] Failed to open SQLite cache: {e}")

        if self.storage_type == "s3":
            self._init_s3_source(verbose=not in_worker)
        else:
            self._init_disk_source(verbose=not in_worker)

        self._initialized = True
        # NOTE: Don't call _advance_to_next_stream() here - let it be lazy
        # The first call to get_next_sample() will open the stream when needed

    def _init_s3_source(self, verbose: bool = True):
        """Initialize source from S3."""
        source_prefix = f"{self.s3_prefix}{self.source}/"

        # Use SQLite cache for TAR file listing to avoid repeated S3 list calls
        tar_files = self.manifest_loader.list_tar_files(
            self.s3_bucket, source_prefix,
            sqlite_cache=self._sqlite_cache,
            source=self.source
        )
        if not tar_files:
            logging.warning(f"[{self.lang}:{self.source}] No TAR files found")
            return

        # Shard TAR files across workers to avoid duplicate downloads
        if self._num_workers > 1:
            tar_files = [t for i, t in enumerate(tar_files) if i % self._num_workers == self._worker_id]
            if verbose:
                logging.info(f"[{self.lang}:{self.source}] Worker {self._worker_id}/{self._num_workers}: {len(tar_files)} TAR files (sharded)")
        else:
            if verbose:
                logging.info(f"[{self.lang}:{self.source}] {len(tar_files)} TAR files")

        self._tar_files = tar_files

        if self._manifest_provider is not None:
            return

        manifest_key = f"{source_prefix}tarred_audio_manifest.json"
        try:
            self._manifest = self.manifest_loader.load_manifest(self.s3_bucket, manifest_key)
            if verbose:
                logging.info(f"[{self.lang}:{self.source}] {len(self._manifest)} manifest entries")
        except Exception as e:
            logging.error(f"[{self.lang}:{self.source}] Failed to load manifest: {e}")

    def _init_disk_source(self, verbose: bool = True):
        """Initialize source from local disk."""
        source_dir = os.path.join(self.data_root, self.source)

        if not os.path.isdir(source_dir):
            logging.warning(f"[{self.lang}:{self.source}] Directory not found: {source_dir}")
            return

        # Use SQLite cache for TAR file listing
        tar_files = self.manifest_loader.list_tar_files(
            source_dir,
            sqlite_cache=self._sqlite_cache,
            source=self.source
        )
        if not tar_files:
            logging.warning(f"[{self.lang}:{self.source}] No TAR files found")
            return

        # Shard TAR files across workers to avoid duplicate downloads
        if self._num_workers > 1:
            tar_files = [t for i, t in enumerate(tar_files) if i % self._num_workers == self._worker_id]
            if verbose:
                logging.info(f"[{self.lang}:{self.source}] Worker {self._worker_id}/{self._num_workers}: {len(tar_files)} TAR files (sharded)")
        else:
            if verbose:
                logging.info(f"[{self.lang}:{self.source}] {len(tar_files)} TAR files")

        self._tar_files = tar_files

        if self._manifest_provider is not None:
            return

        manifest_path = os.path.join(source_dir, "tarred_audio_manifest.json")
        try:
            self._manifest = self.manifest_loader.load_manifest(manifest_path)
            if verbose:
                logging.info(f"[{self.lang}:{self.source}] {len(self._manifest)} manifest entries")
        except Exception as e:
            logging.error(f"[{self.lang}:{self.source}] Failed to load manifest: {e}")

    def _advance_to_next_stream(self) -> bool:
        """Advance to the next TAR file stream."""
        if self._current_tar_idx >= len(self._tar_files):
            return False

        tar_path = self._tar_files[self._current_tar_idx]

        manifest_entries = self._manifest_provider if self._manifest_provider else self._manifest

        logging.debug(f"[{self.lang}:{self.source}] Opening stream: {tar_path}")

        if self.storage_type == "s3":
            from .s3_tar_stream import S3TarStream
            stream = S3TarStream(
                s3_bucket=self.s3_bucket,
                tar_key=tar_path,
                manifest_entries=manifest_entries,
                s3_client=self.s3_client,
            )
        else:
            from .disk_tar_stream import DiskTarStream
            stream = DiskTarStream(
                tar_path=tar_path,
                manifest_entries=manifest_entries,
            )

        self._current_stream = iter(stream)
        self._current_tar_idx += 1
        return True

    def _generate_samples(self) -> Iterator[dict]:
        """Generator that yields samples from all TAR files."""
        if not self._initialized:
            self._initialize()

        while True:
            if self._current_stream is None:
                if not self._advance_to_next_stream():
                    self._exhausted = True
                    return

            try:
                sample = next(self._current_stream)

                if not self.sample_filter(sample):
                    continue

                sample = self.token_augmenter(sample)
                yield sample

            except StopIteration:
                self._current_stream = None
                continue

    def get_next_sample(self) -> Optional[dict]:
        """Get next valid sample from this source."""
        if self.prefetch_buffer_size > 0:
            if self._prefetch_buffer is None:
                self._prefetch_buffer = PrefetchBuffer(
                    source_factory=self._generate_samples,
                    buffer_size=self.prefetch_buffer_size,
                    name=f"src-{self.lang}:{self.source}",
                )
                self._prefetch_iter = iter(self._prefetch_buffer)

            try:
                sample = next(self._prefetch_iter)
                self._samples_yielded += 1
                return sample
            except StopIteration:
                self._exhausted = True
                return None

        if not self._initialized:
            self._initialize()

        # Track empty TARs to avoid downloading all TARs when manifest doesn't match
        empty_tar_count = 0
        max_empty_tars = 2  # Give up after 2 empty TARs
        samples_from_current_tar = 0  # Track samples from current TAR stream

        while True:
            if self._current_stream is None:
                if not self._advance_to_next_stream():
                    self._exhausted = True
                    return None
                samples_from_current_tar = 0  # Reset only when opening new TAR

            try:
                sample = next(self._current_stream)
                samples_from_current_tar += 1

                if not self.sample_filter(sample):
                    continue

                sample = self.token_augmenter(sample)
                self._samples_yielded += 1
                empty_tar_count = 0  # Reset on success
                return sample

            except StopIteration:
                if samples_from_current_tar == 0:
                    empty_tar_count += 1
                    if empty_tar_count >= max_empty_tars:
                        logging.warning(f"[{self.lang}:{self.source}] {empty_tar_count} consecutive empty TARs, giving up")
                        self._exhausted = True
                        return None
                else:
                    empty_tar_count = 0  # TAR had samples, reset
                self._current_stream = None
                continue

    def reset(self):
        """Reset to beginning for new epoch."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
            self._prefetch_iter = None

        self._current_tar_idx = 0
        self._current_stream = None
        self._exhausted = False
        self._samples_yielded = 0
        self._epoch += 1
        # Only log reset in main process (not workers)
        import torch.utils.data
        if torch.utils.data.get_worker_info() is None:
            logging.info(f"[{self.lang}:{self.source}] Reset for epoch {self._epoch}")

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def epoch(self) -> int:
        return self._epoch

    def stop(self):
        """Stop and cleanup resources."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
            self._prefetch_iter = None
        if self._sqlite_cache is not None:
            self._sqlite_cache.close()
            self._sqlite_cache = None
            self._manifest_provider = None

    def __del__(self):
        self.stop()
