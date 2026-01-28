"""
Language source manager for S3 streaming dataset.

Manages multiple TAR sources for a single language with automatic rotation.
"""

from typing import Dict, Iterator, List, Optional

from nemo.utils import logging

from .filters import FilterConfig, SampleFilter
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
    """

    def __init__(
        self,
        lang: str,
        s3_bucket: str,
        s3_prefix: str,
        sources: List[str],
        s3_client,
        sample_filter: SampleFilter,
        token_augmenter: TokenAugmenter,
    ):
        """
        Initialize language source manager.

        Args:
            lang: Language code (e.g., "en", "uk", "zh")
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix (e.g., "asr-data/")
            sources: List of source names (e.g., ["common_en_train", "yodas_en_en000"])
            s3_client: Boto3 S3 client
            sample_filter: Filter for duration/char rate
            token_augmenter: Augmenter for adding <eou> tokens
        """
        self.lang = lang
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix else ''
        self.sources = sources
        self.s3_client = s3_client
        self.sample_filter = sample_filter
        self.token_augmenter = token_augmenter

        self.manifest_loader = S3ManifestLoader(s3_client=s3_client)

        # State tracking
        self._current_source_idx = 0
        self._current_tar_idx = 0
        self._current_stream: Optional[Iterator] = None
        self._tar_files_by_source: Dict[str, List[str]] = {}
        self._manifests_by_source: Dict[str, Dict[str, dict]] = {}
        self._epoch = 0
        self._exhausted = False
        self._initialized = False

    def _initialize(self):
        """Load manifests and discover TAR files for all sources."""
        if self._initialized:
            return

        logging.info(f"[{self.lang}] Initializing {len(self.sources)} sources...")

        for source in self.sources:
            source_prefix = f"{self.s3_prefix}{source}/"

            # List TAR files
            tar_files = self.manifest_loader.list_tar_files(self.s3_bucket, source_prefix)
            if not tar_files:
                logging.warning(f"[{self.lang}] No TAR files found in {source_prefix}")
                continue

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
                continue

        self._initialized = True

        # Start first stream
        self._advance_to_next_stream()

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
                tar_key = tar_files[self._current_tar_idx]
                manifest = self._manifests_by_source.get(source, {})

                logging.debug(f"[{self.lang}] Opening stream: {tar_key}")

                stream = S3TarStream(
                    s3_bucket=self.s3_bucket,
                    tar_key=tar_key,
                    manifest_entries=manifest,
                    s3_client=self.s3_client,
                )
                self._current_stream = iter(stream)
                self._current_tar_idx += 1
                return True

            # Move to next source
            self._current_source_idx += 1
            self._current_tar_idx = 0

        # All sources exhausted for this epoch
        return False

    def get_next_sample(self) -> Optional[dict]:
        """
        Get next valid sample from this language's sources.

        Returns:
            Sample dict or None if all sources exhausted
        """
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
