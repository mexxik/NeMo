"""
Streaming Dataset package for multi-language ASR training.

This package provides components for streaming audio data from S3 or local disk
with round-robin language sampling and on-the-fly token augmentation.

Supports:
- S3-compatible storage (AWS S3, Cloudflare R2, MinIO, etc.)
- Local disk storage
"""

from .audio_merger import AudioMerger, MergeBuffer, MergeConfig
from .dataset import S3MultiLangStreamingDataset
from .disk_tar_stream import DiskManifestLoader, DiskTarStream
from .filters import (
    ALLOWED_CHARS,
    FilterConfig,
    SampleFilter,
    get_allowed_chars,
    has_valid_chars,
    is_valid_text,
)
from .lang_source_manager import LanguageSourceManager, SingleSourceManager
from .prefetch_buffer import MultiSourcePrefetcher, PrefetchBuffer
from .round_robin import RoundRobinInterleaver, SourceRoundRobinInterleaver
from .s3_tar_stream import S3ManifestLoader, S3TarStream
from .sqlite_manifest import (
    DictManifestProvider,
    ManifestProvider,
    SQLiteManifestCache,
    SQLiteManifestProvider,
    get_default_cache_dir,
)
from .token_augmenter import SENTENCE_ENDINGS, TokenAugmenter

# Alias for cleaner naming (supports both S3 and disk)
MultiLangStreamingDataset = S3MultiLangStreamingDataset

__all__ = [
    # Main dataset class
    'S3MultiLangStreamingDataset',
    'MultiLangStreamingDataset',  # Alias
    # Audio merging
    'AudioMerger',
    'MergeBuffer',
    'MergeConfig',
    # Filtering
    'ALLOWED_CHARS',
    'FilterConfig',
    'SampleFilter',
    'get_allowed_chars',
    'has_valid_chars',
    'is_valid_text',
    # Source management
    'LanguageSourceManager',
    'SingleSourceManager',
    'RoundRobinInterleaver',
    'SourceRoundRobinInterleaver',
    # S3 streaming
    'S3ManifestLoader',
    'S3TarStream',
    # Disk streaming
    'DiskManifestLoader',
    'DiskTarStream',
    # SQLite manifest cache
    'SQLiteManifestCache',
    'SQLiteManifestProvider',
    'DictManifestProvider',
    'ManifestProvider',
    'get_default_cache_dir',
    # Prefetching
    'PrefetchBuffer',
    'MultiSourcePrefetcher',
    # Token augmentation
    'TokenAugmenter',
    'SENTENCE_ENDINGS',
]
