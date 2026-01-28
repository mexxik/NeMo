"""
S3 Streaming Dataset package for multi-language ASR training.

This package provides components for streaming audio data directly from S3
with round-robin language sampling and on-the-fly token augmentation.
"""

from .dataset import S3MultiLangStreamingDataset
from .filters import (
    ALLOWED_CHARS,
    FilterConfig,
    SampleFilter,
    get_allowed_chars,
    has_valid_chars,
    is_valid_text,
)
from .lang_source_manager import LanguageSourceManager
from .round_robin import RoundRobinInterleaver
from .s3_tar_stream import S3ManifestLoader, S3TarStream
from .token_augmenter import SENTENCE_ENDINGS, TokenAugmenter

__all__ = [
    'S3MultiLangStreamingDataset',
    'ALLOWED_CHARS',
    'FilterConfig',
    'SampleFilter',
    'get_allowed_chars',
    'has_valid_chars',
    'is_valid_text',
    'LanguageSourceManager',
    'RoundRobinInterleaver',
    'S3ManifestLoader',
    'S3TarStream',
    'TokenAugmenter',
    'SENTENCE_ENDINGS',
]
