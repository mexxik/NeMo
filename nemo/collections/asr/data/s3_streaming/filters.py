"""
Sample filtering for S3 streaming dataset.

Provides configurable filters for duration, character rate, and text quality.
Matches the filtering logic from 01_gather.py for consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from nemo.utils import logging


# Allowed characters per language
# These are the characters that should appear in clean transcriptions
ALLOWED_CHARS: Dict[str, Set[str]] = {
    'en': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-'  # punctuation
    ),
    'fr': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'àâäéèêëïîôùûüÿçœæ'
        'ÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-«»'  # punctuation including French quotes
    ),
    'de': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'äöüß'
        'ÄÖÜ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-„"'  # punctuation including German quotes
    ),
    'es': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'áéíóúüñ'
        'ÁÉÍÓÚÜÑ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-¿¡«»'  # punctuation including Spanish marks
    ),
    'it': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'àèéìíîòóùú'
        'ÀÈÉÌÍÎÒÓÙÚ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-«»'  # punctuation
    ),
    'pt': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'àáâãçéêíóôõú'
        'ÀÁÂÃÇÉÊÍÓÔÕÚ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-«»'  # punctuation
    ),
    'da': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'æøå'
        'ÆØÅ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-'  # punctuation
    ),
    'nl': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'àáâäèéêëïíîóôöúû'
        'ÀÁÂÄÈÉÊËÏÍÎÓÔÖÚÛ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-'  # punctuation
    ),
    'sv': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'åäö'
        'ÅÄÖ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-'  # punctuation
    ),
    'tr': set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'çğıöşü'
        'ÇĞİÖŞÜ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-'  # punctuation
    ),
    'uk': set(
        'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
        'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ'
        '0123456789'
        " 'ʼ'"  # space, ASCII apostrophe, Ukrainian apostrophe (U+02BC), right quote (U+2019)
        '.,!?;:-«»—'  # punctuation including Ukrainian quotes and dash
    ),
    'vi': set(
        # Vietnamese uses Latin script with many diacritics
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # Vietnamese-specific letters with diacritics
        'àáảãạăằắẳẵặâầấẩẫậ'
        'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ'
        'èéẻẽẹêềếểễệ'
        'ÈÉẺẼẸÊỀẾỂỄỆ'
        'ìíỉĩị'
        'ÌÍỈĨỊ'
        'òóỏõọôồốổỗộơờớởỡợ'
        'ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ'
        'ùúủũụưừứửữự'
        'ÙÚỦŨỤƯỪỨỬỮỰ'
        'ỳýỷỹỵ'
        'ỲÝỶỸỴ'
        'đĐ'
        '0123456789'
        " '"  # space and apostrophe
        '.,!?;:-'  # punctuation
    ),
    # Languages with non-Latin scripts or very large character sets
    # skip character validation - rely on is_valid_text() for garbage detection
    'zh': set(),  # Chinese - thousands of characters
    'ar': set(),  # Arabic script
    'he': set(),  # Hebrew script
    'ja': set(),  # Japanese (Kanji + Hiragana + Katakana)
    'ko': set(),  # Korean (Hangul)
}


def get_allowed_chars(languages: List[str]) -> Set[str]:
    """Get combined allowed characters for specified languages."""
    allowed = set()
    for lang in languages:
        if lang in ALLOWED_CHARS:
            allowed.update(ALLOWED_CHARS[lang])
        else:
            # For unknown languages, allow basic Latin + digits + punctuation
            allowed.update(ALLOWED_CHARS.get('en', set()))
    return allowed


def has_valid_chars(text: str, allowed_chars: Set[str]) -> bool:
    """
    Check if text contains only allowed characters.

    Rejects any sample with even a single unsupported character.
    This ensures a clean vocabulary for the tokenizer.

    Args:
        text: Text to check
        allowed_chars: Set of allowed characters

    Returns:
        True if ALL characters are valid, False otherwise
    """
    if not text:
        return False

    # Skip character validation if allowed_chars is empty (e.g., for Chinese)
    if not allowed_chars:
        return True

    for c in text:
        if c not in allowed_chars:
            return False
    return True


def is_valid_text(text: str) -> bool:
    """
    Check if text is valid (not corrupted).

    Filters out samples with:
    - Too many non-alphanumeric characters
    - HTML entities (&lt;, &gt;, &amp;)
    - Too many special characters in short text
    - JSON/code garbage from YouTube metadata
    - Repeating character patterns
    - Music-only tags
    """
    if not text or len(text.strip()) == 0:
        return False

    text_stripped = text.strip()

    # Filter out music-only tags
    if text_stripped in ['[Music]', '[Musique]', '[Música]', '[Musik]', '[音乐]']:
        return False

    # Filter out JSON/code garbage (YouTube metadata leaks)
    garbage_markers = [
        'codecs=', 'bitrate', '.NET CLR', 'ytcfg', 'https://', 'http://',
        'indexRange', 'initRange', 'contentLength', 'clickTrackingParams',
        'navigationEndpoint', 'thumbnailRenderer', 'clientName', 'clientVersion',
    ]
    if any(marker in text for marker in garbage_markers):
        return False

    # Filter out text with too many JSON-like characters
    json_chars = text.count('{') + text.count('}') + text.count('[') + text.count(']')
    if json_chars > 3:
        return False

    # Filter out repeating character patterns (e.g., "AAAAAAA", "0000000", "fffff")
    if len(text) > 20:
        # Check if first 10 chars repeat multiple times
        pattern = text[:10]
        if text.count(pattern) >= 3:
            return False
        # Check for single character repetition
        for char in set(text[:20]):
            if text.count(char * 10) > 0:
                return False

    # Count alphanumeric vs special characters
    alpha_count = sum(c.isalnum() or c.isspace() for c in text)
    total_count = len(text)

    # If less than 50% alphanumeric, likely corrupted
    if total_count > 0 and alpha_count / total_count < 0.5:
        return False

    # Check for HTML entities
    if any(entity in text for entity in ['&lt;', '&gt;', '&amp;', '&quot;']):
        return False

    # Check for excessive special characters in sequence
    special_chars = sum(not c.isalnum() and not c.isspace() for c in text)
    if total_count > 0 and special_chars / total_count > 0.3:
        return False

    return True


@dataclass
class FilterConfig:
    """Configuration for sample filtering."""
    min_duration: float = 0.5
    max_duration: float = 30.0
    min_chars: int = 1
    max_chars: int = 500


class SampleFilter:
    """
    Filters samples based on configurable criteria.

    Filters:
    - Duration bounds (min/max)
    - Text length bounds (min/max chars)
    """

    def __init__(self, config: FilterConfig):
        self.config = config
        self._stats = {
            'total': 0,
            'passed': 0,
            'rejected_min_duration': 0,
            'rejected_max_duration': 0,
            'rejected_min_chars': 0,
            'rejected_max_chars': 0,
        }

    def __call__(self, sample: dict) -> bool:
        """
        Check if sample passes all filters.

        Args:
            sample: Dict with 'duration' and 'text' keys

        Returns:
            True if sample passes all filters, False otherwise
        """
        self._stats['total'] += 1

        duration = sample.get('duration', 0)
        text = sample.get('text', '')

        # Duration filters
        if duration < self.config.min_duration:
            self._stats['rejected_min_duration'] += 1
            return False

        if duration > self.config.max_duration:
            self._stats['rejected_max_duration'] += 1
            return False

        # Text length filters
        text_len = len(text.strip())
        if text_len < self.config.min_chars:
            self._stats['rejected_min_chars'] += 1
            return False

        if text_len > self.config.max_chars:
            self._stats['rejected_max_chars'] += 1
            return False

        self._stats['passed'] += 1
        return True

    def get_stats(self) -> dict:
        """Return filtering statistics."""
        return self._stats.copy()

    def log_stats(self):
        """Log filtering statistics."""
        total = self._stats['total']
        if total == 0:
            return

        passed = self._stats['passed']
        pass_rate = 100.0 * passed / total

        logging.info(f"Filter stats: {passed}/{total} passed ({pass_rate:.1f}%)")
        logging.info(f"  Rejected min_duration: {self._stats['rejected_min_duration']}")
        logging.info(f"  Rejected max_duration: {self._stats['rejected_max_duration']}")
        logging.info(f"  Rejected min_chars: {self._stats['rejected_min_chars']}")
        logging.info(f"  Rejected max_chars: {self._stats['rejected_max_chars']}")
