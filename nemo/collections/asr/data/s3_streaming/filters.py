"""
Sample filtering for S3 streaming dataset.

Provides configurable filters for duration, character rate, and text quality.
"""

from dataclasses import dataclass
from typing import Optional

from nemo.utils import logging


@dataclass
class FilterConfig:
    """Configuration for sample filtering."""
    min_duration: float = 0.5
    max_duration: float = 15.0
    max_chars_per_sec: float = 25.0
    min_chars: int = 1
    max_chars: Optional[int] = None


class SampleFilter:
    """
    Filters samples based on configurable criteria.

    Filters:
    - Duration bounds (min/max)
    - Character rate (chars per second)
    - Text length bounds
    """

    def __init__(self, config: FilterConfig):
        self.config = config
        self._stats = {
            'total': 0,
            'passed': 0,
            'rejected_min_duration': 0,
            'rejected_max_duration': 0,
            'rejected_char_rate': 0,
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

        # Character rate filter (prevents RNN-T loss issues)
        if duration > 0 and len(text) / duration > self.config.max_chars_per_sec:
            self._stats['rejected_char_rate'] += 1
            return False

        # Text length filters
        if len(text) < self.config.min_chars:
            self._stats['rejected_min_chars'] += 1
            return False

        if self.config.max_chars is not None and len(text) > self.config.max_chars:
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
        logging.info(f"  Rejected char_rate: {self._stats['rejected_char_rate']}")
        logging.info(f"  Rejected min_chars: {self._stats['rejected_min_chars']}")
        logging.info(f"  Rejected max_chars: {self._stats['rejected_max_chars']}")
