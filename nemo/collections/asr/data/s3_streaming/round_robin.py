"""
Round-robin interleaver for multi-language sampling.

Cycles through languages, taking one sample from each in order.
"""

from typing import Dict, Iterator, List, Optional

from nemo.utils import logging

from .lang_source_manager import LanguageSourceManager


class RoundRobinInterleaver:
    """
    Interleaves samples from multiple languages in round-robin order.

    Example with 3 languages [en, uk, zh]:
    - Sample 1: en
    - Sample 2: uk
    - Sample 3: zh
    - Sample 4: en
    - ...

    When a language is exhausted, it's skipped until all languages are exhausted,
    then a new epoch begins and all languages reset.
    """

    def __init__(
        self,
        language_managers: Dict[str, LanguageSourceManager],
        languages_order: Optional[List[str]] = None,
    ):
        """
        Initialize round-robin interleaver.

        Args:
            language_managers: Dict mapping language code to LanguageSourceManager
            languages_order: Optional explicit ordering of languages
                           (defaults to sorted keys)
        """
        self.language_managers = language_managers

        if languages_order is not None:
            self.languages_order = [
                lang for lang in languages_order
                if lang in language_managers
            ]
        else:
            self.languages_order = sorted(language_managers.keys())

        if not self.languages_order:
            raise ValueError("No valid languages provided")

        logging.info(f"RoundRobinInterleaver: {len(self.languages_order)} languages: {self.languages_order}")

        self._current_idx = 0
        self._epoch = 0
        self._samples_yielded = 0
        self._active_languages = set(self.languages_order)

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over samples in round-robin order.

        Yields:
            Sample dict with added 'lang' key
        """
        while True:
            sample = self._get_next_sample()
            if sample is None:
                # All languages exhausted, start new epoch
                self._start_new_epoch()
                sample = self._get_next_sample()
                if sample is None:
                    # Still nothing, stop iteration
                    break

            self._samples_yielded += 1
            yield sample

    def _get_next_sample(self) -> Optional[dict]:
        """
        Get next sample using round-robin across active languages.

        Returns:
            Sample dict or None if all languages exhausted
        """
        if not self._active_languages:
            return None

        # Try each language in order, starting from current index
        attempts = 0
        max_attempts = len(self.languages_order)

        while attempts < max_attempts:
            lang = self.languages_order[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self.languages_order)
            attempts += 1

            if lang not in self._active_languages:
                continue

            manager = self.language_managers[lang]
            sample = manager.get_next_sample()

            if sample is not None:
                sample['lang'] = lang
                return sample

            # Language exhausted
            self._active_languages.discard(lang)
            logging.info(f"[{lang}] Exhausted for epoch {self._epoch}")

        return None

    def _start_new_epoch(self):
        """Start a new epoch by resetting all language managers."""
        self._epoch += 1
        logging.info(f"Starting epoch {self._epoch}")

        for lang, manager in self.language_managers.items():
            manager.reset()

        self._active_languages = set(self.languages_order)
        self._current_idx = 0

    @property
    def epoch(self) -> int:
        """Current epoch number."""
        return self._epoch

    @property
    def samples_yielded(self) -> int:
        """Total samples yielded so far."""
        return self._samples_yielded

    def get_stats(self) -> dict:
        """Get interleaver statistics."""
        return {
            'epoch': self._epoch,
            'samples_yielded': self._samples_yielded,
            'active_languages': list(self._active_languages),
            'languages_order': self.languages_order,
        }
