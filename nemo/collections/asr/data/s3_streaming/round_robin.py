"""
Round-robin interleavers for multi-language sampling.

Supports two rotation modes:
- Language-level: Cycles through languages, sources sequential within each
- Source-level: Cycles through languages AND sources (each language round-robins its own sources)
"""

from typing import Dict, Iterator, List, Optional, Tuple

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


class SourceRoundRobinInterleaver:
    """
    Interleaves samples with round-robin across languages AND sources.

    Cycles through languages in order, and each language cycles through
    its own sources independently. This ensures:
    - Equal language representation per cycle
    - Equal source representation within each language over time

    Example with:
      ar: [common_ar_test, common_ar_train, masc_ar_train, yodas_ar_ar000]
      da: [common_da_train, sdp_da_train, yodas_da_da000]
      de: [common_de_test, common_de_train, mls_de_dev, yodas_de_de000]
      en: [spgispeech_en_test, spgispeech_en_train, yodas_en_en000]

    Output order:
    - Sample 1:  common_ar_test (ar[0])
    - Sample 2:  common_da_train (da[0])
    - Sample 3:  common_de_test (de[0])
    - Sample 4:  spgispeech_en_test (en[0])
    - Sample 5:  common_ar_train (ar[1])
    - Sample 6:  sdp_da_train (da[1])
    - Sample 7:  common_de_train (de[1])
    - Sample 8:  spgispeech_en_train (en[1])
    - Sample 9:  masc_ar_train (ar[2])
    - Sample 10: yodas_da_da000 (da[2])
    - Sample 11: mls_de_dev (de[2])
    - Sample 12: yodas_en_en000 (en[2])
    - Sample 13: yodas_ar_ar000 (ar[3])
    - Sample 14: common_da_train (da[0] - wrapped)
    - Sample 15: yodas_de_de000 (de[3])
    - Sample 16: spgispeech_en_test (en[0] - wrapped)
    - ...
    """

    def __init__(
        self,
        language_sources: Dict[str, List[str]],
        source_managers: Dict[str, 'SingleSourceManager'],
        languages_order: Optional[List[str]] = None,
    ):
        """
        Initialize source-level round-robin interleaver.

        Args:
            language_sources: Dict mapping language code to list of source names
            source_managers: Dict mapping "lang:source" to SingleSourceManager
            languages_order: Optional explicit ordering of languages
                            (defaults to sorted order)
        """
        self.language_sources = language_sources
        self.source_managers = source_managers

        # Language order (sorted alphabetically by default)
        if languages_order is not None:
            self.languages_order = [
                lang for lang in languages_order
                if lang in language_sources
            ]
        else:
            self.languages_order = sorted(language_sources.keys())

        if not self.languages_order:
            raise ValueError("No valid languages provided")

        # Build per-language source lists (preserving config order)
        self._sources_by_lang: Dict[str, List[str]] = {}
        for lang in self.languages_order:
            self._sources_by_lang[lang] = list(language_sources[lang])

        # Per-language source index (for round-robin within each language)
        self._source_idx_by_lang: Dict[str, int] = {lang: 0 for lang in self.languages_order}

        # Total sources for logging
        total_sources = sum(len(srcs) for srcs in self._sources_by_lang.values())
        logging.info(
            f"SourceRoundRobinInterleaver: {len(self.languages_order)} languages, "
            f"{total_sources} sources total"
        )
        for lang in self.languages_order:
            logging.info(f"  [{lang}] {len(self._sources_by_lang[lang])} sources: {self._sources_by_lang[lang]}")

        self._current_lang_idx = 0
        self._epoch = 0
        self._samples_yielded = 0
        self._active_languages = set(self.languages_order)

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over samples in round-robin order across languages and sources.

        Yields:
            Sample dict with added 'lang' and 'source' keys
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
        Get next sample using round-robin across languages and sources.

        For each language visit, picks the next source in that language's
        rotation. Sources wrap around when exhausted.

        Returns:
            Sample dict or None if all languages exhausted
        """
        if not self._active_languages:
            return None

        # Try each language in order, starting from current index
        attempts = 0
        max_attempts = len(self.languages_order)

        while attempts < max_attempts:
            lang = self.languages_order[self._current_lang_idx]
            self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)
            attempts += 1

            if lang not in self._active_languages:
                continue

            # Get current source for this language
            sources = self._sources_by_lang[lang]
            source_idx = self._source_idx_by_lang[lang]
            src = sources[source_idx]

            # Advance to next source for this language (round-robin)
            self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)

            # Get sample from this source
            manager_key = f"{lang}:{src}"
            manager = self.source_managers[manager_key]
            sample = manager.get_next_sample()

            if sample is not None:
                sample['lang'] = lang
                sample['source'] = src
                return sample

            # This specific source exhausted - mark language as needing check
            # Try other sources in this language
            sources_tried = 1
            while sources_tried < len(sources):
                source_idx = self._source_idx_by_lang[lang]
                src = sources[source_idx]
                self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)

                manager_key = f"{lang}:{src}"
                manager = self.source_managers[manager_key]
                sample = manager.get_next_sample()

                if sample is not None:
                    sample['lang'] = lang
                    sample['source'] = src
                    return sample

                sources_tried += 1

            # All sources for this language exhausted
            self._active_languages.discard(lang)
            logging.info(f"[{lang}] All sources exhausted for epoch {self._epoch}")

        return None

    def _start_new_epoch(self):
        """Start a new epoch by resetting all source managers."""
        self._epoch += 1
        logging.info(f"Starting epoch {self._epoch}")

        for manager in self.source_managers.values():
            manager.reset()

        self._active_languages = set(self.languages_order)
        self._current_lang_idx = 0
        # Reset source indices for all languages
        self._source_idx_by_lang = {lang: 0 for lang in self.languages_order}

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
            'source_indices': dict(self._source_idx_by_lang),
        }
