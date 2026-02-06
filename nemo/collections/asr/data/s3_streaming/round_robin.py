"""
Round-robin interleavers for multi-language sampling.

Supports two rotation modes:
- Language-level: Cycles through languages, sources sequential within each
- Source-level: Cycles through languages AND sources (each language round-robins its own sources)
"""

from typing import Dict, Iterator, List, Optional, Tuple

import torch.utils.data
from nemo.utils import logging

from .lang_source_manager import LanguageSourceManager


def _in_worker() -> bool:
    """Check if we're running in a DataLoader worker process."""
    return torch.utils.data.get_worker_info() is not None


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
        # Set worker-specific language offset (must be done in __iter__, not __init__)
        # This prevents all workers from synchronizing on the same language
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and len(self.languages_order) > 1:
            self._current_idx = worker_info.id % len(self.languages_order)
            if not _in_worker():
                logging.info(
                    f"Worker {worker_info.id} starting at language index {self._current_idx} "
                    f"({self.languages_order[self._current_idx]})"
                )

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
            if not _in_worker():
                logging.info(f"[{lang}] Exhausted for epoch {self._epoch}")

        return None

    def _start_new_epoch(self):
        """Start a new epoch by resetting all language managers."""
        self._epoch += 1
        if not _in_worker():
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

    **Balancing with Augmentation:**
    When a source is exhausted, it's immediately restarted and its cycle count
    is incremented. Samples from cycle > 0 are marked for augmentation.
    This naturally balances smaller sources by repeating them more often
    with augmentation, giving them equal effective representation.

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
    - ...

    When a small source exhausts, it restarts with cycle=1 (augmented).
    """

    def __init__(
        self,
        language_sources: Dict[str, List[str]],
        source_managers: Dict[str, 'SingleSourceManager'],
        languages_order: Optional[List[str]] = None,
        samples_per_source: int = 50,
        balance_sources: bool = True,
    ):
        """
        Initialize source-level round-robin interleaver.

        Args:
            language_sources: Dict mapping language code to list of source names
            source_managers: Dict mapping "lang:source" to SingleSourceManager
            languages_order: Optional explicit ordering of languages
                            (defaults to sorted order)
            samples_per_source: Number of samples to get from each source before
                               rotating to the next. Higher values = fewer parallel
                               TAR streams but less fine-grained mixing. Default 50.
            balance_sources: If True, restart exhausted sources immediately with
                            augmentation. If False, wait for epoch end (old behavior).
        """
        self.language_sources = language_sources
        self.source_managers = source_managers
        self.samples_per_source = samples_per_source
        self.balance_sources = balance_sources

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

        # Per-source cycle count (for balancing/augmentation)
        # Key: "lang:source", Value: number of times source has been fully cycled
        self._source_cycles: Dict[str, int] = {}
        for lang, sources in language_sources.items():
            for src in sources:
                self._source_cycles[f"{lang}:{src}"] = 0

        # Total sources for logging
        total_sources = sum(len(srcs) for srcs in self._sources_by_lang.values())
        logging.info(
            f"SourceRoundRobinInterleaver: {len(self.languages_order)} languages, "
            f"{total_sources} sources, {samples_per_source} samples/source batch, "
            f"balance_sources={balance_sources}"
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
        # Set worker-specific language offset (must be done in __iter__, not __init__)
        # This prevents all workers from synchronizing on the same language
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and len(self.languages_order) > 1:
            self._current_lang_idx = worker_info.id % len(self.languages_order)
            if not _in_worker():
                logging.info(
                    f"Worker {worker_info.id} starting at language index {self._current_lang_idx} "
                    f"({self.languages_order[self._current_lang_idx]})"
                )

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
        Get next sample using strict round-robin: 1 sample per language, rotating.

        Order: ar→da→de→en→...→zh→ar→da→... (cycling through all languages)
        Each language also rotates through its sources.

        When balance_sources=True:
        - Exhausted sources are restarted immediately with incremented cycle count
        - Sample includes 'source_cycle' for augmentation decision

        Returns:
            Sample dict or None if all languages exhausted
        """
        if not self._active_languages:
            return None

        # Try each language starting from current index
        attempts = 0
        max_attempts = len(self.languages_order)

        while attempts < max_attempts:
            lang = self.languages_order[self._current_lang_idx]
            attempts += 1

            if lang not in self._active_languages:
                self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)
                continue

            # Get current source for this language
            sources = self._sources_by_lang[lang]
            source_idx = self._source_idx_by_lang[lang]
            src = sources[source_idx]
            manager_key = f"{lang}:{src}"

            # Try to get sample from this source
            manager = self.source_managers[manager_key]
            sample = manager.get_next_sample()

            if sample is not None:
                sample['lang'] = lang
                sample['source'] = src
                sample['source_cycle'] = self._source_cycles[manager_key]

                # Rotate to next source for this language
                self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
                # Rotate to next language
                self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)

                return sample

            # Source exhausted
            if self.balance_sources:
                # Balancing mode: restart source immediately with augmentation
                self._source_cycles[manager_key] += 1
                manager.reset()
                if not _in_worker():
                    logging.debug(
                        f"[{lang}:{src}] Restarted (cycle {self._source_cycles[manager_key]})"
                    )

                # Try again with restarted source
                sample = manager.get_next_sample()
                if sample is not None:
                    sample['lang'] = lang
                    sample['source'] = src
                    sample['source_cycle'] = self._source_cycles[manager_key]

                    # Rotate to next source and language
                    self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
                    self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)
                    return sample

            # Try other sources for this language
            sources_tried = 1
            while sources_tried < len(sources):
                self._source_idx_by_lang[lang] = (self._source_idx_by_lang[lang] + 1) % len(sources)
                source_idx = self._source_idx_by_lang[lang]
                src = sources[source_idx]
                manager_key = f"{lang}:{src}"

                manager = self.source_managers[manager_key]
                sample = manager.get_next_sample()

                if sample is not None:
                    sample['lang'] = lang
                    sample['source'] = src
                    sample['source_cycle'] = self._source_cycles[manager_key]
                    # Rotate to next source and language
                    self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
                    self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)
                    return sample

                # This source also exhausted
                if self.balance_sources:
                    self._source_cycles[manager_key] += 1
                    manager.reset()

                sources_tried += 1

            # All sources for this language exhausted (shouldn't happen with balancing)
            if not self.balance_sources:
                self._active_languages.discard(lang)
            self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)

        return None

    def _start_new_epoch(self):
        """Start a new epoch by resetting all source managers."""
        self._epoch += 1
        if not _in_worker():
            # Log cycle stats before reset
            if self.balance_sources:
                max_cycles = max(self._source_cycles.values()) if self._source_cycles else 0
                sources_with_repeats = sum(1 for c in self._source_cycles.values() if c > 0)
                logging.info(
                    f"Epoch {self._epoch} - Source balancing: "
                    f"{sources_with_repeats}/{len(self._source_cycles)} sources repeated, "
                    f"max cycles: {max_cycles}"
                )
            logging.info(f"Starting epoch {self._epoch}")

        for manager in self.source_managers.values():
            manager.reset()

        self._active_languages = set(self.languages_order)
        self._current_lang_idx = 0
        # Reset source indices for all languages
        self._source_idx_by_lang = {lang: 0 for lang in self.languages_order}

        # Note: We don't reset _source_cycles here - they accumulate across epochs
        # This allows tracking total repeats for logging/debugging

    @property
    def epoch(self) -> int:
        """Current epoch number."""
        return self._epoch

    @property
    def samples_yielded(self) -> int:
        """Total samples yielded so far."""
        return self._samples_yielded

    def get_source_cycles(self) -> Dict[str, int]:
        """Get per-source cycle counts."""
        return dict(self._source_cycles)

    def get_stats(self) -> dict:
        """Get interleaver statistics."""
        stats = {
            'epoch': self._epoch,
            'samples_yielded': self._samples_yielded,
            'active_languages': list(self._active_languages),
            'languages_order': self.languages_order,
            'source_indices': dict(self._source_idx_by_lang),
            'balance_sources': self.balance_sources,
        }

        if self.balance_sources:
            stats['source_cycles'] = dict(self._source_cycles)
            stats['max_source_cycle'] = max(self._source_cycles.values()) if self._source_cycles else 0
            stats['sources_repeated'] = sum(1 for c in self._source_cycles.values() if c > 0)

        return stats
