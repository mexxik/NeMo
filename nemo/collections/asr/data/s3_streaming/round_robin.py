"""
Round-robin interleavers for multi-language sampling.

Supports two rotation modes:
- Language-level: Cycles through languages, sources sequential within each
- Source-level: Cycles through languages AND sources (each language round-robins its own sources)
"""

import re
from typing import Dict, Iterator, List, Optional, Tuple

import torch.utils.data
from nemo.utils import logging

from .lang_source_manager import LanguageSourceManager


def _parse_bucket_from_source(source_name: str) -> Optional[int]:
    """Extract bucket number from source name like 'filter5_en' -> 5."""
    m = re.match(r'filter(\d+)_', source_name)
    return int(m.group(1)) if m else None


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

        Yields one epoch of samples, then returns (raises StopIteration).
        PyTorch Lightning calls __iter__ again for the next epoch.
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

        # Reset state for this epoch
        self._active_languages = set(self.languages_order)

        while True:
            sample = self._get_next_sample()
            if sample is None:
                # All languages exhausted — epoch is done
                # Reset for next __iter__ call
                self._start_new_epoch()
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
    its own sources independently.

    **Balance modes:**
    - "none": No balancing. When a source exhausts, skip it. When all sources
      for a language exhaust, remove language from rotation.
    - "max": Restart exhausted sources with augmentation. All languages repeat
      to match the largest. cycle count increments on restart.
    - "min": Cap all languages at the minimum language's sample count.
      No restarts, no augmentation.
    - "distributed": Spread underrepresented languages evenly across the epoch.
      Uses progress-ratio-based source selection instead of round-robin.
      Each source has a budget = its sample count. The source with the lowest
      completion ratio (yielded / budget) gets picked next. Sources never
      restart — each is seen exactly once per epoch. Smaller sources are
      spread evenly across the epoch.
    """

    def __init__(
        self,
        language_sources: Dict[str, List[str]],
        source_managers: Dict[str, 'SingleSourceManager'],
        languages_order: Optional[List[str]] = None,
        samples_per_source: int = 50,
        balance_mode: str = "max",
        lang_sample_counts: Optional[Dict[str, int]] = None,
        source_sample_counts: Optional[Dict[str, int]] = None,
        curriculum_buckets: bool = False,
    ):
        """
        Initialize source-level round-robin interleaver.

        Args:
            language_sources: Dict mapping language code to list of source names
            source_managers: Dict mapping "lang:source" to SingleSourceManager
            languages_order: Optional explicit ordering of languages
            samples_per_source: Number of samples to get from each source before
                               rotating to the next.
            balance_mode: "none", "max", "min", or "distributed"
            lang_sample_counts: Per-language sample counts (required for "min")
            source_sample_counts: Per-source sample counts "lang:source" -> count
                                  (required for "distributed")
        """
        self.language_sources = language_sources
        self.source_managers = source_managers
        self.samples_per_source = samples_per_source

        # Backward compat: bool -> str
        if balance_mode is True:
            balance_mode = "max"
        elif balance_mode is False:
            balance_mode = "none"
        if balance_mode not in ("none", "max", "min", "distributed"):
            raise ValueError(f"balance_mode must be 'none', 'max', 'min', or 'distributed', got: {balance_mode}")
        self.balance_mode = balance_mode

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

        # Per-source restart count and cycle count (for balancing/augmentation)
        self._source_restarts: Dict[str, int] = {}
        self._source_cycles: Dict[str, int] = {}
        for lang, sources in language_sources.items():
            for src in sources:
                self._source_restarts[f"{lang}:{src}"] = 0
                self._source_cycles[f"{lang}:{src}"] = 0

        # Get num_workers for cycle calculation
        worker_info = torch.utils.data.get_worker_info()
        self._num_workers = worker_info.num_workers if worker_info is not None else 1

        # Per-language budgets (for "min" mode)
        self._lang_budget: Dict[str, int] = {}
        self._lang_yielded: Dict[str, int] = {lang: 0 for lang in self.languages_order}

        # Per-source budgets and yield tracking (for "distributed" mode)
        # Each source has its own budget = its sample count. Never restarts.
        # Source with lowest completion ratio gets picked next.
        self._source_budget: Dict[str, int] = {}    # "lang:source" -> total samples
        self._source_yielded: Dict[str, int] = {}   # "lang:source" -> yielded so far
        self._active_sources: set = set()            # active "lang:source" keys

        if balance_mode == "distributed":
            if source_sample_counts:
                for key in source_managers:
                    self._source_budget[key] = source_sample_counts.get(key, 0)
                    self._source_yielded[key] = 0
                self._active_sources = set(
                    k for k, v in self._source_budget.items() if v > 0
                )
            else:
                logging.warning(
                    "balance_mode='distributed' requires source_sample_counts but none provided. "
                    "Falling back to 'none' mode."
                )
                self.balance_mode = "none"
        elif balance_mode == "min":
            if lang_sample_counts:
                min_count = min(lang_sample_counts[l] for l in self.languages_order)
                for lang in self.languages_order:
                    self._lang_budget[lang] = min_count
            else:
                logging.warning(
                    "balance_mode='min' requires lang_sample_counts but none provided. "
                    "Falling back to 'none' mode."
                )
                self.balance_mode = "none"

        # Curriculum bucketing: group sources by duration bucket, process sequentially
        self.curriculum_buckets = curriculum_buckets
        self._bucket_order: List[int] = []
        self._bucket_sources: Dict[int, set] = {}  # bucket -> set of "lang:source" keys
        self._current_bucket_idx = 0

        if curriculum_buckets:
            # Parse bucket from each source name
            for key in source_managers:
                _lang, src = key.split(':', 1)
                bucket = _parse_bucket_from_source(src)
                if bucket is not None:
                    if bucket not in self._bucket_sources:
                        self._bucket_sources[bucket] = set()
                    self._bucket_sources[bucket].add(key)
                else:
                    logging.warning(f"Curriculum buckets: could not parse bucket from '{src}', skipping")

            self._bucket_order = sorted(self._bucket_sources.keys())
            if not self._bucket_order:
                logging.warning("Curriculum buckets enabled but no filterN_ sources found, disabling")
                self.curriculum_buckets = False
            else:
                logging.info(f"Curriculum buckets: {self._bucket_order}")
                for b in self._bucket_order:
                    keys = sorted(self._bucket_sources[b])
                    logging.info(f"  Bucket {b}s: {len(keys)} sources — {keys}")

        # Total sources for logging
        total_sources = sum(len(srcs) for srcs in self._sources_by_lang.values())
        logging.info(
            f"SourceRoundRobinInterleaver: {len(self.languages_order)} languages, "
            f"{total_sources} sources, {samples_per_source} samples/source batch, "
            f"balance_mode={self.balance_mode}"
        )
        if self.balance_mode == "distributed":
            for key in sorted(self._source_budget.keys()):
                logging.info(f"  [{key}] budget={self._source_budget[key]:,}")
        elif self._lang_budget:
            for lang in self.languages_order:
                budget = self._lang_budget.get(lang, 0)
                logging.info(f"  [{lang}] {len(self._sources_by_lang[lang])} sources, "
                             f"budget={budget:,}")
        else:
            for lang in self.languages_order:
                logging.info(f"  [{lang}] {len(self._sources_by_lang[lang])} sources: "
                             f"{self._sources_by_lang[lang]}")

        self._current_lang_idx = 0
        self._epoch = 0
        self._samples_yielded = 0
        self._sample_log_done = False
        self._active_languages = set(self.languages_order)

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over samples in round-robin order across languages and sources.

        Yields one epoch of samples, then returns (raises StopIteration).
        PyTorch Lightning calls __iter__ again for the next epoch.
        """
        # Set worker-specific language offset (must be done in __iter__, not __init__)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and len(self.languages_order) > 1:
            self._current_lang_idx = worker_info.id % len(self.languages_order)
            if not _in_worker():
                logging.info(
                    f"Worker {worker_info.id} starting at language index {self._current_lang_idx} "
                    f"({self.languages_order[self._current_lang_idx]})"
                )

        # Reset state for this epoch
        self._active_languages = set(self.languages_order)
        self._lang_yielded = {lang: 0 for lang in self.languages_order}

        # Reset distributed source tracking
        if self.balance_mode == "distributed":
            self._source_yielded = {k: 0 for k in self._source_budget}
            self._active_sources = set(
                k for k, v in self._source_budget.items() if v > 0
            )

        if self.curriculum_buckets:
            yield from self._iter_curriculum()
        else:
            yield from self._iter_normal()

    def _iter_normal(self) -> Iterator[dict]:
        """Standard iteration (no curriculum)."""
        while True:
            sample = self._get_next_sample()
            if sample is None:
                self._start_new_epoch()
                break

            self._samples_yielded += 1
            self._log_sample(sample)
            yield sample

    def _iter_curriculum(self) -> Iterator[dict]:
        """
        Bucket-stride curriculum: cycle through buckets 5→10→15→20→25→30→5→10→...
        yielding one sample per bucket per round (with distributed language balancing
        within each bucket). This produces mixed-duration output so each dynamic batch
        gets a natural mix of short and long samples, keeping B moderate and memory stable.
        """
        # Initialize per-bucket distributed tracking
        bucket_active = {}  # bucket -> set of active source keys
        bucket_yielded_counts = {}  # bucket -> count for logging
        for bucket in self._bucket_order:
            keys = self._bucket_sources[bucket]
            bucket_active[bucket] = set(
                k for k in keys if self._source_budget.get(k, 0) > 0
            )
            bucket_yielded_counts[bucket] = 0
            # Reset per-source yield counters
            for k in keys:
                self._source_yielded[k] = 0

        if not _in_worker():
            logging.info(f"Curriculum bucket-stride: cycling {self._bucket_order}")

        exhausted_buckets = set()

        while len(exhausted_buckets) < len(self._bucket_order):
            for bucket in self._bucket_order:
                if bucket in exhausted_buckets:
                    continue

                # Temporarily set active_sources to this bucket's sources
                self._active_sources = bucket_active[bucket]

                sample = self._get_next_sample_distributed()
                if sample is None:
                    exhausted_buckets.add(bucket)
                    if not _in_worker():
                        logging.info(
                            f"  Bucket {bucket}s exhausted: "
                            f"{bucket_yielded_counts[bucket]:,} samples"
                        )
                    continue

                self._samples_yielded += 1
                bucket_yielded_counts[bucket] += 1
                self._log_sample(sample)
                yield sample

        # All buckets done — epoch complete
        if not _in_worker():
            logging.info("Curriculum epoch summary:")
            for bucket in self._bucket_order:
                logging.info(f"  Bucket {bucket}s: {bucket_yielded_counts[bucket]:,} samples")
        self._start_new_epoch()

    def _log_sample(self, sample: dict):
        """Log sample (no-op after initial samples)."""
        pass

    def _get_next_sample(self) -> Optional[dict]:
        """Dispatch to mode-specific sample selection."""
        if self.balance_mode in ("min", "distributed"):
            return self._get_next_sample_weighted()
        else:
            return self._get_next_sample_round_robin()

    def _get_next_sample_round_robin(self) -> Optional[dict]:
        """
        Round-robin language selection (for "none" and "max" modes).

        "none": skip exhausted sources, remove exhausted languages.
        "max": restart exhausted sources with cycle increment.
        """
        if not self._active_languages:
            return None

        attempts = 0
        max_attempts = len(self.languages_order)
        restart_sources = (self.balance_mode == "max")

        while attempts < max_attempts:
            lang = self.languages_order[self._current_lang_idx]
            attempts += 1

            if lang not in self._active_languages:
                self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)
                continue

            sample = self._get_sample_from_lang(lang, restart_sources=restart_sources)
            if sample is not None:
                self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)
                return sample

            # All sources for this language exhausted
            if not restart_sources:
                self._active_languages.discard(lang)
            self._current_lang_idx = (self._current_lang_idx + 1) % len(self.languages_order)

        return None

    def _get_next_sample_weighted(self) -> Optional[dict]:
        """
        Progress-ratio-based selection for "min" and "distributed" modes.

        "distributed": picks the SOURCE with the lowest completion ratio
        (yielded / budget). Each source is seen exactly once — never restarts.
        Smaller sources get spread evenly across the epoch.

        "min": picks the LANGUAGE with the lowest completion ratio, capped
        at min(lang_counts). No restarts.
        """
        if self.balance_mode == "distributed":
            return self._get_next_sample_distributed()

        # "min" mode — language-level weighted
        if not self._active_languages:
            return None

        next_lang = min(
            self._active_languages,
            key=lambda l: self._lang_yielded[l] / max(self._lang_budget[l], 1)
        )

        sample = self._get_sample_from_lang(next_lang, restart_sources=False)
        if sample is not None:
            self._lang_yielded[next_lang] += 1
            if self._lang_yielded[next_lang] >= self._lang_budget[next_lang]:
                self._active_languages.discard(next_lang)
            return sample

        self._active_languages.discard(next_lang)
        if self._active_languages:
            return self._get_next_sample_weighted()
        return None

    def _get_next_sample_distributed(self) -> Optional[dict]:
        """
        Source-level distributed interleaving.

        Picks the source with the lowest completion ratio (yielded / budget).
        Never restarts — each source is seen exactly once per epoch.
        Smaller sources are spread evenly across the epoch.
        """
        if not self._active_sources:
            return None

        # Pick source with lowest completion ratio
        next_key = min(
            self._active_sources,
            key=lambda k: self._source_yielded[k] / max(self._source_budget[k], 1)
        )

        manager = self.source_managers[next_key]
        sample = manager.get_next_sample()

        if sample is not None:
            # Parse lang:source from key
            lang, src = next_key.split(':', 1)
            sample['lang'] = lang
            sample['source'] = src
            sample['source_cycle'] = 0  # no restarts in distributed mode

            self._source_yielded[next_key] += 1
            # Remove source when it runs out of data (not when budget reached,
            # because the real data may differ slightly from the counted budget)
            return sample

        # Source exhausted — remove it
        self._active_sources.discard(next_key)
        if not _in_worker():
            logging.debug(
                f"[{next_key}] Exhausted at {self._source_yielded[next_key]:,}"
                f"/{self._source_budget[next_key]:,} samples"
            )

        # Try again with remaining sources
        if self._active_sources:
            return self._get_next_sample_distributed()
        return None

    def _get_sample_from_lang(self, lang: str, restart_sources: bool = False) -> Optional[dict]:
        """
        Get next sample from a specific language, rotating through its sources.

        Args:
            lang: Language code
            restart_sources: If True, restart exhausted sources (for "max" mode)

        Returns:
            Sample dict with 'lang', 'source', 'source_cycle' keys, or None
        """
        sources = self._sources_by_lang[lang]
        sources_tried = 0

        # When curriculum is active, only consider sources in the current bucket
        if self.curriculum_buckets and self._bucket_order:
            current_bucket = self._bucket_order[self._current_bucket_idx]
            allowed_keys = self._bucket_sources[current_bucket]
        else:
            allowed_keys = None

        while sources_tried < len(sources):
            source_idx = self._source_idx_by_lang[lang]
            src = sources[source_idx]
            manager_key = f"{lang}:{src}"

            # Skip sources not in current bucket
            if allowed_keys is not None and manager_key not in allowed_keys:
                self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
                sources_tried += 1
                continue

            manager = self.source_managers[manager_key]

            sample = manager.get_next_sample()
            if sample is not None:
                sample['lang'] = lang
                sample['source'] = src
                sample['source_cycle'] = self._source_cycles[manager_key]
                # Rotate to next source for this language
                self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
                return sample

            # Source exhausted
            if restart_sources:
                self._source_restarts[manager_key] += 1
                self._source_cycles[manager_key] = self._source_restarts[manager_key] // self._num_workers
                manager.reset()
                if not _in_worker():
                    logging.debug(
                        f"[{lang}:{src}] Restarted (restart {self._source_restarts[manager_key]}, "
                        f"cycle {self._source_cycles[manager_key]})"
                    )
                # Try again with restarted source
                sample = manager.get_next_sample()
                if sample is not None:
                    sample['lang'] = lang
                    sample['source'] = src
                    sample['source_cycle'] = self._source_cycles[manager_key]
                    self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
                    return sample

            # Move to next source within this language
            self._source_idx_by_lang[lang] = (source_idx + 1) % len(sources)
            sources_tried += 1

        return None

    def _start_new_epoch(self):
        """Start a new epoch by resetting all source managers."""
        self._epoch += 1
        if not _in_worker():
            if self.balance_mode == "max":
                max_cycles = max(self._source_cycles.values()) if self._source_cycles else 0
                sources_with_repeats = sum(1 for c in self._source_cycles.values() if c > 0)
                logging.info(
                    f"Epoch {self._epoch} - Source balancing (max): "
                    f"{sources_with_repeats}/{len(self._source_cycles)} sources repeated, "
                    f"max cycles: {max_cycles}"
                )
            elif self.balance_mode == "distributed":
                total_yielded = sum(self._source_yielded.values())
                total_budget = sum(self._source_budget.values())
                logging.info(
                    f"Epoch {self._epoch} done - distributed: "
                    f"{total_yielded:,}/{total_budget:,} samples across "
                    f"{len(self._source_budget)} sources"
                )
                for key in sorted(self._source_yielded.keys()):
                    y = self._source_yielded[key]
                    b = self._source_budget[key]
                    logging.info(f"  [{key}] {y:,}/{b:,}")
            elif self.balance_mode == "min":
                logging.info(
                    f"Epoch {self._epoch} - Balance mode 'min': "
                    f"per-lang yields: {dict(self._lang_yielded)}"
                )
            logging.info(f"Starting epoch {self._epoch}")

        for manager in self.source_managers.values():
            manager.reset()

        self._active_languages = set(self.languages_order)
        self._current_lang_idx = 0
        self._source_idx_by_lang = {lang: 0 for lang in self.languages_order}
        # Reset per-language yield counters for budget modes
        self._lang_yielded = {lang: 0 for lang in self.languages_order}

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
            'balance_mode': self.balance_mode,
        }

        if self.balance_mode == "max":
            stats['source_cycles'] = dict(self._source_cycles)
            stats['max_source_cycle'] = max(self._source_cycles.values()) if self._source_cycles else 0
            stats['sources_repeated'] = sum(1 for c in self._source_cycles.values() if c > 0)

        if self.balance_mode in ("min", "distributed"):
            stats['lang_budgets'] = dict(self._lang_budget)
            stats['lang_yielded'] = dict(self._lang_yielded)

        return stats
