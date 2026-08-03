"""
Duration-window curriculum for the streaming dataset.

balance_mode="curriculum" presents an epoch shortest-utterance-first: the
duration range is cut into N equal-count windows and the epoch runs window 0
(shortest) through window N-1 (longest). Within a window, source selection is
exactly "distributed" — each source keeps a budget equal to its sample count
*in that window*, so language/source balance is preserved inside every stage of
the curriculum instead of only across the epoch as a whole.

Unlike the older `curriculum_buckets` path, this needs no `filterN_` source
naming and no re-sharding: window membership is a numeric test on the duration
already stored in the SQLite manifest cache. The gate is applied inside the TAR
streams *before* the member payload is extracted and decoded, so samples outside
the active window cost a header read and a dict lookup, not an audio decode.

Cost: the shards get header-scanned once per window instead of once per epoch.
Measured on distil_uk_common (3.4 GB shard, 25k clips, N=4) this is ~4.5x the
scan time of a single pass, which is far inside the loader's headroom — training
consumes ~105 samples/s while even the gated path delivers ~880 samples/s from a
single worker.
"""

from typing import Dict, List, Optional, Tuple

from nemo.utils import logging

# Histogram resolution used to derive quantile edges. 0.05s over a 0-30s range
# is 600 bins, fine enough that snapping edges to bin boundaries does not
# meaningfully skew the equal-count split.
_BIN_WIDTH = 0.05


class DurationWindow:
    """
    Mutable duration gate shared by reference between the interleaver and every
    TAR stream it drives.

    The interleaver advances the window as the curriculum progresses; streams
    read `lo`/`hi` live on each member, so an advance takes effect immediately
    without rebuilding anything. `active=False` disables the gate entirely
    (every non-curriculum mode).
    """

    __slots__ = ("lo", "hi", "active", "index")

    def __init__(self, lo: float = 0.0, hi: float = float("inf"), active: bool = False):
        self.lo = lo
        self.hi = hi
        self.active = active
        self.index = 0

    def accepts(self, duration: Optional[float]) -> bool:
        """Half-open [lo, hi) test. Samples with no duration are never gated out."""
        if not self.active:
            return True
        if duration is None:
            return True
        return self.lo <= duration < self.hi

    def set(self, index: int, lo: float, hi: float):
        self.index = index
        self.lo = lo
        self.hi = hi
        self.active = True

    def disable(self):
        self.active = False
        self.lo = 0.0
        self.hi = float("inf")

    def __repr__(self):
        if not self.active:
            return "DurationWindow(inactive)"
        return f"DurationWindow(#{self.index} [{self.lo:.2f}, {self.hi:.2f})s)"


def compute_duration_windows(
    sqlite_cache_path: str,
    language_sources: Dict[str, List[str]],
    num_windows: int,
    min_duration: float,
    max_duration: float,
) -> Tuple[List[Tuple[float, float]], List[Dict[str, int]]]:
    """
    Derive equal-count duration windows and per-window source budgets.

    Runs a single grouped scan of the manifest cache and builds a
    (source x duration-bin) histogram, then cuts the global histogram at
    equal-count quantiles.

    Args:
        sqlite_cache_path: path to the manifest cache built by build_cache.py
        language_sources: {lang: [source, ...]} as configured for training
        num_windows: how many curriculum stages to split the epoch into
        min_duration / max_duration: the same bounds the sample filter applies

    Returns:
        (windows, per_window_counts) where windows is a list of [lo, hi) pairs
        ascending by duration, and per_window_counts[i] maps "lang:source" ->
        number of samples that source contributes to window i.

        Returns ([], []) if the cache is unusable or holds no matching rows, so
        the caller can fall back to plain "distributed".
    """
    from .sqlite_manifest import SQLiteManifestCache

    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")

    # source name -> "lang:source" key. A source used by two languages would
    # collide here, but that is already true of the sample-count queries.
    key_by_source: Dict[str, str] = {}
    for lang, sources in language_sources.items():
        for source in sources:
            key_by_source[source] = f"{lang}:{source}"

    if not key_by_source:
        return [], []

    try:
        cache = SQLiteManifestCache(sqlite_cache_path, read_only=True)
        conn = cache._get_connection()

        placeholders = ",".join("?" for _ in key_by_source)
        rows = conn.execute(
            f"""
            SELECT source,
                   CAST(duration / ? AS INTEGER) AS bin,
                   COUNT(*) AS cnt
            FROM manifest_entries
            WHERE source IN ({placeholders})
              AND duration >= ? AND duration <= ?
            GROUP BY source, bin
            """,
            (_BIN_WIDTH, *key_by_source.keys(), min_duration, max_duration),
        ).fetchall()
        cache.close()
    except Exception as e:
        logging.warning(f"Curriculum: could not read duration histogram from cache: {e}")
        return [], []

    if not rows:
        logging.warning("Curriculum: no manifest rows matched the configured sources")
        return [], []

    # bin -> total count, and (source, bin) -> count
    global_hist: Dict[int, int] = {}
    source_hist: Dict[str, Dict[int, int]] = {}
    total = 0
    for row in rows:
        source = row["source"]
        b = int(row["bin"])
        cnt = int(row["cnt"])
        global_hist[b] = global_hist.get(b, 0) + cnt
        source_hist.setdefault(source, {})[b] = cnt
        total += cnt

    if total == 0:
        return [], []

    # Walk the histogram in ascending duration, cutting at equal-count quantiles.
    # Edges land on bin boundaries; a bin is never split across two windows.
    edges: List[float] = [min_duration]
    target = total / num_windows
    cumulative = 0
    next_cut = 1
    for b in sorted(global_hist):
        cumulative += global_hist[b]
        while next_cut < num_windows and cumulative >= target * next_cut:
            edge = (b + 1) * _BIN_WIDTH
            # Degenerate case: one duration value holds more than a window's
            # worth of samples, so several cuts land on the same boundary. Emit
            # the edge once and let the resulting window count come out below
            # num_windows rather than creating empty windows.
            if edge > edges[-1]:
                edges.append(edge)
            next_cut += 1
    # Last window is closed at the top; nudge past max_duration so the half-open
    # [lo, hi) test still accepts a sample sitting exactly on max_duration.
    edges.append(max_duration + 1e-6)

    windows = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    if len(windows) < num_windows:
        logging.warning(
            f"Curriculum: durations too concentrated to cut {num_windows} equal-count "
            f"windows; using {len(windows)}"
        )

    # Per-window budgets, keyed the way the interleaver keys its sources.
    per_window_counts: List[Dict[str, int]] = []
    for lo, hi in windows:
        lo_bin = int(lo / _BIN_WIDTH)
        hi_bin = int(hi / _BIN_WIDTH)
        counts: Dict[str, int] = {}
        for source, hist in source_hist.items():
            key = key_by_source[source]
            n = sum(c for b, c in hist.items() if lo_bin <= b < hi_bin)
            if n > 0:
                counts[key] = n
        per_window_counts.append(counts)

    return windows, per_window_counts


def log_curriculum_plan(
    windows: List[Tuple[float, float]],
    per_window_counts: List[Dict[str, int]],
):
    """Log the derived curriculum so a run's ordering is auditable from the log."""
    total = sum(sum(c.values()) for c in per_window_counts)
    logging.info(f"Curriculum: {len(windows)} duration windows, {total:,} samples/epoch")
    for i, (lo, hi) in enumerate(windows):
        counts = per_window_counts[i]
        n = sum(counts.values())
        pct = 100.0 * n / total if total else 0.0
        logging.info(
            f"  window {i}: [{lo:.2f}, {hi:.2f})s  {n:,} samples ({pct:.1f}%), "
            f"{len(counts)} sources"
        )
        for key in sorted(counts):
            logging.info(f"      [{key}] {counts[key]:,}")
