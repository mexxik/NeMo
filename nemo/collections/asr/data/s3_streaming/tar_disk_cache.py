"""
Opt-in persistent local cache for S3 tar shards.

Disabled by default. When the env var ASR_TAR_CACHE_DIR is set, S3TarStream
downloads each shard to that directory on first touch and reads it from local
disk on every subsequent epoch — so only epoch 1 pays the network cost. The
cache is bounded (LRU by access time) via ASR_TAR_CACHE_MAX_GB (default 200);
oldest shards are evicted once the budget is exceeded.

This is the cross-epoch counterpart to TarPrefetchCache (which only keeps a
sliding window and deletes shards after one read). Use it when the training box
has spare disk; leave it unset for pure zero-disk streaming.

Env:
    ASR_TAR_CACHE_DIR     directory for cached shards; unset => caching disabled
    ASR_TAR_CACHE_MAX_GB  LRU budget in GB (default 200)
"""

import os
import threading

from nemo.utils import logging

_CACHE_DIR = os.environ.get("ASR_TAR_CACHE_DIR", "").strip() or None
try:
    _CACHE_MAX_BYTES = int(float(os.environ.get("ASR_TAR_CACHE_MAX_GB", "200")) * (1024 ** 3))
except ValueError:
    _CACHE_MAX_BYTES = 200 * (1024 ** 3)

# Serialize downloads/eviction across DataLoader worker threads in one process.
# (Separate worker *processes* each hold their own lock but share the dir; the
# atomic .part -> final rename below keeps that safe — a concurrent downloader
# simply re-fetches, and the rename wins last.)
_lock = threading.Lock()


def cache_enabled() -> bool:
    return _CACHE_DIR is not None


def _local_path(tar_key: str) -> str:
    # Flatten the key so distinct sources don't collide on a shared basename
    # like "audio_0.tar". e.g. uk/distil_uk_common/audio_0.tar
    #   -> uk__distil_uk_common__audio_0.tar
    safe = tar_key.strip("/").replace("/", "__")
    return os.path.join(_CACHE_DIR, safe)


def _evict_if_needed(keep_path: str) -> None:
    """Delete least-recently-accessed shards until under the byte budget."""
    try:
        entries = []
        total = 0
        with os.scandir(_CACHE_DIR) as it:
            for e in it:
                if not e.is_file() or e.name.endswith(".part"):
                    continue
                st = e.stat()
                entries.append((st.st_atime, st.st_size, e.path))
                total += st.st_size
        if total <= _CACHE_MAX_BYTES:
            return
        # Oldest access first
        entries.sort(key=lambda x: x[0])
        for _atime, size, path in entries:
            if total <= _CACHE_MAX_BYTES:
                break
            if path == keep_path:
                continue
            try:
                os.remove(path)
                total -= size
                logging.debug(f"[TarDiskCache] evicted {os.path.basename(path)}")
            except OSError:
                pass
    except FileNotFoundError:
        pass


def get_or_fetch(s3_client, s3_bucket: str, tar_key: str):
    """
    Return a local path to the shard, downloading it if not cached.

    Returns None on any failure so the caller can fall back to streaming.
    """
    if _CACHE_DIR is None:
        return None

    path = _local_path(tar_key)

    # Fast path: already cached. Bump atime for LRU and return.
    if os.path.exists(path):
        try:
            os.utime(path, None)
        except OSError:
            pass
        return path

    with _lock:
        # Re-check under lock (another thread may have fetched it).
        if os.path.exists(path):
            return path
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = path + ".part"
        try:
            s3_client.download_file(s3_bucket, tar_key, tmp)
            os.rename(tmp, path)
        except Exception as e:  # noqa: BLE001 — any failure => fall back to stream
            logging.warning(f"[TarDiskCache] fetch failed for {tar_key}: {e}; streaming instead")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return None
        _evict_if_needed(path)
        return path
