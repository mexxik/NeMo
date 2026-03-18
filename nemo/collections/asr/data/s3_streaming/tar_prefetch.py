"""
TAR file prefetch cache for S3/Cloudflare streaming.

Downloads TAR files to local disk ahead of time in a background thread.
When the training loop needs the next TAR, it's already on local disk —
no streaming errors from rate limits or connection drops.

Usage:
    cache = TarPrefetchCache(s3_client, s3_bucket, prefetch_ahead=4, cache_dir="/tmp/tar_cache")
    cache.set_tar_files(["data/en/audio_0.tar", "data/en/audio_1.tar", ...])
    cache.start()

    # Get next TAR — blocks until downloaded, returns local path
    local_path = cache.get_next()   # → "/tmp/tar_cache/audio_0.tar"
    # ... read all samples from local_path ...
    cache.release(local_path)        # delete from disk, free space

    cache.stop()
"""

import os
import tempfile
import threading
import time
from collections import deque
from typing import List, Optional

from nemo.utils import logging


class TarPrefetchCache:
    """
    Background TAR downloader with bounded local cache.

    Maintains a sliding window of pre-downloaded TAR files:
    - Background thread downloads up to `prefetch_ahead` TARs ahead
    - Main thread calls get_next() to get the next local TAR path (blocks if not ready)
    - Main thread calls release() when done reading, which deletes the local file
    - Background thread resumes downloading when slots free up

    Thread-safe: get_next()/release() called from DataLoader workers,
    download loop runs in background thread.
    """

    def __init__(
        self,
        s3_client,
        s3_bucket: str,
        prefetch_ahead: int = 4,
        cache_dir: Optional[str] = None,
        max_retries: int = 5,
        name: str = "",
    ):
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.prefetch_ahead = prefetch_ahead
        self.max_retries = max_retries
        self.name = name

        # Local cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = tempfile.mkdtemp(prefix="tar_prefetch_")

        # TAR file queue (set by set_tar_files)
        self._tar_keys: List[str] = []
        self._download_idx = 0  # Next TAR to download

        # Ready queue: downloaded TARs waiting to be consumed
        self._ready: deque = deque()  # (tar_key, local_path) pairs
        self._ready_lock = threading.Lock()
        self._ready_event = threading.Event()  # Signaled when a TAR is ready

        # Slot semaphore: limits concurrent cached files on disk
        self._slot_sem = threading.Semaphore(prefetch_ahead)

        # Control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def set_tar_files(self, tar_keys: List[str]):
        """Set the list of TAR keys to prefetch (in order)."""
        self._tar_keys = list(tar_keys)
        self._download_idx = 0

    def start(self):
        """Start the background download thread."""
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._download_loop,
            daemon=True,
            name=f"tar-prefetch-{self.name}",
        )
        self._thread.start()
        self._started = True
        logging.info(f"[TarPrefetch:{self.name}] Started (prefetch_ahead={self.prefetch_ahead}, "
                     f"cache_dir={self.cache_dir}, {len(self._tar_keys)} TARs)")

    def stop(self):
        """Stop background thread and clean up cached files."""
        self._stop_event.set()
        # Release semaphore in case thread is blocked on it
        self._slot_sem.release()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._started = False

        # Clean up remaining cached files
        with self._ready_lock:
            while self._ready:
                _, local_path = self._ready.popleft()
                self._safe_delete(local_path)

    def get_next(self) -> Optional[str]:
        """
        Get the next downloaded TAR file path. Blocks until ready.

        Returns:
            Local file path to the downloaded TAR, or None if all TARs exhausted.
        """
        while True:
            with self._ready_lock:
                if self._ready:
                    tar_key, local_path = self._ready.popleft()
                    return local_path

            # Check if all TARs are done and nothing more coming
            if self._stop_event.is_set():
                return None
            if self._download_idx >= len(self._tar_keys) and not self._ready:
                # Wait a bit for any in-flight download
                self._ready_event.wait(timeout=1.0)
                self._ready_event.clear()
                with self._ready_lock:
                    if self._ready:
                        tar_key, local_path = self._ready.popleft()
                        return local_path
                    return None

            # Wait for next TAR to be ready
            self._ready_event.wait(timeout=5.0)
            self._ready_event.clear()

    def release(self, local_path: str):
        """Release a consumed TAR file — deletes from disk and frees a prefetch slot."""
        self._safe_delete(local_path)
        self._slot_sem.release()

    def _download_loop(self):
        """Background thread: downloads TARs ahead of consumption."""
        while not self._stop_event.is_set():
            if self._download_idx >= len(self._tar_keys):
                # All TARs queued for download — signal and exit
                self._ready_event.set()
                return

            # Wait for a free slot (released when consumer calls release())
            acquired = self._slot_sem.acquire(timeout=1.0)
            if not acquired:
                continue
            if self._stop_event.is_set():
                return

            tar_key = self._tar_keys[self._download_idx]
            self._download_idx += 1

            # Download with retries
            local_path = self._download_tar(tar_key)

            if local_path is not None:
                with self._ready_lock:
                    self._ready.append((tar_key, local_path))
                self._ready_event.set()
            else:
                # Download failed — release slot, skip this TAR
                self._slot_sem.release()
                logging.error(f"[TarPrefetch:{self.name}] Skipping {tar_key} after all retries failed")

    def _download_tar(self, tar_key: str) -> Optional[str]:
        """Download a single TAR file to local cache with retries."""
        # Use the TAR filename as local name
        tar_basename = os.path.basename(tar_key)
        local_path = os.path.join(self.cache_dir, tar_basename)

        for attempt in range(self.max_retries):
            if self._stop_event.is_set():
                return None

            try:
                tmp_path = local_path + ".downloading"
                self.s3_client.download_file(self.s3_bucket, tar_key, tmp_path)
                os.rename(tmp_path, local_path)
                logging.debug(f"[TarPrefetch:{self.name}] Downloaded {tar_key} → {local_path}")
                return local_path

            except Exception as e:
                self._safe_delete(tmp_path)
                if attempt < self.max_retries - 1:
                    sleep_time = min(2 ** attempt, 30)  # 1s, 2s, 4s, 8s, 16s, max 30s
                    logging.warning(
                        f"[TarPrefetch:{self.name}] Download failed {tar_key}: {e}, "
                        f"retry {attempt + 1}/{self.max_retries} in {sleep_time}s"
                    )
                    time.sleep(sleep_time)
                else:
                    logging.error(
                        f"[TarPrefetch:{self.name}] Download failed {tar_key} after "
                        f"{self.max_retries} retries: {e}"
                    )

        return None

    @staticmethod
    def _safe_delete(path: str):
        """Delete a file, ignoring errors if it doesn't exist."""
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
