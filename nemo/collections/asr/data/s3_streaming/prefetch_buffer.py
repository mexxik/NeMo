"""
Prefetch buffer for S3 streaming dataset.

Uses a background thread to stream and process samples ahead of consumption,
reducing I/O wait time during training.
"""

import queue
import threading
from typing import Callable, Iterator, Optional

from nemo.utils import logging


class PrefetchBuffer:
    """
    Prefetches samples from an iterator using a background thread.

    The background thread continuously pulls from the source iterator
    and fills a queue. The main thread consumes from the queue,
    allowing streaming I/O to happen in parallel with GPU training.
    """

    # Sentinel value to signal end of iteration
    _STOP = object()

    def __init__(
        self,
        source_factory: Callable[[], Iterator[dict]],
        buffer_size: int = 100,
        name: str = "prefetch",
    ):
        """
        Initialize prefetch buffer.

        Args:
            source_factory: Callable that returns a fresh iterator.
                           Called each time we need to restart iteration.
            buffer_size: Number of samples to buffer ahead
            name: Name for logging
        """
        self.source_factory = source_factory
        self.buffer_size = buffer_size
        self.name = name

        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error: Optional[Exception] = None
        self._started = False

    def _producer_loop(self):
        """Background thread: stream samples into queue."""
        try:
            source = self.source_factory()

            for sample in source:
                if self._stop_event.is_set():
                    break

                # Block until queue has space
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(sample, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        except Exception as e:
            self._error = e
            logging.error(f"[{self.name}] Prefetch error: {e}")

        finally:
            # Signal end of iteration
            try:
                self._queue.put(self._STOP, timeout=1.0)
            except queue.Full:
                pass

    def start(self):
        """Start the background prefetch thread."""
        if self._started:
            return

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name=f"prefetch-{self.name}"
        )
        self._thread.start()
        self._started = True

    def stop(self):
        """Stop the background thread."""
        if not self._started:
            return

        self._stop_event.set()

        # Drain queue to unblock producer
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._started = False

    def __iter__(self):
        """Iterate over prefetched samples."""
        self.start()

        while True:
            try:
                # Get with timeout to allow checking for errors
                item = self._queue.get(timeout=1.0)

                if item is self._STOP:
                    break

                yield item

            except queue.Empty:
                # Check for errors from producer
                if self._error:
                    raise self._error
                # Check if producer is still alive
                if self._thread and not self._thread.is_alive():
                    break
                continue

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class MultiSourcePrefetcher:
    """
    Prefetches from multiple language sources with separate buffers.

    Each language source gets its own background thread, allowing
    parallel streaming from multiple S3 prefixes.
    """

    def __init__(
        self,
        source_factories: dict,  # lang -> callable returning iterator
        buffer_size_per_source: int = 50,
    ):
        """
        Initialize multi-source prefetcher.

        Args:
            source_factories: Dict mapping language to callable that returns iterator
            buffer_size_per_source: Buffer size for each language
        """
        self.buffers = {}
        for lang, factory in source_factories.items():
            self.buffers[lang] = PrefetchBuffer(
                source_factory=factory,
                buffer_size=buffer_size_per_source,
                name=lang,
            )

    def start_all(self):
        """Start all prefetch threads."""
        for buffer in self.buffers.values():
            buffer.start()

    def stop_all(self):
        """Stop all prefetch threads."""
        for buffer in self.buffers.values():
            buffer.stop()

    def get_buffer(self, lang: str) -> PrefetchBuffer:
        """Get buffer for a specific language."""
        return self.buffers.get(lang)
