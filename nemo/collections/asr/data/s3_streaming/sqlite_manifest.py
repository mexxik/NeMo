"""
SQLite-based manifest cache for memory-efficient multi-worker data loading.

Instead of each worker loading full manifests into memory (Dict[str, dict]),
this provides a shared SQLite database that all workers can query.

Memory savings: 8 workers × 2GB manifests = 16GB → ~400MB total
"""

import json
import os
import sqlite3
from typing import Dict, Optional, Protocol, Union

from nemo.utils import logging


class ManifestProvider(Protocol):
    """Protocol for manifest entry lookup."""

    def get(self, audio_filepath: str) -> Optional[dict]:
        """Get manifest entry for an audio file."""
        ...

    def __contains__(self, audio_filepath: str) -> bool:
        """Check if audio file exists in manifest."""
        ...


class DictManifestProvider:
    """Wraps a dict to implement ManifestProvider protocol."""

    def __init__(self, entries: Dict[str, dict]):
        self.entries = entries

    def get(self, audio_filepath: str) -> Optional[dict]:
        return self.entries.get(audio_filepath)

    def __contains__(self, audio_filepath: str) -> bool:
        return audio_filepath in self.entries


class SQLiteManifestCache:
    """
    SQLite-based manifest cache for memory-efficient lookups.

    Features:
    - Single database file shared by all workers
    - WAL mode for concurrent reads
    - Indexed lookups by audio_filepath
    - Automatic schema creation

    Usage:
        # Main process: build cache
        cache = SQLiteManifestCache(db_path)
        cache.add_entries(manifest_dict, source="common_en_train")

        # Workers: query cache
        cache = SQLiteManifestCache(db_path, read_only=True)
        entry = cache.get("audio_0/sample_001.wav")
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str, read_only: bool = False):
        """
        Initialize SQLite manifest cache.

        Args:
            db_path: Path to SQLite database file
            read_only: If True, open in read-only mode (for workers)
        """
        self.db_path = db_path
        self.read_only = read_only
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            if self.read_only:
                # Read-only mode for workers
                uri = f"file:{self.db_path}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            else:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                # WAL mode for concurrent reads
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                # Create schema if needed
                self._create_schema()

            self._conn.row_factory = sqlite3.Row

        return self._conn

    def _create_schema(self):
        """Create database schema if not exists."""
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest_entries (
                audio_filepath TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                text TEXT,
                duration REAL,
                lang TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source
            ON manifest_entries(source)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Store schema version
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION))
        )
        conn.commit()

    def add_entries(self, entries: Dict[str, dict], source: str) -> int:
        """
        Add manifest entries to the cache.

        Args:
            entries: Dict mapping audio_filepath to manifest entry
            source: Source name (e.g., "common_en_train")

        Returns:
            Number of entries added
        """
        conn = self._get_connection()
        count = 0

        with conn:
            for filepath, entry in entries.items():
                conn.execute(
                    """INSERT OR REPLACE INTO manifest_entries
                       (audio_filepath, source, text, duration, lang)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        filepath,
                        source,
                        entry.get('text'),
                        entry.get('duration'),
                        entry.get('lang'),
                    )
                )
                count += 1

        return count

    def add_entries_batch(self, entries: Dict[str, dict], source: str, batch_size: int = 10000) -> int:
        """
        Add manifest entries in batches for better performance with large manifests.

        Args:
            entries: Dict mapping audio_filepath to manifest entry
            source: Source name
            batch_size: Number of entries per batch

        Returns:
            Number of entries added
        """
        conn = self._get_connection()
        count = 0
        batch = []

        for filepath, entry in entries.items():
            batch.append((
                filepath,
                source,
                entry.get('text'),
                entry.get('duration'),
                entry.get('lang'),
            ))
            count += 1

            if len(batch) >= batch_size:
                with conn:
                    conn.executemany(
                        """INSERT OR REPLACE INTO manifest_entries
                           (audio_filepath, source, text, duration, lang)
                           VALUES (?, ?, ?, ?, ?)""",
                        batch
                    )
                batch = []

        # Insert remaining
        if batch:
            with conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO manifest_entries
                       (audio_filepath, source, text, duration, lang)
                       VALUES (?, ?, ?, ?, ?)""",
                    batch
                )

        return count

    def get(self, audio_filepath: str) -> Optional[dict]:
        """
        Get manifest entry for an audio file.

        Args:
            audio_filepath: Path to audio file in TAR

        Returns:
            Manifest entry dict or None if not found
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM manifest_entries WHERE audio_filepath = ?",
            (audio_filepath,)
        ).fetchone()

        if row:
            return {
                'audio_filepath': row['audio_filepath'],
                'text': row['text'],
                'duration': row['duration'],
                'lang': row['lang'],
            }
        return None

    def __contains__(self, audio_filepath: str) -> bool:
        """Check if audio file exists in cache."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT 1 FROM manifest_entries WHERE audio_filepath = ? LIMIT 1",
            (audio_filepath,)
        ).fetchone()
        return row is not None

    def count_entries(self, source: Optional[str] = None) -> int:
        """
        Count entries in the cache.

        Args:
            source: If provided, count only entries from this source

        Returns:
            Number of entries
        """
        conn = self._get_connection()
        if source:
            row = conn.execute(
                "SELECT COUNT(*) FROM manifest_entries WHERE source = ?",
                (source,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM manifest_entries").fetchone()
        return row[0] if row else 0

    def get_sources(self) -> list:
        """Get list of all sources in the cache."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT DISTINCT source FROM manifest_entries ORDER BY source"
        ).fetchall()
        return [row[0] for row in rows]

    def clear(self):
        """Clear all entries from the cache."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM manifest_entries")

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class SQLiteManifestProvider:
    """
    Provides manifest entries from SQLite cache.

    Drop-in replacement for Dict[str, dict] in TarStream classes.
    Implements the ManifestProvider protocol.
    """

    def __init__(self, cache: SQLiteManifestCache, source: Optional[str] = None):
        """
        Initialize provider.

        Args:
            cache: SQLiteManifestCache instance
            source: Optional source filter (not used for lookups, just for info)
        """
        self.cache = cache
        self.source = source

    def get(self, audio_filepath: str) -> Optional[dict]:
        """Get manifest entry for an audio file."""
        return self.cache.get(audio_filepath)

    def __contains__(self, audio_filepath: str) -> bool:
        """Check if audio file exists in manifest."""
        return audio_filepath in self.cache

    def __getitem__(self, audio_filepath: str) -> dict:
        """Get manifest entry (raises KeyError if not found)."""
        entry = self.cache.get(audio_filepath)
        if entry is None:
            raise KeyError(audio_filepath)
        return entry


def get_cache_path(cache_dir: str, identifier: str) -> str:
    """
    Get path for SQLite cache file.

    Args:
        cache_dir: Base cache directory
        identifier: Unique identifier for this cache (e.g., hash of config)

    Returns:
        Full path to SQLite database file
    """
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"manifest_cache_{identifier}.db")


def get_default_cache_dir() -> str:
    """Get default cache directory for manifest caches."""
    return os.path.join(os.path.expanduser("~"), ".cache", "nemo_manifest_cache")
