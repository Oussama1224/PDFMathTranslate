"""Persistent translation memory (SQLite).

Identical source segments translated with identical settings are reused
across runs and documents, which saves cost and guarantees consistency.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class TranslationMemory:
    def __init__(self, path: Optional[Path]):
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS tm (key TEXT PRIMARY KEY, source TEXT, target TEXT,"
                " created REAL)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - filesystem dependent
            log.warning("Translation memory disabled: %s", exc)
            self._conn = None

    @staticmethod
    def make_key(namespace: str, source: str) -> str:
        return hashlib.sha256(f"{namespace}\x1f{source}".encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        if not self._conn:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT target FROM tm WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, source: str, target: str) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO tm (key, source, target, created) VALUES (?, ?, ?, ?)",
                (key, source, target, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
