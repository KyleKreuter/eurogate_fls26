"""Thread-safe SQLite connection helper.

SQLite is fast enough that we open and close a connection per request rather
than maintaining a pool. ``check_same_thread=False`` lets FastAPI's threadpool
workers reuse connections safely within the lifetime of a single context.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from .config import settings


def _ensure_db_exists() -> None:
    """Raise a helpful error if the SQLite file has not been reassembled yet."""
    if not settings.db_path.exists():
        raise RuntimeError(
            f"reefer.db not found at {settings.db_path} — reassemble zips via: "
            "cat reefer.db.zip.0* > reefer.db.zip && unzip reefer.db.zip"
        )


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with ``sqlite3.Row`` factory.

    Usage::

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT ...")
            rows = cur.fetchall()

    The connection is closed automatically on exit, even if an exception
    is raised inside the ``with`` block.
    """
    _ensure_db_exists()
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
