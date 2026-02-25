"""
app/db/database.py
------------------
SQLite connection management and schema initialisation.

We maintain TWO storage layers in the same file:
  1. `conversations` table  — our metadata (title, timestamps).
  2. LangGraph checkpoint tables — managed automatically by SqliteSaver,
     keyed by thread_id = conversation_id.

Both share the same DB file — one file to back up, one file to manage.
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Generator

from app.config import settings

logger = logging.getLogger(__name__)


def init_db() -> None:
    """
    Create the DB file and our metadata table if they don't exist.
    LangGraph creates its own checkpoint tables on first use.
    Called once at application startup via the lifespan hook.
    """
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
    logger.info("Database ready at %s", settings.db_path)


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection.

    - WAL journal mode: allows concurrent reads during writes.
    - row_factory = sqlite3.Row: rows behave like dicts (access by column name).
    - Auto-commits on success, rolls back on any exception.
    """
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()