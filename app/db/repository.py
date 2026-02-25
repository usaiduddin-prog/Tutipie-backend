"""
app/db/repository.py
--------------------
Repository pattern: every SQL query in one place.

Routes and services never write raw SQL — they call functions from here.
This makes queries easy to find, test, and change without touching business logic.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from app.db.database import get_connection

logger = logging.getLogger(__name__)


# ── Conversations ──────────────────────────────────────────────────────────────

def upsert_conversation(conversation_id: str, title: Optional[str] = None) -> None:
    """
    Insert a new conversation row, or touch `updated_at` if it already exists.
    Title is only written on INSERT — never overwritten on update.
    """
    resolved_title = title or "New Conversation"
    now = _utc_now()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (:id, :title, :now, :now)
            ON CONFLICT(id) DO UPDATE SET updated_at = :now
            """,
            {"id": conversation_id, "title": resolved_title, "now": now},
        )


def update_conversation_title(conversation_id: str, title: str) -> None:
    """Overwrite the human-readable title for an existing conversation."""
    now = _utc_now()
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conversation_id),
        )


def get_all_conversations() -> list[dict]:
    """
    Return all conversations, newest-first.
    Each dict: {id, title, created_at, updated_at}
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def conversation_exists(conversation_id: str) -> bool:
    """Return True if a conversation with this ID exists in the DB."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return row is not None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()