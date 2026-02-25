"""
app/api/routes/conversations.py
--------------------------------
Route handlers for the /conversations endpoints.

GET /conversations              — sidebar list, all conversations newest-first.
GET /conversations/{id}         — full message history for one conversation.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.schemas.conversation import ConversationHistoryResponse, ConversationSummary, MessageItem
from app.db.repository import conversation_exists, get_all_conversations
from app.rag.pipeline import get_message_history

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversations"])


@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List all conversations",
)
def list_conversations() -> List[ConversationSummary]:
    """
    Return all conversations ordered by most-recently-updated first.
    Used by the frontend to populate the conversation sidebar.
    """
    try:
        return get_all_conversations()
    except Exception:
        logger.exception("Failed to fetch conversation list.")
        raise HTTPException(status_code=500, detail="Could not retrieve conversations.")


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    summary="Fetch full message history for a conversation",
)
def get_conversation(conversation_id: str) -> ConversationHistoryResponse:
    """
    Return the full ordered message history for a given conversation ID.
    Returns 404 if the conversation ID is not recognised.
    """
    if not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    try:
        messages = get_message_history(conversation_id)
    except Exception:
        logger.exception(
            "Failed to fetch history | conversation=%s", conversation_id
        )
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve conversation history.",
        )

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=[MessageItem(**m) for m in messages],
    )