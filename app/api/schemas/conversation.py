"""
app/api/schemas/conversation.py
--------------------------------
Pydantic schemas for the /conversations endpoints.
"""

from typing import List
from pydantic import BaseModel


class ConversationSummary(BaseModel):
    """Single row in the sidebar list — lightweight, no message history."""
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    """A single message within a conversation history."""
    role: str     # "user" or "assistant"
    content: str


class ConversationHistoryResponse(BaseModel):
    """Full message history for one conversation."""
    conversation_id: str
    messages: List[MessageItem]