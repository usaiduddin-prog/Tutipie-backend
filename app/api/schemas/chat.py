"""
app/api/schemas/chat.py
-----------------------
Pydantic schemas for the /chat endpoint.
"""

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "UUID for the conversation thread. "
            "The frontend generates this on 'New Chat' and reuses it for every "
            "follow-up message in that thread."
        ),
    )
    question: str = Field(..., description="The user's HR question.")
    title: Optional[str] = Field(
        default=None,
        description=(
            "Optional human-readable title. Used only when creating a new conversation. "
            "Defaults to the first 60 characters of the question if omitted."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        return v.strip()

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("conversation_id cannot be empty.")
        return v.strip()


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    follow_up: List[str]