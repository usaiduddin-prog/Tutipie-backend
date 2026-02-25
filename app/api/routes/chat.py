"""
app/api/routes/chat.py
----------------------
Route handler for POST /chat.

Responsibility: validate the request, coordinate between the DB repository
and the RAG pipeline, and return the response. No SQL, no ML logic here.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.db.repository import conversation_exists, upsert_conversation
from app.rag.pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse, summary="Send a message to the HR chatbot")
def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a question within a conversation thread.

    - Creates the conversation record on the first message.
    - On subsequent messages, only `updated_at` is touched.
    - LangGraph handles loading and persisting the full message history.
    - The conversation title defaults to the first 60 chars of the question.
    """
    logger.info(
        "Chat | conversation=%s | question=%r",
        request.conversation_id,
        request.question,
    )

    is_new_conversation = not conversation_exists(request.conversation_id)
    title = (request.title or request.question)[:60] if is_new_conversation else None
    upsert_conversation(request.conversation_id, title)

    try:
        result = run_pipeline(request.question, request.conversation_id)
    except Exception:
        logger.exception(
            "Pipeline failed | conversation=%s", request.conversation_id
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your question. Please try again.",
        )

    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=result["answer"],
        follow_up=result.get("follow_up", []),
    )