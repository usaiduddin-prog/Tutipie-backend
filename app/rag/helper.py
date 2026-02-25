
from collections.abc import Sequence
import json
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

def format_history_for_rewriter(messages: Sequence[BaseMessage]) -> str:
    """Last 6 turns as a compact string for the rewriter's context window."""
    relevant = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-6:]
    lines = []
    for m in relevant:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        if isinstance(m, AIMessage):
            try:
                content = json.loads(m.content).get("answer", m.content)
            except json.JSONDecodeError:
                content = m.content
        else:
            content = m.content
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant HR documents found."
    return "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['content']}" for c in chunks
    )


def ai_message(payload: dict) -> AIMessage:
    return AIMessage(content=json.dumps(payload))

