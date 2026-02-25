import json
import logging
from langchain_core.messages import AIMessage,  HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from app.config import settings
from app.rag.graph import build_graph

logger = logging.getLogger(__name__)

def run_pipeline(query: str, conversation_id: str) -> dict:
    """
    Run the full pipeline for one user turn.
    Returns: { "answer": str, "sources": list[str], "follow_up": list[str] }
    """
    with SqliteSaver.from_conn_string(str(settings.db_path)) as checkpointer:
        graph  = build_graph(checkpointer)
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": conversation_id}},
        )
    return json.loads(result["messages"][-1].content)


def get_message_history(conversation_id: str) -> list[dict]:
    """Return conversation history as [{"role": ..., "content": ...}]."""
    with SqliteSaver.from_conn_string(str(settings.db_path)) as checkpointer:
        state = checkpointer.get({"configurable": {"thread_id": conversation_id}})

    if not state or not state.get("channel_values"):
        return []

    history = []
    for msg in state["channel_values"].get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            try:
                parsed = json.loads(msg.content)
                history.append({"role": "assistant", "content": parsed.get("answer", msg.content)})
            except json.JSONDecodeError:
                history.append({"role": "assistant", "content": msg.content})
    return history