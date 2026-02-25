from typing import Annotated, List, Optional, Sequence
from langchain_core.messages import  BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from app.rag.generator import get_llm, get_rewrite_llm
from app.rag.prompt import QUERY_EXPANSION_PROMPT, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from app.rag.retriever import multi_retrieve
from app.rag.helper import format_history_for_rewriter, format_context, ai_message
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

class HRResponse(BaseModel):
    answer: str
    sources: List[str]
    follow_up: List[str]

class ConversationState(TypedDict):
    messages:         Annotated[Sequence[BaseMessage], add_messages]
    rewritten_query:  Optional[str]
    expanded_queries: Optional[List[str]]

_rewrite_llm           = get_rewrite_llm()
_expand_llm            = get_rewrite_llm()
_answer_llm            = get_llm(temperature=0.1, max_tokens=768)
_answer_llm_structured = _answer_llm.with_structured_output(HRResponse)

_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

_answer_chain = _answer_prompt | _answer_llm_structured

# ── Node 1: Query rewriting ────────────────────────────────────────────────────

def _rewrite_query(state: ConversationState) -> dict:
    """Restructure the user's raw message into an optimised retrieval query."""
    latest = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not latest:
        return {"rewritten_query": "", "expanded_queries": []}

    raw_query   = latest.content.strip()
    history_str = format_history_for_rewriter(state["messages"][:-1])

    prompt    = QUERY_REWRITE_PROMPT.format(history=history_str, raw_query=raw_query)
    rewritten = _rewrite_llm.invoke([HumanMessage(content=prompt)]).content.strip()

    logger.info("Query rewrite: %r → %r", raw_query, rewritten)
    return {"rewritten_query": rewritten, "expanded_queries": [rewritten]}

# ── Node 2: Query expansion ────────────────────────────────────────────────────

def _expand_queries(state: ConversationState) -> dict:
    """
    Ask the LLM to generate 4 alternative phrasings of the rewritten query.
    Merges them with the original to produce [original, v1, v2, v3, v4].
    Falls back gracefully if the LLM returns malformed JSON.
    """
    rewritten = state.get("rewritten_query", "").strip()
    if not rewritten:
        return {"expanded_queries": []}

    prompt   = QUERY_EXPANSION_PROMPT.format(rewritten_query=rewritten)
    raw_resp = _expand_llm.invoke([HumanMessage(content=prompt)]).content.strip()

    variants: list[str] = []
    try:
        clean = raw_resp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            variants = [str(v).strip() for v in parsed if str(v).strip()]
    except (json.JSONDecodeError, ValueError):
        logger.warning("Query expansion returned malformed JSON: %r", raw_resp)

    all_queries = [rewritten] + [v for v in variants if v != rewritten]

    logger.info(
        "Query expansion: %r → %d variants: %s",
        rewritten, len(all_queries), all_queries,
    )
    return {"expanded_queries": all_queries}

# ── Node 3: Retrieve + Answer ─────────────────────────────────────────────────

def _retrieve_and_answer(state: ConversationState) -> dict:
    """
    Run multi-query retrieval using all expanded variants, then generate answer.
    """
    queries = state.get("expanded_queries") or []
    if not queries:
        q = state.get("rewritten_query", "").strip()
        queries = [q] if q else []

    chunks = multi_retrieve(queries) or [] if queries else []

    if chunks:
        logger.info(
            "Retrieved %d chunks from: %s",
            len(chunks), ", ".join({c["title"] for c in chunks}),
        )
    else:
        logger.info("No chunks retrieved for queries: %r", queries)

    context       = format_context(chunks)
    source_titles = list({c["title"] for c in chunks})

    response: HRResponse = _answer_chain.invoke({
        "context":  context,
        "messages": state["messages"],
    })

    payload = response.model_dump()

    if not payload.get("sources") and source_titles:
        payload["sources"] = source_titles

    return {"messages": [ai_message(payload)]}

# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(ConversationState)

    builder.add_node("rewrite_query",       _rewrite_query)
    builder.add_node("expand_queries",      _expand_queries)       
    builder.add_node("retrieve_and_answer", _retrieve_and_answer)

    builder.add_edge(START,                 "rewrite_query")
    builder.add_edge("rewrite_query",       "expand_queries")      
    builder.add_edge("expand_queries",      "retrieve_and_answer") 
    builder.add_edge("retrieve_and_answer", END)

    return builder.compile(checkpointer=checkpointer)