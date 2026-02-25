from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.config import settings

load_dotenv()


def get_llm(
    model_name:  str   = settings.groq_model_name,
    temperature: float = settings.llm_temperature,
    max_tokens:  int   = settings.llm_max_tokens,
) -> ChatGroq:
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_rewrite_llm() -> ChatGroq:
    """
    Fast, low-temperature LLM for query rewriting and expansion.
    Used by both the rewrite node and the expand node.
    max_tokens=256 — enough for a JSON array of 4 query strings.
    """
    return ChatGroq(
        model=settings.groq_rewrite_model,
        temperature=settings.llm_rewrite_temperature,
        max_tokens=256,
    )