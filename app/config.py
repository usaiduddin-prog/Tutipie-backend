"""
app/config.py
─────────────
Central configuration. Every tuneable value lives here.

All values can be overridden via .env file or environment variables.

Usage:
    from app.config import settings
    settings.groq_model_name
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    db_path: Path = Path("app/db/data/chat.db")

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Primary answering model — upgraded to 70B for significantly better reasoning
    groq_model_name: str   = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1    # slight warmth for natural HR tone
    llm_max_tokens:  int   = 768    # enough for detailed policy answers

    # Rewriter model — small/fast is fine here, we just need a query string
    groq_rewrite_model: str       = "llama-3.1-8b-instant"
    llm_rewrite_temperature: float = 0.0
    llm_rewrite_max_tokens:  int   = 128

    # ── Retriever ─────────────────────────────────────────────────────────────
    embedding_model:    str  = "BAAI/bge-base-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    faiss_index_dir:    Path = Path("app/rag/faiss_store")

    # Hybrid search candidate pool sizes
    bm25_fetch_k:  int = 20    # BM25 candidates
    dense_fetch_k: int = 20    # FAISS candidates

    # Cross-encoder shortlist (from merged BM25+FAISS pool)
    cross_encoder_k: int = 20

    # Final output to LLM
    retriever_final_k:             int   = 6
    retriever_mmr_lambda:          float = 0.65   # balance relevance vs diversity
    retriever_similarity_threshold: float = 0.30  # cross-encoder confidence gate

    # RRF fusion constant (higher = less sensitive to rank differences)
    rrf_k: int = 60

    # ── API ───────────────────────────────────────────────────────────────────
    api_title:    str       = "HR Wiki RAG API"
    api_version:  str       = "3.0.0"
    cors_origins: list[str] = ["*"]   # tighten to frontend URL in production


settings = Settings()