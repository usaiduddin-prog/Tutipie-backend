import json
import logging
import os
import pickle
import re
from functools import lru_cache
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

class RetrieverConfig:
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "app/rag/faiss_store")

    # Models
    embedding_model: str     = "BAAI/bge-base-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # BGE query prefix (critical for retrieval quality — model card requirement)
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Stage 1 — candidates per source (increased for better recall)
    bm25_fetch_k: int  = 40
    dense_fetch_k: int = 40

    # Stage 2 — cross-encoder shortlist
    cross_encoder_k: int = 30

    # Stage 3 — final output
    final_k: int       = 6
    mmr_lambda: float  = 0.80   # raised: favour relevance over diversity
    # Lowered threshold — was 0.30, which silently dropped valid results.
    # Set to None to disable (recommended while debugging).
    similarity_threshold: Optional[float] = 0.10

    # RRF constant
    rrf_k: int = 60

    # Heuristic query expansion — disabled because the LLM expansion node
    # now handles this properly via multi_retrieve().
    enable_query_expansion: bool = False
    # Maximum number of sub-queries to run (including original)
    max_sub_queries: int = 3


cfg = RetrieverConfig()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation tokeniser for BM25."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _expand_query(query: str) -> list[str]:
    """
    Lightweight query expansion without an LLM call.

    Heuristics:
    - Split compound "X and Y" questions into individual queries.
    - Add a short noun-phrase version for dense search.
    Returns a deduplicated list starting with the original query.
    """
    variants: list[str] = [query]

    # Split on "and" / "&" / "," when sentence contains a question word
    if re.search(r"\b(what|how|why|when|where|who)\b", query, re.I):
        parts = re.split(r"\s+and\s+|\s*&\s*|\s*,\s*", query, flags=re.I)
        parts = [p.strip() for p in parts if len(p.strip()) > 15]
        variants.extend(parts)

    # Add a keyword-only variant (strip question words + stopwords)
    stopwords = {"what", "is", "are", "the", "a", "an", "of", "in",
                 "how", "does", "do", "can", "i", "to", "for", "on",
                 "at", "by", "with", "this", "that", "it", "be"}
    keywords = [w for w in _tokenize(query) if w not in stopwords and len(w) > 2]
    if keywords:
        kw_query = " ".join(keywords[:8])
        if kw_query not in variants:
            variants.append(kw_query)

    # Deduplicate while preserving order, cap at max_sub_queries
    seen: set[str] = set()
    result: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            result.append(v)
        if len(result) >= cfg.max_sub_queries:
            break
    return result


# ── Resource loading (module-level singleton) ─────────────────────────────────

def _load(name: str, path: str, loader):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    logger.info("Loading %s from %s", name, path)
    return loader(path)


_faiss_index: faiss.Index = _load(
    "FAISS index",
    os.path.join(cfg.faiss_index_dir, "index.faiss"),
    faiss.read_index,
)

with open(os.path.join(cfg.faiss_index_dir, "bm25.pkl"), "rb") as _f:
    _bm25: BM25Okapi = pickle.load(_f)

with open(os.path.join(cfg.faiss_index_dir, "metadata.json"), "r", encoding="utf-8") as _f:
    _metadata: list[dict] = json.load(_f)

_embedder     = SentenceTransformer(cfg.embedding_model)
_cross_encoder = CrossEncoder(cfg.cross_encoder_model, max_length=512)

logger.info(
    "Retriever ready — FAISS: %d vectors | BM25: %d docs",
    _faiss_index.ntotal,
    len(_metadata),
)


# ── Stage 1a: BM25 sparse search ──────────────────────────────────────────────

def _bm25_search(query: str, k: int) -> list[tuple[int, float]]:
    """Returns [(doc_index, bm25_score), ...] sorted descending."""
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_k  = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


# ── Stage 1b: Dense FAISS search ──────────────────────────────────────────────

def _encode_query(query: str) -> np.ndarray:
    """
    Encode a query with the BGE-required prefix for correct retrieval behaviour.
    Returns shape (1, dim) float32 array.
    """
    prefixed = cfg.bge_query_prefix + query
    emb = _embedder.encode(
        [prefixed], normalize_embeddings=True, show_progress_bar=False
    )
    return emb.astype("float32")


def _dense_search(query_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Returns [(doc_index, cosine_score), ...] sorted descending."""
    scores, indices = _faiss_index.search(query_emb, k)
    return [
        (int(idx), float(sc))
        for idx, sc in zip(indices[0], scores[0])
        if idx >= 0
    ]


# ── Stage 2: Reciprocal Rank Fusion ───────────────────────────────────────────

def _rrf_merge(
    ranked_lists: list[list[tuple[int, float]]],
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    Supports any number of lists (BM25 + dense + sub-queries).
    Returns merged list sorted by RRF score descending.
    """
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── Stage 3: Cross-encoder re-ranking ─────────────────────────────────────────

def _cross_encode(
    query: str,
    candidates: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """
    Score each candidate with the cross-encoder and re-sort.
    Uses the full content field — ensure metadata['content'] is not truncated.
    Returns [(doc_index, ce_score), ...] sorted descending.
    """
    pairs     = [(query, _metadata[idx]["content"]) for idx, _ in candidates]
    ce_scores = _cross_encoder.predict(pairs, show_progress_bar=False)

    # sigmoid-normalise so scores are in (0, 1) — makes threshold meaningful
    ce_scores = 1.0 / (1.0 + np.exp(-np.array(ce_scores)))

    reranked = sorted(
        zip([idx for idx, _ in candidates], ce_scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(int(idx), float(sc)) for idx, sc in reranked]


# ── Stage 4: MMR de-duplication ───────────────────────────────────────────────

def _mmr(
    query_emb: np.ndarray,
    candidates: list[tuple[int, float]],
    k: int,
    lambda_param: float,
) -> list[int]:
    """
    Maximal Marginal Relevance.
    Balances relevance to query vs. diversity among selected chunks.
    Returns final list of doc indices in MMR-selection order.
    """
    candidate_indices = [idx for idx, _ in candidates]
    if not candidate_indices:
        return []

    # Reconstruct embeddings from FAISS (avoids storing a separate numpy array)
    all_embs = np.vstack([
        _faiss_index.reconstruct(idx) for idx in candidate_indices
    ]).astype("float32")

    # Normalise query embedding for consistent dot-product = cosine similarity
    q = query_emb / (np.linalg.norm(query_emb) + 1e-10)

    selected: list[int] = []
    remaining = list(range(len(candidate_indices)))

    while len(selected) < min(k, len(candidate_indices)):
        rel_scores = all_embs[remaining] @ q

        if not selected:
            best_pos = remaining[int(np.argmax(rel_scores))]
        else:
            sel_embs   = all_embs[selected]
            redundancy = (all_embs[remaining] @ sel_embs.T).max(axis=1)
            mmr_scores = lambda_param * rel_scores - (1 - lambda_param) * redundancy
            best_pos   = remaining[int(np.argmax(mmr_scores))]

        selected.append(best_pos)
        remaining.remove(best_pos)

    return [candidate_indices[i] for i in selected]


# ── Diagnostic helper ──────────────────────────────────────────────────────────

def debug_retrieve(query: str, top_n: int = 10) -> None:
    """
    Print raw pipeline results without threshold/MMR filtering.
    Use this to check whether relevant content is in the index at all.

    Example:
        from retriever import debug_retrieve
        debug_retrieve("your failing query")
    """
    query_emb = _encode_query(query)
    bm25_hits  = _bm25_search(query, cfg.bm25_fetch_k)
    dense_hits = _dense_search(query_emb, cfg.dense_fetch_k)
    merged     = _rrf_merge([bm25_hits, dense_hits])
    reranked   = _cross_encode(query, merged[:cfg.cross_encoder_k])

    print(f"\n=== debug_retrieve: '{query}' ===")
    for rank, (idx, score) in enumerate(reranked[:top_n], 1):
        snippet = _metadata[idx]["content"][:200].replace("\n", " ")
        print(f"[{rank}] score={score:.4f}  chunk={_metadata[idx].get('chunk_id')}  {snippet!r}")
    print()


# ── Public interface ───────────────────────────────────────────────────────────

def retrieve(query: str) -> Optional[List[Dict]]:
    """
    Full hybrid retrieval pipeline with optional query expansion.

    Pipeline:
        1. Encode query (with BGE prefix)
        2. [Optional] Expand into sub-queries
        3. BM25 + dense search for each sub-query
        4. RRF merge across all ranked lists
        5. Cross-encoder re-ranking (sigmoid-normalised scores)
        6. MMR de-duplication
        7. Threshold filter + build response dicts

    Returns:
        List of dicts: { title, url, chunk_id, content, score }
        None if no chunks pass the confidence threshold.
    """

    # ── 1. Encode query ────────────────────────────────────────────────────────
    query_emb = _encode_query(query)  # shape (1, dim)

    # ── 2. Build sub-query list ────────────────────────────────────────────────
    sub_queries = _expand_query(query) if cfg.enable_query_expansion else [query]
    logger.debug("Sub-queries: %s", sub_queries)

    # ── 3. Dual retrieval for every sub-query ──────────────────────────────────
    all_ranked_lists: list[list[tuple[int, float]]] = []

    for sq in sub_queries:
        sq_emb = _encode_query(sq) if sq != query else query_emb
        all_ranked_lists.append(_bm25_search(sq, cfg.bm25_fetch_k))
        all_ranked_lists.append(_dense_search(sq_emb, cfg.dense_fetch_k))

    # ── 4. RRF merge ───────────────────────────────────────────────────────────
    merged         = _rrf_merge(all_ranked_lists, rrf_k=cfg.rrf_k)
    top_candidates = merged[: cfg.cross_encoder_k]

    if not top_candidates:
        logger.info("No candidates after RRF merge.")
        return None

    # ── 5. Cross-encoder re-ranking ────────────────────────────────────────────
    reranked = _cross_encode(query, top_candidates)

    top_score = reranked[0][1] if reranked else 0.0
    logger.debug(
        "Cross-encoder top score: %.4f (threshold: %s)",
        top_score,
        cfg.similarity_threshold,
    )

    # Early exit guard
    if cfg.similarity_threshold is not None and (
        not reranked or top_score < cfg.similarity_threshold
    ):
        logger.info(
            "Best cross-encoder score %.4f below threshold %.2f — no relevant chunks.",
            top_score,
            cfg.similarity_threshold,
        )
        return None

    # ── 6. MMR de-duplication ──────────────────────────────────────────────────
    final_indices = _mmr(
        query_emb=query_emb[0],
        candidates=reranked,
        k=cfg.final_k,
        lambda_param=cfg.mmr_lambda,
    )

    # ── 7. Build response, apply per-chunk threshold ───────────────────────────
    ce_score_map = {idx: sc for idx, sc in reranked}
    results: list[dict] = []

    for idx in final_indices:
        score = ce_score_map.get(idx, 0.0)
        if cfg.similarity_threshold is not None and score < cfg.similarity_threshold:
            continue
        meta = _metadata[idx]
        results.append({
            "title":    meta.get("title", ""),
            "url":      meta.get("url", ""),
            "chunk_id": meta.get("chunk_id", idx),
            "content":  meta["content"],
            "score":    round(score, 4),
        })

    if not results:
        logger.info("All MMR-selected results below threshold.")
        return None

    logger.info(
        "Returning %d chunks (top score: %.4f)", len(results), results[0]["score"]
    )
    return results


# ── Multi-query public interface ───────────────────────────────────────────────

def _retrieve_candidates(query: str, query_emb: np.ndarray) -> tuple[
    list[tuple[int, float]], list[tuple[int, float]]
]:
    """Return (bm25_hits, dense_hits) for a single query. Used by multi_retrieve."""
    return _bm25_search(query, cfg.bm25_fetch_k), _dense_search(query_emb, cfg.dense_fetch_k)


def multi_retrieve(queries: list[str]) -> Optional[List[Dict]]:
    """
    Run BM25 + dense retrieval for every query in parallel, merge all ranked
    lists with RRF, then cross-encode and MMR-deduplicate using the FIRST query
    (the original rewritten query) as the anchor for scoring and diversity.

    Args:
        queries: [original_query, variant_1, variant_2, ...]
                 First element is treated as the canonical query.

    Returns:
        Same format as retrieve(): list of { title, url, chunk_id, content, score }
        or None if nothing passes the threshold.
    """
    if not queries:
        return None

    canonical = queries[0]

    # ── 1. Encode all queries (parallel-friendly, but encode is fast enough) ──
    embeddings: dict[str, np.ndarray] = {
        q: _encode_query(q) for q in dict.fromkeys(queries)  # deduplicate
    }

    # ── 2. Parallel BM25 + dense for every query ──────────────────────────────
    from concurrent.futures import ThreadPoolExecutor

    all_ranked_lists: list[list[tuple[int, float]]] = []

    with ThreadPoolExecutor(max_workers=min(len(queries) * 2, 8)) as pool:
        futures = [
            pool.submit(_retrieve_candidates, q, embeddings[q])
            for q in dict.fromkeys(queries)
        ]
        for future in futures:
            bm25_hits, dense_hits = future.result()
            all_ranked_lists.extend([bm25_hits, dense_hits])

    logger.info(
        "multi_retrieve: %d queries → %d ranked lists",
        len(queries), len(all_ranked_lists),
    )

    # ── 3. RRF merge across ALL ranked lists ──────────────────────────────────
    merged         = _rrf_merge(all_ranked_lists, rrf_k=cfg.rrf_k)
    top_candidates = merged[: cfg.cross_encoder_k]

    if not top_candidates:
        logger.info("No candidates after multi-query RRF merge.")
        return None

    # ── 4. Cross-encoder re-rank using canonical query ────────────────────────
    reranked  = _cross_encode(canonical, top_candidates)
    top_score = reranked[0][1] if reranked else 0.0

    logger.debug(
        "multi_retrieve cross-encoder top score: %.4f (threshold: %s)",
        top_score, cfg.similarity_threshold,
    )

    if cfg.similarity_threshold is not None and (
        not reranked or top_score < cfg.similarity_threshold
    ):
        logger.info(
            "Best score %.4f below threshold %.2f after multi-query retrieval.",
            top_score, cfg.similarity_threshold,
        )
        return None

    # ── 5. MMR using canonical query embedding ────────────────────────────────
    final_indices = _mmr(
        query_emb=embeddings[canonical][0],
        candidates=reranked,
        k=cfg.final_k,
        lambda_param=cfg.mmr_lambda,
    )

    # ── 6. Build response ─────────────────────────────────────────────────────
    ce_score_map = {idx: sc for idx, sc in reranked}
    results: list[dict] = []

    for idx in final_indices:
        score = ce_score_map.get(idx, 0.0)
        if cfg.similarity_threshold is not None and score < cfg.similarity_threshold:
            continue
        meta = _metadata[idx]
        results.append({
            "title":    meta.get("title", ""),
            "url":      meta.get("url", ""),
            "chunk_id": meta.get("chunk_id", idx),
            "content":  meta["content"],
            "score":    round(score, 4),
        })

    if not results:
        logger.info("All MMR-selected results below threshold (multi_retrieve).")
        return None

    logger.info(
        "multi_retrieve returning %d chunks (top score: %.4f)",
        len(results), results[0]["score"],
    )
    return results