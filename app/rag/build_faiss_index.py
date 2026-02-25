import json
import os
import pickle
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"   
INDEX_OUTPUT_DIR     = "rag/faiss_store"
CHUNKS_INPUT_FILE    = "hr_policies_chunked.json"
BATCH_SIZE           = 32
# ─────────────────────────────────────────────────────────────────────────────


def load_chunks(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Chunk file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def batch_embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[i : i + BATCH_SIZE]
        embs = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embs)
    return np.vstack(all_embeddings).astype("float32")


def build_index() -> None:
    print("🔹 Loading chunks...")
    chunks = load_chunks(CHUNKS_INPUT_FILE)
    if not chunks:
        raise ValueError("No chunks found.")

    texts = [chunk["content"] for chunk in chunks]
    print(f"🔹 Loaded {len(texts)} chunks")

    # ── Dense index ────────────────────────────────────────────────────────────
    print("🔹 Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("🔹 Generating dense embeddings...")
    embeddings = batch_embed(model, texts)
    dim = embeddings.shape[1]
    print(f"🔹 Embedding dimension: {dim}")

    print("🔹 Building FAISS index (IndexFlatIP)...")
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    print(f"🔹 FAISS: {faiss_index.ntotal} vectors indexed")

    # ── BM25 sparse index ──────────────────────────────────────────────────────
    print("🔹 Building BM25 index...")
    tokenized = [tokenize_for_bm25(t) for t in tqdm(texts, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)
    print("🔹 BM25 index built")

    # ── Persist everything ─────────────────────────────────────────────────────
    os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)

    faiss.write_index(faiss_index, os.path.join(INDEX_OUTPUT_DIR, "index.faiss"))
    print("✅ FAISS index saved")

    with open(os.path.join(INDEX_OUTPUT_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print("✅ BM25 index saved")

    metadata_store = [
        {
            "title":    chunk["title"],
            "url":      chunk["url"],
            "chunk_id": chunk["chunk_id"],
            "content":  chunk["content"],
        }
        for chunk in chunks
    ]
    with open(os.path.join(INDEX_OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    print("✅ Metadata saved")

    # Save tokenized corpus for BM25 (needed for online updates, optional)
    with open(os.path.join(INDEX_OUTPUT_DIR, "tokenized_corpus.pkl"), "wb") as f:
        pickle.dump(tokenized, f)
    print("✅ Tokenized corpus saved")

    print(f"\n🎉 Hybrid index ready in: {INDEX_OUTPUT_DIR}/")
    print(f"   Dense  : {faiss_index.ntotal} vectors ({dim}d, BAAI/bge-large)")
    print(f"   Sparse : BM25Okapi over {len(tokenized)} documents")


if __name__ == "__main__":
    build_index()