from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Embedding model — lazy singleton, falls back to BM25-only if unavailable
# ---------------------------------------------------------------------------

_embed_model: Any = None
_embed_model_loaded: bool = False


def _get_embed_model() -> Any:
    global _embed_model, _embed_model_loaded
    if _embed_model_loaded:
        return _embed_model
    _embed_model_loaded = True
    try:
        from fastembed import TextEmbedding
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=cache_dir)
    except Exception:
        _embed_model = None  # no weights or fastembed not installed — BM25 only
    return _embed_model


def _rerank(candidate_chunks: list[str], query: str, top_k: int) -> list[int]:
    """Return indices of top_k candidates ranked by embedding cosine similarity.

    Falls back to returning the first top_k indices (BM25 order) if the
    embedding model is unavailable.  BGE vectors are unit-normalised so the
    dot product equals cosine similarity.
    """
    model = _get_embed_model()
    if model is None:
        return list(range(min(top_k, len(candidate_chunks))))
    try:
        vectors = np.array(list(model.embed([query] + candidate_chunks)))
        q_vec = vectors[0]
        c_vecs = vectors[1:]
        scores = c_vecs @ q_vec
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    except Exception:
        return list(range(min(top_k, len(candidate_chunks))))


# ---------------------------------------------------------------------------
# Tokenisation and chunking
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    # Split camelCase before lowercasing so positionOrder → position + order
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_markdown(text: str, min_chunk_chars: int = 80) -> list[str]:
    """Split markdown into sections by headers; fall back to paragraphs."""
    matches = list(re.finditer(r"^#{1,6}\s+.+", text, re.MULTILINE))
    if matches:
        chunks: list[str] = []
        preamble = text[: matches[0].start()].strip()
        if len(preamble) >= min_chunk_chars:
            chunks.append(preamble)
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[match.start() : end].strip()
            if len(chunk) >= min_chunk_chars:
                chunks.append(chunk)
        if chunks:
            return chunks
    return [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= min_chunk_chars]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    text: str,
    query: str,
    *,
    top_k: int = 10,
    max_chars: int = 8000,
) -> str:
    """Return the most query-relevant markdown chunks, up to max_chars, in document order.

    Two-stage retrieval:
      Stage 1 — BM25 keyword match selects top 2*top_k candidates (fast).
      Stage 2 — fastembed cosine similarity reranks to top_k (handles synonyms).
    Falls back to BM25-only ranking when fastembed weights are not available.
    """
    if len(text) <= max_chars:
        return text

    chunks = _chunk_markdown(text)
    if not chunks:
        return text[:max_chars]

    tokenized = [_tokenize(c) for c in chunks]
    if not any(tokenized):
        return text[:max_chars]

    # Stage 1: BM25 → top 2*top_k candidates
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(_tokenize(query))
    n_candidates = min(top_k * 2, len(chunks))
    candidate_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:n_candidates]
    candidate_chunks = [chunks[i] for i in candidate_indices]

    # Stage 2: rerank candidates with embeddings → top_k
    reranked_local = _rerank(candidate_chunks, query, top_k)
    # Map local indices back to original chunk positions, then restore document order
    final_indices = sorted(candidate_indices[j] for j in reranked_local)

    selected: list[str] = []
    total = 0
    for i in final_indices:
        chunk = chunks[i]
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            if not selected:
                selected.append(chunk[:remaining])
            break
        selected.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(selected)
