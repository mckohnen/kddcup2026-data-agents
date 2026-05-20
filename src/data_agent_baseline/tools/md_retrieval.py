from __future__ import annotations

import os
import re
import time
import threading
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from data_agent_baseline.task_logger import get_logger

# ---------------------------------------------------------------------------
# Embedding model — lazy singleton, falls back to BM25-only if unavailable
# ---------------------------------------------------------------------------

_embed_model: Any = None
_embed_model_loaded: bool = False
_embed_model_lock = threading.Lock()


def _get_embed_model() -> Any:
    global _embed_model, _embed_model_loaded
    with _embed_model_lock:
        if _embed_model_loaded:
            return _embed_model
        _embed_model_loaded = True

        log = get_logger()
        result: dict = {}

        def _load() -> None:
            try:
                from fastembed import TextEmbedding
                cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
                log.info("  EMBED loading model (cache=%s)", cache_dir or "~/.cache/fastembed")
                result["model"] = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=cache_dir)
            except Exception as exc:
                log.warning("  EMBED model load failed (%s) — BM25-only fallback", exc)
                result["model"] = None

        t = threading.Thread(target=_load, daemon=True)
        t0 = time.perf_counter()
        t.start()
        t.join(timeout=15)
        elapsed = time.perf_counter() - t0

        if t.is_alive():
            log.warning("  EMBED model load timed out after %.1fs — BM25-only fallback", elapsed)
            _embed_model = None
        else:
            _embed_model = result.get("model")
            if _embed_model is not None:
                log.info("  EMBED model ready in %.1fs", elapsed)

        return _embed_model


def _rerank(candidate_chunks: list[str], query: str, top_k: int) -> list[int]:
    """Return indices of top_k candidates ranked by embedding cosine similarity.

    Falls back to returning the first top_k indices (BM25 order) if the
    embedding model is unavailable.  BGE vectors are unit-normalised so the
    dot product equals cosine similarity.
    """
    log = get_logger()
    model = _get_embed_model()
    if model is None:
        log.debug("  EMBED skipped (model unavailable) — using BM25 order for top_%d", top_k)
        return list(range(min(top_k, len(candidate_chunks))))
    try:
        t0 = time.perf_counter()
        vectors = np.array(list(model.embed([query] + candidate_chunks)))
        q_vec = vectors[0]
        c_vecs = vectors[1:]
        scores = c_vecs @ q_vec
        elapsed = time.perf_counter() - t0
        log.info("  EMBED reranked %d candidates → top_%d in %.2fs", len(candidate_chunks), top_k, elapsed)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    except Exception as exc:
        log.warning("  EMBED inference failed (%s) — BM25 order fallback", exc)
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
    log = get_logger()
    log.info("  RAG retrieve: file=%d chars query=%r max_chars=%d", len(text), query[:80], max_chars)

    if len(text) <= max_chars:
        log.debug("  RAG skipped (file fits in max_chars)")
        return text

    chunks = _chunk_markdown(text)
    if not chunks:
        return text[:max_chars]

    log.debug("  RAG chunked into %d sections", len(chunks))

    tokenized = [_tokenize(c) for c in chunks]
    if not any(tokenized):
        return text[:max_chars]

    # Stage 1: BM25 → top 2*top_k candidates
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(_tokenize(query))
    n_candidates = min(top_k * 2, len(chunks))
    candidate_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:n_candidates]
    candidate_chunks = [chunks[i] for i in candidate_indices]
    log.info("  RAG BM25: %d chunks → %d candidates in %.2fs", len(chunks), n_candidates, time.perf_counter() - t0)

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

    result = "\n\n---\n\n".join(selected)
    log.info("  RAG result: %d/%d chunks selected, %d chars returned", len(selected), len(chunks), len(result))
    return result
