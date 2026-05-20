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
# Embedding models — lazy singletons, ensemble of BGE + MiniLM
# Falls back to BM25-only if neither model loads.
# ---------------------------------------------------------------------------

_MODEL_NAMES = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]

_embed_models: list[Any] = []
_embed_models_loaded: bool = False
_embed_models_lock = threading.Lock()


def _get_embed_models() -> list[Any]:
    global _embed_models, _embed_models_loaded
    with _embed_models_lock:
        if _embed_models_loaded:
            return _embed_models
        _embed_models_loaded = True

        log = get_logger()
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
        loaded: list[Any] = []

        def _load() -> None:
            try:
                from fastembed import TextEmbedding
                for name in _MODEL_NAMES:
                    try:
                        log.info("  EMBED loading %s (cache=%s)", name, cache_dir or "~/.cache/fastembed")
                        loaded.append(TextEmbedding(name, cache_dir=cache_dir))
                        log.info("  EMBED %s ready", name)
                    except Exception as exc:
                        log.warning("  EMBED %s failed (%s) — skipping", name, exc)
            except Exception as exc:
                log.warning("  EMBED fastembed unavailable (%s) — BM25-only fallback", exc)

        t = threading.Thread(target=_load, daemon=True)
        t0 = time.perf_counter()
        t.start()
        t.join(timeout=30)
        elapsed = time.perf_counter() - t0

        if t.is_alive():
            log.warning("  EMBED model load timed out after %.1fs — BM25-only fallback", elapsed)
        else:
            _embed_models = loaded
            log.info("  EMBED %d model(s) ready in %.1fs", len(_embed_models), elapsed)

        return _embed_models


def _rerank(candidate_chunks: list[str], query: str, top_k: int) -> list[int]:
    """Return indices of top_k candidates ranked by ensemble cosine similarity.

    Averages normalised cosine scores from BGE and MiniLM. Both models produce
    unit-normalised vectors so dot product == cosine similarity. Falls back to
    BM25 order if no models are available.
    """
    log = get_logger()
    models = _get_embed_models()
    if not models:
        log.debug("  EMBED skipped (no models) — using BM25 order for top_%d", top_k)
        return list(range(min(top_k, len(candidate_chunks))))

    texts = [query] + candidate_chunks
    ensemble_scores: np.ndarray | None = None
    t0 = time.perf_counter()
    for model in models:
        try:
            vectors = np.array(list(model.embed(texts)))
            q_vec = vectors[0]
            c_vecs = vectors[1:]
            scores = c_vecs @ q_vec
            # min-max normalise so both models contribute equally
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)
            ensemble_scores = scores if ensemble_scores is None else ensemble_scores + scores
        except Exception as exc:
            log.warning("  EMBED inference failed for model (%s) — skipping", exc)

    if ensemble_scores is None:
        log.warning("  EMBED all models failed — BM25 order fallback")
        return list(range(min(top_k, len(candidate_chunks))))

    elapsed = time.perf_counter() - t0
    log.info(
        "  EMBED ensemble(%d models) reranked %d candidates → top_%d in %.2fs",
        len(models), len(candidate_chunks), top_k, elapsed,
    )
    return sorted(range(len(ensemble_scores)), key=lambda i: ensemble_scores[i], reverse=True)[:top_k]


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
    n_candidates = min(top_k * 4, len(chunks))
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
