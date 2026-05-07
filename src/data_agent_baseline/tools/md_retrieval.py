from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
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

    # Fall back to paragraph splitting
    return [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= min_chunk_chars]


def retrieve_relevant_chunks(
    text: str,
    query: str,
    *,
    top_k: int = 10,
    max_chars: int = 8000,
) -> str:
    """Return the most query-relevant markdown chunks, up to max_chars, in document order."""
    if len(text) <= max_chars:
        return text

    chunks = _chunk_markdown(text)
    if not chunks:
        return text[:max_chars]

    tokenized = [_tokenize(c) for c in chunks]
    # BM25Okapi silently breaks on all-empty token lists; guard against it
    if not any(tokenized):
        return text[:max_chars]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))

    # Pick top_k by score, then restore document order for coherence
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )

    selected: list[str] = []
    total = 0
    for i in top_indices:
        chunk = chunks[i]
        if total + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(selected)
