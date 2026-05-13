from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 20) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": len(data_rows),
    }


def read_json_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _extract_md_toc(text: str) -> list[str]:
    """Return all markdown header lines (the table of contents)."""
    return re.findall(r"^#{1,6} .+", text, re.MULTILINE)


def _extract_md_section(text: str, section: str) -> str:
    """Return a named section including all its subsections.

    Stops at the next header of the same or higher level.
    Matching is case-insensitive and trims leading # chars from the query.
    """
    matches = list(re.finditer(r"^(#{1,6}) .+", text, re.MULTILINE))
    for i, m in enumerate(matches):
        line_end = text.index("\n", m.start()) if "\n" in text[m.start():] else len(text)
        header_text = text[m.start(): line_end].strip()
        if header_text.lower() == section.lower():
            level = len(m.group(1))
            end = len(text)
            for j in range(i + 1, len(matches)):
                if len(matches[j].group(1)) <= level:
                    end = matches[j].start()
                    break
            return text[m.start(): end].strip()
    return ""


# ---------------------------------------------------------------------------
# Document reading
# ---------------------------------------------------------------------------

def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    max_chars: int = 8000,
    query: str | None = None,
    section: str | None = None,
) -> dict[str, object]:
    """Read a document from the task context.

    Behaviour varies by how the caller requests content:

    - ``section`` set   → extract exactly that ## header block (fast, targeted).
    - ``relative_path`` == "knowledge.md" and no section → return the TOC so the
      agent can pick a section rather than loading the full file.
    - Large doc/ files  → BM25 chunk retrieval when ``query`` is provided and
      the file exceeds ``max_chars``.
    - Otherwise        → return the full text (no truncation).
    """
    path = resolve_context_path(task, relative_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Targeted section extraction
    if section:
        extracted = _extract_md_section(text, section)
        return {
            "path": relative_path,
            "section": section,
            "preview": extracted if extracted else f"Section '{section}' not found.",
            "available_sections": _extract_md_toc(text) if not extracted else [],
        }

    # knowledge.md: return TOC first so the agent selects relevant sections
    if relative_path == "knowledge.md" or relative_path.endswith("/knowledge.md"):
        toc = _extract_md_toc(text)
        if len(text) <= max_chars:
            # Small enough to return in full
            return {"path": relative_path, "preview": text, "truncated": False}
        return {
            "path": relative_path,
            "sections": toc,
            "hint": (
                "Call read_doc again with 'section' set to a header from 'sections' "
                "(e.g. '## 2. Core Entities & Fields') to read that section."
            ),
        }

    # Large doc/ files: BM25 retrieval when a query is available
    is_doc_dir_file = (
        relative_path.startswith("doc/")
        or "/doc/" in relative_path
        or relative_path.endswith(".md")
    )
    if query and is_doc_dir_file and len(text) > max_chars:
        from data_agent_baseline.tools.md_retrieval import retrieve_relevant_chunks

        preview = retrieve_relevant_chunks(text, query, max_chars=max_chars)
        return {
            "path": relative_path,
            "preview": preview,
            "truncated": True,
            "retrieval": "bm25",
            "hint": (
                "Showing the most query-relevant chunks. "
                "Use execute_python to read the full file if exhaustive extraction is needed."
            ),
        }

    return {
        "path": relative_path,
        "preview": text,
        "truncated": False,
    }
