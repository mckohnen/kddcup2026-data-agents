"""Coverage gap detection and prose attribute extraction (preflight phase).

Runs during preflight — before the ReAct agent starts — to:
  1. Find (fact_table, lookup_table) pairs where the lookup table does not cover
     all IDs present in the fact table.
  2. Identify which prose docs contain the uncovered IDs.
  3. Extract the missing attribute values via paragraph-level regex + LLM.
  4. Inject a merged *_complete table directly into the shared in-memory SQLite
     connection so the Analyst can query it without any extra steps.

Design principles
-----------------
- Deterministic fallback: if the model is unavailable or times out, the extractor
  skips silently — the Analyst still runs, just without the enriched table.
- Single-attribute focus: when a lookup table has multiple non-ID columns, only
  the first is extracted from prose.  Multi-attribute extraction can be added later.
- Generalizable: works for any (fact, lookup) pair sharing an ID column — not just
  patient/gender.  The attribute hint is auto-generated from column name + observed
  values so the LLM knows what to look for.
"""
from __future__ import annotations

import csv
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter

_MAX_GAP_IDS = 500        # skip if uncovered ID count > this
_MAX_DOC_BYTES = 1_000_000  # skip prose files larger than 1 MB
_MIN_DOC_HITS = 1         # doc must match at least this many gap IDs to be a candidate
_LOOKUP_MAX_COLS = 6       # tables with more columns are not treated as lookup tables
_LOOKUP_MAX_ROWS = 1000    # tables with more rows are treated as fact tables

_EXTRACT_SYSTEM_PROMPT = (
    "You are an attribute extractor. "
    "For each [Entity: <ID>] block below, extract the requested attribute from the text. "
    "Respond with ONLY lines in the format:  <ID>: <VALUE>\n"
    "One line per entity. Skip entities where the attribute is not mentioned. "
    "Use concise, normalised values (e.g. 'M' not 'male', 'F' not 'female')."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _id_col_candidates(cols: list[str]) -> list[str]:
    return [c for c in cols if c.lower() in ("id",) or c.lower().startswith("id_") or c.lower().endswith("_id")]


def _attr_col_candidates(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in _id_col_candidates(cols)]


def _table_info(conn: sqlite3.Connection) -> dict[str, dict]:
    """Return {table_name: {columns, row_count}} for all non-paragraph tables."""
    result = {}
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        if name.endswith("_paragraphs") or name.endswith("_complete"):
            continue
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            cursor = conn.execute(f'SELECT * FROM "{name}" LIMIT 0')
            cols = [d[0] for d in cursor.description or []]
            result[name] = {"columns": cols, "row_count": count}
        except Exception:
            pass
    return result


def _detect_pairs(info: dict[str, dict]) -> list[dict]:
    """Return (fact, lookup) pairs sharing an ID column where lookup is small."""
    lookup = {}
    fact = {}
    for name, meta in info.items():
        cols = meta["columns"]
        row_count = meta["row_count"]
        id_cols = _id_col_candidates(cols)
        attr_cols = _attr_col_candidates(cols)
        if not id_cols:
            continue
        if attr_cols and len(cols) <= _LOOKUP_MAX_COLS and row_count <= _LOOKUP_MAX_ROWS:
            lookup[name] = {"id_col": id_cols[0], "attr_cols": attr_cols, "row_count": row_count}
        if row_count > 0:
            fact[name] = {"id_col": id_cols[0], "row_count": row_count}

    pairs = []
    for fname, fmeta in fact.items():
        for lname, lmeta in lookup.items():
            if fname == lname:
                continue
            if fmeta["id_col"].lower() == lmeta["id_col"].lower():
                pairs.append({
                    "fact_table": fname,
                    "fact_id_col": fmeta["id_col"],
                    "lookup_table": lname,
                    "lookup_id_col": lmeta["id_col"],
                    "attr_cols": lmeta["attr_cols"],
                })
    return pairs


def _gap_ids(conn: sqlite3.Connection, pair: dict) -> list[str]:
    ft, fi = pair["fact_table"], pair["fact_id_col"]
    lt, li = pair["lookup_table"], pair["lookup_id_col"]
    try:
        rows = conn.execute(
            f'SELECT DISTINCT "{fi}" FROM "{ft}" '
            f'WHERE "{fi}" IS NOT NULL AND CAST("{fi}" AS TEXT) != "" '
            f'AND CAST("{fi}" AS TEXT) NOT IN '
            f'(SELECT CAST("{li}" AS TEXT) FROM "{lt}")'
        ).fetchall()
        return [str(r[0]) for r in rows if r[0] is not None]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Prose scanning and extraction
# ---------------------------------------------------------------------------

def _score_doc(doc_path: Path, ids: list[str]) -> int:
    """Count how many of *ids* appear in the doc (word-boundary match)."""
    if doc_path.stat().st_size > _MAX_DOC_BYTES:
        return 0
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"\b(" + "|".join(re.escape(i) for i in ids[:500]) + r")\b")
    return len(set(pat.findall(text)))


def _best_doc(context_dir: Path, ids: list[str]) -> tuple[Path | None, int]:
    """Return (doc_path, hit_count) for the prose doc with most ID hits."""
    doc_dir = context_dir / "doc"
    if not doc_dir.is_dir():
        return None, 0
    best_path: Path | None = None
    best_hits = 0
    for md in sorted(doc_dir.rglob("*.md")):
        hits = _score_doc(md, ids)
        if hits > best_hits:
            best_hits = hits
            best_path = md
    return best_path, best_hits


def _attribute_hint(attr_col: str, conn: sqlite3.Connection, lookup_table: str, lookup_id_col: str) -> str:
    """Generate a short hint string describing the attribute for the LLM."""
    col_label = attr_col.upper()
    try:
        rows = conn.execute(
            f'SELECT DISTINCT "{attr_col}" FROM "{lookup_table}" WHERE "{attr_col}" IS NOT NULL LIMIT 10'
        ).fetchall()
        vals = sorted(str(r[0]) for r in rows if r[0] is not None)
        if vals:
            return col_label + " (observed values: " + " / ".join(vals) + ")"
    except Exception:
        pass
    return col_label


def _extract_from_doc(
    doc_path: Path,
    ids: list[str],
    attr_hint: str,
    model: "ModelAdapter",
) -> dict[str, str]:
    """Paragraph-level extraction: return {id: value} for matched IDs."""
    from data_agent_baseline.agents.model import ModelMessage

    text = doc_path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    ids_set = set(ids)
    id_to_paras: dict[str, list[str]] = {}
    for para in paragraphs:
        for id_val in ids_set:
            if re.search(r"\b" + re.escape(id_val) + r"\b", para):
                bucket = id_to_paras.setdefault(id_val, [])
                if len(id_val) < 4:
                    # Include surrounding paragraphs for short IDs to add context
                    bucket.append(para[:1200])
                else:
                    bucket.append(para[:1000])

    if not id_to_paras:
        return {}

    # Batch all matched entities into one LLM call
    sections = [
        "[Entity: " + id_val + "]\n" + "\n".join(paras)
        for id_val, paras in id_to_paras.items()
    ]
    user_msg = "Extract attribute: " + attr_hint + "\n\n" + "\n\n".join(sections)

    try:
        response = model.complete([
            ModelMessage(role="system", content=_EXTRACT_SYSTEM_PROMPT),
            ModelMessage(role="user", content=user_msg),
        ], extra_body={"enable_thinking": False}).strip()
    except Exception:
        return {}

    extracted: dict[str, str] = {}
    for line in response.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        id_part, _, val_part = line.partition(":")
        id_part = id_part.strip()
        val_part = val_part.strip()
        if id_part in ids_set and val_part:
            extracted[id_part] = val_part
    return extracted


# ---------------------------------------------------------------------------
# Inject into SQLite
# ---------------------------------------------------------------------------

def _inject_complete(
    conn: sqlite3.Connection,
    pair: dict,
    extracted: dict[str, str],
    cache_dir: Path | None = None,
) -> str:
    """Create <lookup>_complete table = existing lookup rows + extracted rows.

    If *cache_dir* is provided the merged table is also written as a CSV so it
    can be restored in resumption subprocesses (which skip preflight).
    """
    lookup = pair["lookup_table"]
    lid_col = pair["lookup_id_col"]
    attr_cols = pair["attr_cols"]
    complete_name = lookup + "_complete"

    try:
        cursor = conn.execute(f'SELECT * FROM "{lookup}"')
        col_names = [d[0] for d in cursor.description or []]
        existing_rows: list[dict] = [dict(zip(col_names, row)) for row in cursor.fetchall()]
    except Exception:
        existing_rows = []

    cols = [lid_col] + attr_cols
    merged: list[dict] = list(existing_rows)

    # Append extracted rows — single attribute only for now
    if attr_cols:
        first_attr = attr_cols[0]
        existing_ids = {str(r.get(lid_col, "")) for r in existing_rows}
        for id_val, val in extracted.items():
            if str(id_val) not in existing_ids:
                merged.append({lid_col: id_val, first_attr: val})

    try:
        conn.execute(f'DROP TABLE IF EXISTS "{complete_name}"')
        col_defs = ", ".join('"' + c + '" TEXT' for c in cols)
        conn.execute(f'CREATE TABLE "{complete_name}" ({col_defs})')
        conn.executemany(
            'INSERT INTO "' + complete_name + '" (' +
            ", ".join('"' + c + '"' for c in cols) + ") VALUES (" +
            ", ".join("?" for _ in cols) + ")",
            [[row.get(c, "") for c in cols] for row in merged],
        )
        conn.commit()
    except Exception:
        return lookup  # fallback: use original table name

    # Persist to disk so resumption subprocesses can restore without re-extracting.
    if cache_dir is not None and merged:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            csv_path = cache_dir / f"extracted_{complete_name}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows({c: row.get(c, "") for c in cols} for row in merged)
        except Exception:
            pass  # disk write failure is non-fatal

    return complete_name


def load_cached_extractions(cache_dir: Path, conn: sqlite3.Connection) -> list[str]:
    """Restore pre-extracted *_complete tables from disk CSVs into SQLite.

    Called during resumption subprocesses (which skip preflight) to make the
    tables available without re-running the LLM extraction.

    Returns the list of table names that were successfully restored.
    """
    restored: list[str] = []
    for csv_path in sorted(cache_dir.glob("extracted_*.csv")):
        table_name = csv_path.stem[len("extracted_"):]  # strip "extracted_" prefix
        try:
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue
            cols = list(rows[0].keys())
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            col_defs = ", ".join('"' + c + '" TEXT' for c in cols)
            conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')
            conn.executemany(
                'INSERT INTO "' + table_name + '" (' +
                ", ".join('"' + c + '"' for c in cols) + ") VALUES (" +
                ", ".join("?" for _ in cols) + ")",
                [[r.get(c, "") for c in cols] for r in rows],
            )
            conn.commit()
            restored.append(table_name)
        except Exception:
            pass
    return restored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_extract_coverage_gaps(
    context_dir: Path,
    model: "ModelAdapter | None",
    timeout_seconds: int = 25,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run gap detection + prose extraction and return summaries for the preflight hint.

    Side effect: injects *_complete tables into the cached SQLite connection so
    the Analyst can query them immediately without any extra steps.

    Returns a list of summary dicts, one per successfully extracted gap:
        fact_table, lookup_table, complete_table, attr_cols,
        gap_ids_total, ids_extracted, source_doc
    """
    from data_agent_baseline.tools.context_sqlite import load_context_to_sqlite

    conn = load_context_to_sqlite(context_dir)
    start = time.monotonic()
    summaries: list[dict[str, Any]] = []

    if model is None:
        return summaries

    info = _table_info(conn)
    pairs = _detect_pairs(info)

    for pair in pairs:
        elapsed = time.monotonic() - start
        if elapsed > timeout_seconds:
            break

        ids = _gap_ids(conn, pair)
        if not ids:
            continue
        if len(ids) > _MAX_GAP_IDS:
            continue

        doc_path, hit_count = _best_doc(context_dir, ids)
        if doc_path is None or hit_count < _MIN_DOC_HITS:
            continue

        # Only extract first attribute column
        if not pair["attr_cols"]:
            continue
        first_attr = pair["attr_cols"][0]
        hint = _attribute_hint(first_attr, conn, pair["lookup_table"], pair["lookup_id_col"])

        remaining = timeout_seconds - (time.monotonic() - start)
        if remaining < 3:
            break

        extracted = _extract_from_doc(doc_path, ids, hint, model)
        if not extracted:
            continue

        complete_name = _inject_complete(conn, pair, extracted, cache_dir=cache_dir)

        summaries.append({
            "fact_table": pair["fact_table"],
            "lookup_table": pair["lookup_table"],
            "lookup_id_col": pair["lookup_id_col"],
            "complete_table": complete_name,
            "attr_cols": pair["attr_cols"],
            "gap_ids_total": len(ids),
            "ids_extracted": len(extracted),
            "source_doc": doc_path.name,
        })

    return summaries
