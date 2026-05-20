"""Pre-flight task analysis: match question terms to schema and raw data.

Given a question and the context schema + raw tables, this module produces a
structured analysis that is injected into the agent's task prompt as grounding
hints before the ReAct loop begins.  It is also passed to the critic agent so
it can check whether proposed columns align with what the question actually
references.

Key capabilities
----------------
- Table matching: finds tables whose name appears in the question.
- Column matching: finds columns whose name (or sub-tokens) appear in the question.
  Two tiers: full-name match (confidence 0.9) and partial token match (confidence 0.65).
  Partial matching uses tokens of ≥ 4 chars to avoid noise from short common words.
- Literal value matching: scans actual row data for quoted literals in the question
  (e.g. ``'Chinese Grand Prix'`` → finds ``races.name``).  Row scan is capped to
  avoid O(n) cost on large tables.
- Join inference: given the candidate tables and the FK relationships from the schema
  profile, suggests the JOIN path(s) the agent should use.
- Metric hint: lightweight detection of aggregation intent (count / sum / avg).

The output format is compatible with ``build_schema_profile`` (from schema_profiler.py)
as the ``context`` argument and ``load_raw_tables`` (from context_sqlite.py) as the
``raw_tables`` argument — no conversion needed.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


# ---------------------------------------------------------------------------
# Fuzzy matching helper
# ---------------------------------------------------------------------------

#: Minimum SequenceMatcher ratio for a LIKE-based substring match to be kept.
#: Below this threshold the cell value is considered too different from the
#: literal (e.g. a 4-char literal matching a 40-char cell) and is discarded.
_FUZZY_MIN_RATIO: float = 0.5


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return SequenceMatcher similarity ratio between two strings (0–1)."""
    return SequenceMatcher(None, a.lower(), str(b).lower()).ratio()


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _singularize(word: str) -> str:
    """Very lightweight singularisation for table-name matching."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _extract_quoted_literals(question: str) -> list[str]:
    """Return all single- or double-quoted string literals in the question."""
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
    return [a or b for a, b in matches if a or b]


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Matching functions
# ---------------------------------------------------------------------------

def _match_tables(question: str, context: dict) -> list[dict]:
    """Find tables whose name (or singularised/pluralised form) appears in the question."""
    q_norm = _normalize(question)
    q_tokens = set(q_norm.split())
    matches: list[dict] = []

    for table_name in context.get("tables", {}):
        # For alias.table format, match against the base table name too
        base = table_name.split(".")[-1] if "." in table_name else table_name
        candidates = set()
        for name in (table_name, base):
            n = _normalize(name)
            candidates |= {n, _singularize(n), n + "s", n.replace("_", " ")}

        for candidate in candidates:
            if candidate and (candidate in q_tokens or candidate in q_norm):
                matches.append({
                    "text": candidate,
                    "match_type": "table_name",
                    "matched_table": table_name,
                    "confidence": 0.95,
                })
                break

    return matches


def _match_columns(question: str, context: dict) -> list[dict]:
    """Find columns whose name (or token sub-parts ≥ 4 chars) appear in the question.

    Two confidence tiers:
      0.90 — full normalised column name found in the question
      0.65 — a token from the column name (≥ 4 chars) found among question tokens
    """
    q_norm = _normalize(question)
    q_tokens = set(q_norm.split())
    matches: list[dict] = []

    for table_name, table_ctx in context.get("tables", {}).items():
        for col_name in table_ctx.get("columns", {}):
            col_norm = _normalize(col_name)
            col_tokens = set(col_norm.split())

            if col_norm in q_norm:
                matches.append({
                    "text": col_name,
                    "match_type": "column_name",
                    "matched_table": table_name,
                    "matched_column": col_name,
                    "confidence": 0.90,
                })
                continue

            # Partial: any token ≥ 4 chars from the column name appears in question tokens
            for token in col_tokens:
                if len(token) >= 4 and token in q_tokens:
                    matches.append({
                        "text": token,
                        "match_type": "column_partial",
                        "matched_table": table_name,
                        "matched_column": col_name,
                        "confidence": 0.65,
                    })
                    break

    return matches


def _match_literal_values_sql(
    question: str,
    context: dict,
    context_dir: "Path",
) -> list[dict]:
    """Match quoted literals from the question against all column values via SQL.

    Uses the unified SQLite connection so .db files, CSVs, and JSON are all
    covered without any row cap.  Three matching tiers, tried in order:

    1. Exact match               → confidence 1.0
    2. Case-insensitive exact    → confidence 1.0  (handles casing differences)
    3. Case-insensitive LIKE     → confidence 0.7  (literal is substring of cell value;
                                                    only tried for literals ≥ 4 chars to
                                                    avoid over-broad short-token matches)

    Each literal stops searching once a match is found (best tier wins).
    """
    from pathlib import Path as _Path
    from data_agent_baseline.tools.context_sqlite import load_context_to_sqlite

    literals = _extract_quoted_literals(question)
    if not literals:
        return []

    conn = load_context_to_sqlite(_Path(context_dir))
    matches: list[dict] = []

    for literal in literals:
        found = False
        for table_name, table_ctx in context.get("tables", {}).items():
            if found:
                break
            if "." in table_name:
                alias, tbl = table_name.split(".", 1)
                from_clause = f'"{alias}"."{tbl}"'
            else:
                from_clause = f'"{table_name}"'

            for col_name in table_ctx.get("columns", {}):
                if found:
                    break
                try:
                    # Build the ordered list of (WHERE clause, params, confidence)
                    candidates = [
                        (f'"{col_name}" = ?', [literal], 1.0),
                        (f'LOWER("{col_name}") = LOWER(?)', [literal], 1.0),
                    ]
                    # Substring match only for literals long enough to be meaningful
                    if len(literal) >= 4:
                        candidates.append((
                            f'LOWER("{col_name}") LIKE LOWER(?)',
                            [f"%{literal}%"],
                            0.7,
                        ))

                    for sql_where, params, base_confidence in candidates:
                        row = conn.execute(
                            f'SELECT "{col_name}" FROM {from_clause}'
                            f' WHERE {sql_where} LIMIT 1',
                            params,
                        ).fetchone()
                        if row is not None:
                            final_confidence = base_confidence
                            # For LIKE (substring) matches apply fuzzy post-filter:
                            # SequenceMatcher penalises short literals matching long
                            # cell values (e.g. "Grand" inside "Chinese Grand Prix 2008").
                            # Matches below the threshold are discarded; passing matches
                            # get their confidence scaled by the fuzzy ratio so the agent
                            # sees a graded signal.
                            if base_confidence < 1.0:
                                ratio = _fuzzy_ratio(literal, str(row[0]))
                                if ratio < _FUZZY_MIN_RATIO:
                                    continue  # too dissimilar — skip, try next candidate
                                final_confidence = round(base_confidence * ratio, 2)
                            matches.append({
                                "text": literal,
                                "match_type": "cell_value",
                                "matched_table": table_name,
                                "matched_column": col_name,
                                "value": row[0],
                                "confidence": final_confidence,
                            })
                            found = True
                            break
                except Exception:
                    pass

    return _dedupe(matches)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_metric(question: str) -> dict[str, Any]:
    q = _normalize(question)
    if any(k in q for k in ("total cost", "total amount", "sum of")):
        return {"metric": "sum", "aggregation": "SUM"}
    if any(k in q for k in ("how many", "count", "number of")):
        return {"metric": "count", "aggregation": "COUNT"}
    if any(k in q for k in ("average", "avg", "mean")):
        return {"metric": "average", "aggregation": "AVG"}
    if any(k in q for k in ("maximum", "max", "highest", "largest", "most")):
        return {"metric": "max", "aggregation": "MAX"}
    if any(k in q for k in ("minimum", "min", "lowest", "smallest", "least")):
        return {"metric": "min", "aggregation": "MIN"}
    return {"metric": "unknown", "aggregation": None}


def _infer_joins(candidate_tables: list[str], context: dict) -> list[dict]:
    """Return JOIN suggestions from the schema relationships.

    Relationships with confidence ≥ 0.95 are marked 'confirmed'; those in the
    0.80–0.95 range are marked 'possible' so the agent knows to verify before
    relying on them.  Both are included — a false negative (missing a real JOIN)
    is worse than a false positive the agent can check and discard.
    """
    joins: list[dict] = []
    for rel in context.get("relationships", []):
        # Include even if only one side is in candidate_tables — the other
        # table may still be relevant even if the question didn't name it.
        if rel["from_table"] in candidate_tables or rel["to_table"] in candidate_tables:
            confidence = rel.get("confidence", 1.0)
            joins.append({
                "from": f"{rel['from_table']}.{rel['from_column']}",
                "to": f"{rel['to_table']}.{rel['to_column']}",
                "confidence": confidence,
                "status": "confirmed" if confidence >= 0.95 else "possible — verify",
            })
    return joins


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_id_coverage(context: dict, context_dir: "Path") -> list[dict]:
    """Detect lookup tables whose IDs don't fully cover the related fact tables.

    A lookup table is a small table (≤ 300 rows, ≤ 4 columns) with a single ID
    column that shares a name with columns in larger tables.  When coverage is
    incomplete (< 95%) the agent should look for supplementary data in prose docs.

    Returns a list of warning dicts, one per incomplete join found.
    """
    from data_agent_baseline.tools.context_sqlite import run_sql_on_context

    tables = context.get("tables", {})
    if not tables:
        return []

    # Separate small (lookup) from large (fact) tables by row count
    small_tables = {
        name: info for name, info in tables.items()
        if info.get("row_count", 0) <= 300
        and len(info.get("columns", [])) <= 4
        and not name.endswith("_paragraphs")
    }
    large_tables = {
        name: info for name, info in tables.items()
        if info.get("row_count", 0) > 300
        and not name.endswith("_paragraphs")
    }

    if not small_tables or not large_tables:
        return []

    warnings: list[dict] = []
    for small_name, small_info in small_tables.items():
        small_cols = list(small_info.get("columns", {}).keys())
        if not small_cols:
            continue
        id_col = small_cols[0]  # first column is typically the ID

        for large_name, large_info in large_tables.items():
            large_col_names = list(large_info.get("columns", {}).keys())
            if id_col not in large_col_names:
                continue

            large_count = large_info.get("row_count", 0)
            if large_count == 0:
                continue

            # Count how many distinct IDs in the large table are missing from the lookup
            try:
                result = run_sql_on_context(
                    context_dir,
                    f'SELECT COUNT(DISTINCT "{id_col}") FROM "{large_name}" '
                    f'WHERE "{id_col}" NOT IN (SELECT "{id_col}" FROM "{small_name}")',
                    limit=1,
                )
                missing = (result.get("rows") or [[0]])[0][0]
                if isinstance(missing, int) and missing > 0:
                    warnings.append({
                        "lookup_table": small_name,
                        "lookup_id_col": id_col,
                        "fact_table": large_name,
                        "missing_ids": missing,
                    })
            except Exception:
                pass  # Never let coverage check crash the preflight

    return warnings


_DOC_SCAN_CHARS = 2000  # chars read from each doc file — first paragraph reveals structure

_DOC_SCAN_SYSTEM_PROMPT = (
    "You are a data analyst. You will be shown opening excerpts from one or more data documents. "
    "For each document, output exactly one line in the format:\n"
    "  <filename>: <one-sentence description of entity types and attributes>\n"
    "Be specific about attribute names where visible (e.g. patient IDs, gender, "
    "measurement values with units, categorical labels, dates). "
    "No preamble, no explanation — only the lines."
)


def count_doc_files(context_dir: "Path") -> int:
    """Count the number of .md files in context_dir/doc/ excluding knowledge.md.

    Returns 0 if the doc/ directory does not exist.
    This count is used to compute a dynamic preflight budget in data_agent.py.
    """
    doc_dir = context_dir / "doc"
    if not doc_dir.is_dir():
        return 0
    count = 0
    for md_file in doc_dir.glob("*.md"):
        if md_file.name.lower() != "knowledge.md":
            count += 1
    return count


def _scan_doc_contents(context_dir: "Path", model: "ModelAdapter | None" = None) -> list[dict]:
    """Use batched LLM calls to describe all doc/*.md files in the context.

    Reads the first _DOC_SCAN_CHARS characters of each doc file, batches them
    in groups of 2, and makes one LLM call per batch. Results from all batches
    are combined into a single list.
    knowledge.md files are excluded (domain reference material, not data).

    Returns a list of dicts:
      {"file": "doc/Patient.md", "description": "Contains patient IDs, gender ..."}

    If no model is provided, or no doc files exist, returns an empty list.
    """
    if model is None:
        return []

    from data_agent_baseline.agents.model import ModelMessage  # noqa: PLC0415

    doc_dir = context_dir / "doc"
    if not doc_dir.is_dir():
        return []

    # Collect (filename, excerpt) pairs — exclude knowledge.md
    excerpts: list[tuple[str, str]] = []
    for md_file in sorted(doc_dir.glob("*.md")):
        if md_file.name.lower() == "knowledge.md":
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")[:_DOC_SCAN_CHARS]
        except OSError:
            continue
        if text.strip():
            excerpts.append((md_file.name, text.strip()))

    if not excerpts:
        return []

    known_names = {name for name, _ in excerpts}

    def _parse_response(response: str) -> list[dict]:
        results: list[dict] = []
        for line in response.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            fname, _, desc = line.partition(":")
            fname = fname.strip()
            # Accept if filename matches (with or without path prefix)
            if fname in known_names or any(fname == n or fname.endswith(n) for n in known_names):
                results.append({"file": f"doc/{fname}", "description": desc.strip()})
        return results

    # Process docs in batches of 2, one LLM call per batch
    _BATCH_SIZE = 2
    all_results: list[dict] = []
    for batch_start in range(0, len(excerpts), _BATCH_SIZE):
        batch = excerpts[batch_start : batch_start + _BATCH_SIZE]
        sections = "\n\n".join(
            f"--- {name} ---\n{text}" for name, text in batch
        )
        try:
            response = model.complete([
                ModelMessage(role="system", content=_DOC_SCAN_SYSTEM_PROMPT),
                ModelMessage(role="user", content=sections),
            ]).strip()
        except Exception:  # noqa: BLE001
            continue  # skip failed batch, try remaining ones
        all_results.extend(_parse_response(response))

    return all_results


def build_task_analysis(
    question: str,
    context: dict,
    context_dir: "Path",
    model: "ModelAdapter | None" = None,
) -> dict[str, Any]:
    """Analyse a question against the context schema and data.

    Args:
        question:     The task question string.
        context:      Schema profile dict — output of ``build_schema_profile()``.
                      Must have ``tables`` and ``relationships`` keys.
        context_dir:  Path to the task context directory.  Used to get the
                      unified SQLite connection for literal value matching
                      across all sources (.db, CSV, JSON).
        model:        Optional model adapter.  When provided, each doc/*.md file
                      (excluding knowledge.md) is summarised by a single LLM call
                      so the agent knows what attributes are available in prose.

    Returns:
        Dict with keys: ``matched_terms``, ``candidate_tables``,
        ``candidate_columns``, ``required_joins``, ``filter_candidates``,
        ``metric``, ``coverage_warnings``, ``doc_contents``.
    """
    table_matches = _match_tables(question, context)
    column_matches = _match_columns(question, context)
    value_matches = _match_literal_values_sql(question, context, context_dir)

    all_matches = _dedupe(table_matches + column_matches + value_matches)

    candidate_tables: list[str] = []
    for m in all_matches:
        t = m.get("matched_table")
        if t and t not in candidate_tables:
            candidate_tables.append(t)

    candidate_columns: list[str] = []
    for m in all_matches:
        t, c = m.get("matched_table"), m.get("matched_column")
        if t and c:
            ref = f"{t}.{c}"
            if ref not in candidate_columns:
                candidate_columns.append(ref)

    filter_candidates = [
        {
            "column": f"{m['matched_table']}.{m['matched_column']}",
            "operator": "=",
            "value": m["value"],
            "confidence": m["confidence"],
        }
        for m in value_matches
    ]

    coverage_warnings = _check_id_coverage(context, context_dir)
    doc_contents = _scan_doc_contents(context_dir, model=model)

    knowledge_content: str = ""
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.exists():
        knowledge_path = context_dir / "doc" / "knowledge.md"
    if knowledge_path.exists():
        try:
            knowledge_content = knowledge_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass

    return {
        "question": question,
        "matched_terms": all_matches,
        "candidate_tables": candidate_tables,
        "candidate_columns": candidate_columns,
        "required_joins": _infer_joins(candidate_tables, context),
        "filter_candidates": filter_candidates,
        "metric": _infer_metric(question),
        "coverage_warnings": coverage_warnings,
        "doc_contents": doc_contents,
        "knowledge_content": knowledge_content,
        "_schema_tables": list(context.get("tables", {}).keys()),
    }


def format_task_analysis_hint(analysis: dict) -> str:
    """Format task analysis as a compact hint block for injection into the agent prompt.

    High-confidence column matches (≥ 0.8) are shown as "Strong col matches".
    Partial matches (< 0.8) are shown as "Weak col matches".
    JOINs are split into confirmed (≥ 0.95) and possible (0.80–0.95) tiers.
    A prominent disclaimer reminds the agent that this is a starting point only.
    """
    schema_tables = analysis.get("_schema_tables", [])
    has_schema_info = bool(schema_tables)

    lines = [
        "[Pre-flight schema analysis]",
        "NOTE: This is a best-effort analysis of the question. It may be incomplete or",
        "contain incorrect links. Verify any suggested joins against the actual data before using them.",
    ]
    if has_schema_info:
        lines.append(
            f"Schema already profiled ({len(schema_tables)} table(s): "
            + ", ".join(schema_tables[:6])
            + (f" … +{len(schema_tables) - 6} more" if len(schema_tables) > 6 else "")
            + "). Do NOT call show_context_schema at step 1 — go directly to data work."
        )

    if analysis["candidate_tables"]:
        lines.append(f"Candidate tables:   {', '.join(analysis['candidate_tables'])}")

    strong = [
        m for m in analysis["matched_terms"]
        if m.get("matched_column") and m["confidence"] >= 0.8
    ]
    weak = [
        m for m in analysis["matched_terms"]
        if m.get("matched_column") and 0.0 < m["confidence"] < 0.8
    ]

    if strong:
        parts = [
            f"{m['matched_table']}.{m['matched_column']} (matched \"{m['text']}\")"
            for m in strong
        ]
        lines.append(f"Strong col matches: {', '.join(parts)}")

    if weak:
        parts = [
            f"{m['matched_table']}.{m['matched_column']} (partial \"{m['text']}\")"
            for m in weak
        ]
        lines.append(f"Weak col matches:   {', '.join(parts)}")

    for f in analysis["filter_candidates"]:
        conf_label = "" if f["confidence"] >= 1.0 else f" (partial match, conf {f['confidence']:.1f})"
        lines.append(f"Literal filter:     {f['column']} = '{f['value']}'{conf_label}")

    confirmed_joins = [j for j in analysis["required_joins"] if j["status"] == "confirmed"]
    possible_joins  = [j for j in analysis["required_joins"] if j["status"] != "confirmed"]

    for j in confirmed_joins:
        lines.append(f"Confirmed join:     {j['from']} → {j['to']}  (conf {j['confidence']:.2f})")
    for j in possible_joins:
        lines.append(f"Possible join:      {j['from']} → {j['to']}  (conf {j['confidence']:.2f} — verify)")

    m = analysis["metric"]
    if m["metric"] != "unknown":
        lines.append(f"Metric hint:        {m['metric'].upper()} ({m['aggregation']}) — verify in docs")

    for w in analysis.get("coverage_warnings", []):
        lines.append(
            f"DATA COVERAGE WARNING: '{w['lookup_table']}' covers only some IDs in "
            f"'{w['fact_table']}' — {w['missing_ids']} distinct {w['lookup_id_col']} values "
            f"in '{w['fact_table']}' are NOT in '{w['lookup_table']}'. "
            f"Check prose doc files for additional '{w['lookup_id_col']}' attribute data "
            f"and parse it to build a complete lookup before filtering."
        )

    knowledge_content = analysis.get("knowledge_content", "")
    if knowledge_content:
        lines.append(f"KNOWLEDGE.MD (full content — do not read this file again):\n{knowledge_content}")

    for doc in analysis.get("doc_contents", []):
        lines.append(
            f"DOC CONTENT ({doc['file']}): {doc['description']} "
            f"Use execute_python with regex to extract structured data from this file when needed."
        )

    # Detect all-docs mode: only *_paragraphs tables are present
    context_schema = analysis.get("_schema_tables", [])
    if context_schema and all(t.endswith("_paragraphs") for t in context_schema):
        lines.append(
            "ALL-DOCS MODE: show_context_schema has ONLY *_paragraphs tables — "
            "ALL structured data is embedded in prose documents. "
            "Use execute_python with regex to extract entity IDs and values from "
            "the full doc files. Do NOT rely on search_doc for data extraction. "
            "See ALL-DOCS MODE instructions in the system prompt."
        )

    return "\n".join(lines)
