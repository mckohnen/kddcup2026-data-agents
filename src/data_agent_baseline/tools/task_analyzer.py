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
from typing import Any


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

def build_task_analysis(
    question: str,
    context: dict,
    context_dir: "Path",
) -> dict[str, Any]:
    """Analyse a question against the context schema and data.

    Args:
        question:     The task question string.
        context:      Schema profile dict — output of ``build_schema_profile()``.
                      Must have ``tables`` and ``relationships`` keys.
        context_dir:  Path to the task context directory.  Used to get the
                      unified SQLite connection for literal value matching
                      across all sources (.db, CSV, JSON).

    Returns:
        Dict with keys: ``matched_terms``, ``candidate_tables``,
        ``candidate_columns``, ``required_joins``, ``filter_candidates``,
        ``metric``.
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

    return {
        "question": question,
        "matched_terms": all_matches,
        "candidate_tables": candidate_tables,
        "candidate_columns": candidate_columns,
        "required_joins": _infer_joins(candidate_tables, context),
        "filter_candidates": filter_candidates,
        "metric": _infer_metric(question),
    }


def format_task_analysis_hint(analysis: dict) -> str:
    """Format task analysis as a compact hint block for injection into the agent prompt.

    High-confidence column matches (≥ 0.8) are shown as "Strong col matches".
    Partial matches (< 0.8) are shown as "Weak col matches".
    JOINs are split into confirmed (≥ 0.95) and possible (0.80–0.95) tiers.
    A prominent disclaimer reminds the agent that this is a starting point only.
    """
    lines = [
        "[Pre-flight schema analysis]",
        "NOTE: This is a best-effort analysis of the question. It may be incomplete or",
        "contain incorrect links. Always call show_context_schema for the full authoritative",
        "schema, and verify any suggested joins against the actual data before using them.",
    ]

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

    return "\n".join(lines)
