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


# ---------------------------------------------------------------------------
# Conditional rule detection — drives targeted preflight hint injections
# ---------------------------------------------------------------------------

# Time values formatted as [M]M:SS[.mmm] — lap times, race times, durations.
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}([.,]\d+)?$")

_TIE_RULE_SYSTEM_PROMPT = (
    "You are a data analysis assistant. Given a question about data, answer with exactly "
    "one word: YES or NO.\n"
    "Answer YES if the question asks for a ranking, extreme value, or superlative where "
    "ties are possible — i.e. multiple rows could share the same maximum, minimum, cheapest, "
    "most expensive, most frequent, least frequent, fastest, slowest, highest, lowest, best, "
    "worst, or similar value, and ALL tied rows should be returned.\n"
    "Answer NO if the question asks for a count, average, sum, percentage, list of all items "
    "without an extreme filter, a specific named entity, or any other aggregation where "
    "tie-breaking is not relevant.\n"
    "Output only YES or NO — no explanation."
)


def _has_ranking_question(question: str, model: "ModelAdapter | None" = None) -> bool:
    """Return True if the question implies a tie-sensitive ranking or extreme value.

    When a model is available, uses a fast non-thinking LLM call for semantic detection.
    Falls back to keyword matching if no model or if the call fails.
    """
    if model is not None:
        from data_agent_baseline.agents.model import ModelMessage  # noqa: PLC0415
        try:
            result = model.complete([
                ModelMessage(role="system", content=_TIE_RULE_SYSTEM_PROMPT),
                ModelMessage(role="user", content=f"Question: {question}"),
            ], extra_body=_NO_THINK).strip().upper()
            return result.startswith("YES")
        except Exception:
            pass
    # Keyword fallback
    _RANKING_KEYWORDS: frozenset[str] = frozenset({
        "highest", "lowest", "best", "worst", "maximum", "minimum",
        "largest", "smallest", "most", "fewest", "top", "bottom",
        "fastest", "slowest", "greatest", "least", "cheapest", "expensive",
        "oldest", "youngest", "earliest", "latest", "longest", "shortest",
    })
    words = set(re.findall(r"\b\w+\b", question.lower()))
    return bool(words & _RANKING_KEYWORDS)


def _detect_dict_columns(context: dict) -> list[str]:
    """Return table.column names whose sample values look like Python dict strings."""
    found: list[str] = []
    for table, info in context.get("tables", {}).items():
        for col, col_info in info.get("columns", {}).items():
            samples = col_info.get("sample_values", [])
            if any(
                isinstance(s, str) and s.strip().startswith("{")
                for s in samples
                if s is not None
            ):
                found.append(f"{table}.{col}")
    return found


def _detect_time_columns(context: dict) -> list[str]:
    """Return table.column names whose sample values look like MM:SS.mmm time strings."""
    found: list[str] = []
    for table, info in context.get("tables", {}).items():
        for col, col_info in info.get("columns", {}).items():
            samples = col_info.get("sample_values", [])
            if any(
                isinstance(s, str) and _TIME_RE.match(s.strip())
                for s in samples
                if s is not None and str(s).strip()
            ):
                found.append(f"{table}.{col}")
    return found


def _detect_bidirectional_tables(context: dict, context_dir: "Path | None" = None) -> list[dict]:
    """Detect symmetric edge/link tables with two FK columns referencing the same entity.

    Pattern 1 — FK-based: two detected relationships from the same table to the same target.
    Pattern 2 — Name-based: a table has column pairs that share a base name (atom_id / atom_id2,
      from_id / to_id, node1_id / node2_id, src_id / dst_id, etc.).

    For each detected table, also checks (via SQL) whether both directions are stored as
    duplicate rows (symmetric storage: A→B AND B→A both present). This determines the
    correct counting strategy:
      - Symmetric storage (both rows present): filter by col1 alone — each entity already
        appears in col1 for every bond it participates in. Using OR or UNION ALL double-counts.
      - Asymmetric storage (each bond stored once): use UNION ALL over col1 and col2 to
        count all connections per entity.

    Returns list of dicts: {table, col1, col2, ref_table, symmetric} for each detected pattern.
    """
    from data_agent_baseline.tools.context_sqlite import run_sql_on_context  # noqa: PLC0415

    found: list[dict] = []
    seen: set[str] = set()

    # Pattern 1: two detected FK relationships from same table → same target
    fk_map: dict[tuple, list[str]] = {}
    for rel in context.get("relationships", []):
        key = (rel["from_table"], rel["to_table"])
        fk_map.setdefault(key, []).append(rel["from_column"])
    for (from_table, to_table), cols in fk_map.items():
        if len(cols) >= 2 and from_table not in seen:
            found.append({"table": from_table, "col1": cols[0], "col2": cols[1], "ref_table": to_table, "symmetric": None})
            seen.add(from_table)

    # Pattern 2: column name pairs suggesting symmetric edges
    _BIDIR_PAIRS = [
        ("_id", "_id2"), ("_id1", "_id2"),
        ("from_id", "to_id"), ("src_id", "dst_id"),
        ("node1_id", "node2_id"), ("source_id", "target_id"),
    ]
    for table_name, table_info in context.get("tables", {}).items():
        if table_name in seen:
            continue
        cols = list(table_info.get("columns", {}).keys())
        col_lower = [c.lower() for c in cols]
        for sfx_a, sfx_b in _BIDIR_PAIRS:
            for i, ca in enumerate(col_lower):
                if ca.endswith(sfx_a):
                    base = ca[: -len(sfx_a)]
                    expected_b = base + sfx_b
                    if expected_b in col_lower:
                        j = col_lower.index(expected_b)
                        ref_table = base.rstrip("_") if base else table_name
                        found.append({
                            "table": table_name,
                            "col1": cols[i],
                            "col2": cols[j],
                            "ref_table": ref_table,
                            "symmetric": None,
                        })
                        seen.add(table_name)
                        break
            if table_name in seen:
                break

    # Detect storage pattern via SQL: symmetric = both (A,B) and (B,A) rows exist
    if context_dir is not None:
        for entry in found:
            tbl, c1, c2 = entry["table"], entry["col1"], entry["col2"]
            try:
                q = (
                    f'SELECT COUNT(*) FROM "{tbl}" t1 '
                    f'WHERE EXISTS (SELECT 1 FROM "{tbl}" t2 '
                    f'WHERE t2."{c1}" = t1."{c2}" AND t2."{c2}" = t1."{c1}") LIMIT 1'
                )
                result = run_sql_on_context(context_dir, q, limit=1)
                count = (result.get("rows") or [[0]])[0][0]
                entry["symmetric"] = isinstance(count, int) and count > 0
            except Exception:
                entry["symmetric"] = None  # unknown

    return found


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return SequenceMatcher similarity ratio between two strings (0–1)."""
    return SequenceMatcher(None, a.lower(), str(b).lower()).ratio()


def _extract_knowledge_toc(knowledge_content: str) -> str:
    """Return markdown section headings from knowledge.md as a compact TOC string.

    Only extracts lines that start with # (headings). Returns a comma-separated
    list of heading titles (without # prefix) for injection into the preflight hint.
    Falls back to first 200 chars of content if no headings found.
    """
    headings = []
    for line in knowledge_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                headings.append(title)
    if headings:
        return "; ".join(headings)
    # No headings — return a short excerpt so agent knows something is there
    return knowledge_content[:200].replace("\n", " ").strip()


def _detect_numeric_text_columns(context: dict) -> list[str]:
    """Return table.column names from CSV tables where values look numeric but are TEXT.

    CSV tables loaded into SQLite store ALL values as TEXT, so MAX(score) on a column
    with values like '9', '14' returns '9' (lexicographic), not 14 (numeric).
    We flag columns where:
    - The table is a plain CSV/JSON source (no "." alias prefix, not a virtual table)
    - The schema profiler inferred type "number" or "integer"
    - There are at least 2 distinct values (to avoid trivially constant columns)
    - The column is not a primary or foreign key (IDs don't need numeric casting)
    """
    found: list[str] = []
    for table_name, table_info in context.get("tables", {}).items():
        # Skip .db tables (attached as "stem.table") — they have native SQLite types
        if "." in table_name:
            continue
        if table_name.endswith("_paragraphs") or table_name.endswith("_complete"):
            continue
        for col, col_info in table_info.get("columns", {}).items():
            if col_info.get("type") not in ("number", "integer"):
                continue
            if col_info.get("unique_count", 0) < 2:
                continue
            if col_info.get("role") in ("primary_key", "foreign_key"):
                continue
            if col_info.get("id_like"):
                continue
            # Also skip camelCase ID columns (e.g. PostId, UserId) that is_id_like misses
            col_lower = col.lower()
            if col_lower == "id" or col_lower.endswith("id") and not col_lower.endswith("_id"):
                # Only skip if the column name IS just an ID carrier (ends in bare "id")
                # but not something like "valid", "void", etc. — require at least 2 chars prefix
                if len(col_lower) > 2 and col_lower.endswith("id"):
                    continue
            found.append(f"{table_name}.{col}")
    return found


def _has_count_scope_question(question: str) -> bool:
    """Return True if the question asks for a total/count/aggregate with a condition.

    These questions are at risk of scope-widening: the agent correctly counts X
    satisfying a condition, then second-guesses and counts all members of qualifying
    containers instead.  A COUNT SCOPE hint warns against this.

    Only fires when the question includes a count/aggregate keyword AND a
    conditional preposition ('with', 'in', 'among', 'that', 'having', 'containing').
    Without the conditional, there is no scope ambiguity risk.
    """
    q_lower = question.lower()
    count_keywords = ("how many", "total", "count", "number of", "sum of", "calculate the")
    scope_keywords = (" with ", " in ", " among ", " that ", " having ", " containing ")
    has_count = any(kw in q_lower for kw in count_keywords)
    has_scope = any(kw in q_lower for kw in scope_keywords)
    return has_count and has_scope


def _detect_date_format_columns(context: dict) -> list[dict]:
    """Detect columns that store dates/periods as all-numeric strings and infer their format.

    Many datasets encode time periods as compact numeric strings:
      YYYYMM      → 6 digits, e.g. '201208'  (year-month)
      YYYYMMDD    → 8 digits, e.g. '20120815' (year-month-day)
      YYYYDDD     → 7 digits, e.g. '2012227'  (year + day-of-year)
      YYYY        → 4 digits, e.g. '2012'     (year only — too short, skip)

    Returns list of dicts: {table, column, format, example, filter_note}
    Only flags columns whose name suggests a date/period concept AND whose sample
    values are all-numeric strings of a consistent length matching a known pattern.
    Skips ID-like columns (they often happen to be 6-8 digit integers).
    """
    _DATE_NAME_TOKENS = frozenset({
        "date", "month", "year", "period", "ym", "ymd", "yearmonth",
        "yearmo", "dt", "time", "week", "quarter",
    })
    _FORMAT_MAP = {
        6: ("YYYYMM", "e.g. '201208' = August 2012",
            "Filter a specific month: WHERE {col} = '201208'  "
            "or a year range: WHERE {col} LIKE '2012%'"),
        7: ("YYYYDDD", "e.g. '2012227' = day 227 of 2012",
            "Filter a year: WHERE {col} LIKE '2012%'"),
        8: ("YYYYMMDD", "e.g. '20120815' = 15 Aug 2012",
            "Filter a month: WHERE {col} LIKE '201208%'  "
            "or a specific day: WHERE {col} = '20120815'"),
    }

    found: list[dict] = []
    for table_name, table_info in context.get("tables", {}).items():
        if table_name.endswith("_paragraphs") or table_name.endswith("_complete"):
            continue
        for col, col_info in table_info.get("columns", {}).items():
            # Must look like a date/period by name
            col_lower = col.lower()
            name_tokens = set(re.split(r"[_\s]", col_lower)) | {col_lower}
            if not (name_tokens & _DATE_NAME_TOKENS):
                continue
            # Skip obvious ID columns
            if col_info.get("role") in ("primary_key", "foreign_key"):
                continue
            if col_info.get("id_like"):
                continue
            # Sample values must be all-numeric strings of a consistent length
            samples = [str(v) for v in col_info.get("sample_values", []) if v is not None]
            if not samples:
                continue
            numeric_samples = [s for s in samples if s.isdigit()]
            if len(numeric_samples) < max(1, len(samples) // 2):
                continue  # majority must be all-digits
            lengths = {len(s) for s in numeric_samples}
            if len(lengths) != 1:
                continue  # inconsistent length — not a clean format
            digit_len = lengths.pop()
            if digit_len not in _FORMAT_MAP:
                continue  # 4-digit (year-only) and others — skip
            fmt, example, filter_note = _FORMAT_MAP[digit_len]
            found.append({
                "table": table_name,
                "column": col,
                "format": fmt,
                "example": example,
                "filter_note": filter_note.format(col=col),
                "sample": numeric_samples[0],
            })
    return found


def _detect_empty_string_columns(context: dict, context_dir: "Path") -> list[str]:
    """Return table.column names that have a significant proportion of empty strings.

    Uses the SQLite context to count empty strings (not just NULLs). Only numeric
    or unknown-type columns are checked, since those are the ones where CAST(''
    AS REAL)=0 would distort AVG/SUM/MIN/MAX aggregations.
    """
    from data_agent_baseline.tools.context_sqlite import run_sql_on_context

    found: list[str] = []
    tables = context.get("tables", {})
    for table_name, table_info in tables.items():
        if table_name.endswith("_paragraphs") or table_name.endswith("_complete"):
            continue
        row_count = table_info.get("row_count", 0)
        if row_count < 10:
            continue
        for col, col_info in table_info.get("columns", {}).items():
            col_type = col_info.get("type", "")
            if col_type not in ("number", "integer", "unknown", "string"):
                continue
            # Skip obvious ID or primary key columns
            if col_info.get("role") in ("primary_key", "foreign_key"):
                continue
            if col_info.get("id_like"):
                continue
            try:
                if "." in table_name:
                    alias, tbl = table_name.split(".", 1)
                    sql = f'SELECT COUNT(*) FROM "{alias}"."{tbl}" WHERE "{col}" = \'\''
                else:
                    sql = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" = \'\''
                result = run_sql_on_context(context_dir, sql, limit=1)
                empty_count = (result.get("rows") or [[0]])[0][0]
                if isinstance(empty_count, int) and empty_count > 0:
                    ratio = empty_count / row_count
                    if ratio >= 0.05:  # ≥5% empty strings — warn
                        found.append(f"{table_name}.{col}")
            except Exception:
                pass
    return found


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

_KNOWLEDGE_EXTRACT_SYSTEM_PROMPT = (
    "You are a precise knowledge extraction assistant. "
    "Given a data analysis question and a knowledge document, extract ONLY the parts "
    "that are directly useful for answering the question. "
    "Keep: column definitions, value encodings, and example SQL queries that relate to "
    "the entities, filters, or metrics mentioned in the question. "
    "Discard: sections about unrelated metrics, unrelated entities, and examples that "
    "do not involve any term from the question. "
    "Preserve original wording — do not paraphrase or summarise. "
    "Output only the extracted text with no preamble. "
    "If nothing is relevant, output the single word: NONE"
)

_KNOWLEDGE_MIN_CHARS_TO_FILTER = 600  # skip LLM call for short documents

# Passed as extra_body to disable Qwen3 chain-of-thought for fast utility calls.
# These calls need a short factual answer, not deep reasoning.
_NO_THINK = {"enable_thinking": False}


def _extract_relevant_knowledge(
    question: str,
    knowledge_content: str,
    model: "ModelAdapter",
) -> str:
    """Return the subset of knowledge_content relevant to the question.

    Falls back to the full content on LLM failure or if the document is short.
    """
    from data_agent_baseline.agents.model import ModelMessage  # noqa: PLC0415

    if len(knowledge_content) < _KNOWLEDGE_MIN_CHARS_TO_FILTER:
        return knowledge_content

    prompt = f"Question: {question}\n\nKnowledge document:\n{knowledge_content}"
    try:
        result = model.complete([
            ModelMessage(role="system", content=_KNOWLEDGE_EXTRACT_SYSTEM_PROMPT),
            ModelMessage(role="user", content=prompt),
        ], extra_body=_NO_THINK).strip()
        if result and result.upper() != "NONE" and len(result) >= 50:
            return result
    except Exception:
        pass
    return knowledge_content  # fallback to full content


_DOMAIN_EXPERT_SYSTEM_PROMPT = (
    "You are a senior domain expert analyst. "
    "Given a data analysis question, dataset schema, and knowledge excerpt, identify the domain "
    "and provide specific, actionable analysis guidance from an expert's perspective.\n\n"
    "Focus especially on:\n"
    "1. Multi-condition filtering on longitudinal data — choose ONE of:\n"
    "   - same-row: ONLY valid when both conditions share a column in the SAME physical row "
    "AND both are always measured together (e.g. a wide table where measurement columns are "
    "rarely empty). If either value is frequently empty/null, same-row will silently exclude "
    "most valid patients.\n"
    "   - any-row: patient ever satisfies condition A (in any row) AND ever satisfies condition B "
    "(in any row, even at different times). Use when the question asks about a patient's "
    "historical status independent of timing.\n"
    "   - temporal-proximity: condition A and condition B must occur within a reasonable time "
    "window of each other. Use when (a) both are measurements in a longitudinal table, "
    "(b) the values are measured on different visits (i.e. the relevant columns are often "
    "empty/null in the same row), AND (c) timing matters clinically (e.g. 'had normal immun marker concentration "
    "when dopamine levels were elevated'). Implement via a self-join on patient_id with a date "
    "window condition.\n"
    "   - unclear: if the question is genuinely ambiguous after reading the schema.\n"
    "2. Domain-specific definitions: what 'normal', 'active', 'current', or other qualitative "
    "terms mean in this context.\n"
    "3. Data structure pitfalls an analyst unfamiliar with this domain might miss.\n\n"
    "CRITICAL table-structure check: look at the schema. If a wide table has many measurement "
    "columns (lab values, scores, metrics) and sample rows show many empty/null cells, the "
    "measurements are collected on different visits — same-row will silently miss most patients. "
    "Choose temporal-proximity or any-row instead.\n\n"
    "IMPORTANT: Do NOT state specific column values, numeric thresholds, or encodings "
    "(e.g. do not say 'Thrombosis = 2' or 'WBC > 11'). The agent will read knowledge.md "
    "directly to obtain those — if you hallucinate them, the agent will use wrong values.\n\n"
    "Output ONLY a JSON object — no preamble, no explanation outside it:\n"
    "{\n"
    '  "domain": "<e.g. clinical/medical | financial/business | sports | manufacturing | general>",\n'
    '  "expert_role": "<e.g. clinical data analyst | financial controller | sports statistician>",\n'
    '  "multi_condition_logic": "<same-row | any-row | temporal-proximity | unclear — '
    'one-sentence structural rationale only, no specific values>",\n'
    '  "guidance": ["<bullet 1>", "<bullet 2>", "<bullet 3>"]\n'
    "}\n\n"
    "Maximum 3 guidance bullets. Structural guidance only — no specific thresholds or encodings."
)


def _format_schema_for_domain_expert(context: dict) -> str:
    """Format table column names concisely for the domain expert prompt."""
    lines = []
    for table_name, table_info in context.get("tables", {}).items():
        if table_name.endswith("_paragraphs") or table_name.endswith("_complete"):
            continue
        cols = list(table_info.get("columns", {}).keys())
        row_count = table_info.get("row_count", "?")
        col_str = ", ".join(cols[:20]) + (" …" if len(cols) > 20 else "")
        lines.append(f"  {table_name} ({row_count} rows): {col_str}")
    return "\n".join(lines) if lines else "(no structured tables)"


def _get_domain_expert_guidance(
    question: str,
    knowledge_content: str,
    model: "ModelAdapter",
    context: dict | None = None,
) -> dict:
    """Return domain-expert analysis guidance for the question.

    Returns a dict with keys: domain, expert_role, multi_condition_logic, guidance.
    Returns an empty dict on failure.
    """
    import json as _json
    from data_agent_baseline.agents.model import ModelMessage  # noqa: PLC0415

    context_block = knowledge_content[:2000] if knowledge_content else "(no knowledge document)"
    schema_block = _format_schema_for_domain_expert(context or {})
    prompt = (
        f"Question: {question}\n\n"
        f"Table schemas:\n{schema_block}\n\n"
        f"Dataset knowledge excerpt:\n{context_block}"
    )

    try:
        raw = model.complete([
            ModelMessage(role="system", content=_DOMAIN_EXPERT_SYSTEM_PROMPT),
            ModelMessage(role="user", content=prompt),
        ], extra_body=_NO_THINK).strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return _json.loads(raw)
    except Exception:
        return {}


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
            ], extra_body=_NO_THINK).strip()
        except Exception:  # noqa: BLE001
            continue  # skip failed batch, try remaining ones
        all_results.extend(_parse_response(response))

    return all_results


def build_task_analysis(
    question: str,
    context: dict,
    context_dir: "Path",
    model: "ModelAdapter | None" = None,
    cache_dir: "Path | None" = None,
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
    knowledge_toc: str = ""
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.exists():
        knowledge_path = context_dir / "doc" / "knowledge.md"
    if knowledge_path.exists():
        try:
            raw_knowledge = knowledge_path.read_text(encoding="utf-8", errors="replace").strip()
            if raw_knowledge:
                # Keep full content for domain expert LLM call (not injected raw into agent)
                knowledge_content = raw_knowledge
                # Extract TOC for lightweight injection — agent reads actual sections on demand
                knowledge_toc = _extract_knowledge_toc(raw_knowledge)
        except OSError:
            pass

    # Domain expert guidance — one LLM call that adopts the appropriate domain persona
    # and advises on multi-condition logic (same-row / any-row / temporal-proximity).
    domain_guidance: dict = {}
    if model is not None and knowledge_content:
        try:
            domain_guidance = _get_domain_expert_guidance(question, knowledge_content, model, context=context)
        except Exception:
            pass

    # Run coverage gap extraction — must come after _match_literal_values_sql above
    # which has already cached the SQLite connection.  The extractor injects
    # *_complete tables directly into that cached connection so the Analyst sees
    # them on its first SQL call without any extra steps.
    extracted_tables: list[dict] = []
    if model is not None:
        try:
            from data_agent_baseline.tools.coverage_extractor import detect_and_extract_coverage_gaps
            extracted_tables = detect_and_extract_coverage_gaps(
                context_dir=context_dir,
                model=model,
                timeout_seconds=25,
                cache_dir=cache_dir,
            )
        except Exception:
            pass

    # Add complete tables to context so column/table matching includes them.
    # This means schema hints will reference patient_sex_complete instead of
    # (or in addition to) patient_sex when the question references those columns.
    schema_tables = list(context.get("tables", {}).keys())
    if extracted_tables:
        for et in extracted_tables:
            complete_name = et["complete_table"]
            lookup_name = et["lookup_table"]
            if lookup_name in context.get("tables", {}) and complete_name not in context.get("tables", {}):
                context["tables"][complete_name] = dict(context["tables"][lookup_name])
        # Re-run table/column matching so complete table shows in candidate hints
        extra_table_matches = _match_tables(question, context)
        extra_col_matches = _match_columns(question, context)
        existing_keys = {json.dumps(m, sort_keys=True) for m in all_matches}
        for m in _dedupe(extra_table_matches + extra_col_matches):
            k = json.dumps(m, sort_keys=True)
            if k not in existing_keys:
                all_matches.append(m)
                t = m.get("matched_table")
                if t and t not in candidate_tables:
                    candidate_tables.append(t)
                t2, c = m.get("matched_table"), m.get("matched_column")
                if t2 and c and f"{t2}.{c}" not in candidate_columns:
                    candidate_columns.append(f"{t2}.{c}")
        schema_tables = list(context.get("tables", {}).keys())

    # Detect columns with significant empty-string rates (distort numeric aggregations)
    empty_string_columns = _detect_empty_string_columns(context, context_dir)

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
        "knowledge_toc": knowledge_toc,
        "extracted_tables": extracted_tables,
        "_schema_tables": schema_tables,
        "domain_guidance": domain_guidance,
        # Conditional rule flags — drive targeted injections in format_task_analysis_hint
        "has_ranking_question": _has_ranking_question(question, model=model),
        "has_count_scope_question": _has_count_scope_question(question),
        "dict_columns": _detect_dict_columns(context),
        "time_columns": _detect_time_columns(context),
        "empty_string_columns": empty_string_columns,
        "numeric_text_columns": _detect_numeric_text_columns(context),
        "date_format_columns": _detect_date_format_columns(context),
        "bidirectional_tables": _detect_bidirectional_tables(context, context_dir=context_dir),
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
            + "). Call show_context_schema if you need column-level detail."
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

    # Conditional rule injections — only present when the schema or question warrants them.
    # These rules were removed from the static system prompt (too edge-case for all tasks)
    # and are instead injected here so the model sees them only when relevant.
    if analysis.get("has_ranking_question"):
        lines.append(
            "TIE HINT: This question asks for an extreme or ranked value where multiple rows "
            "could share the same result. Consider using WHERE col = (SELECT MAX/MIN(col) FROM ...) "
            "instead of ORDER BY ... LIMIT 1 to capture all tied rows. After your query, "
            "verify: could another row have the same value? If yes, include it."
        )

    if analysis.get("has_count_scope_question"):
        lines.append(
            "COUNT SCOPE HINT: 'Total/count of X [with/in/among/containing] Y' means "
            "count X entities that DIRECTLY satisfy the condition — not all entities in "
            "containers that happen to contain some qualifying X. "
            "Example: 'total atoms with triple-bond molecules containing p' = count atoms "
            "WHERE element=p AND molecule has a triple bond (not all atoms in those molecules). "
            "A count of 1 is a perfectly valid final answer — do NOT widen the scope to "
            "increase the count."
        )

    bidir = analysis.get("bidirectional_tables", [])
    if bidir:
        for b in bidir[:2]:
            sym = b.get("symmetric")
            if sym is True:
                lines.append(
                    f"BIDIRECTIONAL TABLE (symmetric storage): '{b['table']}' stores each "
                    f"relationship in BOTH directions — (A,B) and (B,A) are separate rows. "
                    f"This means '{b['col1']}' already contains every entity for all its bonds. "
                    f"To count bonds per entity: GROUP BY {b['col1']}, COUNT(*). "
                    f"Do NOT use OR / UNION ALL — both directions are already in {b['col1']}."
                )
            elif sym is False:
                lines.append(
                    f"BIDIRECTIONAL TABLE (asymmetric storage): '{b['table']}' stores each "
                    f"relationship ONCE — an entity can appear in either '{b['col1']}' or '{b['col2']}'. "
                    f"To count all connections per entity, use UNION ALL:\n"
                    f"  SELECT {b['col1']} AS id FROM {b['table']} UNION ALL "
                    f"SELECT {b['col2']} AS id FROM {b['table']}\n"
                    f"then GROUP BY id and COUNT(*). Do NOT use OR — it double-counts."
                )
            else:
                lines.append(
                    f"BIDIRECTIONAL TABLE: '{b['table']}' has two FK columns "
                    f"({b['col1']}, {b['col2']}) both referencing '{b['ref_table']}'. "
                    f"First check if rows are stored once or twice per bond, then choose: "
                    f"symmetric storage → GROUP BY {b['col1']} only; "
                    f"asymmetric → UNION ALL of both columns."
                )

    dict_cols = analysis.get("dict_columns", [])
    if dict_cols:
        col_list = ", ".join(dict_cols[:3]) + (" …" if len(dict_cols) > 3 else "")
        lines.append(
            f"DICT COLUMN RULE: Column(s) {col_list} store Python dict strings "
            "(e.g. \"{'is_active': True, 'is_verified': False}\"). "
            "These use Python capitalisation (True/False), NOT SQL/JSON booleans. "
            "Filter with: col LIKE '%True%'  or  instr(col, 'True') > 0. "
            "Never use col = 1, col = 'true', or JSON functions — they silently match nothing."
        )

    time_cols = analysis.get("time_columns", [])
    if time_cols:
        col_list = ", ".join(time_cols[:3]) + (" …" if len(time_cols) > 3 else "")
        lines.append(
            f"TIME COLUMN RULE: Column(s) {col_list} store time as TEXT (e.g. \"1:23.456\"). "
            "TEXT ORDER BY is alphabetical — \"1:9\" > \"1:12\" as text! Two mandatory rules: "
            "(1) Filter blanks first: WHERE col != '' AND col IS NOT NULL. "
            "(2) Sort numerically: ORDER BY "
            "(CAST(SUBSTR(col,1,INSTR(col,':')-1) AS INTEGER)*60 "
            "+ CAST(SUBSTR(col,INSTR(col,':')+1) AS REAL)) ASC."
        )

    # Domain expert guidance — only show prescriptive bullets for temporal-proximity logic,
    # where the join pattern is genuinely non-obvious (date arithmetic, self-join on patient_id).
    # For same-row and any-row the classification label alone is the useful signal;
    # bullets for those cases tend to prescribe wrong encodings (e.g. Thrombosis IN (1,2))
    # that override what the agent would correctly read from knowledge.md.
    dg = analysis.get("domain_guidance", {})
    if dg:
        role = dg.get("expert_role", dg.get("domain", "domain expert"))
        mc_logic = dg.get("multi_condition_logic", "")
        guidance_bullets = dg.get("guidance", [])
        mc_logic_type = mc_logic.split("—")[0].strip().lower() if mc_logic else ""
        show_bullets = mc_logic_type == "temporal-proximity"
        lines.append(f"\nDOMAIN ANALYSIS GUIDANCE (perspective: {role}):")
        if mc_logic:
            lines.append(f"  Multi-condition logic: {mc_logic}")
        if show_bullets:
            for bullet in guidance_bullets:
                lines.append(f"  • {bullet}")
        lines.append("")  # blank line after block

    # Extracted tables: show prominently; suppress raw coverage warning for handled gaps.
    extracted = analysis.get("extracted_tables", [])
    extracted_lookup_names = {e["lookup_table"] for e in extracted}
    for e in extracted:
        attrs = ", ".join(e["attr_cols"])
        id_col = e.get("lookup_id_col", "ID")
        lines.append(
            f"EXTRACTED TABLE READY: '{e['complete_table']}' ({e['ids_extracted']} new rows "
            f"from {e['source_doc']} merged in, columns: {id_col}, {attrs}) — "
            f"USE THIS instead of '{e['lookup_table']}' for complete {attrs} coverage. "
            f"Do NOT re-extract from prose — the data is already in the SQL context."
        )

    for w in analysis.get("coverage_warnings", []):
        if w["lookup_table"] in extracted_lookup_names:
            continue  # gap was handled by extractor — no need to warn
        lines.append(
            f"DATA COVERAGE WARNING: '{w['lookup_table']}' covers only some IDs in "
            f"'{w['fact_table']}' — {w['missing_ids']} distinct {w['lookup_id_col']} values "
            f"in '{w['fact_table']}' are NOT in '{w['lookup_table']}'. "
            f"Check prose doc files for additional '{w['lookup_id_col']}' attribute data "
            f"and parse it to build a complete lookup before filtering."
        )

    empty_cols = analysis.get("empty_string_columns", [])
    if empty_cols:
        col_list = ", ".join(empty_cols[:5]) + (" …" if len(empty_cols) > 5 else "")
        lines.append(
            f"EMPTY STRING WARNING: Column(s) {col_list} contain empty strings ('') "
            "that CAST to 0 in SQL — this silently distorts AVG, SUM, MIN, MAX. "
            "This pattern may affect ALL numeric columns in these tables — not just the "
            "ones listed above. Apply the empty-string filter to EVERY numeric column you "
            "aggregate, even those not listed here: "
            "WHERE col != '' AND col IS NOT NULL"
        )

    numeric_text_cols = analysis.get("numeric_text_columns", [])
    if numeric_text_cols:
        col_list = ", ".join(numeric_text_cols[:8]) + (" …" if len(numeric_text_cols) > 8 else "")
        lines.append(
            f"NUMERIC TEXT COLUMNS: {col_list} — these CSV columns contain numbers but "
            "are stored as TEXT in SQLite. Always CAST for MAX, MIN, AVG, SUM, ORDER BY, "
            "WHERE comparisons, and subquery equality checks. "
            "WRONG: MAX(score) → may return '9' not 14. "
            "RIGHT: MAX(CAST(score AS INTEGER)). "
            "WRONG: WHERE col = (SELECT MAX(score) ...) — text vs text, still wrong. "
            "RIGHT: WHERE CAST(col AS INTEGER) = (SELECT MAX(CAST(score AS INTEGER)) ...)"
        )

    date_cols = analysis.get("date_format_columns", [])
    if date_cols:
        for dc in date_cols[:4]:
            lines.append(
                f"DATE FORMAT: '{dc['table']}.{dc['column']}' stores dates as "
                f"{dc['format']} numeric strings ({dc['example']}). "
                f"{dc['filter_note']}. "
                f"Do NOT use ISO formats (YYYY-MM-DD) or LIKE '%%-%%-%%' on this column."
            )

    knowledge_toc = analysis.get("knowledge_toc", "")
    if knowledge_toc:
        lines.append(
            f"KNOWLEDGE.MD EXISTS — sections: {knowledge_toc}. "
            "You MUST call read_knowledge_section to read the relevant sections "
            "BEFORE writing any SQL or Python. Do not skip this step."
        )

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
            "Two-phase strategy: "
            "(1) Use execute_python with regex to extract the full list of matching entity IDs. "
            "(2) Once you have a short list of specific IDs (≤~10), switch to targeted SQL — "
            "SELECT content FROM <stem>_paragraphs WHERE content LIKE '%<id>%' — "
            "and read the returned paragraph text to extract remaining attributes. "
            "Do NOT keep re-scanning the whole file with regex once you have the IDs. "
            "CRITICAL — regex exhaustion fallback: after 2 execute_python attempts, if you "
            "do not yet have COMPLETE results for ALL entities you need (even if you have "
            "partial results), STOP using regex immediately. Switch to targeted SQL for each "
            "specific ID: SELECT content FROM <stem>_paragraphs WHERE content LIKE '%<id>%' "
            "Partial or incomplete regex output is the trigger — you do NOT need 0 results "
            "to switch. The AI can parse the returned paragraph text directly without regex."
        )

    return "\n".join(lines)
