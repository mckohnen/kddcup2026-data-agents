from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.task_logger import get_logger
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.context_sqlite import run_sql_on_context
from data_agent_baseline.tools.filesystem import (
    _extract_md_section,
    _extract_md_toc,
    list_context_tree,
    read_doc_preview,
)
from data_agent_baseline.tools.input_detector import detect_input_files
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.registry import (
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
)
from data_agent_baseline.tools.schema_profiler import build_schema_profile
from data_agent_baseline.tools.task_analyzer import build_task_analysis, format_task_analysis_hint

DATA_AGENT_SYSTEM_PROMPT = """
You are a data agent solving a data analysis task.

Step 1 — Explore context:
  Call list_context to see all available files and their sizes.

Step 2 — Read documentation:
  Read knowledge.md before writing any SQL.  Two cases:
  (a) If the preflight hint contains a "KNOWLEDGE.MD (full content, N chars)" block
      with a fenced code section, the entire knowledge.md is ALREADY in your context —
      do NOT call read_knowledge_section, just re-read the block carefully.
      Pay particular attention to ALL Use Case exemplars (not just the first one that
      mentions the question's keyword) so you pick the right table/column pattern.
  (b) If the preflight only lists section headings ("KNOWLEDGE.MD EXISTS — sections: …"),
      call read_knowledge_section without a section argument to get the TOC, then call it
      again for each relevant section.  Read multiple Use Cases before committing.
  Focus on:
  - Column semantics: the question may use a natural-language term that maps to one specific
    column. Multiple similar-sounding columns often exist (e.g. "net_revenue" vs "gross_revenue",
    "base_salary" vs "total_compensation"). Read the documentation before choosing — never guess.
  - Value encodings: filter strings must match the data exactly (e.g. status='active',
    verified='Y', category='internal'). Do not assume — look up the exact value.
  - Categorical value matching: when the question uses a phrase that corresponds to a column
    value, filter for that exact string (e.g. 'pending orders' → status = 'pending',
    'verified members' → verified = 'Y'). Do not treat it as a non-null check.
  - Example queries in the documentation: replicate their logic, not just their structure.

  Prose document strategy — pick the right tool for the task:
  • Exact identifier lookup (you know a specific ID, code, or name):
      SQL on _paragraphs — SELECT content FROM <stem>_paragraphs WHERE content LIKE '%<id>%'
      More reliable than read_doc; semantic search may return the wrong paragraph for exact IDs.
      Chain lookups: find entity A's ID → use SQL on _paragraphs to fetch its paragraph →
      read the returned text to extract entity B's ID → SQL again for entity B's paragraph.
      NEVER re-scan the whole file with regex when you already have a list of specific IDs.
  • Conceptual or semantic lookup (what a term means, finding a policy or rule):
      read_doc with a focused query — a short concept phrase, not the full question.
      For files larger than ~30 KB, always pass a query parameter; large files without
      a query return low-relevance chunks.
  • Exhaustive extraction (need every occurrence of an entity type across a whole document):
      execute_python — read the full file, split by paragraph, extract with regex, print as JSON.
      Process paragraph by paragraph; print only the final structured result, not raw text.
      After this step you will have a short list of matching IDs — switch to SQL for any
      further attribute lookups on those IDs (see "Exact identifier lookup" above).
  • Regex exhaustion fallback: after 2 execute_python attempts, if you do not yet have
      COMPLETE results for ALL entities you need (even if some partial results were found),
      STOP retrying regex. Partial output is the trigger — you do not need 0 results to switch.
      Instead, for each specific ID you need, run:
        SELECT content FROM <stem>_paragraphs WHERE content LIKE '%<id>%'
      Read the returned paragraph text directly — the AI can parse natural-language prose
      without regex. This always works when the ID appears literally in the text.

Step 3 — Inspect schema and data sources:
  Call show_context_schema. Every file is a potential data source — never conclude "no data
  exists" after checking only one. All sources share one SQLite connection:
  - CSV / JSON files  → plain table names (e.g. SELECT * FROM orders)
  - SQLite .db files  → <db_stem>.<table> (e.g. SELECT * FROM inventory.products)
  - Prose doc/*.md    → <stem>_paragraphs(paragraph_idx INTEGER, content TEXT)

  Pre-extracted tables: if the preflight hint lists an "EXTRACTED TABLE READY" entry
  (e.g. entity_attrs_complete), use that table directly — do not re-extract from prose.

  Do NOT query sqlite_master or sqlite_schema — those omit all ATTACH'd .db tables.
  Always use show_context_schema as the authoritative table list.

Step 4 — Query and analyse:
  Use query_context_tables for SQL. Call get_current_datetime before any age, duration,
  or "current" calculation — never hardcode a year.

  SQL rules:
  - CSV columns are TEXT. Always CAST for numeric comparisons and arithmetic.
    This applies to WHERE filters, ORDER BY, MAX/MIN/AVG/SUM, and subquery comparisons:
      WRONG: WHERE revenue > 1000      ('9' > '10' is TRUE as text)
      RIGHT:  WHERE CAST(revenue AS REAL) > 1000
      WRONG: MAX(score)               (returns '9' not 14 when score is TEXT)
      RIGHT:  MAX(CAST(score AS INTEGER))
      WRONG: WHERE col = (SELECT MAX(score) ...)   (compares text to text — still wrong)
      RIGHT:  WHERE CAST(col AS INTEGER) = (SELECT MAX(CAST(score AS INTEGER)) ...)
    The text-sort hazard is especially dangerous in subquery equality checks: always CAST
    both sides.
  - JSON columns keep native types — no CAST needed.
  - Never use ROUND(), FORMAT(), or Python round() in output. Return raw computed values;
    the evaluation system handles precision normalisation.
      WRONG: ROUND(SUM(a) / SUM(b), 2)
      RIGHT:  CAST(SUM(a) AS REAL) / SUM(b)
  - Empty strings in CSV: CAST('' AS REAL) = 0, distorting AVG, SUM, MIN, MAX.
    Filter before any numeric aggregation:
      RIGHT: AVG(CASE WHEN col != '' AND col IS NOT NULL THEN CAST(col AS REAL) END)
      RIGHT: MIN(CASE WHEN col != '' AND col IS NOT NULL THEN CAST(col AS REAL) END)
  - Never add LIMIT to the final answer query — return all matching rows.
  - Trust your SQL: once the WHERE clause correctly encodes the question, submit every row
    it returns. Do not discard rows based on subjective reasoning.
  - For the 'type of X' questions: GROUP BY the short categorical type column (e.g.
    event.type, category), not a description or name field.
  - For any threshold, cutoff, reference range, or "normal vs abnormal" comparison:
    if 2 read_knowledge_section or search_doc calls fail to surface a definitive
    value for the term, IMMEDIATELY call lookup_reference_range. Do NOT keep
    issuing more searches — the tool already searches the docs first and falls
    back to domain knowledge in one step. This applies to clinical/lab ranges,
    business KPI thresholds, and any other "what counts as X" cutoff.
    The tool returns a structured dict: {normal_range: {lower, upper, units},
    abnormal_condition, confidence, source, raw_extracted_text}. The
    abnormal_condition field is a plain-English WHERE-clause description you
    can translate directly into SQL — do NOT re-interpret the raw text.
    After receiving the range, verify units by running SELECT MIN(col), MAX(col),
    AVG(col), and the empty/null count on the actual data column.
  - Unit-mismatch resolution: when normal_range and the data range don't line
    up cleanly, try standard unit conversions (mg/dL ↔ g/L, cells/µL ↔ ×10⁹/L,
    etc.) first. If NO sensible unit conversion brings normal_range inside the
    observed data range, the data IS in the standard unit and ALL non-empty
    values fall outside the normal range — meaning every measured patient is
    "abnormal" (medically realistic: many tests are only ordered on clinical
    suspicion of abnormality, so the recorded values are pre-selected for
    abnormality). In that case answer with the count of non-empty rows that
    also satisfy the other filters.
  - If a table is referenced in documentation but missing from the schema ("no such table"),
    the data may live in a prose doc file — load it with execute_python instead.
  - Numeric date/period columns (e.g. Date, YearMonth) may use compact formats:
      YYYYMM (201208), YYYYMMDD (20120815), YYYYDDD (2012227).
    Check sample_values in show_context_schema output to confirm the format before
    filtering. The preflight hint will name any detected date-period columns.
    When the question specifies a time period, apply the same date filter to
    EVERY table in the query that has that date column — not just the primary fact table.
  - Compound string IDs (e.g. "entity_1", "entity_2", ..., "entity_10") sort
    lexicographically as text: "entity_10" < "entity_2". To retrieve the Nth item
    by natural numeric order, extract and cast the numeric suffix:
      ORDER BY CAST(SUBSTR(id_col, INSTR(id_col, '_') + 1) AS INTEGER)
    Apply this whenever an ID column contains a text prefix + numeric suffix and
    you need position-based selection (e.g. "4th atom", "last entry").
  - Read the DOMAIN ANALYSIS GUIDANCE block in the preflight hint before writing any
    WHERE clause that combines multiple conditions on a longitudinal or time-series table.

Step 5 — Validate before submitting:
  1. Column count: "how many" / "what is the [aggregate]" → 1 column, 1 row.
     "List [X]" → return only the identifier or name column — no supplementary columns.
     "List X and Y" → exactly 2 columns. Never add extra columns not explicitly requested.
     "Tally" → return only the value column — never add a count/frequency column alongside.
     "Give their X" / "return their X" / "show their X" / "what is their X" → return ONLY
     column X. Do NOT add an ID, key, or name column alongside X for context — those were
     not requested. If X is fully identified by a filter condition already in the WHERE clause,
     the output need only contain the X values themselves.
  2. Text content: "what is the [comment / title / description / body / text / message / post]" →
     return the TEXT column itself, NOT an ID, uuid, or integer column.
     "What is the name of X" → return the name column, not the ID column.
     This applies even if you found the row by ID — select the content column, not the key column.
  3. Column names from source data — never invent aliases or rename columns.
     Use the exact column name as it appears in the schema (e.g. `name`, not `race_name`).
  4. Deduplication: "list distinct values" or "tally" → SELECT DISTINCT or GROUP BY.
     Do not add a count column unless the question explicitly asks for frequency.
  5. Trust your first correct result: if an initial query returns a plausible answer, verify
     the logic once, then submit. Do not keep re-querying at wider and wider scope — each
     revision risks replacing a correct answer with a wrong one.
  6. Check the preflight hint for TIE HINT, BIDIRECTIONAL TABLE, DICT COLUMN, or TIME COLUMN
     notices — apply those patterns before submitting.

Step 6 — Submit:
  Call answer with the final result table.

Response format (mandatory):
  Always respond with exactly one ```json block. Never add any text after the closing ```.
  {
    "thought": "<your reasoning>",
    "action": "<tool_name>",
    "action_input": {<parameters>}
  }
""".strip()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _list_context(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_doc(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 8000))
    section = action_input.get("section")
    # Agent may pass a focused query to override the default (task.question).
    # A targeted single-concept query surfaces far more relevant chunks than
    # the full question when the document is large.
    query = str(action_input["query"]) if action_input.get("query") else task.question
    content = read_doc_preview(
        task,
        path,
        max_chars=max_chars,
        query=query,
        section=section,
    )
    return ToolExecutionResult(ok=True, content=content)


def _read_knowledge_section(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    """Targeted section reader for knowledge.md — returns TOC if no section given."""
    knowledge_path = task.context_dir / "knowledge.md"
    if not knowledge_path.exists():
        return ToolExecutionResult(ok=False, content={"error": "knowledge.md not found in context."})
    text = knowledge_path.read_text(encoding="utf-8", errors="replace")
    section = (action_input.get("section") or "").strip()
    if not section:
        toc = _extract_md_toc(text)
        return ToolExecutionResult(ok=True, content={"sections": toc})
    extracted = _extract_md_section(text, section)
    if not extracted:
        toc = _extract_md_toc(text)
        return ToolExecutionResult(ok=True, content={
            "error": f"Section '{section}' not found.",
            "available_sections": toc,
        })
    return ToolExecutionResult(ok=True, content={"section": section, "content": extracted})


def _show_context_schema(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    del action_input
    profile = build_schema_profile(task.context_dir, question=task.question)
    return ToolExecutionResult(ok=True, content=profile)


def _query_context_tables(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    # Block sqlite_master / sqlite_schema queries — they only expose the main schema
    # and silently omit all ATTACH'd .db tables, causing the agent to loop in confusion.
    sql_compact = sql.upper().replace(" ", "").replace("\n", "")
    if "SQLITE_MASTER" in sql_compact or "SQLITE_SCHEMA" in sql_compact:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": (
                    "Querying sqlite_master or sqlite_schema is not allowed. "
                    "Those system tables only list main-schema tables and will miss "
                    "all .db tables loaded via ATTACH. "
                    "Call show_context_schema to get the complete authoritative table list."
                )
            },
        )
    limit = int(action_input.get("limit", 200))
    result = run_sql_on_context(task.context_dir, sql, limit=limit)
    return ToolExecutionResult(ok=True, content=result)


_EXECUTE_PYTHON_MAX_OUTPUT_CHARS = 3000
_EXECUTE_PYTHON_MAX_OUTPUT_LINES = 60


def _execute_python(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    # Truncate large outputs before they enter the conversation history.
    # Line-based truncation preserves complete lines (no mid-sentence cuts)
    # and tells the agent the total scale so it knows how much was omitted.
    raw_output = content.get("output", "") or ""
    if len(raw_output) > _EXECUTE_PYTHON_MAX_OUTPUT_CHARS:
        content = dict(content)  # don't mutate the original
        lines = raw_output.splitlines()
        head = "\n".join(lines[:_EXECUTE_PYTHON_MAX_OUTPUT_LINES])
        if len(head) > _EXECUTE_PYTHON_MAX_OUTPUT_CHARS:
            head = head[:_EXECUTE_PYTHON_MAX_OUTPUT_CHARS]
        content["output"] = (
            head
            + "\n[truncated — "
            + str(len(lines)) + " lines / " + str(len(raw_output)) + " chars total. "
            "Print only your final structured result, not raw file content.]"
        )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _get_current_datetime(_: PublicTask, action_input: dict) -> ToolExecutionResult:
    del action_input
    now = datetime.now()
    return ToolExecutionResult(ok=True, content={
        "date": now.strftime("%Y-%m-%d"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
    })


_LOOKUP_EXTRACT_SYSTEM_PROMPT = (
    "You are a data extraction assistant. You will be shown text passages about entities. "
    "For each passage, extract the requested attribute for the entity mentioned. "
    "Output exactly one line per entity in this format:\n"
    "  <identifier>: <extracted value>\n"
    "If the attribute is not mentioned in the passage, output:\n"
    "  <identifier>: unknown\n"
    "No preamble, no explanation — only the lines."
)


def _make_lookup_ids_in_doc(model: "ModelAdapter | None"):
    """Factory returning a _lookup_ids_in_doc closure with access to the model."""

    def _lookup_ids_in_doc(task: PublicTask, action_input: dict) -> ToolExecutionResult:
        """Find and extract an attribute from prose for IDs from a structured table.

        Scans `doc_path` paragraph-by-paragraph for exact word-boundary matches of
        each distinct value in `table_name.id_column`.  Works for any identifier type
        (numeric IDs, names, codes) and any reference style ("Patient 43003",
        "Case ID 43003", "John Smith", etc.).

        When `attribute` is provided (e.g. "gender (M/F)", "year of birth", "city"),
        a single batched LLM call extracts that attribute from each matched paragraph
        set and returns `{id: extracted_value}`.  Without `attribute`, returns
        `{id: [paragraph_texts]}` for downstream processing.
        """
        from data_agent_baseline.tools.context_sqlite import run_sql_on_context

        table_name = str(action_input.get("table_name", "")).strip()
        id_column = str(action_input.get("id_column", "")).strip()
        doc_path = str(action_input.get("doc_path", "")).strip()
        attribute = str(action_input.get("attribute", "")).strip()

        if not table_name or not id_column or not doc_path:
            return ToolExecutionResult(ok=False, content={
                "error": "table_name, id_column, and doc_path are all required."
            })

        # Get distinct IDs from SQL table
        try:
            sql_result = run_sql_on_context(
                task.context_dir,
                f'SELECT DISTINCT "{id_column}" FROM "{table_name}" WHERE "{id_column}" IS NOT NULL',
                limit=10000,
            )
        except Exception as exc:
            return ToolExecutionResult(ok=False, content={"error": f"SQL error: {exc}"})

        ids = [str(row[0]).strip() for row in (sql_result.get("rows") or []) if row[0] is not None]
        if not ids:
            return ToolExecutionResult(ok=False, content={
                "error": f"No non-null IDs found in {table_name}.{id_column}"
            })

        # Read and split prose document
        full_doc_path = task.context_dir / doc_path
        if not full_doc_path.exists():
            return ToolExecutionResult(ok=False, content={"error": f"File not found: {doc_path}"})
        if full_doc_path.is_dir():
            return ToolExecutionResult(ok=False, content={
                "error": f"'{doc_path}' is a directory, not a file."
            })

        text = full_doc_path.read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        # Build id→paragraphs mapping via exact word-boundary match
        id_to_paras: dict[str, list[str]] = {}
        for para in paragraphs:
            for id_val in ids:
                if re.search(r"\b" + re.escape(id_val) + r"\b", para):
                    bucket = id_to_paras.setdefault(id_val, [])
                    if len(bucket) < 3:
                        bucket.append(para[:800] + ("…" if len(para) > 800 else ""))

        matched = len(id_to_paras)
        truncated = matched > 100
        results_subset = dict(list(id_to_paras.items())[:100])

        # If no attribute requested, return raw paragraphs
        if not attribute:
            return ToolExecutionResult(ok=True, content={
                "doc_path": doc_path,
                "total_ids_searched": len(ids),
                "matched_ids": matched,
                "results_truncated_to_100": truncated,
                "results": results_subset,
                "hint": "Pass 'attribute' (e.g. 'gender (M/F)', 'birth year') to extract values via LLM instead of raw paragraphs.",
            })

        # LLM-based extraction: one batched call for all matched IDs
        if model is None:
            return ToolExecutionResult(ok=True, content={
                "doc_path": doc_path,
                "total_ids_searched": len(ids),
                "matched_ids": matched,
                "results_truncated_to_100": truncated,
                "results": results_subset,
                "warning": "No model available for attribute extraction — returning raw paragraphs.",
            })

        # Build extraction prompt: one block per ID
        sections = []
        for id_val, paras in results_subset.items():
            combined = "\n".join(paras)
            sections.append(f"[Entity: {id_val}]\n{combined}")
        user_content = (
            f"Extract attribute: {attribute}\n\n"
            + "\n\n".join(sections)
        )

        log = get_logger()
        log.info("  LOOKUP_EXTRACT attribute=%r over %d matched IDs", attribute, len(results_subset))
        try:
            response = model.complete([
                ModelMessage(role="system", content=_LOOKUP_EXTRACT_SYSTEM_PROMPT),
                ModelMessage(role="user", content=user_content),
            ]).strip()
        except Exception as exc:
            return ToolExecutionResult(ok=False, content={"error": f"LLM extraction failed: {exc}"})

        # Parse "identifier: value" lines
        extracted: dict[str, str] = {}
        for line in response.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            id_part, _, val_part = line.partition(":")
            id_part = id_part.strip()
            val_part = val_part.strip()
            if id_part in results_subset:
                extracted[id_part] = val_part

        return ToolExecutionResult(ok=True, content={
            "doc_path": doc_path,
            "total_ids_searched": len(ids),
            "matched_ids": matched,
            "extracted_attribute": attribute,
            "results_truncated_to_100": truncated,
            "extracted": extracted,
            "unmatched_ids": [i for i in ids if i not in id_to_paras],
        })

    return _lookup_ids_in_doc


def _parse_exhausted_searches(prior_summaries: list[str]) -> set[tuple[str, str]]:
    """Extract (path, keyword) pairs that returned 0 results in prior attempts.

    These are hard-blocked in _search_doc so the agent cannot waste steps
    repeating searches that are already known to be empty.
    """
    blocked: set[tuple[str, str]] = set()
    for summary in prior_summaries:
        in_block = False
        for line in summary.splitlines():
            if "ALREADY DONE with ZERO results" in line:
                in_block = True
                continue
            if in_block:
                # Lines look like: "  keyword='foo' in doc/Bar.md"
                m = re.match(r"\s+keyword='(.+)' in (.+)", line)
                if m:
                    blocked.add((m.group(2).strip(), m.group(1).strip().lower()))
                elif line.strip() and not line.startswith(" "):
                    in_block = False
    return blocked


def _make_search_doc(task: PublicTask, blocked_searches: set[tuple[str, str]]):
    """Factory returning a _search_doc closure with pre-blocked exhausted searches."""
    def _search_doc_impl(action_input: dict) -> ToolExecutionResult:
        from data_agent_baseline.tools.filesystem import resolve_context_path
        path = str(action_input["path"])
        keyword = str(action_input["keyword"]).lower()
        max_results = int(action_input.get("max_results", 10))

        # Warn (not hard-error) for searches that already returned 0 results.
        # Returning ok=True keeps the model in a healthy JSON-generation state;
        # ok=False can cause the model to spiral into malformed-output loops.
        if (path, keyword) in blocked_searches:
            return ToolExecutionResult(ok=True, content={
                "keyword": keyword,
                "total_matches": 0,
                "showing": 0,
                "results": [],
                "warning": (
                    f"This search (keyword='{keyword}' in {path}) was already performed "
                    "in a prior attempt and returned 0 results. Do NOT repeat it — "
                    "try a different keyword or use execute_python to read the full file."
                ),
            })

        full_path = resolve_context_path(task, path)
        if full_path.is_dir():
            files = sorted(f.name for f in full_path.iterdir() if f.is_file())
            return ToolExecutionResult(ok=False, content={
                "error": f"'{path}' is a directory, not a file. Specify a file path.",
                "files_in_directory": files,
            })
        text = full_path.read_text(encoding="utf-8", errors="replace")

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        matches = [p for p in paragraphs if keyword in p.lower()]
        return ToolExecutionResult(ok=True, content={
            "keyword": keyword,
            "total_matches": len(matches),
            "showing": min(max_results, len(matches)),
            "results": matches[:max_results],
        })
    return _search_doc_impl


def _answer(_: PublicTask, action_input: dict) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")
    normalized: list[list] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized.append(list(row))
    return ToolExecutionResult(
        ok=True,
        content={"status": "submitted", "column_count": len(columns), "row_count": len(normalized)},
        is_terminal=True,
        answer=AnswerTable(columns=list(columns), rows=normalized),
    )


# ---------------------------------------------------------------------------
# Tool registry — unified, always the same set regardless of file types
# ---------------------------------------------------------------------------

_TOOL_SPECS: dict[str, ToolSpec] = {
    "list_context": ToolSpec(
        name="list_context",
        description="List all files and directories under the task context directory.",
        input_schema={"max_depth": 4},
    ),
    "get_current_datetime": ToolSpec(
        name="get_current_datetime",
        description=(
            "Return today's date and current year/month/day. "
            "Call this before any query involving age, duration, or 'current' calculations. "
            "Never assume or hardcode the current year."
        ),
        input_schema={},
    ),
    "read_doc": ToolSpec(
        name="read_doc",
        description=(
            "Read a document from context. For knowledge.md, returns a table of contents "
            "on the first call; pass 'section' to read a specific ## header block. "
            "For large doc/ files, pass 'query' with a focused search term to surface the "
            "most relevant chunks — use a short concept phrase, not the full question. "
            "If no query is given, uses the task question for relevance ranking. "
            "Use execute_python for exhaustive extraction of all entities in large prose files."
        ),
        input_schema={"path": "knowledge.md", "section": "## 2. Core Entities & Fields", "query": "optional focused search term"},
    ),
    "search_doc": ToolSpec(
        name="search_doc",
        description=(
            "Keyword search inside any prose document in the context directory. "
            "Returns all paragraphs that contain the keyword (case-insensitive). "
            "Use this to locate specific terms, thresholds, or entity mentions in large "
            "Markdown files without needing to write Python code."
        ),
        input_schema={"path": "doc/SomeFile.md", "keyword": "the term to find", "max_results": 10},
    ),
    "read_knowledge_section": ToolSpec(
        name="read_knowledge_section",
        description=(
            "Read a specific ## section from knowledge.md by its exact header text. "
            "Omit 'section' to get the table of contents. "
            "Use this for targeted lookups: column meanings, encodings, example queries."
        ),
        input_schema={"section": "## 2. Core Entities & Fields"},
    ),
    "show_context_schema": ToolSpec(
        name="show_context_schema",
        description=(
            "Return a rich schema profile for all tables in the unified SQLite connection: "
            "CSV/JSON tables (plain names) and SQLite .db tables (<db_stem>.<table>). "
            "Includes column types, null counts, sample values, and inferred relationships."
        ),
        input_schema={},
    ),
    "query_context_tables": ToolSpec(
        name="query_context_tables",
        description=(
            "Run SQL on the unified in-memory SQLite containing ALL context data. "
            "CSV/JSON tables use plain names; .db tables use <db_stem>.<table> prefix. "
            "IMPORTANT: CSV columns are stored as TEXT. Always CAST for numeric comparisons: "
            "  WRONG: WHERE height_cm > 200  (text: '61'>'200' is TRUE!) "
            "  RIGHT:  WHERE CAST(height_cm AS INTEGER) > 200 "
            "JSON columns keep native types (no CAST needed). "
            "Do NOT use sqlite_master to list tables — it only shows CSV/JSON tables. "
            "JOINs across CSV, JSON, and .db sources work in a single query."
        ),
        input_schema={"sql": "SELECT ...", "limit": 200},
    ),
    "execute_python": ToolSpec(
        name="execute_python",
        description=(
            f"Execute Python code with the context directory as working directory. "
            f"Use for: (1) exhaustive extraction from large prose Markdown documents "
            f"(read the full file, split by paragraph, extract structured data, print as JSON); "
            f"(2) multi-step computation not easily expressed in SQL. "
            f"Standard libraries and pandas are available. "
            f"Do NOT open SQLite connections inside Python — use query_context_tables instead. "
            f"Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS}s."
        ),
        input_schema={"code": "import os\nprint(sorted(os.listdir('.')))"},
    ),
    "lookup_reference_range": ToolSpec(
        name="lookup_reference_range",
        description=(
            "Look up the standard reference range (normal/abnormal threshold) for a column. "
            "Searches all context doc files first (up to 3 attempts per file). "
            "If nothing found in docs, falls back to domain knowledge. "
            "Use this INSTEAD of making multiple search_doc calls for a threshold — "
            "it returns a definitive range in one step. "
            "Returns a structured dict with keys: "
            "normal_range = {lower, upper, units}, "
            "abnormal_condition (a plain-English WHERE-clause-style description), "
            "confidence ('high'|'medium'|'low'), "
            "source ('context_documents'|'domain_knowledge'), "
            "raw_extracted_text (the source snippet for verification). "
            "Always verify units against SELECT MIN(col), MAX(col) after receiving the range."
        ),
        input_schema={"column": "WBC", "question_context": "normal range for white blood cells"},
    ),
    "answer": ToolSpec(
        name="answer",
        description=(
            "Submit the final answer table. This is the only valid terminating action. "
            "columns must be the exact source column names. "
            "Only include columns explicitly requested in the question."
        ),
        input_schema={"columns": ["col"], "rows": [["value"]]},
    ),
}

_TOOL_HANDLERS = {
    "get_current_datetime": _get_current_datetime,
    "list_context": _list_context,
    "read_doc": _read_doc,
    "read_knowledge_section": _read_knowledge_section,
    # search_doc is NOT included here — created per-task via factory.
    # See create_data_agent_tool_registry.
    "show_context_schema": _show_context_schema,
    "query_context_tables": _query_context_tables,
    "execute_python": _execute_python,
    "answer": _answer,
}


_DOMAIN_KNOWLEDGE_SYSTEM_PROMPT = (
    "You are a factual domain knowledge assistant. "
    "Answer the question using your training knowledge — be specific and concise. "
    "If the answer involves a numeric threshold or range, state it explicitly in standard units. "
    "If values genuinely vary by context (age, sex, laboratory), mention the most commonly "
    "cited clinical reference range. "
    "If you are uncertain, say so explicitly rather than guessing."
)


_STRUCTURED_RANGE_EXTRACTION_PROMPT = (
    "You are a reference-range extractor.  Given some text describing thresholds, "
    "normal ranges, or abnormality criteria for a measured value, return a single "
    "JSON object with these fields and no other text:\n"
    "  {\n"
    '    "lower": <float | null>   // lower bound of the NORMAL range; null if only an upper bound is given\n'
    '    "upper": <float | null>   // upper bound of the NORMAL range; null if only a lower bound is given\n'
    '    "units": <string | null>  // units (e.g. "mg/dL", "x10^9/L", "%"). null if not stated\n'
    '    "abnormal_condition": <string>  // a SHORT plain-English description of what counts as abnormal, e.g. "value > 1.2 mg/dL" or "value < 4.5 or value > 11.0 x10^9/L"\n'
    '    "confidence": "high" | "medium" | "low"  // "high" if the text explicitly states the range; "medium" if it strongly implies; "low" if you are inferring\n'
    "  }\n\n"
    "Rules:\n"
    "  - If the text gives BOTH bounds, fill both lower and upper.\n"
    "  - If the text says \"above X is abnormal\", set upper=X and lower=null.\n"
    "  - If the text says \"below X is abnormal\", set lower=X and upper=null.\n"
    "  - If the text mentions multiple ranges (e.g. gender-specific), use the most "
    "general or report both with abnormal_condition combining them.\n"
    "  - Numeric values MUST be JSON numbers (not strings).\n"
    "  - If you genuinely cannot extract a numeric range, return: "
    '{"lower": null, "upper": null, "units": null, "abnormal_condition": "<plain text from source>", "confidence": "low"}\n'
    "Output ONLY the JSON object."
)


def _extract_structured_range(
    raw_text: str,
    model: "ModelAdapter",
    column: str,
    question_context: str,
) -> dict:
    """Convert free-text range info into a structured dict.

    Falls back to a minimal envelope when the LLM call or JSON parse fails,
    so callers always get a dict (never raises).
    """
    import json as _json  # noqa: PLC0415
    from data_agent_baseline.agents.model import ModelMessage  # noqa: PLC0415

    user_msg = (
        f"Measured value: '{column}' (context: {question_context})\n\n"
        f"Source text:\n{raw_text}"
    )
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=_STRUCTURED_RANGE_EXTRACTION_PROMPT),
                ModelMessage(role="user", content=user_msg),
            ],
            extra_body={"enable_thinking": False},
        ).strip()
        # Strip code fences if the model added them despite instructions.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()
        parsed = _json.loads(raw)
        # Sanity-coerce types so the agent always sees a consistent shape.
        return {
            "lower": (None if parsed.get("lower") in (None, "") else float(parsed["lower"])),
            "upper": (None if parsed.get("upper") in (None, "") else float(parsed["upper"])),
            "units": parsed.get("units") or None,
            "abnormal_condition": str(parsed.get("abnormal_condition") or "").strip()
                                  or "see raw_extracted_text",
            "confidence": str(parsed.get("confidence") or "low").lower(),
        }
    except Exception:
        return {
            "lower": None,
            "upper": None,
            "units": None,
            "abnormal_condition": raw_text[:200].strip(),
            "confidence": "low",
        }


def create_data_agent_tool_registry(
    input_files: dict | None = None,
    model: ModelAdapter | None = None,
    task: "PublicTask | None" = None,
    blocked_searches: "set[tuple[str, str]] | None" = None,
) -> ToolRegistry:
    """Return the unified tool registry.

    All tools are always active — the unified SQLite handles CSV, JSON, and .db
    sources transparently, so no per-file-type routing is needed.
    The ``input_files`` parameter is kept for backwards compatibility but ignored.
    Pass ``model`` to enable the ``consult_domain_knowledge`` tool.
    Pass ``blocked_searches`` (a set of (path, keyword) pairs that returned 0 results
    in prior attempts) to hard-block those searches and prevent repetition.
    """
    del input_files  # no longer used

    specs = dict(_TOOL_SPECS)
    handlers = dict(_TOOL_HANDLERS)

    # search_doc is wired here (not in _TOOL_HANDLERS) so we can inject the
    # per-attempt blocked-search set that prevents repeating 0-result queries.
    _blocked = blocked_searches or set()
    if task is not None:
        _search_impl = _make_search_doc(task, _blocked)
        handlers["search_doc"] = lambda _task, ai: _search_impl(ai)
    else:
        # Fallback: no task context, create a minimal closure that just calls the raw impl.
        # In practice task is always provided; this guards against test callers.
        handlers["search_doc"] = lambda _task, ai: ToolExecutionResult(
            ok=False, content={"error": "search_doc requires task context."}
        )

    if model is not None:
        def _consult_domain_knowledge(_task: PublicTask, action_input: dict) -> ToolExecutionResult:
            question = str(action_input.get("question", "")).strip()
            if not question:
                return ToolExecutionResult(ok=False, content={"error": "'question' is required."})
            log = get_logger()
            log.info("  DOMAIN_KNOWLEDGE query=%r", question[:120])
            try:
                answer = model.complete([
                    ModelMessage(role="system", content=_DOMAIN_KNOWLEDGE_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=question),
                ], extra_body={"enable_thinking": False})
                log.info("  DOMAIN_KNOWLEDGE answer=%r", answer[:120])
                return ToolExecutionResult(
                    ok=True,
                    content={
                        "answer": answer.strip(),
                        "source": "model_training_knowledge",
                        "next_step": (
                            "Treat as a starting point.  For numeric thresholds, "
                            "prefer lookup_reference_range which combines doc search "
                            "with domain knowledge and returns a structured answer."
                        ),
                    },
                )
            except Exception as exc:
                return ToolExecutionResult(ok=False, content={"error": str(exc)})

        specs["consult_domain_knowledge"] = ToolSpec(
            name="consult_domain_knowledge",
            description=(
                "Query the model's training knowledge for a factual domain question — "
                "e.g. well-known constants, business KPI definitions, or facts not in "
                "the context files.  For lab/clinical reference ranges prefer "
                "lookup_reference_range, which combines doc search + domain knowledge "
                "+ structured extraction in one step.  Treat the answer as a starting "
                "point, not ground truth."
            ),
            input_schema={"question": "What unit is fastestLapTime stored in?"},
        )
        handlers["consult_domain_knowledge"] = _consult_domain_knowledge

        # lookup_reference_range: a multi-step pipeline that gathers all relevant
        # threshold info, structures it, and returns a definitive answer.  The data
        # agent calls this as ONE tool — internal logic does the heavy lifting so
        # the data agent's system prompt stays slim.
        def _make_lookup_reference_range(task_inner=task):
            # Constants for the broadened doc search.
            _MAX_PARAS_PER_FILE = 8   # collect up to N matching paragraphs per file
            _MAX_TOTAL_PARAS = 20     # absolute cap across all files
            _PARA_PREVIEW_CHARS = 400

            def _gather_doc_paragraphs(column_inner: str, ctx: str) -> list[str]:
                """Collect up to _MAX_TOTAL_PARAS matching paragraphs from doc/*.md.

                A paragraph qualifies if it mentions the column name AND any
                threshold-vocabulary keyword.  This is broader than the prior
                "first match per file" behaviour — the structured extractor
                needs multiple data points to triangulate the threshold.
                """
                hits: list[str] = []
                doc_dir = task_inner.context_dir / "doc"
                if not doc_dir.is_dir():
                    return hits
                range_keywords = {
                    "normal", "abnormal", "range", "threshold", "above",
                    "below", "limit", "reference", "elevated", "low",
                    "high", "upper", "lower", "severe", "borderline",
                }
                col_lc = column_inner.lower()
                col_uc = column_inner.upper()
                # Also try the first few words of the context (e.g. "creatinine level").
                ctx_token = (ctx or "").lower().split(",")[0].strip()[:30]
                variants = [v for v in (col_lc, col_uc, ctx_token) if v]

                for doc_file in sorted(doc_dir.glob("*.md")):
                    if len(hits) >= _MAX_TOTAL_PARAS:
                        break
                    try:
                        text = doc_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    paras_this_file = 0
                    seen_paras: set[int] = set()
                    for variant in variants:
                        if variant.lower() not in text.lower():
                            continue
                        for para_idx, para in enumerate(text.split("\n\n")):
                            if para_idx in seen_paras:
                                continue
                            para_lc = para.lower()
                            if variant.lower() not in para_lc:
                                continue
                            if not (range_keywords & set(re.findall(r"\w+", para_lc))):
                                continue
                            hits.append(f"[{doc_file.name}]: {para.strip()[:_PARA_PREVIEW_CHARS]}")
                            seen_paras.add(para_idx)
                            paras_this_file += 1
                            if (paras_this_file >= _MAX_PARAS_PER_FILE
                                    or len(hits) >= _MAX_TOTAL_PARAS):
                                break
                        if (paras_this_file >= _MAX_PARAS_PER_FILE
                                or len(hits) >= _MAX_TOTAL_PARAS):
                            break
                return hits

            def _domain_knowledge_range(column_inner: str, ctx: str) -> str:
                """Ask the model's training knowledge for a clinical reference range."""
                domain_q = (
                    f"What is the standard clinical reference range for '{column_inner}' "
                    f"({ctx})? Give ONLY the normal range in concise standard units."
                )
                try:
                    return model.complete([
                        ModelMessage(role="system", content=_DOMAIN_KNOWLEDGE_SYSTEM_PROMPT),
                        ModelMessage(role="user", content=domain_q),
                    ], extra_body={"enable_thinking": False}).strip()
                except Exception:
                    return ""

            _NEXT_STEP = (
                "Use 'abnormal_condition' directly in your WHERE clause "
                "(remember to CAST text columns to REAL/INTEGER first). "
                "Verify units: run SELECT MIN(col), MAX(col), AVG(col), and the "
                "count of empty/null cells on the actual data column. "
                "If NO unit conversion makes the normal_range overlap with the "
                "observed data range, the standard unit applies and ALL "
                "non-empty rows are abnormal (medically: tests ordered only on "
                "clinical suspicion)."
            )

            def _wrap_result(structured: dict, source: str, raw: str) -> ToolExecutionResult:
                return ToolExecutionResult(ok=True, content={
                    "column": structured.get("_column", ""),
                    "normal_range": {
                        "lower": structured["lower"],
                        "upper": structured["upper"],
                        "units": structured["units"],
                    },
                    "abnormal_condition": structured["abnormal_condition"],
                    "confidence": structured["confidence"],
                    "source": source,
                    "raw_extracted_text": raw[:600],
                    "next_step": _NEXT_STEP,
                })

            def _lookup_reference_range_impl(_task: "PublicTask", action_input: dict) -> ToolExecutionResult:
                column = str(action_input.get("column", "")).strip()
                question_context = str(action_input.get("question_context", "")).strip()
                if not column:
                    return ToolExecutionResult(ok=False, content={"error": "'column' is required."})

                log_inner = get_logger()

                # --- Phase 1: gather all matching paragraphs across context docs ---
                doc_hits = _gather_doc_paragraphs(column, question_context)
                doc_structured: dict | None = None
                doc_raw = ""
                if doc_hits:
                    doc_raw = "\n\n".join(doc_hits)
                    log_inner.info(
                        "  LOOKUP_RANGE doc-search: %d paragraph(s) for column=%r",
                        len(doc_hits), column,
                    )
                    doc_structured = _extract_structured_range(
                        doc_raw, model, column, question_context,
                    )
                    doc_structured["_column"] = column

                # --- Phase 2: also consult domain knowledge IF docs gave low-confidence
                # or null bounds.  Many tasks have prose with one data point but no
                # explicit range — domain knowledge complements the doc context.
                docs_definitive = (
                    doc_structured is not None
                    and doc_structured.get("confidence") == "high"
                    and (
                        doc_structured.get("lower") is not None
                        or doc_structured.get("upper") is not None
                    )
                )

                if docs_definitive:
                    return _wrap_result(doc_structured, "context_documents", doc_raw)

                # Fall through to domain knowledge.
                log_inner.info(
                    "  LOOKUP_RANGE consulting domain knowledge for column=%r (docs: %s)",
                    column,
                    "low_confidence" if doc_structured else "no_hits",
                )
                domain_raw = _domain_knowledge_range(column, question_context)
                domain_structured: dict | None = None
                if domain_raw:
                    domain_structured = _extract_structured_range(
                        domain_raw, model, column, question_context,
                    )
                    domain_structured["_column"] = column

                # Prefer the answer with explicit numeric bounds + higher confidence.
                def _score(s: dict | None) -> tuple[int, int]:
                    if s is None:
                        return (-1, -1)
                    has_bounds = int(
                        s.get("lower") is not None or s.get("upper") is not None
                    )
                    conf_rank = {"high": 2, "medium": 1, "low": 0}.get(
                        s.get("confidence", "low"), 0
                    )
                    return (has_bounds, conf_rank)

                if _score(domain_structured) > _score(doc_structured):
                    return _wrap_result(domain_structured, "domain_knowledge", domain_raw)
                if doc_structured is not None:
                    return _wrap_result(doc_structured, "context_documents", doc_raw)
                # Both empty: return a low-confidence envelope rather than failing,
                # so the agent can decide to commit (e.g. via the all-abnormal fallback).
                return _wrap_result(
                    {
                        "_column": column,
                        "lower": None,
                        "upper": None,
                        "units": None,
                        "abnormal_condition": (
                            f"No reference range found for '{column}'. "
                            f"Run SELECT MIN({column}), MAX({column}), AVG({column}) "
                            f"and decide whether all non-empty values are likely abnormal."
                        ),
                        "confidence": "low",
                    },
                    "no_source",
                    "",
                )

            return _lookup_reference_range_impl

        handlers["lookup_reference_range"] = _make_lookup_reference_range()

    return ToolRegistry(specs=specs, handlers=handlers)


# ---------------------------------------------------------------------------
# Prior-attempt summarisation helpers (used for resumption after max_steps)
# ---------------------------------------------------------------------------

def _extract_domain_guidance(hint: str) -> str:
    """Extract the DOMAIN ANALYSIS GUIDANCE block from a preflight hint string."""
    if not hint:
        return ""
    idx = hint.find("DOMAIN ANALYSIS GUIDANCE")
    if idx == -1:
        return ""
    # Include up to the next blank-line-separated section (or end of string)
    block = hint[idx:]
    # Trim at the next double-newline that starts a new all-caps section header
    m = re.search(r"\n\n[A-Z][A-Z ]", block)
    if m:
        block = block[: m.start()]
    return block.strip()


def _extract_data_warnings(hint: str) -> str:
    """Extract data-quality warnings from a preflight hint for carry-forward into resumption.

    Pulls any lines that start with a known warning tag. These warnings name specific
    columns and are critical for SQL correctness — a resumed agent that skips them
    will silently produce wrong results (e.g. empty-string FG values treated as abnormal).

    Tags extracted: EMPTY STRING WARNING, NUMERIC TEXT COLUMNS, DATE FORMAT, TIE HINT,
    COUNT SCOPE HINT, BIDIRECTIONAL TABLE, DICT COLUMN RULE, TIME COLUMN RULE.
    """
    if not hint:
        return ""
    _WARNING_TAGS = (
        "EMPTY STRING WARNING",
        "NUMERIC TEXT COLUMNS",
        "DATE FORMAT",
        "TIE HINT",
        "COUNT SCOPE HINT",
        "BIDIRECTIONAL TABLE",
        "DICT COLUMN RULE",
        "TIME COLUMN RULE",
    )
    lines = hint.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(tag) for tag in _WARNING_TAGS):
            result.append(stripped)
    return "\n".join(result)


def summarise_trace_for_resumption(trace_dict: dict, preflight_hint: str = "") -> str:
    """Convert a completed (but unanswered) agent trace into a compact summary.

    The summary is injected as context into the next attempt so the agent can
    skip already-explored paths.  It highlights:
    - Tables confirmed to exist in the schema
    - SQL queries that returned non-empty results (with row counts)
    - Tools that caused repeated errors (e.g. API content-filter rejections)
    - The last few non-empty thoughts (what the agent was trying to do)
    - The blocking issue (why the attempt didn't produce an answer)
    """
    steps = trace_dict.get("steps", [])
    failure_reason = trace_dict.get("failure_reason", "unknown")
    is_content_filter_stop = "content_filter_triggered" in failure_reason

    # Collect confirmed tables (from show_context_schema observations)
    confirmed_tables: list[str] = []
    # Collect productive SQL (queries that returned rows), with sample rows
    productive_sql: list[dict] = []
    # Collect productive Python outputs (non-empty stdout with actual data)
    productive_python: list[dict] = []
    # Collect key factual findings (reference ranges, domain knowledge, search hits)
    key_findings: list[str] = []
    # Track search_doc calls that returned 0 results — agent must not repeat these
    exhausted_searches: list[str] = []
    # Collect the last N non-empty thoughts
    last_thoughts: list[str] = []
    # Detect consecutive API errors (e.g. content-filter rejections)
    failed_actions: list[str] = []  # actions that produced errors
    consecutive_errors = 0
    # Track large tool results for content-filter diagnosis
    # Each entry: (action, file_or_table, result_size_chars)
    large_results: list[tuple[str, str, int]] = []

    for step in steps:
        action = step.get("action", "")
        obs = step.get("observation", {})
        content = obs.get("content", {})
        thought = step.get("thought", "").strip()

        if action == "show_context_schema" and obs.get("ok"):
            tables = list(content.get("tables", {}).keys()) if isinstance(content, dict) else []
            confirmed_tables = tables  # keep the latest (most complete) schema view

        if action == "query_context_tables" and obs.get("ok"):
            sql = step.get("action_input", {}).get("sql", "")
            rows = content.get("rows", []) if isinstance(content, dict) else []
            columns = content.get("columns", []) if isinstance(content, dict) else []
            if rows and sql:
                productive_sql.append({
                    "sql": sql[:200].strip(),
                    "row_count": len(rows),
                    "columns": columns,
                    "sample_rows": rows[:3],
                })

        if action == "execute_python" and obs.get("ok"):
            output = content.get("output", "") if isinstance(content, dict) else ""
            code = step.get("action_input", {}).get("code", "")
            if output and len(output.strip()) > 20 and "[truncated" not in output[:30]:
                productive_python.append({
                    "code_preview": code[:300].strip(),
                    "output_preview": output.strip()[:500],
                })
            elif output and len(output.strip()) > 20:
                # Truncated output still has useful data up to the cut
                productive_python.append({
                    "code_preview": code[:300].strip(),
                    "output_preview": output.strip()[:500],
                })

        if action == "lookup_reference_range" and obs.get("ok"):
            col = step.get("action_input", {}).get("column", "")
            range_str = content.get("range", "") if isinstance(content, dict) else ""
            source = content.get("source", "") if isinstance(content, dict) else ""
            if range_str:
                key_findings.append(
                    f"Reference range for '{col}': {range_str} (source: {source})"
                )

        if action in ("consult_domain_knowledge",) and obs.get("ok"):
            answer_text = content.get("answer", "") if isinstance(content, dict) else ""
            question_text = step.get("action_input", {}).get("question", "")
            if answer_text:
                key_findings.append(
                    f"Domain knowledge — Q: {question_text[:80]} → {answer_text[:200]}"
                )

        if action == "search_doc" and obs.get("ok"):
            total = content.get("total_matches", -1) if isinstance(content, dict) else -1
            if total == 0:
                keyword = step.get("action_input", {}).get("keyword", "")
                file_path = step.get("action_input", {}).get("path", "")
                exhausted_searches.append(f"  keyword='{keyword}' in {file_path}")
            elif total > 0:
                keyword = step.get("action_input", {}).get("keyword", "")
                file_path = step.get("action_input", {}).get("path", "")
                results = content.get("results", []) if isinstance(content, dict) else []
                if results:
                    key_findings.append(
                        f"search_doc '{keyword}' in {file_path}: {total} hits. "
                        f"First match: {str(results[0])[:200]}"
                    )

        # Track large tool result observations (potential content-filter triggers)
        if obs.get("ok") and action not in ("show_context_schema",):
            result_size = len(str(content))
            if result_size > 3000:
                file_ref = (
                    step.get("action_input", {}).get("path")
                    or step.get("action_input", {}).get("sql", "")[:60]
                    or action
                )
                large_results.append((action, str(file_ref), result_size))

        # Track repeated model-level errors (content filter, parse failures, etc.)
        if action == "__error__":
            error_msg = obs.get("error", "")
            consecutive_errors += 1
            if consecutive_errors >= 3:
                failed_actions.append(error_msg[:120])
        else:
            consecutive_errors = 0

        if thought:
            last_thoughts.append(thought)

    stop_reason = (
        "agent stopped early — content moderation triggered by sensitive data in context"
        if is_content_filter_stop
        else "agent exhausted max steps without submitting an answer"
    )
    lines = [f"[Prior attempt summary — {stop_reason}]"]

    if confirmed_tables:
        lines.append(f"Confirmed tables in schema: {', '.join(confirmed_tables[:12])}")

    # Key factual findings first — these are the most actionable carry-forwards
    if key_findings:
        lines.append(
            "KEY FINDINGS from prior attempt — accept these as established facts, "
            "do NOT re-search for them:"
        )
        for f in key_findings[-8:]:
            lines.append(f"  • {f}")

    if productive_python:
        lines.append(
            "Python executions that produced output — use these results directly, "
            "do NOT re-run the same extraction:"
        )
        for p in productive_python[-3:]:
            lines.append(f"  Code: {p['code_preview'][:150]}")
            lines.append(f"  Output: {p['output_preview'][:400]}")

    if productive_sql:
        lines.append("SQL queries that returned data — reuse or refine these, do NOT re-run identical queries:")
        for s in productive_sql[-5:]:
            cols = s["columns"]
            sample = s["sample_rows"]
            lines.append(f"  [{s['row_count']} rows] cols={cols} SQL={s['sql'][:120]}")
            if sample:
                lines.append(f"  Sample rows: {sample[:2]}")

    if exhausted_searches:
        lines.append(
            "Searches ALREADY DONE with ZERO results — do NOT repeat any of these "
            "(searching again will waste steps and produce the same empty result):"
        )
        for s in exhausted_searches[-10:]:
            lines.append(s)

    # Last 3 non-empty thoughts show what the agent was trying to do
    if last_thoughts:
        lines.append("Last agent thoughts (context on what was being attempted):")
        for t in last_thoughts[-3:]:
            lines.append(f"  - {t[:200]}")

    # Warn if repeated API errors occurred — the next attempt must avoid that approach
    if failed_actions:
        representative = failed_actions[-1]
        lines.append(
            f"WARNING: This attempt hit {len(failed_actions)} consecutive model errors "
            f"(e.g. '{representative}'). "
            "The approach that triggered them must NOT be repeated. "
            "If execute_python produced large text output that caused content-filter errors, "
            "use read_doc with a focused query instead — it returns only relevant sections."
        )

    if is_content_filter_stop and large_results:
        last_large = large_results[-1]
        lines.append(
            f"CONTENT FILTER WARNING: This attempt was halted because accumulated context "
            f"triggered API content moderation (sensitive patient/medical data in context). "
            f"The last large result before the stop was: action='{last_large[0]}' on "
            f"'{last_large[1]}' ({last_large[2]} chars). "
            f"In this attempt, DO NOT repeat that broad query on sensitive data. "
            f"Use more targeted queries: search_doc with specific IDs rather than broad keywords, "
            f"or use SQL to aggregate/filter before reading raw data. "
            f"Avoid loading large raw patient records into context."
        )

    # Preserve domain analysis guidance and data-quality warnings from the preflight.
    # These are skipped on resumption attempts (preflight only runs once) but are
    # critical for SQL correctness: empty-string filters, CAST requirements, date
    # formats, and multi-condition logic must all be re-applied by the resumed agent.
    _hint_source = preflight_hint or trace_dict.get("preflight", {}).get("hint", "")
    domain_guidance = _extract_domain_guidance(_hint_source)
    data_warnings = _extract_data_warnings(_hint_source)
    preserved: list[str] = []
    if domain_guidance:
        preserved.append(f"PRESERVED DOMAIN GUIDANCE (carry-forward from preflight):\n{domain_guidance}")
    if data_warnings:
        preserved.append(
            "PRESERVED DATA WARNINGS (carry-forward from preflight — still apply to ALL SQL):\n"
            + data_warnings
        )
    if preserved:
        lines.insert(1, "\n\n".join(preserved) + "\n")

    lines.append(f"Blocking issue: {failure_reason}")
    lines.append(
        "Continue from this context — the schema is already known, "
        "avoid repeating the same SQL queries, and focus on finding "
        "the correct answer rather than re-exploring the schema."
    )
    return "\n".join(lines)


def _format_context_files(input_files: dict) -> str:
    """Format the authoritative list of context files for injection into the task prompt.

    Excludes 'other' category (e.g. .DS_Store) and any blank entries.
    The block is prepended to the task hint so the model sees exactly which
    files exist before it starts reasoning — preventing path hallucination.
    """
    lines = ["[Available context files — this is the complete list]"]
    for category in ("csv", "db", "json", "doc"):
        for path in input_files.get(category, []):
            if path:
                lines.append(f"  [{category}] {path}")
    lines.append("Do not attempt to read any file not listed above.")
    return "\n".join(lines)


def _format_prior_attempts(summaries: list[str]) -> str:
    """Format one or more prior attempt summaries as a block for the task prompt."""
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0]
    parts = []
    for i, s in enumerate(summaries, 1):
        parts.append(f"[Attempt {i} of {len(summaries)}]\n{s}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# DataAgent
# ---------------------------------------------------------------------------

class DataAgent:
    """
    General-purpose data agent that handles tasks of any difficulty.

    All data sources (CSV, JSON, SQLite .db files) are unified in a single
    in-memory SQLite connection so the agent can JOIN across them without any
    tool switching. Tool selection is fixed regardless of which file types are
    present in the context.
    """

    _EXTRACTOR_TIMEOUT_SECONDS = 25  # additive budget beyond base preflight

    def __init__(
        self,
        *,
        model: ModelAdapter,
        max_steps: int = 16,
        preflight_timeout_seconds: int = 30,
        cache_dir: "Path | None" = None,
        live_trace_path: "Path | None" = None,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.preflight_timeout_seconds = preflight_timeout_seconds
        self._cache_dir = cache_dir
        self._live_trace_path = live_trace_path

    def run(self, task: PublicTask, prior_attempts: list[str] | None = None):
        """Run the agent on a task.

        Args:
            task: The task to solve.
            prior_attempts: Optional list of compact summaries of previous attempts
                that exhausted max_steps without submitting an answer.  Each summary
                is injected into the task prompt as additional context so the agent
                can avoid repeating the same dead-ends.
        """
        # Pre-flight: analyse the question against the schema and raw data.
        # The resulting hint is injected into the first user message so the
        # agent starts with candidate tables, columns, literal filters, and
        # join paths already identified.
        #
        # IMPORTANT: pre-flight runs in a background thread capped at
        # preflight_timeout_seconds.  For tasks with very large context files
        # (hundreds of MB) the pre-flight would exhaust the per-task budget
        # before the agent runs a single step.  If the timeout fires, the agent
        # continues without hints — always better than timing out with 0 steps.

        log = get_logger()
        log.info("TASK %s | question=%r", task.task_id, task.question[:120])

        task_hint: str | None = None
        task_analysis: dict = {}

        def _run_preflight() -> None:
            try:
                schema = build_schema_profile(task.context_dir, question=task.question)
                analysis = build_task_analysis(
                    task.question, schema, task.context_dir, model=self.model,
                    cache_dir=self._cache_dir,
                )
                hint = format_task_analysis_hint(analysis)
                task_analysis.update(analysis)
                _preflight_result["hint"] = hint
            except Exception as exc:
                log.warning("PREFLIGHT error: %s", exc)

        from data_agent_baseline.tools.task_analyzer import count_doc_files  # noqa: PLC0415
        n_docs = count_doc_files(task.context_dir)
        n_batches = max(1, (n_docs + 1) // 2)
        # Extractor timeout is additive on top of the base preflight budget.
        # On resumptions (preflight_timeout_seconds == 0) we skip preflight entirely,
        # so do NOT add the extractor bonus — that would run a spurious 25s thread.
        dynamic_preflight_secs = (
            self.preflight_timeout_seconds
            + (n_batches - 1) * 30
            + (self._EXTRACTOR_TIMEOUT_SECONDS if self.preflight_timeout_seconds > 0 else 0)
        )
        _preflight_result: dict = {}
        if dynamic_preflight_secs > 0:
            log.info("PREFLIGHT start (budget=%ds, docs=%d, batches=%d)", dynamic_preflight_secs, n_docs, n_batches)
            _t = threading.Thread(target=_run_preflight, daemon=True)
            _t.start()
            _t.join(timeout=dynamic_preflight_secs)
        else:
            log.info("PREFLIGHT skipped (resumption or zero budget)")
        task_hint = _preflight_result.get("hint")
        if task_hint:
            log.info("PREFLIGHT done: hint=%d chars", len(task_hint))
        elif dynamic_preflight_secs > 0:
            log.warning("PREFLIGHT timed out or produced no hint")
        # Store for the caller (runner.py saves this as preflight.json).
        self._last_preflight: dict = {
            "hint": task_hint or "",
            "task_analysis": dict(task_analysis),
        }

        # On resumption (preflight skipped): restore *_complete tables from the
        # disk cache written during the first attempt.  Without this, the resumed
        # subprocess starts with a fresh in-memory SQLite that is missing any
        # tables injected by the extractor, causing "no such table" errors.
        restored_tables: list[str] = []
        if self.preflight_timeout_seconds == 0 and self._cache_dir is not None:
            try:
                from data_agent_baseline.tools.context_sqlite import load_context_to_sqlite  # noqa: PLC0415
                from data_agent_baseline.tools.coverage_extractor import load_cached_extractions  # noqa: PLC0415
                conn = load_context_to_sqlite(task.context_dir)
                restored_tables = load_cached_extractions(self._cache_dir, conn)
                if restored_tables:
                    log.info("RESUMPTION: restored extracted tables: %s", restored_tables)
            except Exception as exc:
                log.warning("RESUMPTION: failed to restore cached extractions: %s", exc)

        # Prepend the authoritative file listing so the model knows exactly
        # which paths exist before it starts reasoning.  This prevents it from
        # hallucinating file paths (e.g. assuming doc/Laboratory.md when only
        # csv/Laboratory.csv is present) which causes infinite error loops.
        input_files = detect_input_files(task)
        file_listing = _format_context_files(input_files)
        if restored_tables:
            file_listing += (
                "\n[Restored pre-extracted tables: "
                + ", ".join(f"'{t}'" for t in restored_tables)
                + " — these are available in the SQL context. Use them as directed by the prior attempt summary.]"
            )
        task_hint = (file_listing + "\n\n" + task_hint) if task_hint else file_listing

        # Extract exhausted searches from prior summaries and hard-block them.
        # This prevents the model from repeating search_doc calls that already
        # returned 0 results — the tool itself will reject those queries.
        blocked_searches = _parse_exhausted_searches(prior_attempts or [])
        if blocked_searches:
            log.info("BLOCKED SEARCHES from prior attempts: %d", len(blocked_searches))

        tools = create_data_agent_tool_registry(
            model=self.model, task=task, blocked_searches=blocked_searches
        )

        # Inject prior attempt summaries BEFORE the task question so the model
        # reads them with high attention before anchoring to its default strategy.
        # (Appending after the question causes the model to ignore them.)
        if prior_attempts:
            prior_block = _format_prior_attempts(prior_attempts)
            task_hint = (prior_block + "\n\n" + task_hint) if task_hint else prior_block

        agent = ReActAgent(
            model=self.model,
            tools=tools,
            config=ReActAgentConfig(max_steps=self.max_steps),
            system_prompt=DATA_AGENT_SYSTEM_PROMPT,
            task_hint=task_hint,
            task_analysis=task_analysis,
            live_trace_path=self._live_trace_path,
        )
        return agent.run(task)
