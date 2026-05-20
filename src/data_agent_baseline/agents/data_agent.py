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
You are a data agent solving a data analysis task. Follow these steps carefully:

Step 1 — Explore the context:
  Call list_context to see all available files and their sizes.

Step 2 — Read documentation:
  Call read_doc for knowledge.md first. For knowledge.md, you will receive a table of
  contents; then call read_doc again with 'section' set to the relevant header(s) to read
  only what you need (e.g. '## 2. Core Entities & Fields').
  Pay close attention to:
  - Column semantics: a question may use a natural-language term (e.g. "ranked", "active",
    "rate") — find the EXACT column that matches it. Multiple similar-sounding columns may
    exist (e.g. "positionOrder" vs "rank", "points" vs "score") — read ALL relevant sections
    in knowledge.md before choosing. Pick the column the documentation explicitly links to
    the question's concept, not just the one with the most intuitive name.
    When two columns sound similar, the documentation will define which one maps to the
    question's intent — never guess; always look it up.
  - Value encodings: filters like label='+', status='Y', type='A' must match exactly.
    Read the documentation to find the exact string value used in the data.
  - Categorical value matching: when the question uses a phrase that corresponds to
    a column value, filter for that exact string (e.g. 'confirmed orders' →
    `orders = 'confirmed'`, 'active members' → `members = 'active'`, 'valid type' → `type = 'valid'`). Do not treat
    the phrase as a description of having any non-null value — match the literal
    string you observed in the schema or documentation.
  - Example queries: replicate their logic, not just their structure.
  For large doc/ files, call read_doc with the task question as context — relevant sections
  will be surfaced automatically via relevance ranking.
  Do NOT use execute_python to search for keywords or values inside prose documents.
  Always try read_doc first; only escalate to execute_python for exhaustive entity
  extraction (e.g. listing ALL entities of a type from a document).

Step 3 — Inspect schema and consider ALL data sources:
  Treat every file in the context as a potential data source. Never conclude "no data
  exists" after checking only one table. Cross-check all CSV, JSON, and database files —
  each may contain different records or reference data that is needed to answer the
  question. If one data source returns no results, look in the others before giving up.
  Call show_context_schema to see all tables, columns, row counts, type hints, and inferred
  relationships. ALL data sources are unified in one SQLite connection:

  Pre-extracted lookup tables: the preflight phase has already detected coverage
  gaps and extracted missing attributes from prose documents into *_complete tables.
  Check the preflight hint for "EXTRACTED TABLE READY" entries — if one exists for a
  lookup table you need (e.g. patient_sex_complete instead of patient_sex), USE IT.
  These tables are already in the SQL context with full ID coverage.
  Do NOT attempt to re-extract from prose or call lookup_ids_in_doc — the data is ready.
  - CSV and JSON files → accessible as plain table names (e.g. SELECT * FROM atom)
  - SQLite .db files   → accessible as <db_stem>.<table> (e.g. SELECT * FROM hero_power.hero_power)
  - Large prose doc/*.md files → indexed as <filename_stem>_paragraphs(paragraph_idx INTEGER, content TEXT)
    Use SQL LIKE to search them: SELECT content FROM <stem>_paragraphs WHERE content LIKE '%term%'
  You can JOIN across all sources in a single SQL query.

  ALL-DOCS MODE: If show_context_schema reveals ONLY *_paragraphs tables (no CSV/JSON/DB
  tables), ALL structured data is embedded in prose documents. In this mode:
  1. Do NOT rely on search_doc for data extraction — it returns snippets, not complete records.
  2. Use execute_python to read the full document files and extract structured data with regex.
     Process paragraph by paragraph. Extract patient IDs + values into a Python dict/list.
  3. Once you have extracted the data into Python variables, compute the answer in Python.
  4. Do NOT spend more than 2 steps on keyword searches before switching to execute_python.
  Example pattern for prose data extraction:
    import re
    with open('doc/Laboratory.md', 'r') as f: content = f.read()
    # split by paragraph, extract patient_id + numeric value per paragraph
    records = []
    for para in content.split('\n\n'):
        pid = re.search(r'patient (\d+)', para, re.IGNORECASE)
        val = re.search(r'creatinine.*?(\d+\.\d+) mg/dL', para, re.IGNORECASE)
        if pid and val: records.append({'id': pid.group(1), 'cre': float(val.group(1))})
  CRITICAL: Do NOT run "SELECT name FROM sqlite_master WHERE type='table'" or any query
  against sqlite_master / sqlite_schema to discover tables. That system table only shows
  the main (CSV/JSON) schema and will NOT list .db tables. Always trust show_context_schema
  as the authoritative list of all available tables.

Step 4 — Query and analyse:
  Use query_context_tables with SQL for all data retrieval.

  Before writing any query involving age, duration, or 'current' date:
    Call get_current_datetime to get today's date. Never assume or hardcode the current year.

  For locating a specific term, threshold, or value in a large prose document:
    Call search_doc with the exact keyword before reaching for execute_python.
    search_doc returns all paragraphs containing the keyword — no code needed.
    COMMIT-ON-FIRST-FIND: once an observation clearly states the threshold or range you
    need (e.g. "creatinine upper limit of normal is 1.2 mg/dL"), STOP searching. Do NOT
    call search_doc or read_doc again for the same fact. Proceed directly to execute_python
    or query_context_tables using that value. Repeating the same search wastes steps.

  Important SQL rules:
  - Multi-condition filtering on longitudinal data:
    Before writing any WHERE clause that combines two or more conditions on a
    time-series table, read the
    DOMAIN ANALYSIS GUIDANCE block in the preflight hint. That block tells you whether
    the question calls for same-row (concurrent), any-row (independent), or temporal-
    proximity logic — the right approach depends on domain and question phrasing.
    Never combine independent conditions into a single WHERE clause without considering
    whether the measurements must co-occur on the same date.
  - CSV columns are stored as TEXT — ALWAYS use CAST for numeric comparisons and arithmetic.
    WRONG: WHERE height_cm > 200          (text comparison: '61' > '200' is TRUE!)
    RIGHT:  WHERE CAST(height_cm AS INTEGER) > 200
    This applies to every numeric filter or sort on CSV-sourced columns.
  - JSON columns preserve native types (integers stay integers — no CAST needed).
  - When ordering by a numeric ID suffix (e.g. atom_id like 'TR001_12'), always sort
    numerically: ORDER BY CAST(SUBSTR(col, INSTR(col, '_') + 1) AS INTEGER)
  - Never add LIMIT to the final answer query — return all matching rows.
  - Trust your SQL: once your WHERE clause correctly encodes the question's condition,
    trust ALL rows it returns. Never discard or manually filter rows from the result
    based on subjective reasoning (e.g. "closer to the target value"). The query
    result IS the answer — submit every row it produces.
  - When a question asks for the 'type of X', GROUP BY the short categorical `type`
    column on the entity table (e.g. `event.type`, `category`), not by a description
    or name field. Type columns hold values like 'Meeting', 'Election', 'Purchase'.
  - Do NOT round or truncate numeric results. Never use ROUND(), FORMAT(), or Python's
    round(). Return the exact value computed by SQL or Python — the evaluation system
    handles precision normalization.
    WRONG: ROUND(SUM(a) / SUM(b), 2)
    RIGHT:  CAST(SUM(a) AS REAL) / SUM(b)
  - "Average monthly X": compute AVG(X) over monthly-granularity rows, NOT SUM(X) / 12.
    If each row represents one month: SELECT AVG(value_col).
    If each row is a yearly total: SELECT AVG(yearly_col) / 12.
  - Empty strings in CSV columns: CAST('' AS REAL) = 0 in SQLite, which silently distorts
    AVG and SUM. Always filter empty strings from numeric aggregations:
    WRONG: AVG(CAST(col AS REAL))                    -- '' treated as 0
    RIGHT:  AVG(CASE WHEN col != '' THEN CAST(col AS REAL) END)
    Whether to also exclude 0-valued rows depends on domain context — do not assume 0
    means "unknown" unless the question or documentation says so.
  - Time strings (e.g. "1:23.456", "0:47.832") are stored as TEXT. TEXT ORDER BY is
    alphabetical, not numeric — "1:09" > "1:8" as text!
    Two mandatory rules for any time-column query:
    1. ALWAYS filter out rows where the time is empty or null FIRST:
       WHERE time_col != '' AND time_col IS NOT NULL
       (missing times are stored as '' and sort BEFORE all valid times alphabetically,
        so without this filter an empty string would be "returned as the fastest time")
    2. THEN convert to seconds for correct numeric ordering:
       ORDER BY (CAST(SUBSTR(col, 1, INSTR(col,':')-1) AS INTEGER) * 60
                 + CAST(SUBSTR(col, INSTR(col,':')+1) AS REAL)) ASC
  - If a table is referenced in the documentation but missing from the SQL schema
    (you get "no such table" error), use execute_python to load the relevant .md or
    .csv file into a pandas DataFrame and run the analysis there. Do not give up after
    a "no such table" error — the data may live in a doc file.
  - Standard domain thresholds: use lookup_reference_range to find any lab or clinical
    reference range in one step. It searches all context docs first, then falls back to
    domain knowledge automatically — you do NOT need to call search_doc multiple times.
    After receiving a range, ALWAYS verify units by running SELECT MIN(col), MAX(col), AVG(col)
    on the actual data column first. The threshold may be in different units than the dataset
    (e.g. cells/µL vs ×10⁹/L). Scale the threshold to match what you observe in the data.

    DISTRIBUTION-BASED ABNORMALITY RULE: After checking MIN/MAX/AVG, if you find that
    ALL non-empty values in the column fall entirely outside the known reference range on
    the same side (all below it, or all above it), do NOT keep searching for the "correct"
    threshold or unit conversion. Instead, treat every non-empty value as abnormal and
    filter with WHERE col IS NOT NULL AND col != ''. Do not waste further steps on this.

    COMMIT RULE: If after 10 steps total you still do not have an answer, stop searching
    and commit to your best estimate. An imperfect answer always scores better than no answer
    (no answer = 0 score). Use whatever evidence you have: domain knowledge thresholds,
    partial data ranges, or the most defensible assumption. Submit an answer even if uncertain.
    If you know the upper limit but not the lower limit of a normal range, use just the upper
    limit (e.g. creatinine > 1.2 mg/dL = abnormal) — partial criteria beat no answer.

  For exhaustive entity extraction from prose documents (ALL-DOCS MODE or when the
  preflight extracted table is missing), use execute_python:
    - Open the file by path under the context directory.
    - Process it paragraph by paragraph (not sentence by sentence) to handle cases where
      an entity ID and its classification appear in different sentences of the same paragraph.
    - Print structured output (e.g. JSON list) to stdout; keep the script focused on a
      single task. Do NOT mix SQLite connections into the same Python step as file reading.
    - Keep printed output concise — print only the final structured result, not every
      intermediate line you inspect. Large raw text dumps will be truncated.
    - Never use f-strings with {variable} expressions — they break JSON encoding.
      Use string concatenation instead:
        WRONG: print(f"Found {len(results)} rows")
        RIGHT:  print("Found " + str(len(results)) + " rows")

Step 5 — Validate before submitting:
  Before calling answer, verify:
  1. Row count is non-zero and no key columns are entirely NULL. (Zero rows IS a valid
     answer if no data genuinely matches the filter — submit it.)
  2. Column count matches the question: "how many" / "what is X" → 1 column;
     "list X and Y" → 2 columns. Do NOT add extra columns (counts, IDs, labels) unless
     the question explicitly asks for them.
     "List all [X]" or "List all [X] that [condition]" → return ONLY the primary identifier
     column (the natural key or ID column) that identifies each X.
     Do NOT add supplementary columns (amounts, dates, counts, descriptions, status) unless
     the question explicitly asks for those attributes too.
     EXCEPTION: "what is the [content]" questions (e.g. "what is the comment/message/text/
     description/title") ask for the content itself, not an ID. Return the content column.
  3. Column names come directly from the source data. Never invent aliases or rename columns.
  4. Result shape matches the question's intent:
     - "how many" → 1 row, 1 column (a single count).
     - "list / tally / enumerate [values or entities]" → one row per DISTINCT value; use
       SELECT DISTINCT or GROUP BY to deduplicate. Output ONLY the values themselves —
       never add a count/frequency column alongside them unless the question explicitly
       uses words like "count", "how many", "how often", "number of", or "frequency".
     - "list [entities]" → one row per unique entity, not one row per relationship record.
     - "which/what X has the highest/lowest/maximum/minimum Y" → use WHERE Y = (SELECT
       MAX/MIN(Y) ...) to capture ALL tied rows, not ORDER BY ... LIMIT 1.

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
                ])
                log.info("  DOMAIN_KNOWLEDGE answer=%r", answer[:120])
                return ToolExecutionResult(
                    ok=True,
                    content={
                        "answer": answer.strip(),
                        "source": "model_training_knowledge",
                        "next_step": (
                            "IMPORTANT: run SELECT MIN(col), MAX(col), AVG(col) on the actual "
                            "data column to verify units match. Then apply the "
                            "DISTRIBUTION-BASED ABNORMALITY RULE from your instructions: if ALL "
                            "non-empty values fall entirely outside the reference range on one "
                            "side, treat every non-empty value as abnormal and filter with "
                            "WHERE col IS NOT NULL AND col != '' — do not keep searching."
                        ),
                    },
                )
            except Exception as exc:
                return ToolExecutionResult(ok=False, content={"error": str(exc)})

        specs["consult_domain_knowledge"] = ToolSpec(
            name="consult_domain_knowledge",
            description=(
                "Query the model's training knowledge about a factual domain question — "
                "e.g. standard lab reference ranges, clinical thresholds, or well-known "
                "constants that are not explicitly defined in the context files. "
                "Use this ONLY after 5 or more searches of knowledge.md and doc/ files "
                "have not yielded a definitive threshold or value. Do NOT use it as a "
                "first resort — always check the context first. "
                "CRITICAL: after receiving the answer, immediately verify units by running "
                "SELECT MIN(col), MAX(col), AVG(col) on the relevant data column. "
                "General knowledge thresholds may be in different units than the dataset "
                "(e.g. cells/µL vs ×10⁹/L, g/dL vs g/L). Scale to match the data before filtering. "
                "The answer reflects general knowledge and may vary by population; treat it as "
                "a starting point, not ground truth."
            ),
            input_schema={"question": "What is the normal range for creatinine in mg/dL?"},
        )
        handlers["consult_domain_knowledge"] = _consult_domain_knowledge

        # lookup_reference_range: searches doc files (max 3 per file) then
        # falls back to domain knowledge — saves the agent many search_doc steps.
        def _make_lookup_reference_range(task_inner=task):
            def _lookup_reference_range_impl(_task: "PublicTask", action_input: dict) -> ToolExecutionResult:
                column = str(action_input.get("column", "")).strip()
                question_context = str(action_input.get("question_context", "")).strip()
                if not column:
                    return ToolExecutionResult(ok=False, content={"error": "'column' is required."})

                log_inner = get_logger()
                col_lower = column.lower()

                # --- Phase 1: search doc files (max 3 keyword variants per file) ---
                search_hits: list[str] = []
                doc_dir = task_inner.context_dir / "doc"
                if doc_dir.is_dir():
                    range_keywords = {"normal", "range", "threshold", "above", "below", "limit", "reference"}
                    for doc_file in sorted(doc_dir.glob("*.md")):
                        text = doc_file.read_text(encoding="utf-8", errors="replace")
                        text_lower = text.lower()
                        variants = [col_lower, column.upper(), question_context.lower()[:25]]
                        found_in_file = False
                        for variant in variants[:3]:
                            if variant.lower() in text_lower:
                                for para in text.split("\n\n"):
                                    if variant.lower() in para.lower() and range_keywords & set(para.lower().split()):
                                        search_hits.append(f"[{doc_file.name}]: {para.strip()[:500]}")
                                        found_in_file = True
                                        break
                            if found_in_file:
                                break

                if search_hits:
                    log_inner.info("  LOOKUP_RANGE found in docs for column=%r", column)
                    prompt = (
                        f"From the text below, extract the standard reference range for '{column}' "
                        f"(context: {question_context}). "
                        f"Return ONLY the range — e.g. '4.5–11.0 ×10⁹/L' or 'below 200 mg/dL'. "
                        f"If multiple ranges exist, list all briefly.\n\n"
                        + "\n\n".join(search_hits)
                    )
                    try:
                        range_str = model.complete([ModelMessage(role="user", content=prompt)]).strip()
                        return ToolExecutionResult(ok=True, content={
                            "column": column,
                            "range": range_str,
                            "source": "context_documents",
                            "next_step": (
                                "Verify units: run SELECT MIN(col), MAX(col), AVG(col) on the actual "
                                "data column to confirm the range matches the dataset's scale."
                            ),
                        })
                    except Exception:
                        pass  # fall through to domain knowledge

                # --- Phase 2: domain knowledge fallback ---
                log_inner.info("  LOOKUP_RANGE fallback to domain knowledge for column=%r", column)
                domain_q = (
                    f"What is the standard clinical reference range for '{column}' "
                    f"({question_context})? Give ONLY the normal range in concise standard units."
                )
                try:
                    range_str = model.complete([
                        ModelMessage(role="system", content=_DOMAIN_KNOWLEDGE_SYSTEM_PROMPT),
                        ModelMessage(role="user", content=domain_q),
                    ]).strip()
                    return ToolExecutionResult(ok=True, content={
                        "column": column,
                        "range": range_str,
                        "source": "domain_knowledge",
                        "next_step": (
                            "IMPORTANT: verify units by running SELECT MIN(col), MAX(col), AVG(col) "
                            "on the actual data column before applying this threshold. "
                            "Domain knowledge values may be in different units than the dataset."
                        ),
                    })
                except Exception as exc:
                    return ToolExecutionResult(ok=False, content={"error": str(exc)})
            return _lookup_reference_range_impl

        handlers["lookup_reference_range"] = _make_lookup_reference_range()

    return ToolRegistry(specs=specs, handlers=handlers)


# ---------------------------------------------------------------------------
# Prior-attempt summarisation helpers (used for resumption after max_steps)
# ---------------------------------------------------------------------------

def summarise_trace_for_resumption(trace_dict: dict) -> str:
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
    # Collect productive SQL (queries that returned rows)
    productive_sql: list[str] = []
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
            if rows and sql:
                row_count = len(rows)
                productive_sql.append(f"  [{row_count} rows] {sql[:120].strip()}")

        if action == "search_doc" and obs.get("ok"):
            total = content.get("total_matches", -1) if isinstance(content, dict) else -1
            if total == 0:
                keyword = step.get("action_input", {}).get("keyword", "")
                file_path = step.get("action_input", {}).get("path", "")
                exhausted_searches.append(f"  keyword='{keyword}' in {file_path}")

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

    if productive_sql:
        lines.append("SQL queries that returned data (reuse these as a starting point):")
        for s in productive_sql[-5:]:  # last 5 productive queries
            lines.append(s)

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
            lines.append(f"  - {t[:150]}")

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
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.preflight_timeout_seconds = preflight_timeout_seconds
        self._cache_dir = cache_dir

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
        dynamic_preflight_secs = (
            self.preflight_timeout_seconds
            + (n_batches - 1) * 30
            + self._EXTRACTOR_TIMEOUT_SECONDS
        )
        log.info("PREFLIGHT start (budget=%ds, docs=%d, batches=%d)", dynamic_preflight_secs, n_docs, n_batches)
        _preflight_result: dict = {}
        _t = threading.Thread(target=_run_preflight, daemon=True)
        _t.start()
        _t.join(timeout=dynamic_preflight_secs)
        task_hint = _preflight_result.get("hint")
        if task_hint:
            log.info("PREFLIGHT done: hint=%d chars", len(task_hint))
        else:
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
        )
        return agent.run(task)
