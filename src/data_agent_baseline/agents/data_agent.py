from __future__ import annotations

import threading

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
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
  - CSV and JSON files → accessible as plain table names (e.g. SELECT * FROM atom)
  - SQLite .db files   → accessible as <db_stem>.<table> (e.g. SELECT * FROM hero_power.hero_power)
  You can JOIN across all sources in a single SQL query.
  CRITICAL: Do NOT run "SELECT name FROM sqlite_master WHERE type='table'" or any query
  against sqlite_master / sqlite_schema to discover tables. That system table only shows
  the main (CSV/JSON) schema and will NOT list .db tables. Always trust show_context_schema
  as the authoritative list of all available tables.

Step 4 — Query and analyse:
  Use query_context_tables with SQL for all data retrieval.
  Important SQL rules:
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
  - Standard domain thresholds: first search ALL context files (knowledge.md, every doc/,
    every CSV header, every table) for explicitly defined thresholds. Only if none are
    found anywhere in the context, fall back to your training knowledge of well-known
    reference values (e.g. clinical normal ranges, physical constants, industry standards).
    When using training knowledge, state in your thought exactly which values you assumed
    and confirm no definition was found in the context.
    IMPORTANT: if you have already made 2 or more doc/knowledge searches for a threshold
    and found nothing, treat the search as exhausted — do NOT repeat the same search.
    Immediately apply standard domain knowledge values and proceed to your SQL query.

  For exhaustive entity extraction from prose documents (e.g. listing every patient whose
  label is X, collecting all values of a field across a large Markdown file), use
  execute_python ONLY after read_doc has failed to surface what you need:
    - Open the file by path under the context directory.
    - Process it paragraph by paragraph (not sentence by sentence) to handle cases where
      an entity ID and its classification appear in different sentences of the same paragraph.
    - Print structured output (e.g. JSON list) to stdout; keep the script focused on a
      single task. Do NOT mix SQLite connections into the same Python step as file reading.
    - Keep printed output concise — print only the final structured result, not every
      intermediate line you inspect. Large raw text dumps will be truncated.

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
    content = read_doc_preview(
        task,
        path,
        max_chars=max_chars,
        query=task.question,
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


def _execute_python(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    # Truncate large outputs before they enter the conversation history.
    # Without this, a single execute_python call that returns thousands of
    # characters of raw document text (e.g. medical lab reports) fills the
    # context window and triggers content-filter rejections on every
    # subsequent model call, crashing the remaining agent steps.
    raw_output = content.get("output", "") or ""
    if len(raw_output) > _EXECUTE_PYTHON_MAX_OUTPUT_CHARS:
        content = dict(content)  # don't mutate the original
        content["output"] = (
            raw_output[:_EXECUTE_PYTHON_MAX_OUTPUT_CHARS]
            + f"\n[output truncated — {len(raw_output)} chars total, showing first {_EXECUTE_PYTHON_MAX_OUTPUT_CHARS}]"
        )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


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
    "read_doc": ToolSpec(
        name="read_doc",
        description=(
            "Read a document from context. For knowledge.md, returns a table of contents "
            "on the first call; pass 'section' to read a specific ## header block. "
            "For large doc/ files, returns the most query-relevant chunks automatically. "
            "Use execute_python for exhaustive extraction of all entities in large prose files."
        ),
        input_schema={"path": "knowledge.md", "section": "## 2. Core Entities & Fields"},
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
    "list_context": _list_context,
    "read_doc": _read_doc,
    "read_knowledge_section": _read_knowledge_section,
    "show_context_schema": _show_context_schema,
    "query_context_tables": _query_context_tables,
    "execute_python": _execute_python,
    "answer": _answer,
}


def create_data_agent_tool_registry(input_files: dict | None = None) -> ToolRegistry:
    """Return the unified tool registry.

    All tools are always active — the unified SQLite handles CSV, JSON, and .db
    sources transparently, so no per-file-type routing is needed.
    The ``input_files`` parameter is kept for backwards compatibility but ignored.
    """
    del input_files  # no longer used
    return ToolRegistry(specs=_TOOL_SPECS, handlers=_TOOL_HANDLERS)


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

    # Collect confirmed tables (from show_context_schema observations)
    confirmed_tables: list[str] = []
    # Collect productive SQL (queries that returned rows)
    productive_sql: list[str] = []
    # Collect the last N non-empty thoughts
    last_thoughts: list[str] = []
    # Detect consecutive API errors (e.g. content-filter rejections)
    failed_actions: list[str] = []  # actions that produced errors
    consecutive_errors = 0

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

    lines = ["[Prior attempt summary — agent exhausted max steps without submitting an answer]"]

    if confirmed_tables:
        lines.append(f"Confirmed tables in schema: {', '.join(confirmed_tables[:12])}")

    if productive_sql:
        lines.append("SQL queries that returned data (reuse these as a starting point):")
        for s in productive_sql[-5:]:  # last 5 productive queries
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

    lines.append(f"Blocking issue: {failure_reason}")
    lines.append(
        "Continue from this context — the schema is already known, "
        "avoid repeating the same SQL queries, and focus on finding "
        "the correct answer rather than re-exploring the schema."
    )
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

    def __init__(
        self,
        *,
        model: ModelAdapter,
        max_steps: int = 16,
        preflight_timeout_seconds: int = 30,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.preflight_timeout_seconds = preflight_timeout_seconds

    def run(self, task: PublicTask, prior_attempts: list[str] | None = None):
        """Run the agent on a task.

        Args:
            task: The task to solve.
            prior_attempts: Optional list of compact summaries of previous attempts
                that exhausted max_steps without submitting an answer.  Each summary
                is injected into the task prompt as additional context so the agent
                can avoid repeating the same dead-ends.
        """
        tools = create_data_agent_tool_registry()

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

        task_hint: str | None = None
        task_analysis: dict = {}

        def _run_preflight() -> None:
            try:
                schema = build_schema_profile(task.context_dir, question=task.question)
                analysis = build_task_analysis(task.question, schema, task.context_dir)
                hint = format_task_analysis_hint(analysis)
                task_analysis.update(analysis)
                _preflight_result["hint"] = hint
            except Exception:
                pass  # Never let analysis failure block the agent run

        _preflight_result: dict = {}
        _t = threading.Thread(target=_run_preflight, daemon=True)
        _t.start()
        _t.join(timeout=self.preflight_timeout_seconds)
        task_hint = _preflight_result.get("hint")
        # Store for the caller (runner.py saves this as preflight.json).
        self._last_preflight: dict = {
            "hint": task_hint or "",
            "task_analysis": dict(task_analysis),
        }

        # Prepend prior attempt summaries to the task hint so the agent can
        # skip already-explored paths and focus on what's left to try.
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
