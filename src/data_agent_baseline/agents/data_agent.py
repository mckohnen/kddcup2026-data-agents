from __future__ import annotations

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
  - Value encodings: filters like label='+', status='Y', type='A' must match exactly.
  - Example queries: replicate their logic, not just their structure.
  For large doc/ files, call read_doc with the task question as context — relevant sections
  will be surfaced automatically. For exhaustive extraction from large prose documents
  (e.g. listing every entity with a certain label), use execute_python instead.

Step 3 — Inspect schema:
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
  For prose document extraction (e.g. extracting entity labels from a Markdown file),
  use execute_python:
    - Open the file by path under the context directory.
    - Process it paragraph by paragraph (not sentence by sentence) to handle cases where
      an entity ID and its classification appear in different sentences of the same paragraph.
    - Print structured output (e.g. JSON list) to stdout; keep the script focused on a
      single task. Do NOT mix SQLite connections into the same Python step as file reading.

Step 5 — Validate before submitting:
  Before calling answer, verify:
  1. Row count is non-zero and no key columns are entirely NULL.
  2. Column count matches the question: "how many" / "what is X" → 1 column;
     "list X and Y" → 2 columns. Do NOT add extra columns (counts, IDs, labels) unless
     the question explicitly asks for them.
  3. Column names come directly from the source data. Never invent aliases or rename columns.
  4. Result shape matches the question's intent:
     - "how many" → 1 row, 1 column (a single count).
     - "list / tally / enumerate [values or entities]" → one row per DISTINCT value; use
       SELECT DISTINCT or GROUP BY to deduplicate. Output ONLY the values themselves —
       never add a count/frequency column alongside them unless the question explicitly
       uses words like "count", "how many", "how often", "number of", or "frequency".
     - "list [entities]" → one row per unique entity, not one row per relationship record.

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
    profile = build_schema_profile(task.context_dir)
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


def _execute_python(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
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

    def __init__(self, *, model: ModelAdapter, max_steps: int = 16) -> None:
        self.model = model
        self.max_steps = max_steps

    def run(self, task: PublicTask):
        tools = create_data_agent_tool_registry()

        # Pre-flight: analyse the question against the schema and raw data.
        # The resulting hint is injected into the first user message so the
        # agent starts with candidate tables, columns, literal filters, and
        # join paths already identified.  The structured analysis is also
        # passed to the critic for column-level disambiguation.
        task_hint: str | None = None
        task_analysis: dict = {}
        try:
            schema = build_schema_profile(task.context_dir)
            task_analysis = build_task_analysis(task.question, schema, task.context_dir)
            task_hint = format_task_analysis_hint(task_analysis)
        except Exception:
            pass  # Never let analysis failure block the agent run

        agent = ReActAgent(
            model=self.model,
            tools=tools,
            config=ReActAgentConfig(max_steps=self.max_steps),
            system_prompt=DATA_AGENT_SYSTEM_PROMPT,
            task_hint=task_hint,
            task_analysis=task_analysis,
        )
        return agent.run(task)
