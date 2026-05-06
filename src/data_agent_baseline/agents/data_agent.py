from __future__ import annotations

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.context_sqlite import get_context_schema, run_sql_on_context
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.input_detector import detect_input_files
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.registry import (
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
)
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

DATA_AGENT_SYSTEM_PROMPT = """
You are a data agent solving a data analysis task. Follow these steps:

Step 1 — Explore the context:
  Use list_context to see all available files. Note the file types present.

Step 2 — Read documentation:
  If a knowledge.md, README.md, or similar document exists, read it with read_doc first.
  It may define column semantics, field encodings, and contain example queries.

Step 3 — Inspect data schemas:
  - For CSV and JSON files: use show_context_schema to load them into in-memory SQLite
    and inspect table names, columns, row counts, and sample values.
  - For .db or .sqlite files: use inspect_sqlite_schema with the relative file path,
    then sample rows with execute_context_sql to understand the data.

Step 4 — Query and analyze:
  - For CSV/JSON data (loaded into in-memory SQLite): use query_context_tables with SQL.
    JSON-sourced columns preserve native types (no CAST needed for integers).
    CSV-sourced columns are TEXT — use CAST(col AS INTEGER/REAL) for numeric comparisons.
  - For .db/.sqlite files: use execute_context_sql with the relative file path and SQL.
  - For complex multi-step analysis: use execute_python.
    The context directory is the working directory. Standard libraries and pandas are available.

Step 5 — Validate:
  Verify the result has non-zero rows and no key columns are entirely NULL.
  Re-query with corrected logic if the result looks wrong.

Step 6 — Submit:
  Call answer with exactly the columns requested in the question.
  Do not include columns that were not explicitly asked for.
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
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _read_csv(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _show_context_schema(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    del action_input
    tables = get_context_schema(task.context_dir)
    return ToolExecutionResult(ok=True, content={"tables": tables})


def _query_context_tables(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    result = run_sql_on_context(task.context_dir, sql, limit=limit)
    return ToolExecutionResult(ok=True, content=result)


def _inspect_sqlite_schema(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


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
# Context-aware tool registry
# ---------------------------------------------------------------------------

_ALL_SPECS: dict[str, ToolSpec] = {
    "answer": ToolSpec(
        name="answer",
        description="Submit the final answer table. This is the only valid terminating action.",
        input_schema={"columns": ["col"], "rows": [["value"]]},
    ),
    "list_context": ToolSpec(
        name="list_context",
        description="List all files and directories under the task context.",
        input_schema={"max_depth": 4},
    ),
    "read_doc": ToolSpec(
        name="read_doc",
        description="Read a text or markdown file from context (e.g. knowledge.md).",
        input_schema={"path": "knowledge.md", "max_chars": 8000},
    ),
    "read_csv": ToolSpec(
        name="read_csv",
        description="Preview a CSV file from context (returns header + sample rows).",
        input_schema={"path": "relative/path.csv", "max_rows": 20},
    ),
    "read_json": ToolSpec(
        name="read_json",
        description="Preview a JSON file from context.",
        input_schema={"path": "relative/path.json", "max_chars": 4000},
    ),
    "show_context_schema": ToolSpec(
        name="show_context_schema",
        description=(
            "Load all CSV and JSON context files into in-memory SQLite and return "
            "table names, columns, row counts, and sample rows."
        ),
        input_schema={},
    ),
    "query_context_tables": ToolSpec(
        name="query_context_tables",
        description=(
            "Run SQL on in-memory SQLite containing all CSV/JSON context files. "
            "JSON columns keep native types; CSV columns are TEXT (CAST for numerics)."
        ),
        input_schema={"sql": "SELECT ...", "limit": 200},
    ),
    "inspect_sqlite_schema": ToolSpec(
        name="inspect_sqlite_schema",
        description="Inspect tables and column definitions in a .db or .sqlite file in context.",
        input_schema={"path": "relative/path.sqlite"},
    ),
    "execute_context_sql": ToolSpec(
        name="execute_context_sql",
        description="Run a read-only SQL query against a .db or .sqlite file in context.",
        input_schema={"path": "relative/path.sqlite", "sql": "SELECT ...", "limit": 200},
    ),
    "execute_python": ToolSpec(
        name="execute_python",
        description=(
            f"Execute Python code with the context directory as working directory. "
            f"Standard libraries and pandas are available. Returns stdout as output. "
            f"Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS}s."
        ),
        input_schema={"code": "import os\nprint(sorted(os.listdir('.')))"},
    ),
}

_ALL_HANDLERS = {
    "answer": _answer,
    "list_context": _list_context,
    "read_doc": _read_doc,
    "read_csv": _read_csv,
    "read_json": _read_json,
    "show_context_schema": _show_context_schema,
    "query_context_tables": _query_context_tables,
    "inspect_sqlite_schema": _inspect_sqlite_schema,
    "execute_context_sql": _execute_context_sql,
    "execute_python": _execute_python,
}


def create_data_agent_tool_registry(input_files: dict) -> ToolRegistry:
    """Build a tool registry tailored to the file types present in the task context."""
    has_tabular = bool(input_files.get("csv") or input_files.get("json"))
    has_db = bool(input_files.get("db"))

    active: set[str] = {"answer", "list_context", "read_doc", "execute_python"}

    if has_tabular or not has_db:
        active.update({"show_context_schema", "query_context_tables"})

    if has_db:
        active.update({"inspect_sqlite_schema", "execute_context_sql"})

    if not has_tabular and not has_db:
        # Unknown context — add raw file readers for exploration
        active.update({"read_csv", "read_json"})

    return ToolRegistry(
        specs={k: v for k, v in _ALL_SPECS.items() if k in active},
        handlers={k: v for k, v in _ALL_HANDLERS.items() if k in active},
    )


# ---------------------------------------------------------------------------
# DataAgent
# ---------------------------------------------------------------------------


class DataAgent:
    """
    General-purpose data agent that handles tasks of any difficulty.

    Tool selection adapts automatically based on the file types detected in the
    task's context directory (CSV/JSON → in-memory SQLite; .db → real SQLite;
    mixed → both; always includes Python execution as a fallback).
    """

    def __init__(self, *, model: ModelAdapter, max_steps: int = 16) -> None:
        self.model = model
        self.max_steps = max_steps

    def run(self, task: PublicTask):
        input_files = detect_input_files(task)
        tools = create_data_agent_tool_registry(input_files)
        agent = ReActAgent(
            model=self.model,
            tools=tools,
            config=ReActAgentConfig(max_steps=self.max_steps),
            system_prompt=DATA_AGENT_SYSTEM_PROMPT,
        )
        return agent.run(task)
