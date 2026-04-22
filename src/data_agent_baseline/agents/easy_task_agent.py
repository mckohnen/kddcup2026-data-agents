from __future__ import annotations

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.context_sqlite import get_context_schema, run_sql_on_context
from data_agent_baseline.tools.filesystem import read_doc_preview
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
)

EASY_TASK_SYSTEM_PROMPT = """
You are a data agent solving an easy data task. Follow these steps in order:

Step 1 — Read knowledge.md:
  Use read_doc with path "knowledge.md". This file defines column semantics, field encodings,
  and contains Use Case SQL examples that often directly answer the question. Read it first.

Step 2 — Inspect schema:
  Use show_context_schema to see all available tables, columns, row counts, and sample values.
  Identify which tables contain the data needed for the question.

Step 3 — Run SQL:
  Use query_context_tables with SQL that answers the question. Rules:
  - Mirror SQL patterns from knowledge.md Use Case examples as closely as possible
  - Output only the columns explicitly requested in the question
  - Match filter values exactly to the question wording (knowledge.md defines encodings)
  - Use raw/event tables rather than derived views
  - JSON-sourced columns preserve native types (integers stay integers — no CAST needed)
  - CSV-sourced columns are stored as TEXT — use CAST(col AS INTEGER) for numeric comparisons

Step 4 — Validate:
  After running SQL check that row count is non-zero and no key columns are all NULL.
  If the result looks wrong, revise the SQL and re-run.

Step 5 — Submit:
  Call the answer tool with the final result table.
""".strip()


def _show_context_schema(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    del action_input
    tables = get_context_schema(task.context_dir)
    return ToolExecutionResult(ok=True, content={"tables": tables})


def _query_context_tables(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    result = run_sql_on_context(task.context_dir, sql, limit=limit)
    return ToolExecutionResult(ok=True, content=result)


def _read_doc(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 8000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


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


def create_easy_task_tool_registry() -> ToolRegistry:
    specs = {
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text/markdown file from context (use for knowledge.md).",
            input_schema={"path": "knowledge.md", "max_chars": 8000},
        ),
        "show_context_schema": ToolSpec(
            name="show_context_schema",
            description=(
                "Load all JSON and CSV files from context into in-memory SQLite and return "
                "table names, columns, row counts, and sample rows."
            ),
            input_schema={},
        ),
        "query_context_tables": ToolSpec(
            name="query_context_tables",
            description=(
                "Load all JSON and CSV context files into in-memory SQLite and run a SQL query. "
                "JSON columns preserve native types. CSV columns are TEXT (use CAST for numerics)."
            ),
            input_schema={"sql": "SELECT ...", "limit": 200},
        ),
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={"columns": ["col"], "rows": [["value"]]},
        ),
    }
    handlers = {
        "read_doc": _read_doc,
        "show_context_schema": _show_context_schema,
        "query_context_tables": _query_context_tables,
        "answer": _answer,
    }
    return ToolRegistry(specs=specs, handlers=handlers)


class EasyTaskAgent:
    def __init__(self, *, model: ModelAdapter, max_steps: int = 8) -> None:
        self._agent = ReActAgent(
            model=model,
            tools=create_easy_task_tool_registry(),
            config=ReActAgentConfig(max_steps=max_steps),
            system_prompt=EASY_TASK_SYSTEM_PROMPT,
        )

    def run(self, task: PublicTask):
        return self._agent.run(task)
