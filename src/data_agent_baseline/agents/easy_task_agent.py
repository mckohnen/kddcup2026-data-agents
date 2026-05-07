from __future__ import annotations

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.context_sqlite import get_context_schema, run_sql_on_context
from data_agent_baseline.tools.filesystem import read_doc_preview, _extract_md_toc, _extract_md_section
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
)

EASY_TASK_SYSTEM_PROMPT = """
You are a data agent solving a data task. Follow these steps:

Step 1 — Inspect schema:
  Call show_context_schema to see all available tables, columns, row counts, and sample values.

Step 2 — Resolve ambiguities (only if needed):
  If the question or schema is ambiguous (unclear encoding, filter value, or calculation), call
  read_knowledge_section to look up the relevant part of knowledge.md. Available sections:
    "## 2. Core Entities & Fields"  — column meanings and value encodings
    "## 3. Metric Definitions"      — KPI formulas and calculation logic
    "## 4. Constraints & Conventions" — filtering rules, units, formats
    "## 5. Exemplar Use Cases"      — SQL patterns that directly model the question
    "## 6. Ambiguity Resolution"    — field priority and disambiguation rules

Step 3 — Run SQL:
  Write SQL that answers the question directly from the schema. Rules:
  - Output only the columns explicitly requested in the question
  - JSON-sourced columns preserve native types (integers stay integers — no CAST needed)
  - CSV-sourced columns are stored as TEXT — use CAST(col AS INTEGER) for numeric comparisons

Step 4 — Validate:
  Check that row count is non-zero and no key columns are all NULL.
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


def _read_knowledge_section(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    from pathlib import Path
    knowledge_path = task.context_dir / "knowledge.md"
    if not knowledge_path.exists():
        return ToolExecutionResult(ok=False, content={"error": "knowledge.md not found in context."})
    text = knowledge_path.read_text(encoding="utf-8", errors="replace")
    section = action_input.get("section", "").strip()
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


def _read_doc(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 8000))
    content = read_doc_preview(task, path, max_chars=max_chars, query=task.question)
    return ToolExecutionResult(ok=True, content=content)


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
        "read_knowledge_section": ToolSpec(
            name="read_knowledge_section",
            description=(
                "Read a section from knowledge.md by its ## header. "
                "Omit 'section' to get the table of contents. "
                "Use only when the schema or question is ambiguous."
            ),
            input_schema={"section": "## 5. Exemplar Use Cases"},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a file from the doc/ directory (e.g. doc/Patient.md). "
                "Use when additional domain context beyond knowledge.md is needed."
            ),
            input_schema={"path": "doc/filename.md"},
        ),
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={"columns": ["col"], "rows": [["value"]]},
        ),
    }
    handlers = {
        "show_context_schema": _show_context_schema,
        "query_context_tables": _query_context_tables,
        "read_knowledge_section": _read_knowledge_section,
        "read_doc": _read_doc,
        "answer": _answer,
    }
    return ToolRegistry(specs=specs, handlers=handlers)


class EasyTaskAgent:
    def __init__(self, *, model: ModelAdapter, max_steps: int = 10) -> None:
        self._agent = ReActAgent(
            model=model,
            tools=create_easy_task_tool_registry(),
            config=ReActAgentConfig(max_steps=max_steps),
            system_prompt=EASY_TASK_SYSTEM_PROMPT,
        )

    def run(self, task: PublicTask):
        return self._agent.run(task)
