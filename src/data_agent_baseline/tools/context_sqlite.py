from __future__ import annotations

import csv
import json
import sqlite3
from itertools import islice
from pathlib import Path
from typing import Any

# Per-process cache: built once per context_dir, reused across all queries in the same task.
_conn_cache: dict[str, sqlite3.Connection] = {}


def _sanitize(name: str) -> str:
    return name.replace('"', "")


def _create_table(conn: sqlite3.Connection, table_name: str, columns: list[str]) -> None:
    col_defs = ", ".join(f'"{_sanitize(c)}" TEXT' for c in columns)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{_sanitize(table_name)}" ({col_defs})')


def _insert_rows(
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[str],
    records: list[dict[str, Any]],
) -> None:
    col_names = ", ".join(f'"{_sanitize(c)}"' for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    sql = f'INSERT INTO "{_sanitize(table_name)}" ({col_names}) VALUES ({placeholders})'
    # Preserve native Python types from JSON (int stays INTEGER in SQLite).
    # CSV values arrive as strings and are stored as TEXT.
    rows = [[record.get(c) for c in columns] for record in records]
    conn.executemany(sql, rows)
    conn.commit()


def _load_json_file(conn: sqlite3.Connection, path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        table_name = str(payload.get("table", path.stem))
        records: list[dict[str, Any]] = payload["records"]
    elif isinstance(payload, list):
        table_name = path.stem
        records = payload
    else:
        return

    if not records:
        return

    columns = list(records[0].keys())
    _create_table(conn, table_name, columns)
    _insert_rows(conn, table_name, columns, records)


def _load_csv_file(conn: sqlite3.Connection, path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records_str = list(reader)

    if not records_str:
        return

    table_name = path.stem
    columns = list(records_str[0].keys())
    _create_table(conn, table_name, columns)
    _insert_rows(conn, table_name, columns, records_str)  # type: ignore[arg-type]


def load_context_to_sqlite(context_dir: Path) -> sqlite3.Connection:
    """Load all JSON and CSV files from context into an in-memory SQLite database.

    The result is cached per context_dir for the lifetime of the process, so
    repeated calls within the same task subprocess pay the loading cost only once.
    """
    key = str(context_dir.resolve())
    if key in _conn_cache:
        return _conn_cache[key]

    conn = sqlite3.connect(":memory:")

    for json_file in sorted(context_dir.rglob("*.json")):
        try:
            _load_json_file(conn, json_file)
        except Exception:
            pass

    for csv_file in sorted(context_dir.rglob("*.csv")):
        try:
            _load_csv_file(conn, csv_file)
        except Exception:
            pass

    _conn_cache[key] = conn
    return conn


def load_raw_tables(context_dir: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Load JSON, CSV, and SQLite files from context as raw record dicts.

    max_rows caps the number of records per table (used by the schema profiler
    to avoid reading millions of rows just for type inference).
    """
    raw: list[dict[str, Any]] = []

    for json_file in sorted(context_dir.rglob("*.json")):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "records" in payload:
                table_name = str(payload.get("table", json_file.stem))
                records: list[dict[str, Any]] = payload["records"]
            elif isinstance(payload, list):
                table_name = json_file.stem
                records = payload
            else:
                continue
            if records:
                raw.append({
                    "table": table_name,
                    "records": records[:max_rows] if max_rows is not None else records,
                })
        except Exception:
            pass

    for csv_file in sorted(context_dir.rglob("*.csv")):
        try:
            with csv_file.open(newline="", encoding="utf-8") as f:
                records_str = list(islice(csv.DictReader(f), max_rows))
            if records_str:
                raw.append({"table": csv_file.stem, "records": records_str})
        except Exception:
            pass

    db_extensions = ("*.db", "*.sqlite", "*.sqlite3")
    for pattern in db_extensions:
        for db_file in sorted(context_dir.rglob(pattern)):
            try:
                db_conn = sqlite3.connect(str(db_file))
                db_conn.row_factory = sqlite3.Row
                db_tables = db_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                for (table_name,) in db_tables:
                    limit_clause = f" LIMIT {max_rows}" if max_rows is not None else ""
                    rows = db_conn.execute(
                        f'SELECT * FROM "{_sanitize(table_name)}"{limit_clause}'
                    ).fetchall()
                    records_db: list[dict[str, Any]] = [dict(r) for r in rows]
                    if records_db:
                        raw.append({"table": table_name, "records": records_db})
                db_conn.close()
            except Exception:
                pass

    return raw


def get_context_schema(context_dir: Path) -> list[dict[str, Any]]:
    """Return schema + sample rows for all tables loaded from context."""
    conn = load_context_to_sqlite(context_dir)
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    tables = []
    for (name,) in table_rows:
        count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        cursor = conn.execute(f'SELECT * FROM "{name}" LIMIT 3')
        col_names = [d[0] for d in cursor.description or []]
        sample = [list(r) for r in cursor.fetchall()]
        tables.append(
            {
                "table": name,
                "row_count": count,
                "columns": col_names,
                "sample_rows": sample,
            }
        )
    return tables


def run_sql_on_context(context_dir: Path, sql: str, limit: int = 200) -> dict[str, Any]:
    """Load all context files into in-memory SQLite and execute a SQL query."""
    conn = load_context_to_sqlite(context_dir)
    cursor = conn.execute(sql)
    col_names = [d[0] for d in cursor.description or []]
    rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    limited = rows[:limit]
    return {
        "columns": col_names,
        "rows": [list(r) for r in limited],
        "row_count": len(limited),
        "truncated": truncated,
    }
