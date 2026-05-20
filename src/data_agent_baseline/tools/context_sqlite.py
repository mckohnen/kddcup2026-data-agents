from __future__ import annotations

import csv
import json
import re
import sqlite3
from itertools import islice
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Per-process cache: built once per context_dir, reused across all queries
# in the same task subprocess.  Keyed by resolved context directory path.
# ---------------------------------------------------------------------------
_conn_cache: dict[str, sqlite3.Connection] = {}
# Stores the list of ATTACH alias names for each cached connection.
_db_aliases: dict[str, list[str]] = {}

_DB_EXTENSIONS = ("*.db", "*.sqlite", "*.sqlite3")

# CSV files larger than this threshold get column-pruned before loading.
# Only columns relevant to the question (plus all ID/key columns) are loaded,
# reducing memory use and load time dramatically for wide tables.
_LARGE_CSV_THRESHOLD_BYTES = 20 * 1024 * 1024  # 20 MB

# Prose Markdown files in doc/ larger than this are indexed as paragraph tables
# so the agent can query them with SQL LIKE expressions instead of reading raw text.
_MD_PARAGRAPH_MIN_FILE_BYTES = 5_000   # skip tiny .md files
_MD_PARAGRAPH_MIN_LENGTH = 80          # skip very short paragraphs (headers, captions)


def _question_words(question: str) -> set[str]:
    """Extract lowercase, punctuation-stripped words from a question string."""
    return set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())


def _select_columns_for_large_csv(
    all_columns: list[str], question: str
) -> list[str]:
    """Choose which columns to load from a large CSV.

    Strategy (column is kept if ANY rule matches):
    1. Always-keep: columns whose name ends with ``_id``, starts with ``id``,
       or contains common structural keywords (date, time, season, stage, year,
       month, name, type, code, key, flag) — these are needed for joins, filters,
       and group-bys regardless of the question.
    2. Question-match: any column whose name (lowercased, underscores → spaces)
       shares at least one word with the question.

    Falls back to the full column list when the heuristic selects fewer than 5
    columns (guards against empty/degenerate questions).
    """
    q_words = _question_words(question)

    _STRUCTURAL_KEYWORDS = {
        "id", "date", "time", "year", "month", "season", "stage", "name",
        "type", "code", "key", "flag", "status", "rank", "score", "goal",
        "result", "label", "category", "class", "group", "level", "value",
    }

    selected: list[str] = []
    for col in all_columns:
        col_lower = col.lower()
        col_words = set(re.sub(r"[^a-z0-9 ]", " ", col_lower).split())

        # Rule 1: structural / key columns
        if (
            col_lower == "id"
            or col_lower.startswith("id_")
            or col_lower.endswith("_id")
            or bool(col_words & _STRUCTURAL_KEYWORDS)
        ):
            selected.append(col)
            continue

        # Rule 2: question-word match
        if col_words & q_words:
            selected.append(col)

    # Fallback: return all columns if heuristic is too aggressive
    if len(selected) < 5:
        return all_columns

    return selected


def _sanitize(name: str) -> str:
    return name.replace('"', "")


def _safe_alias(stem: str) -> str:
    """Turn a file stem into a valid SQLite schema alias."""
    alias = _sanitize(stem)
    # SQLite schema names must not start with a digit
    if alias and alias[0].isdigit():
        alias = "db_" + alias
    return alias or "db"


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


def _md_table_name(path: Path) -> str:
    """Derive a safe SQLite table name from a Markdown file path."""
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem + "_paragraphs"


def _load_md_as_paragraphs(conn: sqlite3.Connection, path: Path) -> None:
    """Index a prose Markdown file as a paragraph table in SQLite.

    Creates table <stem>_paragraphs(paragraph_idx INTEGER, content TEXT).
    The agent can then run SQL LIKE queries against prose content without
    needing to write Python or use search_doc.  Only called for files that
    exceed _MD_PARAGRAPH_MIN_FILE_BYTES.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [
        p.strip() for p in re.split(r"\n{2,}", text)
        if len(p.strip()) >= _MD_PARAGRAPH_MIN_LENGTH
    ]
    if not paragraphs:
        return
    table_name = _md_table_name(path)
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{_sanitize(table_name)}" '
        "(paragraph_idx INTEGER, content TEXT)"
    )
    conn.executemany(
        f'INSERT INTO "{_sanitize(table_name)}" (paragraph_idx, content) VALUES (?, ?)',
        enumerate(paragraphs),
    )
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


def _read_csv_headers(path: Path) -> list[str]:
    """Read only the header row of a CSV file without loading all data."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _load_csv_file(
    conn: sqlite3.Connection,
    path: Path,
    columns_to_load: list[str] | None = None,
) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if columns_to_load is not None:
            records_str: list[Any] = [
                {c: row[c] for c in columns_to_load if c in row}
                for row in reader
            ]
        else:
            records_str = list(reader)

    if not records_str:
        return

    table_name = path.stem
    columns = list(records_str[0].keys())
    _create_table(conn, table_name, columns)
    _insert_rows(conn, table_name, columns, records_str)  # type: ignore[arg-type]


def load_context_to_sqlite(
    context_dir: Path,
    question: str | None = None,
) -> sqlite3.Connection:
    """Load all context files into a unified in-memory SQLite database.

    - CSV and JSON files are loaded directly into the main schema.
    - SQLite .db / .sqlite / .sqlite3 files are ATTACHed under an alias equal
      to their file stem (e.g. hero_power.db → schema alias 'hero_power').
      Tables from attached databases are addressed as <alias>.<table>.
    - For CSV files larger than _LARGE_CSV_THRESHOLD_BYTES, only columns
      relevant to *question* (plus structural/ID columns) are loaded.

    The result is cached per context_dir for the lifetime of the process so
    repeated calls within the same task subprocess pay the loading cost only once.
    """
    key = str(context_dir.resolve())
    if key in _conn_cache:
        return _conn_cache[key]

    conn = sqlite3.connect(":memory:", check_same_thread=False)

    for json_file in sorted(context_dir.rglob("*.json")):
        try:
            _load_json_file(conn, json_file)
        except Exception:
            pass

    for csv_file in sorted(context_dir.rglob("*.csv")):
        try:
            columns_to_load: list[str] | None = None
            if question and csv_file.stat().st_size > _LARGE_CSV_THRESHOLD_BYTES:
                headers = _read_csv_headers(csv_file)
                if headers:
                    columns_to_load = _select_columns_for_large_csv(headers, question)
            _load_csv_file(conn, csv_file, columns_to_load)
        except Exception:
            pass

    # Index large prose doc/*.md files as paragraph tables for SQL LIKE queries.
    # knowledge.md is excluded — it is navigated via read_knowledge_section instead.
    doc_dir = context_dir / "doc"
    if doc_dir.is_dir():
        for md_file in sorted(doc_dir.rglob("*.md")):
            try:
                if md_file.stat().st_size >= _MD_PARAGRAPH_MIN_FILE_BYTES:
                    _load_md_as_paragraphs(conn, md_file)
            except Exception:
                pass

    # ATTACH .db files — zero-copy, reads from disk on demand.
    aliases: list[str] = []
    used_aliases: set[str] = set()
    for pattern in _DB_EXTENSIONS:
        for db_file in sorted(context_dir.rglob(pattern)):
            alias = _safe_alias(db_file.stem)
            # Deduplicate aliases across extensions
            base = alias
            counter = 1
            while alias in used_aliases:
                alias = f"{base}_{counter}"
                counter += 1
            try:
                conn.execute(f'ATTACH DATABASE ? AS "{alias}"', (str(db_file.resolve()),))
                aliases.append(alias)
                used_aliases.add(alias)
            except Exception:
                pass

    _conn_cache[key] = conn
    _db_aliases[key] = aliases
    return conn


def load_raw_tables(
    context_dir: Path,
    max_rows: int | None = None,
    question: str | None = None,
) -> list[dict[str, Any]]:
    """Load JSON, CSV, and SQLite files from context as raw record dicts.

    max_rows caps the number of records per table (used by the schema profiler
    to avoid reading millions of rows just for type inference).
    Includes tables from .db / .sqlite / .sqlite3 files alongside CSV/JSON.
    For CSV files larger than _LARGE_CSV_THRESHOLD_BYTES, only columns
    relevant to *question* (plus structural/ID columns) are loaded.
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
            columns_to_load: list[str] | None = None
            if question and csv_file.stat().st_size > _LARGE_CSV_THRESHOLD_BYTES:
                headers = _read_csv_headers(csv_file)
                if headers:
                    columns_to_load = _select_columns_for_large_csv(headers, question)
            with csv_file.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if columns_to_load is not None:
                    records_str = [
                        {c: row[c] for c in columns_to_load if c in row}
                        for row in islice(reader, max_rows)
                    ]
                else:
                    records_str = list(islice(reader, max_rows))
            if records_str:
                raw.append({"table": csv_file.stem, "records": records_str})
        except Exception:
            pass

    # Include large prose doc/*.md files as paragraph tables for the schema profiler.
    doc_dir = context_dir / "doc"
    if doc_dir.is_dir():
        for md_file in sorted(doc_dir.rglob("*.md")):
            try:
                if md_file.stat().st_size >= _MD_PARAGRAPH_MIN_FILE_BYTES:
                    text = md_file.read_text(encoding="utf-8", errors="replace")
                    paragraphs = [
                        p.strip() for p in re.split(r"\n{2,}", text)
                        if len(p.strip()) >= _MD_PARAGRAPH_MIN_LENGTH
                    ]
                    if paragraphs:
                        capped = paragraphs[:max_rows] if max_rows is not None else paragraphs
                        raw.append({
                            "table": _md_table_name(md_file),
                            "records": [
                                {"paragraph_idx": i, "content": p}
                                for i, p in enumerate(capped)
                            ],
                        })
            except Exception:
                pass

    for pattern in _DB_EXTENSIONS:
        for db_file in sorted(context_dir.rglob(pattern)):
            try:
                db_conn = sqlite3.connect(str(db_file))
                db_conn.row_factory = sqlite3.Row
                db_tables = db_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                alias = _safe_alias(db_file.stem)
                for (table_name,) in db_tables:
                    limit_clause = f" LIMIT {max_rows}" if max_rows is not None else ""
                    rows = db_conn.execute(
                        f'SELECT * FROM "{_sanitize(table_name)}"{limit_clause}'
                    ).fetchall()
                    records_db: list[dict[str, Any]] = [dict(r) for r in rows]
                    if records_db:
                        raw.append({
                            "table": f"{alias}.{table_name}",
                            "records": records_db,
                        })
                db_conn.close()
            except Exception:
                pass

    return raw


def get_context_schema(context_dir: Path) -> list[dict[str, Any]]:
    """Return schema + sample rows for all tables in the unified SQLite connection.

    Includes tables from CSV/JSON (main schema) and tables from ATTACHed .db files
    (shown with their alias prefix, e.g. 'hero_power.hero_power').
    """
    conn = load_context_to_sqlite(context_dir)
    key = str(context_dir.resolve())
    aliases = _db_aliases.get(key, [])

    tables = []

    # Main schema: CSV/JSON tables
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            cursor = conn.execute(f'SELECT * FROM "{name}" LIMIT 3')
            col_names = [d[0] for d in cursor.description or []]
            sample = [list(r) for r in cursor.fetchall()]
            tables.append({
                "table": name,
                "schema": "main",
                "row_count": count,
                "columns": col_names,
                "sample_rows": sample,
            })
        except Exception:
            pass

    # Attached schemas: .db file tables, addressed as <alias>.<table>
    for alias in aliases:
        try:
            db_tables = conn.execute(
                f'SELECT name FROM "{alias}".sqlite_master WHERE type=\'table\' ORDER BY name'
            ).fetchall()
            for (name,) in db_tables:
                try:
                    full_name = f"{alias}.{name}"
                    count = conn.execute(f'SELECT COUNT(*) FROM "{alias}"."{name}"').fetchone()[0]
                    cursor = conn.execute(f'SELECT * FROM "{alias}"."{name}" LIMIT 3')
                    col_names = [d[0] for d in cursor.description or []]
                    sample = [list(r) for r in cursor.fetchall()]
                    tables.append({
                        "table": full_name,
                        "schema": alias,
                        "row_count": count,
                        "columns": col_names,
                        "sample_rows": sample,
                    })
                except Exception:
                    pass
        except Exception:
            pass

    return tables


def run_sql_on_context(context_dir: Path, sql: str, limit: int = 200) -> dict[str, Any]:
    """Execute a SQL query against the unified in-memory SQLite connection.

    The connection contains all CSV/JSON tables (main schema) and all .db tables
    (accessible as <alias>.<table>, e.g. hero_power.hero_power).
    """
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
