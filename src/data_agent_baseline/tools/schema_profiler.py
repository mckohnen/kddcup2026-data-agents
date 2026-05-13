from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Value normalisation helpers
# ---------------------------------------------------------------------------

def normalize_null(value: Any) -> Any:
    if value in {"null", "NULL", "None", ""}:
        return None
    return value


def normalize_records(records: list[dict]) -> list[dict]:
    return [{k: normalize_null(v) for k, v in row.items()} for row in records]


def looks_like_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def looks_like_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            datetime.strptime(value, pattern)
            return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Column / table profiling
# ---------------------------------------------------------------------------

def infer_type(values: list[Any]) -> str:
    non_null = [normalize_null(v) for v in values if normalize_null(v) is not None]
    if not non_null:
        return "unknown"
    if all(isinstance(v, bool) for v in non_null):
        return "boolean"
    if all(str(v).lower() in {"true", "false"} for v in non_null):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "integer"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "number"
    if all(looks_like_number(v) for v in non_null):
        return "number"
    if all(looks_like_date(v) for v in non_null):
        return "date"
    unique_ratio = len(set(map(str, non_null))) / len(non_null)
    if unique_ratio <= 0.5:
        return "categorical"
    return "string"


def is_id_like(column_name: str) -> bool:
    col = column_name.lower()
    return col == "id" or col.endswith("_id") or col.startswith("link_to_")


def is_primary_key_candidate(
    table_name: str, column_name: str, values: list[Any], row_count: int
) -> bool:
    non_null = [normalize_null(v) for v in values if normalize_null(v) is not None]
    if len(non_null) != row_count:
        return False
    if len(set(map(str, non_null))) != row_count:
        return False
    col = column_name.lower()
    tbl = table_name.lower()
    return col == f"{tbl}_id" or col == "id" or col.endswith("_id")


def profile_table(table_name: str, records: list[dict]) -> dict[str, Any]:
    records = normalize_records(records)
    columns = sorted({key for row in records for key in row.keys()})
    row_count = len(records)

    profile: dict[str, Any] = {"row_count": row_count, "primary_key": None, "columns": {}}

    for col in columns:
        values = [row.get(col) for row in records]
        non_null = [normalize_null(v) for v in values if normalize_null(v) is not None]
        unique_values = list(dict.fromkeys(map(str, non_null)))

        col_profile: dict[str, Any] = {
            "type": infer_type(values),
            "null_count": row_count - len(non_null),
            "unique_count": len(set(map(str, non_null))),
            "sample_values": unique_values[:10],
        }

        if is_id_like(col):
            col_profile["id_like"] = True

        if is_primary_key_candidate(table_name, col, values, row_count):
            col_profile["role"] = "primary_key"
            if profile["primary_key"] is None:
                profile["primary_key"] = col

        profile["columns"][col] = col_profile

    return profile


# ---------------------------------------------------------------------------
# Relationship detection
# ---------------------------------------------------------------------------

def _get_reference_columns(raw_tables: dict, context: dict) -> list[dict]:
    refs = []
    for table_name, records in raw_tables.items():
        for col, col_ctx in context["tables"][table_name]["columns"].items():
            values = [
                str(normalize_null(row.get(col)))
                for row in records
                if normalize_null(row.get(col)) is not None
            ]
            if not values:
                continue
            col_lower = col.lower()
            tbl_lower = table_name.lower()
            if len(set(values)) == len(values) and (
                col == context["tables"][table_name].get("primary_key")
                or col_lower == f"{tbl_lower}_id"
                or col_lower == "id"
                or col_lower.endswith("_id")
            ):
                refs.append({"table": table_name, "column": col, "values": set(values)})
    return refs


def _relationship_name_score(from_col: str, to_table: str, to_col: str) -> float:
    score = 0.0
    fn, tt, tc = from_col.lower(), to_table.lower(), to_col.lower()
    if fn == f"link_to_{tt}":
        score += 0.3
    if tt in fn:
        score += 0.2
    if tc in fn:
        score += 0.1
    return score


def _detect_relationships(raw_tables: dict, context: dict) -> list[dict]:
    relationships: list[dict] = []
    reference_columns = _get_reference_columns(raw_tables, context)

    for from_table, records in raw_tables.items():
        for from_col, from_col_ctx in context["tables"][from_table]["columns"].items():
            if not is_id_like(from_col) or from_col_ctx.get("role") == "primary_key":
                continue
            values = [
                str(normalize_null(row.get(from_col)))
                for row in records
                if normalize_null(row.get(from_col)) is not None
            ]
            if not values:
                continue
            for ref in reference_columns:
                to_table, to_col, ref_values = ref["table"], ref["column"], ref["values"]
                if from_table == to_table:
                    continue
                coverage = sum(v in ref_values for v in values) / len(values)
                if coverage >= 0.80:
                    confidence = min(
                        1.0, coverage + _relationship_name_score(from_col, to_table, to_col)
                    )
                    rel = {
                        "from_table": from_table,
                        "from_column": from_col,
                        "to_table": to_table,
                        "to_column": to_col,
                        "confidence": round(confidence, 3),
                        "coverage": round(coverage, 3),
                        "evidence": (
                            f"{coverage:.1%} of non-null values in "
                            f"{from_table}.{from_col} match {to_table}.{to_col}"
                        ),
                    }
                    if rel not in relationships:
                        relationships.append(rel)
                    from_col_ctx["role"] = "foreign_key"
                    from_col_ctx["references"] = f"{to_table}.{to_col}"

    return relationships


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context(database_name: str, raw_tables_input: list[dict]) -> dict[str, Any]:
    raw_tables = {
        obj["table"]: normalize_records(obj["records"]) for obj in raw_tables_input
    }
    context: dict[str, Any] = {
        "database": database_name,
        "tables": {},
        "relationships": [],
    }
    for table_name, records in raw_tables.items():
        context["tables"][table_name] = profile_table(table_name, records)
    context["relationships"] = _detect_relationships(raw_tables, context)
    return context


def build_schema_profile(context_dir: Path, sample_rows: int = 50000) -> dict[str, Any]:
    """Build a rich schema profile for all data sources in context_dir.

    Loads CSV/JSON/SQLite data, infers column types, detects primary/foreign
    keys, and identifies likely JOIN relationships between tables.

    Row counts are corrected against the true SQLite counts after profiling
    (sample_rows may have capped what was loaded for type inference).
    """
    from data_agent_baseline.tools.context_sqlite import load_raw_tables, load_context_to_sqlite

    raw_tables = load_raw_tables(context_dir, max_rows=sample_rows)
    result = build_context(context_dir.parent.name, raw_tables)

    # Overwrite row_count with the true count from the unified SQLite connection.
    conn = load_context_to_sqlite(context_dir)
    for table_name in list(result["tables"].keys()):
        try:
            # Support both plain names (CSV/JSON) and alias.table (.db) names
            if "." in table_name:
                alias, tbl = table_name.split(".", 1)
                actual = conn.execute(
                    f'SELECT COUNT(*) FROM "{alias}"."{tbl}"'
                ).fetchone()[0]
            else:
                actual = conn.execute(
                    f'SELECT COUNT(*) FROM "{table_name}"'
                ).fetchone()[0]
            result["tables"][table_name]["row_count"] = actual
        except Exception:
            pass

    return result
