from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAMBDA = 0.5  # penalty weight for extra (unmatched) predicted columns

# ---------------------------------------------------------------------------
# Column-name patterns that identify first-name / last-name gold columns.
# The competition considers a single "full name" prediction column to be
# equivalent to separate first- and last-name gold columns.
#
# Headers are matched after _normalize_header(), which lowercases and strips
# all whitespace, underscores, and dashes.  This means "first_name",
# "first-name", "FirstName", and "firstname" all resolve to "firstname" and
# match the same entry in the set.
# ---------------------------------------------------------------------------
_FIRST_NAME_HEADERS = {"firstname", "forename", "givenname"}
_LAST_NAME_HEADERS = {"lastname", "surname", "familyname"}


def _normalize_header(h: str) -> str:
    """Lowercase a column header and strip whitespace, underscores, and dashes.

    Examples: "first_name" → "firstname", "Last-Name" → "lastname",
    "Given Name" → "givenname".
    """
    return re.sub(r"[\s_\-]+", "", h).lower()


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def _normalize_value(val: str) -> str:
    """Normalize a single cell value to match competition scoring rules.

    Competition rule: "Numeric data should retain sufficient precision; the
    evaluation system will normalize values to 2 decimal places (rounding)
    before comparison."

    Numeric values are rounded to 2 dp and formatted with exactly two decimal
    places so that "2.727272", "2.73", and "2.730" all compare equal, and
    "5", "5.0", "5.00" all compare equal.  Non-numeric values are returned
    with surrounding whitespace stripped and otherwise unchanged.
    """
    v = val.strip()
    if not v:
        return v
    try:
        f = float(v)
        return f"{round(f, 2):.2f}"
    except (ValueError, OverflowError):
        return v


def _read_csv_columns(path: Path) -> list[tuple[str, ...]]:
    """Return each column as a sorted tuple of normalized cell values (the column signature).

    The header row is skipped; only data rows are used.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    data_rows = rows[1:]
    num_cols = len(rows[0])

    columns = []
    for col_idx in range(num_cols):
        values = tuple(
            sorted(
                _normalize_value(row[col_idx])
                for row in data_rows
                if col_idx < len(row)
            )
        )
        columns.append(values)

    return columns


def _read_csv_with_headers(path: Path) -> tuple[list[str], list[tuple[str, ...]]]:
    """Return (header_names, column_signatures) for a CSV file.

    Header names are returned as-is (lowercased for comparison elsewhere).
    Column signatures are sorted tuples of normalized cell values.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return [], []

    headers = rows[0]
    data_rows = rows[1:]
    num_cols = len(headers)

    columns = []
    for col_idx in range(num_cols):
        values = tuple(
            sorted(
                _normalize_value(row[col_idx])
                for row in data_rows
                if col_idx < len(row)
            )
        )
        columns.append(values)

    return headers, columns


# ---------------------------------------------------------------------------
# Name-column equivalence
# ---------------------------------------------------------------------------

def _find_name_pairs(
    headers: list[str], columns: list[tuple[str, ...]]
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Find (first_name_sig, last_name_sig) pairs from gold column headers.

    Uses *gold* column headers (which are well-defined by the competition) to
    detect first-name / last-name column pairs.  This avoids the ambiguity of
    trying to split a prediction column whose values might contain multi-word
    first names (e.g. "John Chris Smith").

    Returns a list of (first_col_sig, last_col_sig) pairs.  There will be at
    most one pair for typical tasks but the function handles multiple pairs
    correctly.
    """
    first_indices = [
        i for i, h in enumerate(headers)
        if _normalize_header(h) in _FIRST_NAME_HEADERS
    ]
    last_indices = [
        i for i, h in enumerate(headers)
        if _normalize_header(h) in _LAST_NAME_HEADERS
    ]

    pairs = []
    for fi in first_indices:
        for li in last_indices:
            if fi < len(columns) and li < len(columns):
                pairs.append((columns[fi], columns[li]))
    return pairs


def _merge_name_pair(
    first_sig: tuple[str, ...], last_sig: tuple[str, ...]
) -> tuple[str, ...] | None:
    """Merge a (first_name_sig, last_name_sig) pair into a combined full-name signature.

    Both columns must have the same length (same number of data rows).
    The values are zipped (both are already sorted), concatenated with a space,
    and the result is sorted to form the merged column signature.

    Returns None if the columns have different lengths.
    """
    if len(first_sig) != len(last_sig):
        return None
    return tuple(sorted(f"{fn} {ln}" for fn, ln in zip(first_sig, last_sig)))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_task(prediction_path: Path, gold_path: Path) -> dict[str, Any]:
    """Score a single task prediction against its gold file.

    Applies two competition-equivalent normalizations before matching:

    1. Numeric precision: cell values are rounded to 2 dp, so "2.727272" and
       "2.73" compare equal (competition rule: "normalize values to 2 decimal
       places before comparison").

    2. Name-column equivalence: if the gold file has explicit ``first_name``
       and ``last_name`` columns (detected by header name), a prediction that
       uses a single combined "full name" column is treated as matching both
       gold columns.  The merged virtual signature is derived from the gold
       columns themselves — we never attempt to split prediction values, which
       would be ambiguous for multi-word first names.
    """
    if not prediction_path.exists():
        return {
            "matched": 0,
            "gold_cols": 0,
            "predicted_cols": 0,
            "extra_cols": 0,
            "recall": 0.0,
            "score": 0.0,
            "error": "prediction.csv not found",
        }

    if not gold_path.exists():
        return {
            "matched": 0,
            "gold_cols": 0,
            "predicted_cols": 0,
            "extra_cols": 0,
            "recall": 0.0,
            "score": 0.0,
            "error": "gold.csv not found",
        }

    pred_cols = _read_csv_columns(prediction_path)
    gold_headers, gold_cols = _read_csv_with_headers(gold_path)

    pred_counter = Counter(pred_cols)
    gold_counter = Counter(gold_cols)

    # --- Normal column matching ---
    matched = 0
    for sig, gold_count in gold_counter.items():
        matched += min(pred_counter.get(sig, 0), gold_count)

    # --- Name-column equivalence (gold-header-driven) ---
    #
    # For each first_name+last_name pair identified in gold headers, build a
    # merged "full name" signature and check whether the prediction contains it.
    # Only credit the merge if neither original gold column was already matched
    # by the prediction (prevents double-counting when the prediction correctly
    # used the split form).
    #
    # When the prediction has the merged column:
    #   matched += 2  (covers both first_name and last_name gold columns)
    #   extra is computed as max(n_pred - matched, 0), which naturally becomes 0
    #   when the prediction only has the merged column (n_pred < matched is fine
    #   because extra is clamped to 0).
    name_pairs = _find_name_pairs(gold_headers, gold_cols)
    for first_sig, last_sig in name_pairs:
        merged_sig = _merge_name_pair(first_sig, last_sig)
        if merged_sig is None:
            continue

        pred_has_merged = pred_counter.get(merged_sig, 0) > 0
        if not pred_has_merged:
            continue

        # Check whether the split form was already matched normally
        first_already_matched = min(
            pred_counter.get(first_sig, 0), gold_counter.get(first_sig, 0)
        )
        last_already_matched = min(
            pred_counter.get(last_sig, 0), gold_counter.get(last_sig, 0)
        )

        if first_already_matched == 0 and last_already_matched == 0:
            # The merged pred column covers both gold name columns
            matched += 2

    # --- Final score ---
    n_gold = len(gold_cols)
    n_pred = len(pred_cols)
    extra = max(n_pred - matched, 0)

    recall = matched / n_gold if n_gold > 0 else 0.0
    penalty = LAMBDA * (extra / n_pred) if n_pred > 0 else 0.0
    score = max(recall - penalty, 0.0)

    return {
        "matched": matched,
        "gold_cols": n_gold,
        "predicted_cols": n_pred,
        "extra_cols": extra,
        "recall": round(recall, 4),
        "score": round(score, 4),
        "error": None,
    }


@dataclass
class EvaluationResult:
    task_scores: dict[str, dict[str, Any]]  # task_id -> score dict
    mean_score: float          # mean over evaluated tasks only (skipped excluded)
    mean_score_all: float      # mean over all gold tasks (skipped count as 0)
    evaluated: int
    skipped: int  # tasks with no gold or no prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_score": round(self.mean_score, 4),
            "mean_score_all": round(self.mean_score_all, 4),
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "tasks": self.task_scores,
        }


def evaluate_run(run_output_dir: Path, evaluation_dir: Path) -> EvaluationResult:
    """Evaluate all tasks in a run against the gold files in evaluation_dir."""
    task_scores: dict[str, dict[str, Any]] = {}
    scores: list[float] = []
    skipped = 0

    gold_task_ids = {p.parent.name for p in evaluation_dir.glob("*/gold.csv")}

    for task_dir in sorted(run_output_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        if task_id not in gold_task_ids:
            continue

        prediction_path = task_dir / "prediction.csv"
        gold_path = evaluation_dir / task_id / "gold.csv"

        result = score_task(prediction_path, gold_path)
        task_scores[task_id] = result

        if result["error"] is not None:
            skipped += 1
        else:
            scores.append(result["score"])

    total_gold = len(gold_task_ids)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_score_all = sum(scores) / total_gold if total_gold else 0.0

    return EvaluationResult(
        task_scores=task_scores,
        mean_score=mean_score,
        mean_score_all=mean_score_all,
        evaluated=len(scores),
        skipped=skipped,
    )


def write_evaluation_report(result: EvaluationResult, run_output_dir: Path) -> Path:
    report_path = run_output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path
