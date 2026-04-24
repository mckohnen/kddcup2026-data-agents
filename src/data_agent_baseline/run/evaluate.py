from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAMBDA = 0.5  # penalty weight for extra (unmatched) predicted columns


def _read_csv_columns(path: Path) -> list[tuple[str, ...]]:
    """Return each column as a sorted tuple of its cell values (the column signature)."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    # rows[0] is the header — skip it, work on data rows
    data_rows = rows[1:]
    num_cols = len(rows[0])

    columns = []
    for col_idx in range(num_cols):
        values = tuple(sorted(row[col_idx] for row in data_rows if col_idx < len(row)))
        columns.append(values)

    return columns


def score_task(prediction_path: Path, gold_path: Path) -> dict[str, Any]:
    """Score a single task prediction against its gold file."""
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
    gold_cols = _read_csv_columns(gold_path)

    pred_counter = Counter(pred_cols)
    gold_counter = Counter(gold_cols)

    matched = 0
    for sig, gold_count in gold_counter.items():
        matched += min(pred_counter.get(sig, 0), gold_count)

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
    mean_score: float
    evaluated: int
    skipped: int  # tasks with no gold or no prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_score": round(self.mean_score, 4),
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

    mean_score = sum(scores) / len(scores) if scores else 0.0

    return EvaluationResult(
        task_scores=task_scores,
        mean_score=mean_score,
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
