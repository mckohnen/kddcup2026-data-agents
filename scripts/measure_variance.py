"""Measure per-task variance by running the same benchmark config N times.

Run `dabench run-benchmark` N times sequentially (each with a distinct run_id),
then aggregate the predictions and scores per task to report:
  * how often the agent produced an identical answer across runs
    (answer agreement rate, post-canonicalisation)
  * score min / max / mean / stdev
  * which runs disagreed when they did

Canonicalisation uses the same value-normalisation as the evaluator
(round to 2 dp, sort rows per column) so that "5", "5.0", "5.00" all
compare equal — the goal is to measure whether the AGENT produced the
same answer, independent of cosmetic differences.

Usage:
    uv run python scripts/measure_variance.py \\
        --config configs/react_baseline.failing.yaml \\
        --runs 5

Outputs:
    artifacts/variance_reports/<base_run_id>_<timestamp>.json
    Plus a printable summary table on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Make the package importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_agent_baseline.run.evaluate import _read_csv_with_headers, _normalize_value  # noqa: E402


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

def _canonicalise_prediction(pred_path: Path) -> str | None:
    """Return a stable canonical string for the contents of a prediction CSV.

    The canonical form ignores column ORDER and column NAMES — two runs that
    produced the same set of columns (by value signature) and the same row
    values are considered identical. Numeric values are rounded to 2 dp
    (matching the competition evaluator).

    Returns None if the file is missing or empty.
    """
    if not pred_path.exists():
        return None
    try:
        _headers, columns = _read_csv_with_headers(pred_path)
    except Exception:
        return None
    if not columns:
        # Header-only file = "empty answer". Distinguish from missing.
        return "<empty>"
    # Sort the columns themselves so two runs with the same data in different
    # column order compare equal.  Each column is already a sorted tuple of
    # normalised values from _read_csv_with_headers().
    canonical = sorted(list(c) for c in columns)
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False)


def _header_signature(pred_path: Path) -> tuple[str, ...]:
    """Return the header row of a prediction CSV (lowercased, stripped)."""
    if not pred_path.exists():
        return ()
    try:
        headers, _cols = _read_csv_with_headers(pred_path)
    except Exception:
        return ()
    return tuple(h.strip().lower() for h in headers)


# ---------------------------------------------------------------------------
# Per-run benchmark execution
# ---------------------------------------------------------------------------

def _write_temp_config(
    base_payload: dict,
    new_run_id: str,
    dst: Path,
) -> None:
    """Write a copy of base_payload with run.run_id overridden to new_run_id."""
    payload = dict(base_payload)
    run_block = dict(payload.get("run", {}))
    run_block["run_id"] = new_run_id
    payload["run"] = run_block
    dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _execute_run(temp_config_path: Path, limit: int | None) -> int:
    """Invoke `uv run dabench run-benchmark` and return the exit code."""
    cmd = [
        "uv", "run", "dabench", "run-benchmark",
        "--config", str(temp_config_path),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print(f"  exec: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _load_run_evaluation(run_dir: Path) -> dict[str, float]:
    """Return {task_id: score} from a run's evaluation.json, empty dict if missing."""
    eval_file = run_dir / "evaluation.json"
    if not eval_file.exists():
        return {}
    try:
        data = json.loads(eval_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {tid: float(t.get("score", 0.0)) for tid, t in data.get("tasks", {}).items()}


def _aggregate_variance(
    output_dir: Path,
    run_ids: list[str],
    base_run_id: str,
    timestamp: str,
) -> Path:
    """Compute per-task variance metrics and write a JSON report.

    Returns the path to the written report.
    """
    # Gather all task IDs that appeared in at least one run.
    per_run: dict[str, dict[str, Any]] = {}
    all_task_ids: set[str] = set()

    for run_id in run_ids:
        run_dir = output_dir / run_id
        if not run_dir.exists():
            print(f"  WARNING: run dir {run_dir} not found — skipping")
            continue
        scores = _load_run_evaluation(run_dir)
        predictions: dict[str, dict[str, Any]] = {}
        for task_dir in sorted(run_dir.glob("task_*")):
            if not task_dir.is_dir():
                continue
            tid = task_dir.name
            pred_path = task_dir / "prediction.csv"
            predictions[tid] = {
                "canonical": _canonicalise_prediction(pred_path),
                "headers": list(_header_signature(pred_path)),
            }
            all_task_ids.add(tid)
        per_run[run_id] = {"scores": scores, "predictions": predictions}

    # Per-task analysis.
    task_report: dict[str, dict[str, Any]] = {}
    for tid in sorted(all_task_ids):
        scores_per_run = []
        canonicals_per_run = []
        headers_per_run = []
        for run_id in run_ids:
            run_block = per_run.get(run_id, {})
            scores_per_run.append(run_block.get("scores", {}).get(tid))
            pred_block = run_block.get("predictions", {}).get(tid, {})
            canonicals_per_run.append(pred_block.get("canonical"))
            headers_per_run.append(pred_block.get("headers", []))

        scores_clean = [s for s in scores_per_run if s is not None]
        counter = Counter(c for c in canonicals_per_run if c is not None)
        if counter:
            most_common_count = counter.most_common(1)[0][1]
            agreement_rate = most_common_count / len(canonicals_per_run)
            distinct_answers = len(counter)
        else:
            agreement_rate = 0.0
            distinct_answers = 0

        task_report[tid] = {
            "n_runs": len(canonicals_per_run),
            "scores": scores_per_run,
            "score_min": min(scores_clean) if scores_clean else None,
            "score_max": max(scores_clean) if scores_clean else None,
            "score_mean": (statistics.mean(scores_clean) if scores_clean else None),
            "score_stdev": (statistics.stdev(scores_clean) if len(scores_clean) >= 2 else 0.0),
            "distinct_answers": distinct_answers,
            "agreement_rate": agreement_rate,
            "headers_per_run": headers_per_run,
            "missing_predictions": sum(1 for c in canonicals_per_run if c is None),
        }

    # Overall summary.
    all_scores_per_run = [
        statistics.mean([s for s in per_run.get(r, {}).get("scores", {}).values()])
        for r in run_ids
        if per_run.get(r, {}).get("scores")
    ]
    summary = {
        "base_run_id": base_run_id,
        "timestamp": timestamp,
        "n_runs": len(run_ids),
        "run_ids": run_ids,
        "mean_score_per_run": all_scores_per_run,
        "mean_score_min": min(all_scores_per_run) if all_scores_per_run else None,
        "mean_score_max": max(all_scores_per_run) if all_scores_per_run else None,
        "tasks": task_report,
    }

    variance_dir = PROJECT_ROOT / "artifacts" / "variance_reports"
    variance_dir.mkdir(parents=True, exist_ok=True)
    report_path = variance_dir / f"{base_run_id}_{timestamp}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path


def _print_summary(report_path: Path) -> None:
    """Render a printable variance summary table to stdout."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    n_runs = report["n_runs"]
    tasks = report["tasks"]

    print()
    print("=" * 92)
    print(f"VARIANCE REPORT  ·  base={report['base_run_id']}  ·  N={n_runs} runs")
    print("=" * 92)

    mean_per_run = report.get("mean_score_per_run", [])
    if mean_per_run:
        print(
            f"Mean score per run: "
            f"min={min(mean_per_run):.4f}  max={max(mean_per_run):.4f}  "
            f"stdev={statistics.stdev(mean_per_run):.4f}"
            if len(mean_per_run) >= 2
            else f"Mean score per run: {mean_per_run[0]:.4f}"
        )
        print(f"  per-run values: {[round(m, 4) for m in mean_per_run]}")

    print()
    header = (
        f"{'Task':<12} {'Agree':>6} {'Distinct':>9} "
        f"{'ScoreMin':>9} {'ScoreMax':>9} {'ScoreStd':>9} {'Scores':>30}"
    )
    print(header)
    print("-" * 92)
    for tid in sorted(tasks.keys()):
        t = tasks[tid]
        agree = f"{t['agreement_rate']*100:.0f}%"
        s_min = f"{t['score_min']:.2f}" if t["score_min"] is not None else "—"
        s_max = f"{t['score_max']:.2f}" if t["score_max"] is not None else "—"
        s_std = f"{t['score_stdev']:.3f}"
        scores_str = " ".join(
            "—" if s is None else f"{s:.2f}" for s in t["scores"]
        )
        print(
            f"{tid:<12} {agree:>6} {t['distinct_answers']:>9} "
            f"{s_min:>9} {s_max:>9} {s_std:>9} {scores_str:>30}"
        )

    print()
    unstable = [
        (tid, t) for tid, t in tasks.items() if t["agreement_rate"] < 1.0
    ]
    if unstable:
        print(f"Unstable tasks (answer agreement < 100%):  {len(unstable)} of {len(tasks)}")
        for tid, t in sorted(unstable, key=lambda kv: kv[1]["agreement_rate"]):
            print(
                f"  {tid:<12} agreement={t['agreement_rate']*100:.0f}%  "
                f"distinct_answers={t['distinct_answers']}  "
                f"scores={[round(s, 2) if s is not None else None for s in t['scores']]}"
            )
    else:
        print("All tasks agreed across runs.")
    print()
    print(f"Full report: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-task variance across N sequential benchmark runs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML benchmark config (e.g. configs/react_baseline.failing.yaml).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of full benchmark passes to perform (default: 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on tasks per run (passes --limit to dabench).",
    )
    parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep the per-run YAMLs in configs/ instead of deleting them.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help=(
            "Skip the benchmark execution and only aggregate existing run dirs. "
            "Requires --existing-run-ids."
        ),
    )
    parser.add_argument(
        "--existing-run-ids",
        type=str,
        default=None,
        help="Comma-separated list of existing run_ids to aggregate (used with --skip-runs).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    base_payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_run_id = (
        (base_payload.get("run", {}) or {}).get("run_id")
        or args.config.stem
        or "variance"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "artifacts" / "runs"

    if args.skip_runs:
        if not args.existing_run_ids:
            print("ERROR: --skip-runs requires --existing-run-ids", file=sys.stderr)
            sys.exit(2)
        run_ids = [r.strip() for r in args.existing_run_ids.split(",") if r.strip()]
    else:
        run_ids = [f"{base_run_id}_var_{i:02d}" for i in range(args.runs)]
        configs_dir = args.config.parent
        for i, run_id in enumerate(run_ids):
            print(f"\n[{i+1}/{args.runs}] starting run with run_id={run_id}")
            temp_path = configs_dir / f"_variance_{timestamp}_{i:02d}.yaml"
            _write_temp_config(base_payload, run_id, temp_path)
            try:
                rc = _execute_run(temp_path, args.limit)
                if rc != 0:
                    print(f"  WARNING: run {run_id} exited with code {rc}")
            finally:
                if not args.keep_temp_configs:
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

    print("\nAggregating variance report ...")
    report_path = _aggregate_variance(output_dir, run_ids, base_run_id, timestamp)
    _print_summary(report_path)


if __name__ == "__main__":
    main()
