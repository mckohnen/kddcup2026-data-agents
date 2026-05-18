#!/usr/bin/env python3
"""
Analyze a completed benchmark run.

Usage:
    python analyze_run.py <run_dir>
    python analyze_run.py artifacts/runs/my_run
    python analyze_run.py artifacts/runs/my_run --eval-dir public/output
    python analyze_run.py artifacts/runs/my_run --task task_11   # single task deep-dive
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Trace analysis helpers
# ---------------------------------------------------------------------------

def classify_failure(trace: dict) -> str:
    if trace.get("succeeded"):
        return "succeeded"

    failure_reason = (trace.get("failure_reason") or "").lower()
    if "timeout" in failure_reason or "timed out" in failure_reason:
        return "timeout"

    steps = trace.get("steps", [])
    if not steps:
        return "no_steps"

    actions = [s.get("action") for s in steps]

    # Last step was a JSON parse failure
    if actions and actions[-1] == "__error__":
        return "parse_error"

    # Any parse errors in the run (but recovered)
    has_parse_error = "__error__" in actions

    # Tool returned ok=False
    tool_errors = [s for s in steps if not s.get("ok") and s.get("action") not in (None, "__error__")]
    if tool_errors and actions[-1] != "answer":
        bad_tool = tool_errors[-1].get("action", "unknown")
        return f"tool_error:{bad_tool}"

    if has_parse_error:
        return "recovered_parse_error_then_succeeded"  # shouldn't reach here (succeeded check above)

    return "max_steps_exceeded"


def analyze_trace(trace: dict) -> dict:
    steps = trace.get("steps", [])
    actions = [s.get("action") for s in steps]
    tools_used = Counter(a for a in actions if a and a != "__error__")

    answer_step = next((s for s in steps if s.get("action") == "answer"), None)
    answer_rows = 0
    answer_cols = 0
    if answer_step:
        ai = answer_step.get("action_input", {})
        answer_rows = len(ai.get("rows", []))
        answer_cols = len(ai.get("columns", []))

    return {
        "failure_mode": classify_failure(trace),
        "steps_used": len(steps),
        "tools_used": dict(tools_used),
        "parse_errors": sum(1 for a in actions if a == "__error__"),
        "tool_errors": sum(1 for s in steps if not s.get("ok") and s.get("action") not in (None, "__error__")),
        "last_action": actions[-1] if actions else None,
        "failure_reason": trace.get("failure_reason"),
        "elapsed_seconds": trace.get("e2e_elapsed_seconds"),
        "answer_rows": answer_rows,
        "answer_cols": answer_cols,
    }


def score_band(score: float) -> str:
    if score >= 0.9:
        return "excellent (≥0.9)"
    if score >= 0.7:
        return "good     (≥0.7)"
    if score >= 0.4:
        return "partial  (≥0.4)"
    if score > 0:
        return "poor     (>0)"
    return "zero"


# ---------------------------------------------------------------------------
# Deep-dive: single task
# ---------------------------------------------------------------------------

def print_task_deep_dive(task_id: str, task_dir: Path, score_info: dict) -> None:
    trace_path = task_dir / "trace.json"
    detection_path = task_dir / "input_detection.json"

    if not trace_path.exists():
        print(f"No trace found for {task_id}")
        return

    trace = json.loads(trace_path.read_text())
    analysis = analyze_trace(trace)

    print(f"\n{'=' * 70}")
    print(f"DEEP DIVE: {task_id}")
    print(f"{'=' * 70}")

    if detection_path.exists():
        detection = json.loads(detection_path.read_text())
        print(f"\nContext files detected: {json.dumps(detection, indent=2)}")

    if score_info:
        print(f"\nScore:         {score_info.get('score', 'N/A')}")
        print(f"Recall:        {score_info.get('recall', 'N/A')}")
        print(f"Matched cols:  {score_info.get('matched', 'N/A')} / {score_info.get('gold_cols', 'N/A')}")
        print(f"Extra cols:    {score_info.get('extra_cols', 'N/A')}")
        if score_info.get("error"):
            print(f"Eval error:    {score_info['error']}")

    print(f"\nFailure mode:  {analysis['failure_mode']}")
    print(f"Steps used:    {analysis['steps_used']}")
    print(f"Elapsed:       {analysis.get('elapsed_seconds', 'N/A')}s")
    print(f"Answer rows:   {analysis['answer_rows']}, cols: {analysis['answer_cols']}")
    if analysis.get("failure_reason"):
        print(f"Failure reason: {analysis['failure_reason']}")

    print(f"\n{'─' * 70}")
    print("STEP-BY-STEP TRACE")
    print(f"{'─' * 70}")

    for step in trace.get("steps", []):
        idx = step.get("step_index", "?")
        action = step.get("action", "?")
        thought = (step.get("thought") or "").strip()
        ok = step.get("ok", True)

        print(f"\n[Step {idx}] Action: {action}  ok={ok}")
        if thought:
            # Truncate long thoughts
            short_thought = thought[:300] + "..." if len(thought) > 300 else thought
            print(f"  Thought: {short_thought}")

        obs = step.get("observation", {})
        if obs:
            content = obs.get("content", {})
            obs_ok = obs.get("ok", True)
            if not obs_ok:
                # Prefer explicit error message; fall back to content
                err_detail = obs.get("error") or content
                print(f"  !! Tool error: {err_detail}")
            elif action == "show_context_schema":
                tables = content.get("tables", [])
                # New schema profiler returns a dict {table_name: profile};
                # old format was a list of {"table": name, ...} dicts.
                if isinstance(tables, dict):
                    table_names = list(tables.keys())
                else:
                    table_names = [t["table"] for t in tables]
                rels = content.get("relationships", [])
                rel_info = f", {len(rels)} relationships" if rels else ""
                print(f"  Schema: {len(table_names)} tables{rel_info} — {table_names}")
            elif action == "query_context_tables":
                sql = step.get("action_input", {}).get("sql", "")
                short_sql = sql[:200] + "..." if len(sql) > 200 else sql
                print(f"  SQL: {short_sql}")
                rows = content.get("row_count", "?")
                cols = content.get("columns", [])
                truncated = content.get("truncated", False)
                print(f"  Result: {rows} rows, cols={cols}{' (truncated)' if truncated else ''}")
            elif action == "answer":
                ai = step.get("action_input", {})
                print(f"  Submitted: {len(ai.get('columns', []))} cols, {len(ai.get('rows', []))} rows")
                print(f"  Columns: {ai.get('columns', [])}")
            elif action == "read_doc":
                length = len(str(content))
                print(f"  Doc content: {length} chars")
            elif action == "execute_python":
                # Key is "output" (not "stdout") in execute_python results
                out = (content.get("output") or content.get("stdout", ""))[:200]
                err = content.get("stderr", "")[:200]
                if out:
                    print(f"  stdout: {out}")
                if err:
                    print(f"  stderr: {err}")
            elif action == "__error__":
                # For __error__ steps the error lives in obs["error"], not obs["content"]
                err_msg = obs.get("error") or content
                print(f"  Parse error: {err_msg}")
            else:
                short_content = str(content)[:300]
                print(f"  Content: {short_content}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a benchmark run output directory")
    parser.add_argument("run_dir", type=Path, help="Path to run output directory (contains summary.json)")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Gold evaluation directory (public/output)")
    parser.add_argument("--task", type=str, default=None, help="Print deep-dive for a specific task_id")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load summary
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    # Load or compute evaluation
    eval_path = run_dir / "evaluation.json"
    evaluation: dict = {}
    if eval_path.exists():
        evaluation = json.loads(eval_path.read_text())
    elif args.eval_dir:
        # Try to trigger evaluate on the fly
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from data_agent_baseline.run.evaluate import evaluate_run, write_evaluation_report
            eval_result = evaluate_run(run_dir, args.eval_dir)
            write_evaluation_report(eval_result, run_dir)
            evaluation = eval_result.to_dict()
            print(f"(Evaluation computed and saved to {eval_path})")
        except Exception as exc:
            print(f"Warning: could not compute evaluation: {exc}", file=sys.stderr)

    task_scores: dict[str, dict] = evaluation.get("tasks", {})

    # Collect task dirs
    task_dirs = sorted(run_dir.glob("task_*"), key=lambda p: p.name)

    # Single-task deep-dive mode
    if args.task:
        target = next((d for d in task_dirs if d.name == args.task), None)
        if target is None:
            print(f"Task {args.task} not found in {run_dir}", file=sys.stderr)
            sys.exit(1)
        print_task_deep_dive(args.task, target, task_scores.get(args.task, {}))
        return

    # Full run analysis
    records: list[dict] = []
    for task_dir in task_dirs:
        task_id = task_dir.name
        trace_path = task_dir / "trace.json"
        score_info = task_scores.get(task_id, {})

        if not trace_path.exists():
            records.append({"task_id": task_id, "analysis": None, "score_info": score_info})
            continue

        trace = json.loads(trace_path.read_text())
        analysis = analyze_trace(trace)
        records.append({"task_id": task_id, "analysis": analysis, "score_info": score_info})

    # --- Header ---
    print("=" * 70)
    print(f"BENCHMARK RUN ANALYSIS  /  {run_dir.name}")
    print("=" * 70)

    if summary:
        print(f"\nRun ID:     {summary.get('run_id', run_dir.name)}")
        print(f"Tasks run:  {summary.get('task_count', len(records))}")
        print(f"Succeeded:  {summary.get('succeeded_task_count', 'N/A')}")

    if evaluation:
        mean = evaluation.get("mean_score", 0)
        mean_all = evaluation.get("mean_score_all", mean)
        evaled = evaluation.get("evaluated", 0)
        skipped = evaluation.get("skipped", 0)
        print(f"\nMean score (excl. skipped): {mean:.4f}  ({evaled} evaluated, {skipped} skipped)")
        print(f"Mean score (incl. skipped): {mean_all:.4f}  (skipped tasks count as 0)")

    # --- Per-task table ---
    print(f"\n{'Task':<14} {'Failure Mode':<30} {'Steps':>5}  {'Score':>6}  {'Recall':>6}  {'Extra':>5}  {'Elapsed':>8}")
    print("─" * 80)

    failure_counts: Counter = Counter()
    score_band_counts: Counter = Counter()

    for r in records:
        task_id = r["task_id"]
        an = r.get("analysis") or {}
        si = r.get("score_info") or {}

        failure_mode = an.get("failure_mode", "no_trace")
        steps = an.get("steps_used", "-")
        score = si.get("score")
        recall = si.get("recall")
        extra = si.get("extra_cols")
        elapsed = an.get("elapsed_seconds")
        eval_err = si.get("error")

        score_str = f"{score:.3f}" if score is not None else ("ERR" if eval_err else "N/A")
        recall_str = f"{recall:.3f}" if recall is not None else "N/A"
        extra_str = str(extra) if extra is not None else "N/A"
        elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "N/A"

        failure_counts[failure_mode] += 1
        if score is not None:
            score_band_counts[score_band(score)] += 1

        print(f"{task_id:<14} {failure_mode:<30} {str(steps):>5}  {score_str:>6}  {recall_str:>6}  {extra_str:>5}  {elapsed_str:>8}")

    # --- Failure mode breakdown ---
    print(f"\nFailure mode breakdown:")
    for mode, count in failure_counts.most_common():
        pct = 100 * count / len(records) if records else 0
        bar = "█" * count
        print(f"  {mode:<35} {bar}  {count} ({pct:.0f}%)")

    # --- Score distribution ---
    if score_band_counts:
        print(f"\nScore distribution (evaluated tasks):")
        order = ["excellent (≥0.9)", "good     (≥0.7)", "partial  (≥0.4)", "poor     (>0)", "zero"]
        total_scored = sum(score_band_counts.values())
        for band in order:
            count = score_band_counts.get(band, 0)
            pct = 100 * count / total_scored if total_scored else 0
            bar = "█" * count
            print(f"  {band}  {bar}  {count} ({pct:.0f}%)")

    # --- Tool usage ---
    all_tools: Counter = Counter()
    for r in records:
        for tool, cnt in (r.get("analysis") or {}).get("tools_used", {}).items():
            all_tools[tool] += cnt

    if all_tools:
        print(f"\nTool call totals across all tasks:")
        for tool, count in all_tools.most_common():
            print(f"  {tool:<35} {count:>4}")

    # --- Succeeded but scored 0 (wrong answer delivered) ---
    wrong_answer = [
        r for r in records
        if (r.get("analysis") or {}).get("failure_mode") == "succeeded"
        and (r.get("score_info") or {}).get("score", 1) == 0
    ]
    if wrong_answer:
        print(f"\nAnswered but scored 0  ({len(wrong_answer)} tasks — column mismatch?):")
        for r in wrong_answer:
            si = r["score_info"]
            an = r["analysis"]
            print(f"  {r['task_id']:<14}  predicted={si.get('predicted_cols',0)} cols,"
                  f" gold={si.get('gold_cols',0)} cols, extra={si.get('extra_cols',0)},"
                  f" answer_rows={an.get('answer_rows',0)}")

    # --- Slowest tasks ---
    timed = [(r["task_id"], r["analysis"]["elapsed_seconds"])
             for r in records if r.get("analysis") and r["analysis"].get("elapsed_seconds")]
    if timed:
        timed.sort(key=lambda x: x[1], reverse=True)
        print(f"\nSlowest tasks (top 5):")
        for task_id, elapsed in timed[:5]:
            print(f"  {task_id:<14}  {elapsed:.1f}s")

    # --- Hint for deep-dive ---
    failed = [r["task_id"] for r in records if (r.get("analysis") or {}).get("failure_mode") != "succeeded"]
    if failed:
        print(f"\nTo inspect a failed task in detail:")
        print(f"  python analyze_run.py {run_dir} --task {failed[0]}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
