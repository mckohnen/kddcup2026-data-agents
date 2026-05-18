from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.data_agent import DataAgent, summarise_trace_for_resumption
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.input_detector import detect_input_files
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _resolve_attempt_timeouts(run_config) -> list[int]:
    """Return per-attempt wall-clock budgets (seconds), excluding preflight.

    If ``task_timeout_seconds_per_attempt`` is set in config it is used directly
    (length determines the total number of attempts).  Otherwise the flat
    ``task_timeout_seconds`` value is repeated ``1 + max_resumptions`` times for
    backward compatibility.
    """
    if run_config.task_timeout_seconds_per_attempt:
        return list(run_config.task_timeout_seconds_per_attempt)
    return [run_config.task_timeout_seconds] * (1 + max(0, run_config.max_resumptions))


def _run_one_attempt_core(
    *,
    task_id: str,
    config: AppConfig,
    prior_summaries: list[str] | None = None,
    is_first_attempt: bool = True,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Run a SINGLE agent attempt (no internal resumption loop).

    Returns the attempt result dict including ``preflight``, ``input_tokens``,
    and ``output_tokens`` keys that the caller may pop before writing the trace.
    """
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    model_instance = model or build_model_adapter(config)

    # Only run preflight on the first attempt.  Resumptions already receive the
    # schema via the prior_summaries block, so re-running preflight wastes budget.
    preflight_secs = config.run.preflight_timeout_seconds if is_first_attempt else 0
    agent = DataAgent(
        model=model_instance,
        max_steps=config.agent.max_steps,
        preflight_timeout_seconds=preflight_secs,
    )

    run_result = agent.run(task, prior_attempts=prior_summaries or []).to_dict()

    # Attach per-attempt metadata for the caller to accumulate / write.
    run_result["preflight"] = getattr(agent, "_last_preflight", {})
    run_result["input_tokens"] = getattr(model_instance, "total_input_tokens", 0)
    run_result["output_tokens"] = getattr(model_instance, "total_output_tokens", 0)
    return run_result


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Run all attempts in-process (used when model/tools overrides are provided).

    This path is only taken in single-worker / test mode.  Production runs go
    through ``_run_single_task_with_timeout`` which spawns a subprocess per attempt.
    """
    attempt_timeouts = _resolve_attempt_timeouts(config.run)
    prior_summaries: list[str] = []
    cumulative_input = 0
    cumulative_output = 0
    preflight_saved = False
    run_result: dict[str, Any] = _failure_run_result_payload(task_id, "No attempts were made.")

    for attempt_idx in range(len(attempt_timeouts)):
        run_result = _run_one_attempt_core(
            task_id=task_id,
            config=config,
            prior_summaries=prior_summaries,
            is_first_attempt=(attempt_idx == 0),
            model=model,
            tools=tools,
        )
        cumulative_input += run_result.pop("input_tokens", 0)
        cumulative_output += run_result.pop("output_tokens", 0)
        if not preflight_saved:
            run_result.setdefault("preflight", {})
            preflight_saved = True
        else:
            run_result.pop("preflight", None)

        if run_result.get("answer") is not None:
            break
        failure_reason = run_result.get("failure_reason", "")
        if "max_steps" not in failure_reason.lower():
            break
        if attempt_idx < len(attempt_timeouts) - 1:
            prior_summaries.append(summarise_trace_for_resumption(run_result))

    run_result["input_tokens"] = cumulative_input
    run_result["output_tokens"] = cumulative_output
    return run_result


def _run_one_attempt_in_subprocess(
    task_id: str,
    config: AppConfig,
    prior_summaries: list[str],
    attempt_idx: int,
    queue: multiprocessing.Queue[Any],
) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_one_attempt_core(
                    task_id=task_id,
                    config=config,
                    prior_summaries=prior_summaries,
                    is_first_attempt=(attempt_idx == 0),
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc)})


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    """Run a task with per-attempt escalating wall-clock timeouts.

    Each attempt runs in its own subprocess so the timeout is enforced reliably.
    The preflight budget is added on top of the first attempt's agent budget only.

    Resumptions are only triggered when an attempt exhausted ``max_steps`` — a
    timeout or crash is not retried (the subprocess was already killed / gave up).
    """
    attempt_timeouts = _resolve_attempt_timeouts(config.run)
    preflight_budget = config.run.preflight_timeout_seconds

    prior_summaries: list[str] = []
    cumulative_input = 0
    cumulative_output = 0
    preflight_result: dict = {}
    last_result: dict[str, Any] = _failure_run_result_payload(task_id, "No attempts were made.")

    for attempt_idx, attempt_timeout in enumerate(attempt_timeouts):
        # First attempt: add preflight budget.  Resumptions skip preflight.
        subprocess_budget = attempt_timeout + (preflight_budget if attempt_idx == 0 else 0)

        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue[Any] = ctx.Queue()
        process = ctx.Process(
            target=_run_one_attempt_in_subprocess,
            args=(task_id, config, prior_summaries, attempt_idx, queue),
        )
        process.start()
        process.join(subprocess_budget)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            run_result = _failure_run_result_payload(
                task_id,
                f"Attempt {attempt_idx + 1} timed out after {subprocess_budget}s.",
            )
        elif queue.empty():
            exit_code = process.exitcode
            msg = (
                f"Attempt {attempt_idx + 1} exited unexpectedly (code {exit_code})."
                if exit_code not in (None, 0)
                else f"Attempt {attempt_idx + 1} exited without returning a result."
            )
            run_result = _failure_run_result_payload(task_id, msg)
        else:
            item = queue.get()
            if item.get("ok"):
                run_result = dict(item["run_result"])
                cumulative_input += run_result.pop("input_tokens", 0)
                cumulative_output += run_result.pop("output_tokens", 0)
                if attempt_idx == 0:
                    preflight_result = run_result.pop("preflight", {})
                else:
                    run_result.pop("preflight", None)
            else:
                run_result = _failure_run_result_payload(
                    task_id,
                    f"Attempt {attempt_idx + 1} failed with uncaught error: {item.get('error', 'unknown')}",
                )

        last_result = run_result

        # Done — got an answer.
        if run_result.get("answer") is not None:
            break

        # Only resume after step-exhaustion, not after timeout/crash.
        failure_reason = run_result.get("failure_reason", "")
        if "max_steps" not in failure_reason.lower():
            break

        # Prepare summary for the next attempt (if one remains).
        if attempt_idx < len(attempt_timeouts) - 1:
            summary = summarise_trace_for_resumption(run_result)
            prior_summaries.append(summary)
            # Persist the intermediate attempt trace and resumption summary so
            # they can be inspected for debugging even if the final attempt
            # succeeds (and overwrites trace.json).
            task_output_dir = Path(config.run.output_dir) / config.run.run_id / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            attempt_num = attempt_idx + 1
            _write_json(task_output_dir / f"trace_attempt_{attempt_num}.json", run_result)
            (task_output_dir / f"resumption_summary_{attempt_num}.txt").write_text(
                summary, encoding="utf-8"
            )

    last_result["preflight"] = preflight_result
    last_result["input_tokens"] = cumulative_input
    last_result["output_tokens"] = cumulative_output
    return last_result


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Save preflight analysis separately so it can be inspected independently.
    preflight = run_result.pop("preflight", None)
    if preflight:
        _write_json(task_output_dir / "preflight.json", preflight)

    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
        input_tokens=run_result.get("input_tokens", 0),
        output_tokens=run_result.get("output_tokens", 0),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    # Step 1: run input detection before the agent starts
    task = DABenchPublicDataset(config.dataset.root_path).get_task(task_id)
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    input_files = detect_input_files(task)
    _write_json(task_output_dir / "input_detection.json", {"task_id": task_id, "input_files": input_files})

    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
