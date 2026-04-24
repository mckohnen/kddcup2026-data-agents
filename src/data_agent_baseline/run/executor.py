from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.input_detector import detect_input_files


class TaskExecutor:
    def __init__(self, output_root: Path, run_id: str | None = None):
        self.run_id = self._resolve_run_id(run_id)
        self.run_output_dir = output_root / self.run_id

    @staticmethod
    def _create_run_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _resolve_run_id(run_id: str | None = None) -> str:
        if run_id is None:
            return TaskExecutor._create_run_id()

        normalized = run_id.strip()
        if not normalized:
            raise ValueError("run_id must not be empty.")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("run_id must be a single directory name, not a path.")
        return normalized

    def execute_task(self, task: PublicTask) -> Path:
        task_output_dir = self.run_output_dir / task.task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        input_files = detect_input_files(task)

        result = {
            "task_id": task.task_id,
            "input_files": input_files,
        }

        result_path = task_output_dir / "input_detection.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        return task_output_dir
