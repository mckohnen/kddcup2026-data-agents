#!/usr/bin/env python3
"""Docker submission entrypoint for KDD Cup 2026 Data Agents challenge.

Reads MODEL_API_URL, MODEL_API_KEY, MODEL_NAME from environment variables,
runs the benchmark against /input/, and writes prediction CSVs to /output/.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
LOGS_DIR = Path("/logs")
_RUN_SUBDIR = "run"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    model_api_url = os.environ["MODEL_API_URL"]
    model_api_key = os.environ["MODEL_API_KEY"]
    model_name = os.environ["MODEL_NAME"]

    from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
    from data_agent_baseline.run.runner import TaskRunArtifacts, run_benchmark

    config = AppConfig(
        dataset=DatasetConfig(root_path=INPUT_DIR),
        agent=AgentConfig(
            model=model_name,
            api_base=model_api_url,
            api_key=model_api_key,
            max_steps=16,
            temperature=0.0,
        ),
        run=RunConfig(
            output_dir=OUTPUT_DIR,
            run_id=_RUN_SUBDIR,
            max_workers=4,
            # 3 escalating attempts per task: 400 s → 600 s → 800 s.
            # Resumptions are only triggered on max_steps exhaustion, not timeout.
            task_timeout_seconds_per_attempt=(400, 600, 800),
            preflight_timeout_seconds=30,
        ),
    )

    def on_task_complete(artifact: TaskRunArtifacts) -> None:
        # Copy prediction immediately so already-completed tasks survive SIGTERM.
        if artifact.prediction_csv_path is not None:
            dest = OUTPUT_DIR / artifact.task_id / "prediction.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.prediction_csv_path, dest)
            log.info("prediction written: %s", dest)
        else:
            log.warning("no prediction for %s: %s", artifact.task_id, artifact.failure_reason)

    log.info(
        "starting: input=%s output=%s model=%s workers=%d attempts=%s",
        INPUT_DIR,
        OUTPUT_DIR,
        model_name,
        config.run.max_workers,
        config.run.task_timeout_seconds_per_attempt,
    )

    _, artifacts = run_benchmark(config=config, progress_callback=on_task_complete)

    succeeded = sum(1 for a in artifacts if a.succeeded)
    log.info("done: %d/%d tasks succeeded", succeeded, len(artifacts))


if __name__ == "__main__":
    main()
