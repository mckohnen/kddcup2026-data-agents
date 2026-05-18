# CLAUDE.md — KDD Cup 2026 Data Agents (Team team1210)

This file is read by Claude Code at the start of every session. Keep it current.

**Maintenance rule:** After completing any change that affects architecture, tools, agents, commands, or the submission workflow, flag it to the user with: *"This change may be worth reflecting in CLAUDE.md — want me to update it?"* Do not update this file unilaterally. Do not flag it for bug fixes, prompt tuning, or minor improvements.

## What this is

Competition fork of the official KDD Cup 2026 DataAgent-Bench starter kit.
Team **team1210 "Zero Sugar Full Intelligence"**.
Rules and submission spec: https://dataagent.top/rules

The agent receives tabular-data tasks and must produce a `prediction.csv` answering each one.

## Repository layout

```
src/data_agent_baseline/
├── agents/
│   ├── react.py          # Generic ReAct loop (Thought → Action → Observation)
│   ├── data_agent.py     # General-purpose agent wrapping ReActAgent (all difficulties)
│   ├── model.py          # OpenAI-compatible model adapter (retry, timeout, token tracking)
│   ├── prompt.py         # System / task / observation prompt builders
│   └── runtime.py        # AgentRunResult, AgentRuntimeState, StepRecord
├── benchmark/
│   ├── dataset.py        # DABenchPublicDataset — scans task_*/ directories
│   └── schema.py         # PublicTask, TaskAssets, TaskRecord, AnswerTable
├── tools/
│   ├── registry.py       # ToolRegistry, ToolSpec, ToolExecutionResult
│   ├── filesystem.py     # list_context, read_csv, read_json, read_doc
│   ├── python_exec.py    # execute_python tool
│   ├── sqlite.py         # inspect_sqlite_schema, execute_context_sql (original)
│   ├── context_sqlite.py # get_context_schema, run_sql_on_context (in-memory SQLite variant)
│   └── input_detector.py # Classify context files by type before running agent
├── run/
│   ├── runner.py         # run_single_task, run_benchmark, TaskRunArtifacts
│   ├── executor.py       # TaskExecutor
│   └── evaluate.py       # evaluate_run, write_evaluation_report
├── config.py             # AppConfig, AgentConfig, DatasetConfig, RunConfig + YAML loader
└── cli.py                # Typer CLI entry point (dabench run-benchmark)

configs/                  # YAML config files (see react_baseline.example.yaml)
submit.py                 # Docker submission entry point (reads env vars, not YAML)
Dockerfile                # Competition image definition
scripts/build_submission.sh  # Build + package image as <team_id>_v<N>.tar.gz
```

## Architecture

### Agent flow

```
task.json + context/  →  InputDetector  →  DataAgent  →  prediction.csv
                                               ↓
                                    preflight schema analysis
                                               ↓
                                         ReActAgent loop
                                   (Thought → Action → Observation)
                                               ↓
                                   JSON action protocol:
                                   { "thought": "...", "action": "...", "action_input": {...} }
                                               ↓
                                        ToolRegistry
                         read_doc | show_context_schema | query_context_tables | answer
```

### Key design decisions

- **JSON action protocol**: the model must emit a single JSON object per step. `parse_model_step` in `react.py` strips code fences and validates the schema.
- **DataAgent** (`data_agent.py`) handles tasks of all difficulties. It runs a preflight schema-analysis step before the ReAct loop to build a task hint injected into the first user message.
- **Input detection** runs before the agent on every task and writes `input_detection.json` alongside the trace. It classifies context files into csv / db / json / doc / other.
- **Parallel execution** uses `ThreadPoolExecutor` (benchmark) and `multiprocessing` per task (for timeout enforcement). Pass `model=` or `tools=` to `run_benchmark` to force single-worker mode (for shared state).
- **SQLite in-memory connections** in `context_sqlite.py` are created with `check_same_thread=False` so they can be shared between the preflight thread and the main ReAct thread safely.
- **Model resilience** (`model.py`): API calls are wrapped in up to 6 retry attempts with exponential backoff + jitter for both `RateLimitError` and `APIConnectionError`. A hard `httpx.Timeout(read=120s)` prevents hung connections from blocking a task indefinitely.
- **Answer critic**: after the model proposes an `answer` action, a second model call checks the structural shape (extra columns, wrong scalar shape) before the answer is accepted. Factual correctness is not checked.

## Local development

### Setup

```bash
uv sync
```

### Config files

Copy the example and fill in your credentials:

```bash
cp configs/react_baseline.example.yaml configs/react_baseline.local.yaml
# edit: set api_base, api_key; adjust run_id before each new run
```

Key config parameters (see `react_baseline.example.yaml` for annotated defaults):

| Parameter | Description |
|---|---|
| `dataset.root_path` | Path to the `task_*/` input directories |
| `agent.max_steps` | Max ReAct iterations per task (default 16) |
| `run.run_id` | Unique identifier — bump before each run to avoid overwriting |
| `run.max_workers` | Parallel task workers |
| `run.task_timeout_seconds` | Wall-clock limit per task (default 900 s) |
| `run.preflight_timeout_seconds` | Budget for schema pre-analysis (default 60 s) |
| `run.max_resumptions` | How many times a timed-out task may be resumed (default 2) |

### Available configs

| Config file | Dataset | Purpose |
|---|---|---|
| `react_baseline.local.yaml` | `data/public/input` (all 50 tasks) | Full local test / submission prep |
| `react_baseline.imperfect.yaml` | `data/imperfect/input` (16 tasks) | Fast iteration on previously failing tasks |
| `react_baseline.test3.yaml` | `data/test3/input` | Test3 dataset runs |

### Run benchmark

```bash
uv run dabench run-benchmark --config configs/react_baseline.local.yaml
# add --limit N to cap task count
```

### Run evaluation (needs gold files)

Evaluation runs automatically at the end of `run-benchmark` if a matching `evaluation/` directory exists next to `input/` (e.g. `data/public/evaluation/task_<id>/gold.csv`).

The local evaluator matches competition scoring:
- **Numeric precision**: cell values are normalized to 2 decimal places before comparison.
- **Name-column equivalence**: if gold has explicit `first_name` + `last_name` columns, a prediction with a single combined full-name column is treated as correct.

### Run a single task (Python API)

```python
from pathlib import Path
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import create_run_output_dir, run_single_task

config = load_app_config(Path("configs/react_baseline.local.yaml"))
_, run_output_dir = create_run_output_dir(config.run.output_dir, run_id="debug")
artifact = run_single_task(task_id="task_163", config=config, run_output_dir=run_output_dir)
print(artifact)
```

## Submission

### Build and package

```bash
./scripts/build_submission.sh team1210 <N>
# produces: team1210_v<N>.tar.gz
```

Version numbers must increment; prior versions cannot be reused.
Max image size: 10 GB. Max 1 submission/day; 30 total in Phase 1.

### Email to organizers

Subject: `[KDDCup2026 Data Agents] Submission - team1210 - v<N>`
Body: team ID, version number, Google Drive shareable link (set to "Anyone with link can view").

### How the evaluator runs the container

```bash
docker run --rm \
  --cpus=16 --memory=64g \
  -v /host/input:/input:ro \
  -v /host/output:/output \
  -v /host/logs:/logs \
  -e MODEL_API_URL=<url> \
  -e MODEL_API_KEY=<key> \
  -e MODEL_NAME=qwen3.5-35b-a3b \
  team1210:vN
```

`submit.py` reads these three env vars and writes `/output/task_<id>/prediction.csv` per task.

### Test the container locally

```bash
docker build -t team1210:vlocal .
docker run --rm \
  -v "$(pwd)/data/public/input:/input:ro" \
  -v "$(pwd)/artifacts/docker_out:/output" \
  -v "$(pwd)/artifacts/docker_logs:/logs" \
  -e MODEL_API_URL=<your_url> \
  -e MODEL_API_KEY=<your_key> \
  -e MODEL_NAME=<your_model> \
  team1210:vlocal
```

## Competition constraints (affect all code changes)

- **No internet at eval time.** Only `MODEL_API_URL` is reachable. No HuggingFace downloads, no pip installs at runtime.
- **No hardcoded credentials.** `submit.py` reads everything from env vars.
- **`/input/` is read-only.** Never write to it.
- **No GPU.** All compute is CPU-only.
- **12-hour total wall-clock limit** across all tasks. Per-task timeout in config is 900 s.
- **Scoring:** column-level matching, `Score = Recall − λ × (ExtraColumns / PredictedColumns)`. Extra columns hurt — only output what the question asks for.
- The model name injected is always `qwen3.5-35b-a3b`; do not hard-code any other model name in submission code.

## Inviolable rules for all code and prompt changes

1. **Never modify task data.** The files under `data/` (task.json, context files, knowledge.md, gold CSVs, etc.) must never be edited to help the agent. They are read-only competition inputs.
2. **Prompt and tool changes must be generalizable.** No dataset-specific instructions, no hardcoded column names, topics, or domain knowledge. Changes must work correctly across all tasks.
3. **No rounding in agent output.** The competition normalizes numerics to 2 dp at scoring time. Never use `ROUND()`, `FORMAT()`, or Python's `round()` in the agent's SQL or Python — return raw computed values.
4. **Name column format is irrelevant.** The competition accepts both split (`first_name` + `last_name`) and combined (`full_name`) forms. Do not add instructions that force one format over the other.
