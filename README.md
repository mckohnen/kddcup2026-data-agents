<div align="center">

# KDD Cup 2026 Data Agents — Team team1210

[![Official Website](https://img.shields.io/badge/Official%20Website-dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

Team **"Zero Sugar Full Intelligence"** (team1210) — competition fork of the [official KDD Cup 2026 DataAgent-Bench starter kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit).

The agent receives tabular-data tasks, reasons over the provided context files, and writes a `prediction.csv` answering each question.

## Quick start

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python ≥ 3.10.

```bash
# 1. Install dependencies
uv sync

# 2. Create a local config from the example
cp configs/react_baseline.example.yaml configs/react_baseline.local.yaml
# Edit the copy: set model, api_base, api_key, and dataset root path

# 3. Run the benchmark against the public dataset
uv run dabench run-benchmark --config configs/react_baseline.local.yaml

# Optional: cap the number of tasks
uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 10
```

`configs/react_baseline.local.yaml` is git-ignored — never commit API keys.

## Dataset

Download the Phase 1 public demo dataset and place it at `data/public/input/`.

Each task directory has this structure:

```
data/public/input/task_<id>/
├── task.json        # task_id, difficulty, question
└── context/         # csv/, db/, json/, doc/ files — varies per task
```

Public ground-truth answers (for local evaluation) live at `data/public/evaluation/task_<id>/gold.csv`.
Evaluation runs automatically at the end of `run-benchmark` when that directory is present.

Run outputs are written to:

```
artifacts/runs/<run_id>/
├── summary.json
└── task_<id>/
    ├── trace.json          # full agent step trace
    ├── prediction.csv      # answer table
    └── input_detection.json
```

## Configuration

```yaml
dataset:
  root_path: data/public/input   # relative to project root, or absolute

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:                        # leave blank for a UTC timestamp; must be unique
  max_workers: 4
  task_timeout_seconds: 600
```

## Architecture

```
task.json + context/
       │
       ▼
 InputDetector          classifies context files (csv / db / json / doc)
       │
       ▼
 EasyTaskAgent          specialised for "easy" difficulty
       │
       ▼
 ReActAgent loop        Thought → Action (JSON) → Observation → repeat
       │
       ▼
 ToolRegistry           routes actions to tool implementations
       │
       ▼
 prediction.csv
```

**Tools available to the easy-task agent:**

| Tool | Purpose |
| --- | --- |
| `read_doc` | Read a text/markdown file (primarily `knowledge.md`) |
| `show_context_schema` | Load all JSON/CSV files into in-memory SQLite and return schema + sample rows |
| `query_context_tables` | Run a SQL query over the in-memory SQLite tables |
| `answer` | Submit the final result table (terminates the agent) |

Only `easy` difficulty tasks are handled in the current version.

## Submission

### Build and package the Docker image

```bash
./scripts/build_submission.sh team1210 <N>
# Produces: team1210_v<N>.tar.gz  (upload this to Google Drive)
```

Version numbers must increment; max image size is 10 GB; max 1 submission per day.

### Email the organizers

```
Subject: [KDDCup2026 Data Agents] Submission - team1210 - v<N>
```

Include: team ID, version number, shareable Google Drive link (set to "Anyone with link can view").

### Test the container locally before submitting

```bash
docker build -t team1210:vlocal .
docker run --rm \
  -v "$(pwd)/data/public/input:/input:ro" \
  -v "$(pwd)/artifacts/docker_out:/output" \
  -v "$(pwd)/artifacts/docker_logs:/logs" \
  -e MODEL_API_URL=<url> \
  -e MODEL_API_KEY=<key> \
  -e MODEL_NAME=<model> \
  team1210:vlocal
```

The evaluator injects `MODEL_API_URL`, `MODEL_API_KEY`, and `MODEL_NAME` at runtime — no credentials are baked into the image.

## Key modules

| Module | Responsibility |
| --- | --- |
| `src/data_agent_baseline/agents/react.py` | Generic ReAct loop with JSON action protocol |
| `src/data_agent_baseline/agents/easy_task_agent.py` | Easy-task agent and its tool registry |
| `src/data_agent_baseline/agents/model.py` | OpenAI-compatible model adapter |
| `src/data_agent_baseline/benchmark/dataset.py` | Dataset loader (`DABenchPublicDataset`) |
| `src/data_agent_baseline/tools/context_sqlite.py` | In-memory SQLite over JSON/CSV context |
| `src/data_agent_baseline/tools/input_detector.py` | Context file classifier |
| `src/data_agent_baseline/run/runner.py` | Single-task and benchmark execution |
| `src/data_agent_baseline/run/evaluate.py` | Local scoring against gold CSVs |
| `src/data_agent_baseline/config.py` | Config dataclasses and YAML loader |
| `submit.py` | Docker entrypoint (reads env vars, not YAML) |
