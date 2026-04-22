from __future__ import annotations

from data_agent_baseline.benchmark.schema import PublicTask

_EXT_TO_KIND: dict[str, str] = {
    ".csv": "csv",
    ".db": "db",
    ".sqlite": "db",
    ".json": "json",
    ".md": "doc",
}


def detect_input_files(task: PublicTask) -> dict[str, list[str]]:
    """Return context files grouped by type as relative paths from context_dir."""
    result: dict[str, list[str]] = {"csv": [], "db": [], "json": [], "doc": [], "other": []}

    context_dir = task.context_dir
    if not context_dir.exists():
        return result

    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        kind = _EXT_TO_KIND.get(path.suffix.lower(), "other")
        result[kind].append(path.relative_to(context_dir).as_posix())

    return result
