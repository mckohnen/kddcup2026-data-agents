from __future__ import annotations

import logging
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_task_logger(log_path: Path, *, attempt: int = 1) -> None:
    """Configure the 'agent' logger to write to *log_path* (appended).

    Safe to call multiple times (once per attempt inside a subprocess).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("=== ATTEMPT %d START ===", attempt)


def get_logger() -> logging.Logger:
    return logging.getLogger("agent")


def close_task_logger() -> None:
    logger = logging.getLogger("agent")
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
