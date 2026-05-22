"""Self-consistency voting across N sub-runs of the same task.

When ``run.consistency_runs > 1``, the runner launches N independent agent
runs for each task.  Each writes its outputs under ``task_<id>/run_<i>/``.
After all N complete, ``vote_on_answer`` picks a winner and the runner
copies that sub-run's ``prediction.csv`` to ``task_<id>/prediction.csv``
so the existing evaluator path keeps working unchanged.

Vote semantics
--------------
1. Canonicalise each candidate prediction by value, ignoring column ORDER
   and column NAMES (same canonicalisation as ``scripts/measure_variance.py``).
2. If one canonical answer has a strict majority (> N/2) → it wins.
3. If two or more answers tie at the top → ask the LLM judge to pick.
4. If a tie-breaker call is unavailable or fails → fall back to the
   candidate with the fewest columns (penalty for extras) and the
   smallest row count (penalty for inflation), breaking ties by sub-run
   index for determinism.

The vote summary is written to ``task_<id>/voting_summary.json`` so the
decision is auditable.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_agent_baseline.run.evaluate import _read_csv_with_headers

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter


# ---------------------------------------------------------------------------
# Column-signature comparison (adaptive voting helpers)
# ---------------------------------------------------------------------------
#
# Adaptive voting compares two predictions at the COLUMN-VALUE level
# (matching the evaluator's logic — no dependency on column names).
# Each column becomes a sorted tuple of normalised values; a prediction
# becomes a frozenset of those tuples.  Two predictions are then either:
#   - identical:    signatures match exactly
#   - a_subset_of_b: a's signature is a strict subset of b's (b has extra cols)
#   - b_subset_of_a: b's signature is a strict subset of a's (a has extra cols)
#   - unclear:      neither identical nor subset (different content)
#   - no_predictions: both empty / missing
# When clear (identical, a_subset_of_b, b_subset_of_a) we can decide in two runs
# and skip phase-2 escalation. When unclear we escalate to 5 runs for full voting.


def _signature_set(pred_path: Path) -> frozenset:
    """Return a frozenset of per-column sorted-value tuples (evaluator-style).

    Two predictions with the same set of column-value tuples are considered
    equivalent regardless of column NAMES or ORDER — that matches the way the
    competition evaluator does column matching.
    """
    if not pred_path.exists():
        return frozenset()
    try:
        _headers, columns = _read_csv_with_headers(pred_path)
    except Exception:
        return frozenset()
    # Each column is already (sorted, normalised) by _read_csv_with_headers.
    return frozenset(tuple(col) for col in columns)


# Outcome labels for ``classify_signature_pair``.
SIG_IDENTICAL = "identical"
SIG_A_SUBSET_OF_B = "a_subset_of_b"   # a's columns are all in b; b has extras
SIG_B_SUBSET_OF_A = "b_subset_of_a"   # b's columns are all in a; a has extras
SIG_UNCLEAR = "unclear"
SIG_NO_PREDICTIONS = "no_predictions"


def classify_signature_pair(sig_a: frozenset, sig_b: frozenset) -> str:
    """Compare two prediction signatures and return one of SIG_* constants.

    Note: when only one side has predictions and the other is empty, this is
    considered ``unclear`` (escalate to phase 2). The user's adaptive design
    requires identical/subset/subset-reversed for phase-1 termination.
    """
    if not sig_a and not sig_b:
        return SIG_NO_PREDICTIONS
    if sig_a == sig_b:
        return SIG_IDENTICAL
    # Strict subset (b has extras beyond a).
    if sig_a and sig_a < sig_b:
        return SIG_A_SUBSET_OF_B
    if sig_b and sig_b < sig_a:
        return SIG_B_SUBSET_OF_A
    return SIG_UNCLEAR


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Candidate:
    subrun_idx: int
    prediction_path: Path
    canonical: str | None  # None when prediction is missing
    headers: list[str]
    n_columns: int
    n_rows: int
    succeeded: bool
    failure_reason: str | None


def _canonicalise(pred_path: Path) -> tuple[str | None, list[str], int, int]:
    """Return (canonical_str, headers, n_columns, n_rows) for a prediction CSV.

    Canonical form is a JSON-encoded sorted list of sorted columns (each column
    a sorted tuple of normalised values).  Ignores column order and names so
    two runs that produced the same data in different layouts compare equal.

    Returns (None, [], 0, 0) when the file is missing or unreadable.
    Returns ("<empty>", headers, n_cols, 0) for header-only files.
    """
    if not pred_path.exists():
        return None, [], 0, 0
    try:
        headers, columns = _read_csv_with_headers(pred_path)
    except Exception:
        return None, [], 0, 0
    n_cols = len(headers)
    n_rows = max((len(c) for c in columns), default=0)
    if not columns or n_rows == 0:
        return "<empty>", headers, n_cols, 0
    canonical = sorted(list(c) for c in columns)
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False), headers, n_cols, n_rows


def _load_candidates(task_output_dir: Path, n_runs: int) -> list[_Candidate]:
    """Read all N sub-run predictions for a task."""
    candidates: list[_Candidate] = []
    for i in range(n_runs):
        subdir = task_output_dir / f"run_{i}"
        pred_path = subdir / "prediction.csv"
        canonical, headers, n_cols, n_rows = _canonicalise(pred_path)
        # Pull the sub-run's trace to learn success/failure.
        trace_path = subdir / "trace.json"
        succeeded = False
        failure_reason: str | None = None
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                succeeded = bool(trace.get("succeeded"))
                failure_reason = trace.get("failure_reason")
            except Exception:
                pass
        candidates.append(
            _Candidate(
                subrun_idx=i,
                prediction_path=pred_path,
                canonical=canonical,
                headers=headers,
                n_columns=n_cols,
                n_rows=n_rows,
                succeeded=succeeded,
                failure_reason=failure_reason,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# LLM judge for tie-breaking
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are a careful data-analysis judge.  Given a question and several "
    "candidate answers, your job is to pick the answer most likely to be "
    "correct based on the question's intent.\n\n"
    "Apply these structural cues in priority order:\n"
    "1. Aggregate questions ('how many', 'what is the total', 'what is the "
    "average', 'what percentage', 'count of') expect EXACTLY one column "
    "and one row.\n"
    "2. 'List X' / 'Which X' / 'Give the X' questions expect a single column "
    "(the X being asked for), not the X plus an ID or context column.\n"
    "3. 'List X and Y' expects exactly 2 columns.\n"
    "4. Ranking questions (highest, lowest, best, worst, fastest, slowest) "
    "may legitimately return multiple tied rows.\n"
    "5. Empty answers are almost always wrong unless the question explicitly "
    "asks about non-existence.\n\n"
    "Reply with EXACTLY one line in the form:\n"
    "  PICK: <subrun_index>\n"
    "where <subrun_index> is the integer index of the candidate you select. "
    "No other text, no explanation."
)


def _llm_judge(
    question: str,
    candidates: list[_Candidate],
    model: "ModelAdapter | None",
) -> int | None:
    """Ask the LLM to pick one candidate. Returns subrun_idx or None on failure."""
    if model is None or not candidates:
        return None
    # Import lazily to avoid circular dependencies at module load.
    from data_agent_baseline.agents.model import ModelMessage  # noqa: PLC0415

    # Build a compact textual representation of each candidate.
    blocks: list[str] = []
    for cand in candidates:
        # Read up to first 6 data rows for preview.
        preview_rows: list[list[str]] = []
        if cand.prediction_path.exists():
            try:
                import csv  # local
                with cand.prediction_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    rows = list(reader)
                preview_rows = rows[1:7] if len(rows) > 1 else []
            except Exception:
                preview_rows = []
        block = (
            f"--- candidate subrun_index={cand.subrun_idx} ---\n"
            f"columns: {cand.headers}\n"
            f"n_columns: {cand.n_columns}  n_rows: {cand.n_rows}\n"
            f"first rows: {preview_rows}\n"
        )
        blocks.append(block)

    user_msg = (
        f"Question: {question}\n\n"
        f"Candidate answers:\n\n"
        + "\n".join(blocks)
        + "\nPick the candidate index most likely to be correct."
    )
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
                ModelMessage(role="user", content=user_msg),
            ],
            extra_body={"enable_thinking": False},
        ).strip()
    except Exception:
        return None
    # Parse "PICK: N"
    for token in raw.replace(":", " ").split():
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < len(candidates):
                return idx
    return None


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def _heuristic_pick(candidates: list[_Candidate]) -> int:
    """Deterministic fallback: prefer succeeded runs with the fewest columns
    and fewest rows.  Tie-break by subrun_idx ascending.

    Rationale:
      - succeeded runs are more trustworthy than failed (no-prediction) ones.
      - extra columns are penalised by the scorer, so fewer is better.
      - inflated row counts (e.g. 154 rows when 9 were expected) suggest
        a missing filter.  Fewer rows is usually closer to correct when
        all else is equal.
    """
    if not candidates:
        return 0

    def _key(c: _Candidate) -> tuple[int, int, int, int]:
        # Lower is better for each tuple element.
        missing = 0 if (c.canonical is not None and c.canonical != "<empty>") else 1
        failed = 0 if c.succeeded else 1
        return (missing, failed, c.n_columns, c.subrun_idx)

    return min(candidates, key=_key).subrun_idx


def classify_phase1(
    *,
    task_id: str,
    task_output_dir: Path,
) -> dict[str, Any]:
    """Inspect the two phase-1 sub-run predictions and decide whether to
    terminate adaptive voting at phase 1 (clear outcome) or escalate to phase 2.

    Returns a dict:
      {
        "task_id": ..., "phase": "1",
        "outcome": <SIG_*>,
        "terminate": bool,
        "winner_subrun_idx": int | None,  # only set when terminate=True
        "sig_0_size": int, "sig_1_size": int,
      }

    Termination cases (per user-confirmed design):
      - SIG_IDENTICAL                → take subrun 0
      - SIG_A_SUBSET_OF_B (a smaller) → take subrun 0 (the minimal one)
      - SIG_B_SUBSET_OF_A (b smaller) → take subrun 1 (the minimal one)

    Escalation cases (proceed to phase 2):
      - SIG_UNCLEAR
      - SIG_NO_PREDICTIONS  (neither phase-1 sub-run produced output)
      - Anything else (incl. only-one-has-prediction; the user's design treats
        that as not-clearly-resolved → escalate)

    This function does NOT copy any prediction.csv yet; the caller decides
    whether to copy the winner (on terminate) or wait for phase 2.
    """
    pred_0 = task_output_dir / "run_0" / "prediction.csv"
    pred_1 = task_output_dir / "run_1" / "prediction.csv"
    sig_0 = _signature_set(pred_0)
    sig_1 = _signature_set(pred_1)
    outcome = classify_signature_pair(sig_0, sig_1)

    terminate = False
    winner_idx: int | None = None
    if outcome == SIG_IDENTICAL:
        terminate = True
        winner_idx = 0
    elif outcome == SIG_A_SUBSET_OF_B:
        # a (subrun 0) has the minimal column set → take it (strips extras).
        terminate = True
        winner_idx = 0
    elif outcome == SIG_B_SUBSET_OF_A:
        # b (subrun 1) has the minimal column set → take it.
        terminate = True
        winner_idx = 1

    return {
        "task_id": task_id,
        "phase": "1",
        "outcome": outcome,
        "terminate": terminate,
        "winner_subrun_idx": winner_idx,
        "sig_0_size": len(sig_0),
        "sig_1_size": len(sig_1),
    }


def write_phase1_winner(
    *,
    task_id: str,
    task_output_dir: Path,
    classification: dict[str, Any],
) -> None:
    """Copy the phase-1 winning sub-run's prediction.csv to the top level and
    write a voting_summary.json reflecting the phase-1 decision."""
    winner_idx = classification.get("winner_subrun_idx")
    if winner_idx is None:
        return  # nothing to copy; the caller should escalate

    winner_path = task_output_dir / f"run_{winner_idx}" / "prediction.csv"
    final_path = task_output_dir / "prediction.csv"
    if winner_path.exists():
        final_path.write_bytes(winner_path.read_bytes())
    else:
        if final_path.exists():
            final_path.unlink(missing_ok=True)

    # Build a minimal summary so audits look like the standard voting case.
    candidates: list[dict] = []
    for i in (0, 1):
        sub = task_output_dir / f"run_{i}"
        pred = sub / "prediction.csv"
        canonical, headers, n_cols, n_rows = _canonicalise(pred)
        candidates.append({
            "subrun_idx": i,
            "n_columns": n_cols,
            "n_rows": n_rows,
            "headers": headers,
            "has_prediction": canonical is not None,
            "canonical": (
                canonical[:300] + "…" if canonical and len(canonical) > 300 else canonical
            ),
        })

    summary = {
        "task_id": task_id,
        "n_runs": 2,
        "phase": "1",
        "decision": f"phase1_{classification['outcome']}",
        "winner_subrun_idx": winner_idx,
        "candidates": candidates,
    }
    (task_output_dir / "voting_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def vote_on_answer(
    *,
    task_id: str,
    task_output_dir: Path,
    question: str,
    n_runs: int,
    model: "ModelAdapter | None" = None,
) -> dict[str, Any]:
    """Read the N sub-run predictions, vote on a winner, and write a summary.

    The winning candidate's ``prediction.csv`` is copied to
    ``<task_output_dir>/prediction.csv`` so the evaluator can find it.

    Returns a summary dict (also persisted as ``voting_summary.json``).
    """
    candidates = _load_candidates(task_output_dir, n_runs)

    # Count canonical answers.
    counter: Counter[str] = Counter()
    for c in candidates:
        if c.canonical is not None:
            counter[c.canonical] += 1

    decision: str
    winner_idx: int
    if not counter:
        # All N sub-runs missing predictions — fall back to heuristic.
        decision = "all_missing_heuristic"
        winner_idx = _heuristic_pick(candidates)
    else:
        # Find top vote count.
        ranked = counter.most_common()
        top_count = ranked[0][1]
        top_candidates_canonicals = [c for c, n in ranked if n == top_count]
        majority_threshold = (n_runs // 2) + 1
        if top_count >= majority_threshold and len(top_candidates_canonicals) == 1:
            # Clear majority — pick the lowest subrun_idx with that canonical.
            winning_canonical = top_candidates_canonicals[0]
            winner_idx = next(
                c.subrun_idx for c in candidates if c.canonical == winning_canonical
            )
            decision = "majority"
        elif len(top_candidates_canonicals) == 1:
            # Plurality but no strict majority — still pick it.
            winning_canonical = top_candidates_canonicals[0]
            winner_idx = next(
                c.subrun_idx for c in candidates if c.canonical == winning_canonical
            )
            decision = "plurality"
        else:
            # Top is a tie between distinct canonicals — use LLM judge or heuristic.
            tied_candidates = [
                c for c in candidates if c.canonical in top_candidates_canonicals
            ]
            judge_pick = _llm_judge(question, tied_candidates, model)
            if judge_pick is not None:
                winner_idx = judge_pick
                decision = "llm_judge"
            else:
                winner_idx = _heuristic_pick(tied_candidates)
                decision = "heuristic_tiebreak"

    # Copy the winning sub-run's prediction.csv to the top level.
    winner_path = task_output_dir / f"run_{winner_idx}" / "prediction.csv"
    final_path = task_output_dir / "prediction.csv"
    if winner_path.exists():
        # Use binary copy so identical content is preserved exactly.
        final_path.write_bytes(winner_path.read_bytes())
    else:
        # Winner had no prediction file — remove any stale top-level file so the
        # evaluator treats this task as "no prediction".
        if final_path.exists():
            final_path.unlink(missing_ok=True)

    summary = {
        "task_id": task_id,
        "n_runs": n_runs,
        "decision": decision,
        "winner_subrun_idx": winner_idx,
        "candidates": [
            {
                "subrun_idx": c.subrun_idx,
                "succeeded": c.succeeded,
                "failure_reason": c.failure_reason,
                "n_columns": c.n_columns,
                "n_rows": c.n_rows,
                "headers": c.headers,
                "has_prediction": c.canonical is not None,
                "canonical": (
                    c.canonical[:300] + "…" if c.canonical and len(c.canonical) > 300
                    else c.canonical
                ),
            }
            for c in candidates
        ],
        "vote_counts": dict(counter),
    }
    (task_output_dir / "voting_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
