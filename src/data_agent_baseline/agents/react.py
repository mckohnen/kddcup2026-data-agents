from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.task_logger import get_logger
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

_RECOVERY_HINT = (
    "Your previous response could not be parsed as a valid JSON action. "
    "You MUST respond with exactly one ```json block containing a single JSON object:\n"
    "```json\n"
    '{\"thought\": \"<your reasoning>\", \"action\": \"<tool_name>\", \"action_input\": {...}}\n'
    "```\n"
    "Do not add any text before or after the code block."
)

_RECOVERY_HINT_ESCALATED = (
    "CRITICAL: You have now produced invalid JSON multiple times in a row. "
    "Stop whatever you were doing and output ONLY this minimal valid action — "
    "copy it exactly, changing nothing except the thought string:\n"
    "```json\n"
    '{\"thought\": \"Recovering: will re-examine schema before continuing.\", '
    '\"action\": \"show_context_schema\", \"action_input\": {}}\n'
    "```\n"
    "No text before. No text after. Only the ```json block."
)


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _escape_literal_control_chars(text: str) -> str:
    """Escape unescaped control characters inside JSON string values.

    When a model emits multi-line Python code inside a JSON string without
    properly escaping newlines, json.JSONDecoder.raw_decode() raises
    JSONDecodeError before any other repair can run.  This pass walks the text
    character-by-character and replaces bare \\n / \\r / \\t inside string
    values with their JSON escape sequences so downstream repairs can proceed.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and in_string:
            # Already-escaped sequence — copy both chars verbatim.
            result.append(c)
            i += 1
            if i < len(text):
                result.append(text[i])
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and c == "\n":
            result.append("\\n")
        elif in_string and c == "\r":
            result.append("\\r")
        elif in_string and c == "\t":
            result.append("\\t")
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _try_repair_json(raw_response: str) -> str | None:
    """Attempt lightweight repair of common LLM JSON formatting errors.

    Handles three cases in order:
    0. Literal control characters (bare newlines/tabs) inside string values —
       happens when the model emits multi-line execute_python code without
       escaping.  raw_decode() rejects these before any other repair runs.
    1. Trailing commas before } or ]  — e.g. {"a":1,}
    2. Trailing closing braces after a valid object  — e.g. {"a":1}}
       Some models (qwen3.5) emit an extra } after the JSON object when the
       action_input itself contains nested braces.  raw_decode() parses the
       first complete object and leaves the remainder as "}"; we strip it.
    Returns a repaired string on success, or None if the response is
    unrecoverable.
    """
    text = _strip_json_fence(raw_response)

    # Repair 0: escape literal newlines/tabs inside JSON string values so that
    # raw_decode() can at least parse the structure.  Without this, Repair 2
    # (trailing-brace strip) never gets a chance to run because raw_decode()
    # throws JSONDecodeError on the first unescaped newline it encounters.
    text = _escape_literal_control_chars(text)

    # Repair 1: trailing commas before closing braces / brackets
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)

    try:
        _load_single_json_object(cleaned)
        return cleaned
    except (json.JSONDecodeError, ValueError):
        pass

    # Repair 2: strip trailing bare } or ] characters that appear after a
    # complete JSON object.  raw_decode gives us the end index of the first
    # valid object; anything after that which is only whitespace and closing
    # braces/brackets is dropped.
    try:
        _, end = json.JSONDecoder().raw_decode(cleaned)
        remainder = cleaned[end:].strip()
        if remainder and re.fullmatch(r"[}\]]+", remainder):
            trimmed = cleaned[:end]
            _load_single_json_object(trimmed)
            return trimmed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


_CRITIC_SYSTEM_PROMPT = (
    "You are a structural data quality critic. "
    "Your ONLY job is to check the SHAPE of the proposed answer — not its content, values, or "
    "column naming style.\n\n"
    "Check ONLY these two structural issues:\n"
    "1. Too many columns: did the agent include columns that were NOT explicitly requested?\n"
    "   Examples to flag:\n"
    "   - 'What is the name?' answered with [name, id] → flag the id column\n"
    "   - 'List the values' answered with [value, count] → flag count (unless question asked for it)\n"
    "   - 'What is the average?' answered with [average, total, count] → flag total and count\n"
    "2. Wrong scalar shape: ONLY flag this when the question uses EXPLICIT aggregation language "
    "such as 'how many', 'what is the total', 'what is the count', 'what is the average', "
    "'what is the sum', 'what percentage'. These expect exactly 1 row and 1 column.\n"
    "   NEVER flag multi-row answers for questions using 'what is the X of Y', 'list', 'which', "
    "   'find', 'identify', or any phrasing that does NOT explicitly ask for a single aggregate "
    "   — multiple matching entities are a perfectly valid answer.\n\n"
    "CRITICAL — do NOT flag any of the following:\n"
    "- Column naming style: aliases, SQL expressions like AVG(...), source column names — all fine\n"
    "- Zero rows: 0 rows is a perfectly valid answer when no data matches the filter\n"
    "- Multiple rows for non-aggregate questions: e.g. 'What is the product code?' or "
    "  'What is the category?' can legitimately return multiple rows when multiple entities match "
    "  — do NOT assume singular phrasing means exactly one result\n"
    "- Factual correctness: you have NO access to ground truth. Never use your training knowledge "
    "  to override what the SQL returned (e.g. do not say 'X should be Y based on your knowledge')\n"
    "- Column names not matching schema candidates: schema candidates are heuristic and incomplete\n"
    "- Aggregated values: computing SUM, AVG, COUNT is valid even if the column is named differently\n\n"
    "Reply with exactly one of:\n"
    "  OK\n"
    "  CONCERN: <one sentence describing the specific structural problem>\n"
    "No other text."
)


def _run_critic_check(
    model: ModelAdapter,
    question: str,
    columns: list,
    rows: list,
    task_analysis: dict | None = None,
) -> str | None:
    """Call the model as a structural critic to review a proposed answer.

    Checks only shape issues (extra columns, wrong scalar shape).
    Does NOT validate column names, factual correctness, or zero-row results.
    The ``task_analysis`` parameter is kept for API compatibility but is no longer
    injected into the critic prompt — schema candidates are heuristic and caused
    false positives blocking correct answers.

    Returns a concern string if a structural issue is found, or None if OK.
    Errors are silently swallowed so they never block a valid answer.
    """
    preview_rows = rows[:5]

    critic_user = (
        f"Question: {question}\n"
        f"Proposed answer — columns: {columns}, "
        f"rows (first {len(preview_rows)} of {len(rows)}): {preview_rows}\n\n"
        "Is there a structural problem with this answer?"
    )
    try:
        response = model.complete(
            [
                ModelMessage(role="system", content=_CRITIC_SYSTEM_PROMPT),
                ModelMessage(role="user", content=critic_user),
            ]
        )
        response = response.strip()
        if response.upper().startswith("CONCERN"):
            return response
    except Exception:
        pass
    return None


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        enable_answer_critic: bool = True,
        task_hint: str | None = None,
        task_analysis: dict | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.enable_answer_critic = enable_answer_critic
        # Pre-flight task analysis hint injected into the first user message
        self.task_hint = task_hint
        # Structured analysis dict passed to the critic for column disambiguation
        self.task_analysis = task_analysis or {}

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        task_content = build_task_prompt(task)
        if self.task_hint:
            hint = self.task_hint
            # If the hint starts with a [Prior attempt summary] block, place it
            # BEFORE the task question so the model reads it with high attention
            # before anchoring to its default strategy.  Remaining hint (schema
            # analysis) stays appended after the question as usual.
            if "[Prior attempt summary" in hint:
                prior_end = hint.find("\n\n[Pre-flight")
                if prior_end == -1:
                    # No schema analysis follows — entire hint is prior-attempt context
                    task_content = hint + "\n\n" + task_content
                else:
                    prior_part = hint[:prior_end]
                    rest = hint[prior_end + 2:]
                    task_content = prior_part + "\n\n" + task_content + "\n\n" + rest
            else:
                task_content = task_content + "\n\n" + hint
        messages.append(ModelMessage(role="user", content=task_content))

        # Count consecutive __error__ steps at the tail of the trace so we can escalate.
        consecutive_errors = 0
        for step in reversed(state.steps):
            if step.action == "__error__":
                consecutive_errors += 1
            else:
                break

        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            if step.action == "__error__":
                # After 2+ consecutive parse errors, escalate to a concrete rescue hint
                # that tells the model to emit a specific minimal valid action.
                # This breaks the confusion loop caused by very long or complex contexts.
                if consecutive_errors >= 2:
                    messages.append(ModelMessage(role="user", content=_RECOVERY_HINT_ESCALATED))
                else:
                    messages.append(ModelMessage(role="user", content=_RECOVERY_HINT))
            else:
                messages.append(
                    ModelMessage(role="user", content=build_observation_prompt(step.observation))
                )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        log = get_logger()
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""  # safe default — ensures the except branch has a valid value
            try:
                messages = self._build_messages(task, state)
                ctx_chars = sum(len(m.content) for m in messages)
                log.info("STEP %d/%d | context %d chars (%d msgs)", step_index, self.config.max_steps, ctx_chars, len(messages))

                t0 = time.perf_counter()
                raw_response = self.model.complete(messages)
                llm_elapsed = time.perf_counter() - t0
                model_step = parse_model_step(raw_response)

                thought_preview = model_step.thought[:120].replace("\n", " ")
                log.info("  action=%s thought=%r llm=%.1fs", model_step.action, thought_preview, llm_elapsed)

                # --- Critic gate: review proposed answers before executing them ---
                if self.enable_answer_critic and model_step.action == "answer":
                    proposed_cols = model_step.action_input.get("columns", [])
                    proposed_rows = model_step.action_input.get("rows", [])
                    concern = _run_critic_check(
                        self.model, task.question, proposed_cols, proposed_rows,
                        task_analysis=self.task_analysis,
                    )
                    if concern:
                        log.warning("  CRITIC BLOCKED: %s", concern)
                        observation = {
                            "ok": False,
                            "tool": "answer",
                            "content": {
                                "error": (
                                    f"Answer blocked by quality check. {concern} "
                                    "Please revise your answer and call answer again."
                                )
                            },
                        }
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation=observation,
                                ok=False,
                            )
                        )
                        continue
                # -----------------------------------------------------------------

                t0 = time.perf_counter()
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                tool_elapsed = time.perf_counter() - t0
                result_chars = len(str(tool_result.content))
                log.info("  tool=%s ok=%s result=%d chars tool=%.2fs", model_step.action, tool_result.ok, result_chars, tool_elapsed)

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    log.info("DONE answer accepted at step %d", step_index)
                    break
            except Exception as exc:
                exc_str = str(exc)

                # Content filter: the accumulated context contains data that triggered
                # the API's moderation layer.  Adding more error turns cannot fix this
                # because the offending content stays in the context window.  Stop
                # immediately so the runner can start a fresh attempt with clean context.
                if "data_inspection_failed" in exc_str:
                    log.error(
                        "CONTENT FILTER at step %d — stopping attempt early; "
                        "runner will resume with fresh context (%d steps consumed)",
                        step_index, len(state.steps),
                    )
                    state.failure_reason = (
                        f"content_filter_triggered at step {step_index}: context accumulated "
                        "data that caused API content moderation. Fresh attempt needed."
                    )
                    break

                # Layer 1: attempt silent JSON repair before recording the error.
                repaired = _try_repair_json(raw_response)
                if repaired is not None:
                    try:
                        model_step = parse_model_step(repaired)
                        log.warning("  JSON repaired — retrying action=%s", model_step.action)
                        t0 = time.perf_counter()
                        tool_result = self.tools.execute(
                            task, model_step.action, model_step.action_input
                        )
                        tool_elapsed = time.perf_counter() - t0
                        log.info("  tool=%s ok=%s tool=%.2fs (after repair)", model_step.action, tool_result.ok, tool_elapsed)
                        observation = {
                            "ok": tool_result.ok,
                            "tool": model_step.action,
                            "content": tool_result.content,
                        }
                        step_record = StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=repaired,
                            observation=observation,
                            ok=tool_result.ok,
                        )
                        state.steps.append(step_record)
                        if tool_result.is_terminal:
                            state.answer = tool_result.answer
                            log.info("DONE answer accepted at step %d (after repair)", step_index)
                            break
                        continue
                    except Exception:
                        pass  # Repair succeeded but tool failed — fall through to error record

                # Layer 2: record the error; _build_messages will inject the recovery hint
                # as the next user message so the model knows exactly what went wrong.
                log.error("  PARSE ERROR step=%d: %s", step_index, exc_str)
                observation = {
                    "ok": False,
                    "error": exc_str,
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
            log.warning("FAILED max_steps=%d exhausted", self.config.max_steps)

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
