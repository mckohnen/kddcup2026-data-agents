from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
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


def _try_repair_json(raw_response: str) -> str | None:
    """Attempt lightweight repair of common LLM JSON formatting errors.

    Handles the most frequent case: trailing commas before } or ].
    Returns a repaired string on success, or None if the response is
    unrecoverable.
    """
    text = _strip_json_fence(raw_response)

    # Remove trailing commas before closing braces / brackets
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)

    try:
        _load_single_json_object(cleaned)
        return cleaned
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
    "You are a concise data quality critic. "
    "Review the proposed answer to the question and check for these issues:\n"
    "1. Extra columns: are there columns not explicitly requested (e.g. a 'count' alongside "
    "   values when only values were asked for, or an 'id' column when only names were asked)?\n"
    "2. Wrong column names: do the column names look renamed or aliased rather than taken "
    "   directly from the source data?\n"
    "3. Shape mismatch: 'how many' → 1 row 1 col; 'list/tally values' → distinct values only.\n"
    "4. Obvious wrong answer: e.g. zero rows when some result is clearly expected.\n"
    "Reply with exactly one of:\n"
    "  OK\n"
    "  CONCERN: <one sentence describing the specific problem>\n"
    "No other text."
)


def _run_critic_check(
    model: ModelAdapter,
    question: str,
    columns: list,
    rows: list,
    task_analysis: dict | None = None,
) -> str | None:
    """Call the model as a critic to review a proposed answer.

    If ``task_analysis`` is provided (output of ``build_task_analysis``), the
    critic receives the list of strong candidate columns so it can flag cases
    where the proposed column doesn't match what the question references.

    Returns a concern string if the critic flags an issue, or None if the
    answer looks fine.  Errors are silently swallowed so they never block a
    valid answer.
    """
    preview_rows = rows[:5]

    # Build optional column-grounding context for the critic
    grounding = ""
    if task_analysis:
        strong_cols = [
            m for m in task_analysis.get("matched_terms", [])
            if m.get("matched_column") and m["confidence"] >= 0.8
        ]
        if strong_cols:
            col_hints = ", ".join(
                f"{m['matched_table']}.{m['matched_column']} (matched \"{m['text']}\")"
                for m in strong_cols
            )
            grounding = (
                f"\nQuestion-to-schema analysis found these strong column candidates: {col_hints}. "
                "If the proposed column doesn't appear in this list, check whether a better "
                "column exists."
            )

    critic_user = (
        f"Question: {question}\n"
        f"Proposed answer columns: {columns}\n"
        f"Proposed answer rows (first {len(preview_rows)} of {len(rows)}): {preview_rows}"
        f"{grounding}\n\n"
        "Is there a problem with this answer?"
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
            task_content = task_content + "\n\n" + self.task_hint
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
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)

                # --- Critic gate: review proposed answers before executing them ---
                if self.enable_answer_critic and model_step.action == "answer":
                    proposed_cols = model_step.action_input.get("columns", [])
                    proposed_rows = model_step.action_input.get("rows", [])
                    concern = _run_critic_check(
                        self.model, task.question, proposed_cols, proposed_rows,
                        task_analysis=self.task_analysis,
                    )
                    if concern:
                        # Inject the concern as a tool error so the agent can revise
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

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
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
                    break
            except Exception as exc:
                # Layer 1: attempt silent JSON repair before recording the error.
                repaired = _try_repair_json(raw_response)
                if repaired is not None:
                    try:
                        model_step = parse_model_step(repaired)
                        tool_result = self.tools.execute(
                            task, model_step.action, model_step.action_input
                        )
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
                            break
                        continue
                    except Exception:
                        pass  # Repair succeeded but tool failed — fall through to error record

                # Layer 2: record the error; _build_messages will inject the recovery hint
                # as the next user message so the model knows exactly what went wrong.
                observation = {
                    "ok": False,
                    "error": str(exc),
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

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
