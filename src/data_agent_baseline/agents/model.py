from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APIError, OpenAI, RateLimitError


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


_MAX_RETRIES = 6          # up to 6 attempts total (5 retries after first failure)
_BASE_RETRY_DELAY = 2.0   # seconds; doubles each attempt (capped at _MAX_RETRY_DELAY)
_MAX_RETRY_DELAY = 60.0   # never wait more than 60 s before retrying
_JITTER_RANGE = 3.0       # add up to 3 s of random jitter to spread concurrent retries

# Per-request wall-clock timeout passed to the OpenAI SDK.
# The SDK's default read timeout resets on every byte received, so a slowly-
# streaming server can hold the connection open indefinitely.  We cap the
# total time we are willing to wait for a single API call.
# connect: 10 s — fail fast if the server is unreachable
# read:   120 s — if the server hasn't sent a new byte in 2 min, the call is hung
# write:  30  s — sending the prompt should be fast
# pool:   10  s — acquiring an httpx connection from the pool
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        # Reuse a single client for the lifetime of this adapter.
        # Pass the request timeout explicitly so the SDK uses it instead of its
        # default read=600s (which is time-between-bytes, not total call time).
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=httpx.Client(verify=False),
            timeout=_REQUEST_TIMEOUT,
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    temperature=self.temperature,
                )
            except RateLimitError as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES - 1:
                    break  # exhausted retries — raise below
                delay = _retry_delay(exc, attempt)
                time.sleep(delay)
                continue
            except APIConnectionError as exc:
                # Covers network errors, connection resets, and read timeouts.
                # These are transient — retry with backoff just like rate limits.
                last_exc = exc
                if attempt >= _MAX_RETRIES - 1:
                    break
                delay = _retry_delay(exc, attempt)
                time.sleep(delay)
                continue
            except APIError as exc:
                raise RuntimeError(f"Model request failed: {exc}") from exc

            choices = response.choices or []
            if not choices:
                raise RuntimeError("Model response missing choices.")
            content = choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Model response missing text content.")

            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

            return content

        raise RuntimeError(
            f"Model request failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc


def _retry_delay(exc: APIError, attempt: int) -> float:
    """Compute how long to wait before the next retry.

    Prefers the ``Retry-After`` header from the API response when present
    (only available for RateLimitError, not connection errors).
    Falls back to exponential backoff with random jitter.

    The jitter is critical when multiple workers hit the rate limit at the same
    moment — it spreads their retry times so they don't all pile up again.
    """
    # Try to honour the server's Retry-After header (rate limit responses only).
    retry_after: float | None = None
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            header_val = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-requests")
            if header_val:
                retry_after = float(header_val)
    except Exception:
        pass

    if retry_after is not None:
        # Honour the server hint but cap it at _MAX_RETRY_DELAY so a large
        # Retry-After value (e.g. 600 s) cannot burn the entire task budget
        # across six retries.  The cap means we may retry before the server is
        # fully ready, but _MAX_RETRIES limits the total number of attempts.
        return min(retry_after, _MAX_RETRY_DELAY) + random.uniform(0, _JITTER_RANGE)

    # Exponential backoff: 2s, 4s, 8s, 16s, 32s … capped at 60s, plus jitter.
    backoff = min(_MAX_RETRY_DELAY, _BASE_RETRY_DELAY * (2 ** attempt))
    return backoff + random.uniform(0, _JITTER_RANGE)


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
