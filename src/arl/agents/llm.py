"""Agent backed by a real chat model.

The model is asked to reply with exactly one JSON object per step:

    {"action": "tool_call", "tool": "<name>", "arguments": {...}, "rationale": "..."}
    {"action": "finish", "rationale": "..."}

Two clients are provided: Anthropic Messages API and any OpenAI-compatible
`/chat/completions` endpoint (this covers Ollama and llama.cpp servers).
Both use only the standard library. The HTTP layer is injected as
``post_json`` so the agent is testable without a network.

STATUS: exercised in tests against a fake transport only. It has not been
run against a live API in this repository's history; treat the first real
run as the first real test.

Infrastructure failures (HTTP errors, timeouts) raise ``LLMClientError``.
They are *not* converted into agent decisions: a 500 from the provider says
nothing about agent reliability and must not end up in the failure dataset.
Malformed model output *is* agent behaviour; it gets one repair attempt and
then ends the run with an explicit rationale.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from arl.agents.base import ActionKind, Agent, AgentDecision
from arl.domain.events import EventType, TrajectoryEvent

PostJson = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]

SYSTEM_PROMPT = """You are an agent that completes a task by calling tools one step at a time.
Reply with exactly one JSON object and nothing else.
To call a tool:
{"action": "tool_call", "tool": "<tool name>", "arguments": {...}, "rationale": "<short>"}
When the task is complete: {"action": "finish", "rationale": "<short>"}
"""

UNPARSEABLE = "unparseable model output"


class LLMClientError(RuntimeError):
    """Transport or provider failure; not an agent failure."""


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class ChatClient(Protocol):
    def complete(self, system: str, messages: list[ChatMessage]) -> str: ...


def urllib_post_json(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float = 120.0
) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - scheme is fixed by the caller
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            parsed = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise LLMClientError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMClientError(f"request to {url} failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError(f"unexpected response shape from {url}")
    return parsed


class AnthropicClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        post_json: PostJson = urllib_post_json,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMClientError("ANTHROPIC_API_KEY is not set")
        self._model = model
        self._key = key
        self._max_tokens = max_tokens
        self._post = post_json

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self._key, "anthropic-version": "2023-06-01"},
            {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            },
        )
        try:
            blocks = data["content"]
            return "".join(b["text"] for b in blocks if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMClientError(f"unexpected Anthropic response: {str(data)[:200]}") from exc


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        post_json: PostJson = urllib_post_json,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._post = post_json

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        data = self._post(
            self._url,
            self._headers,
            {
                "model": self._model,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}]
                + [{"role": m.role, "content": m.content} for m in messages],
            },
        )
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"unexpected chat response: {str(data)[:200]}") from exc


def parse_decision(text: str) -> AgentDecision | None:
    """Extract the first JSON object from ``text`` and validate it.

    Returns None if there is no valid decision. Tolerates markdown fences and
    prose around the object, because models add both.
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    rationale = str(obj.get("rationale", ""))
    action = obj.get("action")
    if action == "finish":
        return AgentDecision(kind=ActionKind.FINISH, rationale=rationale)
    if action == "tool_call":
        tool = obj.get("tool")
        arguments = obj.get("arguments", {})
        if isinstance(tool, str) and tool and isinstance(arguments, dict):
            return AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name=tool,
                arguments=arguments,
                rationale=rationale,
            )
    return None


def _history_to_messages(
    task_description: str, tool_specs: list[dict[str, Any]], history: list[TrajectoryEvent]
) -> list[ChatMessage]:
    first = f"Task:\n{task_description}\n\nTools:\n{json.dumps(tool_specs, indent=2)}"
    turns: list[ChatMessage] = [ChatMessage("user", first)]

    def add(role: Literal["user", "assistant"], content: str) -> None:
        if turns[-1].role == role:  # keep strict alternation required by the APIs
            turns[-1] = ChatMessage(role, turns[-1].content + "\n" + content)
        else:
            turns.append(ChatMessage(role, content))

    for event in history:
        payload = event.payload
        if event.event_type == EventType.TOOL_CALL:
            add(
                "assistant",
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool": payload.get("tool_name"),
                        "arguments": payload.get("arguments", {}),
                    }
                ),
            )
        elif event.event_type == EventType.TOOL_RESULT:
            add("user", f"Tool result: {payload.get('result_summary', '')}")
        elif event.event_type == EventType.TOOL_ERROR:
            add("user", f"Tool error: {payload.get('message', 'unknown error')}")
    return turns


class LLMAgent(Agent):
    def __init__(self, client: ChatClient, model_label: str) -> None:
        self._client = client
        self.name = f"llm:{model_label}"

    def decide(
        self,
        task_description: str,
        tool_specs: list[dict[str, Any]],
        history: list[TrajectoryEvent],
    ) -> AgentDecision:
        messages = _history_to_messages(task_description, tool_specs, history)
        text = self._client.complete(SYSTEM_PROMPT, messages)
        decision = parse_decision(text)
        if decision is not None:
            return decision

        repair = [
            *messages,
            ChatMessage("assistant", text or "(empty)"),
            ChatMessage(
                "user",
                "That was not a valid reply. Respond with only one JSON object in the "
                "format described in the instructions.",
            ),
        ]
        decision = parse_decision(self._client.complete(SYSTEM_PROMPT, repair))
        if decision is not None:
            return decision
        return AgentDecision(kind=ActionKind.FINISH, rationale=UNPARSEABLE)
