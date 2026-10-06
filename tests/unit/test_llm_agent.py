from __future__ import annotations

from typing import Any

import pytest

from arl.agents.base import ActionKind
from arl.agents.llm import (
    UNPARSEABLE,
    AnthropicClient,
    ChatMessage,
    LLMAgent,
    LLMClientError,
    OpenAICompatClient,
    parse_decision,
)
from arl.domain.events import EventType, TrajectoryEvent


class ScriptedClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        del system
        self.calls.append(messages)
        return self.replies.pop(0)


def _ev(i: int, t: EventType, payload: dict[str, Any]) -> TrajectoryEvent:
    return TrajectoryEvent(run_id="r", step_index=i, event_type=t, payload=payload)


def test_parse_tool_call_inside_markdown_fence() -> None:
    body = '{"action":"tool_call","tool":"calculator","arguments":{"expression":"1+1"}}'
    text = f"Sure:\n```json\n{body}\n```"
    d = parse_decision(text)
    assert d is not None
    assert d.kind == ActionKind.TOOL_CALL
    assert d.tool_name == "calculator"
    assert d.arguments == {"expression": "1+1"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        '{"action": "tool_call"}',  # missing tool
        '{"action": "tool_call", "tool": "x", "arguments": "oops"}',
        '{"action": "dance"}',
        "[1, 2, 3]",
        '{"action": "finish"',  # truncated
    ],
)
def test_parse_rejects_invalid(text: str) -> None:
    assert parse_decision(text) is None


def test_finish_decision() -> None:
    d = parse_decision('{"action": "finish", "rationale": "done"}')
    assert d is not None
    assert d.kind == ActionKind.FINISH
    assert d.rationale == "done"


def test_repair_attempt_recovers_from_one_bad_reply() -> None:
    client = ScriptedClient(["I think we should list files", '{"action":"finish"}'])
    agent = LLMAgent(client, "m")
    d = agent.decide("task", [], [])
    assert d.kind == ActionKind.FINISH
    assert len(client.calls) == 2
    assert client.calls[1][-1].role == "user"  # repair prompt follows the bad assistant turn


def test_two_bad_replies_end_the_run_with_explicit_rationale() -> None:
    agent = LLMAgent(ScriptedClient(["nope", "still nope"]), "m")
    d = agent.decide("task", [], [])
    assert d.kind == ActionKind.FINISH
    assert d.rationale == UNPARSEABLE


def test_history_is_rendered_with_strictly_alternating_roles() -> None:
    history = [
        _ev(1, EventType.TOOL_CALL, {"tool_name": "filesystem", "arguments": {"op": "list"}}),
        _ev(2, EventType.TOOL_ERROR, {"message": "boom"}),
        _ev(3, EventType.TOOL_CALL, {"tool_name": "filesystem", "arguments": {"op": "list"}}),
        _ev(4, EventType.TOOL_CALL, {"tool_name": "calculator", "arguments": {}}),  # no outcome
        _ev(5, EventType.TOOL_RESULT, {"result_summary": "42"}),
    ]
    client = ScriptedClient(['{"action":"finish"}'])
    LLMAgent(client, "m").decide("task", [], history)
    roles = [m.role for m in client.calls[0]]
    assert roles[0] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))
    assert "Tool error: boom" in client.calls[0][2].content


def test_anthropic_client_request_and_response_shape() -> None:
    seen: dict[str, Any] = {}

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        seen.update(url=url, headers=headers, body=body)
        return {"content": [{"type": "text", "text": '{"action":"finish"}'}]}

    client = AnthropicClient("some-model", api_key="k", post_json=post)
    out = client.complete("sys", [ChatMessage("user", "hi")])
    assert out == '{"action":"finish"}'
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMClientError, match="ANTHROPIC_API_KEY"):
        AnthropicClient("m")


def test_openai_compat_puts_system_first_and_parses_choice() -> None:
    seen: dict[str, Any] = {}

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        seen.update(url=url, body=body)
        return {"choices": [{"message": {"content": "ok"}}]}

    client = OpenAICompatClient("llama", base_url="http://host:1/v1/", post_json=post)
    assert client.complete("sys", [ChatMessage("user", "hi")]) == "ok"
    assert seen["url"] == "http://host:1/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_malformed_provider_response_is_a_client_error_not_an_agent_decision() -> None:
    client = OpenAICompatClient("m", post_json=lambda *_: {"nope": 1})
    with pytest.raises(LLMClientError):
        client.complete("s", [ChatMessage("user", "x")])
