from __future__ import annotations

from typing import Any

from arl.tools.base import Tool, ToolError


class _Crashing(Tool):
    name = "crashing"
    description = "always raises"
    input_schema: dict[str, Any] = {"type": "object"}

    def execute(self, arguments: dict[str, Any]) -> str:
        raise KeyError("boom")


class _Failing(Tool):
    name = "failing"
    description = "raises ToolError"
    input_schema: dict[str, Any] = {"type": "object"}

    def execute(self, arguments: dict[str, Any]) -> str:
        raise ToolError("nope", error_type="custom")


def test_unexpected_exception_becomes_a_failed_result() -> None:
    result = _Crashing().run({})
    assert not result.success
    assert result.error is not None
    assert result.error.startswith("unexpected_error: KeyError")


def test_tool_error_keeps_its_type() -> None:
    result = _Failing().run({})
    assert not result.success
    assert result.error == "custom: nope"
