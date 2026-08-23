"""Tool protocol and registry.

A Tool is a named, schema-described capability an agent can invoke. Tools
never touch the real filesystem or a real shell on the host — see
docs/security.md for the sandbox model and its actual limits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from arl.domain.tool import ToolResult


class ToolError(Exception):
    """Raised by a tool implementation when execution fails."""

    def __init__(self, message: str, error_type: str = "execution_error") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class Tool(ABC):
    """Base class every tool implements."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        """Run the tool and return a plain-text result.

        Raise ToolError on failure; do not swallow errors into the return
        value, since the runtime distinguishes tool_result from tool_error
        events based on whether this raises.
        """
        raise NotImplementedError

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        import time

        start = time.monotonic()
        try:
            output = self.execute(arguments)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                duration_seconds=time.monotonic() - start,
            )
        except ToolError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"{exc.error_type}: {exc.message}",
                duration_seconds=time.monotonic() - start,
            )


class ToolRegistry:
    """Holds the tools available to an agent for a given environment."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]
