"""Example: implementing a custom tool.

Shows the full `Tool` interface — a real tool (word count), not a stub.

Run:
    uv run python examples/custom_tool.py
"""

from __future__ import annotations

from typing import Any

from arl.tools.base import Tool, ToolError


class WordCountTool(Tool):
    name = "word_count"
    description = "Count words, lines, and characters in a given string."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, arguments: dict[str, Any]) -> str:
        text = arguments.get("text")
        if text is None:
            raise ToolError("'text' is required", error_type="invalid_arguments")
        words = len(text.split())
        lines = len(text.splitlines()) or (1 if text else 0)
        chars = len(text)
        return f"words={words} lines={lines} chars={chars}"


def main() -> None:
    tool = WordCountTool()
    result = tool.run({"text": "the quick brown fox\njumps over the lazy dog"})
    print("success:", result.success)
    print("output: ", result.output)

    # Missing required argument -> a ToolError, surfaced as a failed ToolResult
    bad_result = tool.run({})
    print("success:", bad_result.success, "error:", bad_result.error)


if __name__ == "__main__":
    main()
