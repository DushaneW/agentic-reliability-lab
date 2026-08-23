"""Search tool.

Searches a small fixed in-memory document corpus rather than the live web.
This is intentional: benchmark tasks must be reproducible, and a tool that
hits the real internet would make grading nondeterministic and would also
require sending task content to an external service.
"""

from __future__ import annotations

from typing import Any

from arl.tools.base import Tool, ToolError


class SearchTool(Tool):
    name = "search"
    description = "Search a fixed reference document corpus by keyword."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self, corpus: dict[str, str]) -> None:
        self.corpus = corpus

    def execute(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not query:
            raise ToolError("'query' is required", error_type="invalid_arguments")
        terms = query.lower().split()
        hits = [
            title
            for title, body in self.corpus.items()
            if any(t in body.lower() or t in title.lower() for t in terms)
        ]
        if not hits:
            return "no results"
        return "\n".join(f"- {title}: {self.corpus[title]}" for title in hits[:5])
