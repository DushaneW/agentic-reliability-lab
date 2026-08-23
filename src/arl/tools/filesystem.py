"""Filesystem tool. Operates on a VirtualFilesystem, never the host disk."""

from __future__ import annotations

from typing import Any

from arl.runtime.environment import VirtualFileNotFoundError, VirtualFilesystem
from arl.tools.base import Tool, ToolError

_VALID_OPS = {"read", "write", "append", "delete", "list"}


class FilesystemTool(Tool):
    name = "filesystem"
    description = (
        "Read, write, append, delete, or list files in the task's sandboxed "
        "virtual filesystem. Operations: read(path), write(path, content), "
        "append(path, content), delete(path), list()."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": sorted(_VALID_OPS)},
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["op"],
    }

    def __init__(self, fs: VirtualFilesystem) -> None:
        self.fs = fs

    def execute(self, arguments: dict[str, Any]) -> str:
        op = arguments.get("op")
        if op not in _VALID_OPS:
            raise ToolError(f"Unknown op '{op}'", error_type="invalid_arguments")

        if op == "list":
            return "\n".join(self.fs.list_files())

        path = arguments.get("path")
        if not path:
            raise ToolError("'path' is required for op '" + str(op) + "'", "invalid_arguments")

        try:
            if op == "read":
                return self.fs.read(path)
            if op == "write":
                self.fs.write(path, arguments.get("content", ""))
                return f"wrote {path}"
            if op == "append":
                self.fs.append(path, arguments.get("content", ""))
                return f"appended to {path}"
            if op == "delete":
                self.fs.delete(path)
                return f"deleted {path}"
        except VirtualFileNotFoundError as exc:
            raise ToolError(f"File not found: {exc}", error_type="not_found") from exc

        raise ToolError(f"Unhandled op '{op}'", error_type="invalid_arguments")
