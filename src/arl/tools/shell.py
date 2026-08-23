"""Restricted shell tool.

Executes an allow-listed set of read-only, side-effect-free commands as a
subprocess, inside a temporary directory, with a wall-clock timeout and no
inherited environment variables (so host secrets are never visible to the
subprocess).

This is process-level restriction, not container isolation. There is no
seccomp/namespace/cgroup sandboxing here. It should NOT be treated as safe
against arbitrary hostile code — see docs/security.md for the exact threat
model this tool does and does not defend against.
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from arl.tools.base import Tool, ToolError

ALLOWED_COMMANDS = {"echo", "ls", "cat", "wc", "grep", "sort", "uniq", "head", "tail"}
DEFAULT_TIMEOUT_SECONDS = 5.0


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a restricted, allow-listed shell command "
        f"({', '.join(sorted(ALLOWED_COMMANDS))}) in an isolated temp directory."
    )
    input_schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(
        self, workdir_files: dict[str, str] | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        self.workdir_files = workdir_files or {}
        self.timeout = timeout

    def execute(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not command:
            raise ToolError("'command' is required", error_type="invalid_arguments")

        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"Could not parse command: {exc}", "invalid_arguments") from exc

        if not parts:
            raise ToolError("Empty command", error_type="invalid_arguments")

        program = parts[0]
        if program not in ALLOWED_COMMANDS:
            raise ToolError(
                f"Command '{program}' is not on the allow-list", error_type="command_denied"
            )

        with tempfile.TemporaryDirectory(prefix="arl-shell-") as tmpdir:
            tmp_path = Path(tmpdir)
            for name, content in self.workdir_files.items():
                (tmp_path / name).write_text(content)

            try:
                proc = subprocess.run(  # noqa: S603
                    parts,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={},  # deliberately empty: no host secrets reach the subprocess
                )
            except subprocess.TimeoutExpired as exc:
                raise ToolError(
                    f"Command exceeded {self.timeout}s timeout", error_type="timeout"
                ) from exc

            if proc.returncode != 0:
                raise ToolError(
                    proc.stderr.strip() or f"exit code {proc.returncode}",
                    error_type="nonzero_exit",
                )
            return proc.stdout
