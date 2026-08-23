"""Restricted shell tool.

Executes an allow-listed set of read-only, side-effect-free commands
inside a temporary directory, with a wall-clock timeout and no inherited
environment variables (so host secrets are never visible to the child
process).

This is process-level restriction, not container isolation. There is no
seccomp/namespace/cgroup sandboxing here. It should NOT be treated as safe
against arbitrary hostile code — see docs/security.md for the exact threat
model this tool does and does not defend against.

Cross-platform execution strategy
----------------------------------
An earlier version of this tool ran the allow-listed command name directly
as the subprocess target, e.g. `subprocess.run(["echo", "hello"], ...)`.
That works on POSIX systems, where `echo`, `ls`, `cat`, `wc`, `grep`,
`sort`, `uniq`, `head`, and `tail` are real coreutils executables on PATH.
It fails on Windows: none of those names (aside from `sort.exe`) exist as
standalone executables reachable via `CreateProcess`/`shell=False` — they
are either PowerShell aliases/functions or cmd.exe builtins, not files on
disk, so `subprocess.run([...], shell=False)` raises `FileNotFoundError`
(WinError 2).

The fix is not `shell=True` (that would reintroduce shell metacharacter
injection risk — the exact thing the argv-list form exists to prevent) and
is not a platform-conditional dispatch to `cmd.exe`/PowerShell (same
problem, plus doubling the surface to maintain and audit).

Instead, each allow-listed command is implemented as a small, fixed,
embedded Python script (`_RUNNER_SCRIPT`) and dispatched via
`subprocess.run([sys.executable, "-c", _RUNNER_SCRIPT, *parts], ...)`.
`sys.executable` — the Python interpreter already running this process —
is guaranteed to exist and be resolvable on every platform this project
runs on, unlike coreutils or Windows equivalents. This keeps every
existing security property:

- still `shell=False`; arguments are passed as separate argv entries, so
  there is no shell string for user-controlled input to be interpolated
  into or escape from
- the allow-list check happens in Python *before* any subprocess is
  started, exactly as before — this file, not the script text, decides
  what commands exist; the embedded script cannot be reached with a
  command name that isn't in `ALLOWED_COMMANDS`
- still a real, separate OS process, so `env={}` (no host environment
  variables reach it) and the wall-clock `timeout=` argument to
  `subprocess.run` both still apply exactly as before, including as a
  safety net against e.g. a pathological `grep` regex — see
  `_grep` below
- no new capability is introduced: the embedded script implements the
  same 9 read-only operations the allow-list already named, nothing more
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from arl.tools.base import Tool, ToolError

ALLOWED_COMMANDS = {"echo", "ls", "cat", "wc", "grep", "sort", "uniq", "head", "tail"}
DEFAULT_TIMEOUT_SECONDS = 5.0

# Fixed, non-configurable implementation of each allow-listed command.
# `sys.argv[1]` is the command name, `sys.argv[2:]` are its arguments.
# Deliberately minimal: this exists to make the allow-listed operations
# work identically across platforms, not to be a general reimplementation
# of coreutils. See module docstring for why this exists instead of
# shelling out to real OS binaries.
_RUNNER_SCRIPT = r"""
import re
import sys


def _echo(args):
    sys.stdout.write(" ".join(args) + "\n")
    return 0


def _ls(args):
    import os

    for entry in sorted(os.listdir(".")):
        sys.stdout.write(entry + "\n")
    return 0


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def _write_line(line):
    sys.stdout.write(line if line.endswith("\n") else line + "\n")


def _cat(args):
    if not args:
        sys.stderr.write("cat: missing operand\n")
        return 1
    code = 0
    for path in args:
        try:
            with open(path, encoding="utf-8") as f:
                sys.stdout.write(f.read())
        except OSError as exc:
            sys.stderr.write(f"cat: {path}: {exc}\n")
            code = 1
    return code


def _wc(args):
    if not args:
        sys.stderr.write("wc: missing operand\n")
        return 1
    code = 0
    for path in args:
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            sys.stderr.write(f"wc: {path}: {exc}\n")
            code = 1
            continue
        lines = text.count("\n")
        words = len(text.split())
        chars = len(text)
        sys.stdout.write(f"{lines:>7} {words:>7} {chars:>7} {path}\n")
    return code


def _grep(args):
    # Real grep's convention: exit 0 if at least one line matched, exit 1
    # if the pattern was valid but nothing matched, exit 2 on a usage or
    # pattern error. Kept for consistency with the tool this replaces.
    if len(args) < 2:
        sys.stderr.write("grep: usage: grep PATTERN FILE...\n")
        return 2
    pattern, paths = args[0], args[1:]
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        sys.stderr.write(f"grep: invalid pattern: {exc}\n")
        return 2
    matched = False
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if regex.search(line):
                        _write_line(line)
                        matched = True
        except OSError as exc:
            sys.stderr.write(f"grep: {path}: {exc}\n")
            return 2
    return 0 if matched else 1


def _sort(args):
    if not args:
        sys.stderr.write("sort: missing operand\n")
        return 1
    lines = []
    for path in args:
        try:
            lines.extend(_read_lines(path))
        except OSError as exc:
            sys.stderr.write(f"sort: {path}: {exc}\n")
            return 1
    for line in sorted(lines):
        _write_line(line)
    return 0


def _uniq(args):
    if not args:
        sys.stderr.write("uniq: missing operand\n")
        return 1
    try:
        lines = _read_lines(args[0])
    except OSError as exc:
        sys.stderr.write(f"uniq: {args[0]}: {exc}\n")
        return 1
    previous = None
    for line in lines:
        if line != previous:
            _write_line(line)
        previous = line
    return 0


def _head_or_tail(args, take_from_end):
    n = 10
    rest = list(args)
    if rest[:1] == ["-n"] and len(rest) >= 2:
        n = int(rest[1])
        rest = rest[2:]
    if not rest:
        sys.stderr.write("missing operand\n")
        return 1
    try:
        lines = _read_lines(rest[0])
    except OSError as exc:
        sys.stderr.write(f"{rest[0]}: {exc}\n")
        return 1
    for line in (lines[-n:] if take_from_end else lines[:n]):
        _write_line(line)
    return 0


_COMMANDS = {
    "echo": _echo,
    "ls": _ls,
    "cat": _cat,
    "wc": _wc,
    "grep": _grep,
    "sort": _sort,
    "uniq": _uniq,
    "head": lambda args: _head_or_tail(args, take_from_end=False),
    "tail": lambda args: _head_or_tail(args, take_from_end=True),
}


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("no command given\n")
        return 2
    handler = _COMMANDS.get(sys.argv[1])
    if handler is None:
        sys.stderr.write(f"unknown command: {sys.argv[1]}\n")
        return 2
    return handler(sys.argv[2:])


sys.exit(main())
"""


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
                    [sys.executable, "-c", _RUNNER_SCRIPT, *parts],
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
