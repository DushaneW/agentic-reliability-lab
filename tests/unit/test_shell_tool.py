from __future__ import annotations

from arl.tools.shell import ShellTool


def test_allowed_command_runs() -> None:
    tool = ShellTool()
    result = tool.run({"command": "echo hello"})
    assert result.success
    assert result.output.strip() == "hello"


def test_disallowed_command_is_denied() -> None:
    tool = ShellTool()
    result = tool.run({"command": "rm -rf /"})
    assert not result.success
    assert "command_denied" in (result.error or "")


def test_subprocess_receives_empty_environment(monkeypatch) -> None:  # noqa: ANN001
    import subprocess

    captured_kwargs = {}
    real_run = subprocess.run

    def spying_run(*args, **kwargs):  # noqa: ANN002, ANN003
        captured_kwargs.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spying_run)
    tool = ShellTool()
    tool.run({"command": "echo hi"})
    assert captured_kwargs["env"] == {}


def test_command_timeout_is_handled(monkeypatch) -> None:  # noqa: ANN001
    import subprocess

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd="cat", timeout=0.01)

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = ShellTool(timeout=0.01)
    result = tool.run({"command": "cat"})
    assert not result.success
    assert "timeout" in (result.error or "")


# --- Regression coverage for the embedded cross-platform runner script ---
#
# The four tests above use monkeypatched subprocess.run (or only exercise
# `echo`) and would NOT have caught a bug in _RUNNER_SCRIPT's actual command
# implementations, since they never let a real subprocess run one of the
# file-based commands end to end. These do.


def test_cat_reads_a_workdir_file() -> None:
    tool = ShellTool(workdir_files={"a.txt": "hello world\n"})
    result = tool.run({"command": "cat a.txt"})
    assert result.success
    assert result.output == "hello world\n"


def test_cat_missing_file_is_nonzero_exit() -> None:
    tool = ShellTool()
    result = tool.run({"command": "cat does-not-exist.txt"})
    assert not result.success
    assert "nonzero_exit" in (result.error or "")


def test_ls_lists_workdir_files() -> None:
    tool = ShellTool(workdir_files={"a.txt": "1", "b.txt": "2"})
    result = tool.run({"command": "ls"})
    assert result.success
    assert set(result.output.split()) == {"a.txt", "b.txt"}


def test_wc_counts_words_and_lines() -> None:
    tool = ShellTool(workdir_files={"a.txt": "one two three\nfour five\n"})
    result = tool.run({"command": "wc a.txt"})
    assert result.success
    # lines, words, chars, filename
    lines, words, chars, name = result.output.split()
    assert lines == "2"
    assert words == "5"
    assert name == "a.txt"


def test_grep_finds_matching_lines() -> None:
    tool = ShellTool(workdir_files={"a.txt": "apple\nbanana\navocado\n"})
    result = tool.run({"command": "grep ^a a.txt"})
    assert result.success
    assert result.output.splitlines() == ["apple", "avocado"]


def test_grep_no_match_is_nonzero_exit() -> None:
    tool = ShellTool(workdir_files={"a.txt": "banana\n"})
    result = tool.run({"command": "grep zzz a.txt"})
    assert not result.success
    assert "nonzero_exit" in (result.error or "")


def test_sort_orders_lines() -> None:
    tool = ShellTool(workdir_files={"a.txt": "banana\napple\ncherry\n"})
    result = tool.run({"command": "sort a.txt"})
    assert result.success
    assert result.output.splitlines() == ["apple", "banana", "cherry"]


def test_uniq_collapses_consecutive_duplicates() -> None:
    tool = ShellTool(workdir_files={"a.txt": "a\na\nb\na\n"})
    result = tool.run({"command": "uniq a.txt"})
    assert result.success
    assert result.output.splitlines() == ["a", "b", "a"]


def test_head_returns_first_n_lines() -> None:
    tool = ShellTool(workdir_files={"a.txt": "\n".join(str(i) for i in range(20)) + "\n"})
    result = tool.run({"command": "head -n 3 a.txt"})
    assert result.success
    assert result.output.splitlines() == ["0", "1", "2"]


def test_tail_returns_last_n_lines() -> None:
    tool = ShellTool(workdir_files={"a.txt": "\n".join(str(i) for i in range(20)) + "\n"})
    result = tool.run({"command": "tail -n 3 a.txt"})
    assert result.success
    assert result.output.splitlines() == ["17", "18", "19"]


def test_shell_tool_never_invokes_a_real_os_binary_named_after_the_command() -> None:
    """The whole point of the fix: the subprocess target must always be
    sys.executable, never a resolved 'echo'/'ls'/etc. binary — this is
    what makes the tool work without coreutils on PATH (i.e. on Windows).
    """
    import subprocess
    import sys

    captured_args = []
    real_run = subprocess.run

    def spying_run(args, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        captured_args.append(args)
        return real_run(args, *a, **kw)

    tool = ShellTool()
    import unittest.mock as mock

    with mock.patch.object(subprocess, "run", side_effect=spying_run):
        tool.run({"command": "echo hi"})

    assert captured_args[0][0] == sys.executable
    assert "echo" not in captured_args[0][:2]  # not the direct subprocess target
