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
