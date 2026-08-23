from __future__ import annotations

from arl.runtime.environment import VirtualFilesystem
from arl.tools.filesystem import FilesystemTool


def test_write_then_read() -> None:
    fs = VirtualFilesystem()
    tool = FilesystemTool(fs)
    write_result = tool.run({"op": "write", "path": "a.txt", "content": "hello"})
    assert write_result.success
    read_result = tool.run({"op": "read", "path": "a.txt"})
    assert read_result.success
    assert read_result.output == "hello"


def test_read_missing_file_fails() -> None:
    fs = VirtualFilesystem()
    tool = FilesystemTool(fs)
    result = tool.run({"op": "read", "path": "missing.txt"})
    assert not result.success
    assert "not_found" in (result.error or "")


def test_list_returns_all_paths() -> None:
    fs = VirtualFilesystem({"a.txt": "1", "b.txt": "2"})
    tool = FilesystemTool(fs)
    result = tool.run({"op": "list"})
    assert result.success
    assert set(result.output.split("\n")) == {"a.txt", "b.txt"}


def test_delete_removes_file() -> None:
    fs = VirtualFilesystem({"a.txt": "1"})
    tool = FilesystemTool(fs)
    tool.run({"op": "delete", "path": "a.txt"})
    assert not fs.exists("a.txt")


def test_unknown_op_is_invalid_arguments() -> None:
    fs = VirtualFilesystem()
    tool = FilesystemTool(fs)
    result = tool.run({"op": "format_disk"})
    assert not result.success
    assert "invalid_arguments" in (result.error or "")
