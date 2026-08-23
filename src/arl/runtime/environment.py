"""In-memory sandboxed environment for a single run.

Each run gets its own VirtualFilesystem instance. It never touches the host
filesystem. This is deliberate: it makes benchmark tasks fully deterministic
and removes an entire class of host-escape risk, at the cost of not being a
general-purpose code execution sandbox. See docs/security.md.
"""

from __future__ import annotations


class VirtualFileNotFoundError(Exception):
    pass


class VirtualFilesystem:
    """A minimal flat-namespace in-memory filesystem for benchmark tasks."""

    def __init__(self, initial_files: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = dict(initial_files or {})

    def read(self, path: str) -> str:
        if path not in self._files:
            raise VirtualFileNotFoundError(path)
        return self._files[path]

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def append(self, path: str, content: str) -> None:
        self._files[path] = self._files.get(path, "") + content

    def delete(self, path: str) -> None:
        if path not in self._files:
            raise VirtualFileNotFoundError(path)
        del self._files[path]

    def exists(self, path: str) -> bool:
        return path in self._files

    def list_files(self) -> list[str]:
        return sorted(self._files)

    def snapshot(self) -> dict[str, str]:
        return dict(self._files)
