"""Best-effort secret redaction for anything written into a trajectory.

This is defense in depth, not a guarantee: the shell tool already runs with
an empty environment (arl/tools/shell.py) so host secrets should never reach
a tool argument or result in the first place. This catches the case where a
task's own virtual filesystem happens to contain something that looks like a
credential, and keeps it out of persisted trajectories and logs.
"""

from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+"),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),  # long base64-looking blobs
]

REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    result = text
    for pattern in _PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result
