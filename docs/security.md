# Security

The full threat model, sandbox limitations, and secret-handling policy
live in [`SECURITY.md`](../SECURITY.md) at the repository root (GitHub
surfaces that file specially, so it's kept as the canonical copy rather
than duplicated here).

Short version: the `filesystem`, `calculator`, and `search` tools never
leave the Python process. The `shell` tool runs a real subprocess under an
allow-list, empty environment, temp directory, and timeout — process-level
restriction, not container isolation. See `SECURITY.md` for exactly what
that does and does not protect against before extending the allow-list or
pointing it at anything beyond the bundled benchmark suite.
