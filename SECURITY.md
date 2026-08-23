# Security

## Threat model

Agentic Reliability Lab executes agent-proposed tool calls. The tools an
agent can invoke, and what each one can actually touch, are:

| Tool | What it touches | Isolation |
|---|---|---|
| `filesystem` | An in-process `VirtualFilesystem` (a Python dict) | Full — never touches the host disk |
| `calculator` | Nothing external | A restricted AST evaluator (`arl/tools/calculator.py`); no `eval()`, no name lookups, arithmetic node types only |
| `search` | A fixed in-memory corpus supplied by the task | Full — no network access |
| `shell` | A real subprocess | **Partial** — see below |

**Only the `shell` tool executes outside the Python process, and its
isolation is weaker than the others.** Read the rest of this document before
relying on it for anything beyond running the bundled benchmark suite.

## What the shell tool actually does

`arl/tools/shell.py`:

- Allow-lists a fixed set of read-only commands (`echo`, `ls`, `cat`, `wc`,
  `grep`, `sort`, `uniq`, `head`, `tail`). Anything else is rejected before
  a subprocess is even started.
- Does not run the allow-listed command name as the subprocess target.
  It dispatches to a fixed, embedded Python implementation of that
  command via `[sys.executable, "-c", <script>, ...]`. This is a
  cross-platform fix, not a design preference: `echo`/`ls`/`cat`/etc. are
  real coreutils executables on PATH on POSIX but do not exist as
  standalone Windows executables (aside from `sort.exe`), so running the
  command name directly failed on Windows with `FileNotFoundError`. The
  embedded script is fixed and non-configurable — the allow-list check
  still happens in Python before any subprocess starts, and arguments are
  still passed as separate argv entries (`shell=False`), never
  interpolated into a shell string.
- Runs in a fresh `tempfile.TemporaryDirectory()`, populated only with the
  files explicitly passed to the tool.
- Runs with `env={}` — the subprocess receives no environment variables at
  all, so host secrets (API keys, credentials, etc. that might be in the
  calling process's environment) cannot leak into a tool call via `os.environ`.
- Has a wall-clock timeout (`subprocess.run(..., timeout=...)`), default 5s.

## What it does NOT do

This is process-level restriction via an allow-list, a scratch directory,
timeout, and an empty environment. It is **not**:

- container isolation (no Docker, no namespaces)
- a seccomp/AppArmor/gVisor sandbox
- protection against kernel-level exploits, resource exhaustion via
  allowed commands (e.g. `cat` on a very large file), or symlink tricks
  within the temp directory
- a general-purpose code execution sandbox — there is no `python`, `bash
  -c`, or arbitrary-binary execution path, by design

**Do not add commands to the allow-list, and do not point this tool at
anything other than the benchmark suite's own scratch files, without
adding real container/VM isolation first.** The allow-listed commands were
chosen because they are read-only and side-effect-free against the temp
directory; adding a command that writes, deletes, or makes network calls
changes this risk profile and should not be done without revisiting this
document.

There is no Docker daemon available in the environment this project was
built in, so a container-based sandbox for the shell tool is **documented
as future work, not implemented** — see `docs/security.md` cross-reference
below and the repository's `Dockerfile`, which containerizes the
application itself (for a reproducible dev/CI environment) but does not yet
sandbox individual tool calls.

## Secrets

- `.env.example` documents the only environment variable the project reads
  today (`ARL_DB_PATH`, a local file path — not a secret). There is no API
  key in this repository because the reference agent does not call an
  external model provider (see `docs/research/limitations.md`).
- `arl/telemetry/redaction.py` applies best-effort pattern-based redaction
  (API-key-shaped strings, `key=`/`token=`/`password=` patterns, long
  base64-looking blobs) to every string value written into a trajectory
  event, as defense in depth. This is a safety net, not the primary
  control — the primary control is that the shell subprocess never
  receives host environment variables in the first place, so there is
  nothing secret for a tool result to contain by default.
- Redaction is regex-based and will miss secrets that don't match the
  patterns in `_PATTERNS`. Do not rely on it as the only thing standing
  between a real credential and a persisted trajectory.

## Known limitations

- No rate limiting or resource quota (CPU/memory) on the shell subprocess
  beyond the wall-clock timeout.
- No protection against a task's own `initial_files` content being
  adversarially large (a benchmark author could write a task with a
  multi-gigabyte `input.txt`); task files are currently trusted input.
- The SQLite database has no encryption at rest and no access control —
  it's a local single-user tool, and `arl.db` should be treated like any
  other local file containing your run history.
- `arl reliability train` and `arl benchmark run` execute the tasks
  bundled in `benchmarks/` (or any directory you pass); they do not
  validate that a third-party benchmark directory is safe before loading
  it. Only run benchmark suites you trust.

## Reporting a vulnerability

This is an open-source research/tooling project without a dedicated
security team. If you find a vulnerability, please open a GitHub issue
using the "Bug report" template and mark it clearly as a security issue in
the title, or reach the maintainers through the contact method listed in
the repository's GitHub profile if one is more appropriate for the
severity. There is currently no bug bounty program.
