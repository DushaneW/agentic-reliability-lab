# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Security

- `shell` tool: the embedded runner opened any path it was given, so an agent
  could `cat /etc/passwd` or `cat ../../.env` and read host files with the host
  user's permissions. Every path argument is now resolved and must stay inside
  the scratch directory. The interpreter also starts in isolated mode (`-I`),
  and workdir file names are validated.
- Dashboard: run ids, task ids and tool payloads were interpolated into HTML
  strings and inline `onclick` attributes (stored XSS via agent output or an
  imported JSONL file). The page now builds DOM nodes with `textContent` only
  and sends a nonce-based CSP.
- Reliability models are stored as JSON instead of pickle. `ReliabilityModel.load`
  refuses `.pkl` files, and `models/reliability_model.pkl` is removed. Retraining
  (`arl reliability train --seeds-per-profile 10`) reproduces
  `models/reliability_metrics.json` byte for byte.

### Added

- Regression tests for the shell escape, calculator edge cases, tool crash
  handling, model persistence and the dashboard (128 -> 148 tests).

- `arl reliability cv`: grouped cross-validation (by run, task, profile), five
  predictors including single-feature baselines, cluster-bootstrap CIs, Brier
  score, per-held-out-group AUROC. Report in `docs/research/cv_report.md`.
- `arl/agents/llm.py`: LLM agent adapter (Anthropic, OpenAI-compatible) and
  `arl benchmark run --agent llm`. Tested with fake transports only.
- `arl reliability score` and `arl/importers/jsonl.py`: score external JSONL
  trajectories.
- `benchmarks/diverse`: 10 hand-written tasks with oracle-checked graders.
- `incorrect_output` failure category and detector.
- Named fault profiles (`FAULT_PROFILES` is now a dict) and `profile_id` on
  dataset examples.

### Fixed

- `Tool.run` only caught `ToolError`, so any other exception aborted the run.
  It now records `unexpected_error: <type>: <message>` as a failed tool call.
- `calculator`: `9**9**9`, `10**400` (OverflowError) and `(-8)**0.5` (TypeError)
  escaped the tool as exceptions; they are now tool errors, as are non-finite
  results, non-string expressions and null bytes.
- `shell`: `head -n abc f` printed a Python traceback; file contents are written
  and read as UTF-8 on every platform.
- README: broken code blocks (`[url](url)` inside `curl`/`git clone` commands),
  wrong Python version (3.11+ vs `requires-python >=3.12`), a nonexistent
  `.[dev]` extra, wrong suite sizes, and a "0.893 AUROC at 80-100%" claim that
  contradicted `cv_report.md`. The results section now states what the
  synthetic data can and cannot show. The failure taxonomy lists which
  categories actually have detectors.
- `experiments/intervention.py` docstring described a `reset_on_repeat`
  strategy that does not exist; the code implements stop-on-predicted-failure.
- `.github/workflows/security.yml` only ran on pushes to `main`; the default
  branch is `master`.
- `Dockerfile` no longer hides `uv sync` errors; `docker-compose.yml` publishes
  the unauthenticated dashboard on 127.0.0.1 only.
- `consecutive_error_pairs` was always 0 (it counted adjacent `tool_error`
  events, which never occur). It now counts back-to-back failed tool outcomes.
- CI did not run on pushes: workflow listed `main`, default branch is `master`.
- Run ids no longer depend on Python's `hash()` of a dataclass.
- README/docs overstated results: headline AUROC 0.893 included 80-100%
  prefixes; early-warning AUROC is about 0.77-0.79. `experiments.md` rewritten
  (it also contained a wrong explanation for the dead feature and
  inconsistent numbers).
- README claimed every failure category had a detector; four have none.

## [0.1.0] - Unreleased

### Added

- Core domain model (`Task`, `Run`, `Trajectory`, `TrajectoryEvent`,
  `EvaluationResult`, `FailureDiagnosis`, `Experiment`, `Metric`).
- Sandboxed tool system: `filesystem` (in-memory virtual FS), `calculator`
  (restricted AST evaluator), `search` (fixed corpus), `shell`
  (allow-listed subprocess; see `SECURITY.md`).
- Deterministic reference agent with configurable fault injection
  (`FaultProfile`) for generating labeled failure trajectories.
- Runtime execution loop with typed trajectory recording and secret
  redaction.
- 23-task benchmark suite (generated via
  `scripts/generate_benchmark_tasks.py`) across 5 suites.
- Deterministic evaluation engine with 4 grader kinds, including a
  tolerance-based numeric grader (added after finding a floating-point
  exact-match bug during development — see
  `docs/research/experiments.md`).
- Rule-based failure analysis engine covering 7 failure categories with
  event-level evidence.
- SQLite storage layer (`Repository`) for runs, trajectories, evaluations,
  diagnoses, experiments, and metrics.
- Baseline reliability predictor (logistic regression over 9 hand-built
  trajectory features), trained and evaluated on real generated data —
  see `docs/research/experiments.md` for measured results.
- Minimal post-hoc intervention mechanism (stop-on-predicted-failure).
- CLI (`arl benchmark`, `arl runs`, `arl experiment`, `arl reliability`,
  `arl report`).
- Local dashboard (FastAPI + vanilla-JS single page).
- 128 tests across unit, integration, and end-to-end layers.
- CI (lint, typecheck, test, build), issue/PR templates, security policy,
  and research documentation (methodology, experiments, limitations).

### Known limitations

See `docs/research/limitations.md` — most importantly, the reference
agent is a deterministic scripted policy, not a real LLM.
