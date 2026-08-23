# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- 38 tests across unit, integration, and end-to-end layers.
- CI (lint, typecheck, test, build), issue/PR templates, security policy,
  and research documentation (methodology, experiments, limitations).

### Known limitations

See `docs/research/limitations.md` — most importantly, the reference
agent is a deterministic scripted policy, not a real LLM.
