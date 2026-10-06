# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
