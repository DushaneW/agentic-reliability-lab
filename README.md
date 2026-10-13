<div align="center">

# Agentic Reliability Lab (ARL)

**Benchmark, Telemetry, Fault-Injection & Early-Warning Failure Prediction Framework for Autonomous AI Agents**

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](#installation)
[![Package Manager](https://img.shields.io/badge/uv-supported-purple.svg)](https://github.com/astral-sh/uv)
[![Typechecked](https://img.shields.io/badge/mypy-strict-success.svg)](#development--quality-assurance)
[![Code Style](https://img.shields.io/badge/linter-ruff-orange.svg)](#development--quality-assurance)
[![Platform](https://img.shields.io/badge/platform-Windows_%7C_macOS_%7C_Linux-lightgrey.svg)](#installation)

[Overview](#overview) · [Architecture](#system-architecture) · [Installation](#installation) · [CLI & Benchmarks](#benchmarking--llm-providers) · [Failure Taxonomy](#failure-analysis-taxonomy) · [ML Reliability](#early-warning-reliability-prediction)

</div>

---

## Overview

Autonomous agent runs frequently fail in subtle, structural ways long before a task times out or returns a final response: a tool gets invoked repeatedly with identical arguments, an execution error is blindly retried instead of adapted to, or the agent terminates prematurely without fulfilling the objective.

**Agentic Reliability Lab (ARL)** provides the core engineering and research infrastructure to capture, diagnose, and predict these failure modes in real time:

1. **Typed Trajectory Telemetry:** Captures every model decision, tool invocation, argument payload, and environment response inside a sandboxed virtual filesystem.
2. **Deterministic Evaluation:** Grades task outcomes strictly against verifiable environment state—zero stochastic "LLM-as-a-judge" heuristics.
3. **Evidence-Backed Diagnosis:** Rule-based failure detectors analyze execution traces and cite specific event IDs responsible for the breakdown.
4. **Early-Warning Failure Prediction:** Extracts structural trajectory features at `20%`, `40%`, `60%`, `80%`, and `100%` completion horizons to predict whether an active run will fail *before* it finishes executing.

> **Status.** Every trajectory behind the reliability results below is synthetic: a scripted
> agent with injected faults, not an LLM. The LLM adapters are tested against fake transports
> only and have not been run against a live model. Read
> [`docs/research/limitations.md`](docs/research/limitations.md) before citing any number.

---

## Key Capabilities

| Module | Description |
| :--- | :--- |
| **Sandboxed Execution Engine** | Runs agents against isolated benchmark suites inside a virtual filesystem with strict process-level allow-lists, clean environments, and execution timeouts. |
| **Controlled Fault Injection** | Includes a deterministic `FaultProfile` harness and LLM adapters for the Anthropic Messages API and any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio). The adapters have not been run against a live model yet. The fault harness generates the labeled reliability datasets. |
| **Deterministic Graders** | Evaluates final workspace states, file outputs, and side effects deterministically without relying on secondary LLM grading prompts ([`docs/evaluation.md`](docs/evaluation.md)). |
| **Rule-Based Diagnostics** | Classifies failed trajectories into actionable failure categories (`looping_failure`, `error_recovery_failure`, `tool_selection_failure`, etc.) with exact event-level citations. |
| **Predictive Reliability ML** | Trains and evaluates **Logistic Regression** and **Gradient Boosting** models over 9 structural trajectory features. The only intervention implemented is stopping a run once its predicted failure probability crosses a threshold (saves steps; does not recover the run). |
| **Local Web Dashboard & CLI** | Ships with a full-featured `arl` CLI and an interactive FastAPI/Uvicorn telemetry dashboard backed by a unified SQLite storage layer. |

---

## System Architecture

```text
+-----------------------------------------------------------------------+
|                        CLI  /  Web Dashboard                          |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                            Runtime Engine                             |
|        [ Agent Policy ]  <--->  [ Sandboxed Tools ]                   |
|                \                      /                               |
|                 v                    v                                |
|              [ Typed Trajectory Recorder ]                            |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    Evaluation & Diagnostic Pipeline                   |
|   [ Deterministic Graders ]  --->  [ Rule-Based Failure Detectors ]   |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    Reliability & Prediction Engine                    |
|   [ Prefix Feature Extractor (20%..100%) ] ---> [ ML Classifiers ]    |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                       SQLite Telemetry Storage                        |
+-----------------------------------------------------------------------+
```

For a deep dive into component boundaries, data schemas, and execution flow, see [`docs/architecture.md`](docs/architecture.md).

---

## Installation

ARL runs natively on **Linux**, **macOS**, and **Windows** (Python `3.12+`). We recommend using [**uv**](https://github.com/astral-sh/uv) for fast, reproducible environment management, though standard `venv` + `pip` is fully supported.

### 1. Install `uv` (Optional but Recommended)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone & Set Up Workspace

#### Linux & macOS
```bash
git clone https://github.com/DushaneW/agentic-reliability-lab.git
cd agentic-reliability-lab

# Using uv:
uv sync
uv pip install -e .

# Or using standard Python venv:
python3 -m venv .venv
source .venv/bin/activate
pip install -e . --group dev   # needs pip >= 25.1

cp .env.example .env
```

#### Windows (PowerShell)
```powershell
git clone https://github.com/DushaneW/agentic-reliability-lab.git
cd agentic-reliability-lab

# Using uv:
uv sync
uv pip install -e .

# Or using standard Python venv:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e . --group dev   # needs pip >= 25.1

Copy-Item .env.example .env
```

---

## Quickstart & CLI Walkthrough

### 1. Run the Smoke Benchmark Suite
```bash
$ arl benchmark run benchmarks/smoke
Task        Run ID    Result  Steps  Failure category
smoke-001   97f1fd94  PASS    5      none
smoke-002   e1a3ad1b  PASS    5      none

2/2 tasks passed
```

### 2. Inspect & Diagnose Trajectories
List recorded runs and run the diagnostic engine against any trajectory ID:

```bash
$ arl runs list
$ arl runs inspect <RUN_ID>
$ arl runs analyze 8f29c1a4
Run 8f29c1a4-...
Diagnosis: looping_failure
Confidence: 0.83
Wasted tool calls: 2
Recovered: False
  - 4 events are part of an identical repeated tool call (events: a1b2c3d4, ...)
```

### 3. Launch the Local Telemetry Dashboard
```bash
make dashboard
# Or directly via uvicorn:
uvicorn arl.dashboard.app:app --reload
```

See [`docs/getting-started.md`](docs/getting-started.md) for the complete interactive walkthrough.

---

## Benchmarking & LLM Providers

ARL organizes benchmark evaluations into specialized suites targeting distinct operational behaviors:

| Suite | Tasks | Purpose |
| :--- | :---: | :--- |
| `benchmarks/smoke` | 2 | Fast sanity check; always passes with the clean reference agent. |
| `benchmarks/coding` | 10 | Varying input size and magnitude. |
| `benchmarks/tool_use` | 5 | Larger input sets, longer tool-call sequences. |
| `benchmarks/recovery` | 3 | Degenerate inputs (zero, cancelling pairs, large magnitude). |
| `benchmarks/adversarial` | 3 | Tight step budgets, so competence matters. |
| `benchmarks/diverse` | 10 | Hand-written tasks: word count, log filtering, sorting, CSV filtering, multi-file reads, a misleading path, append semantics, a prompt-injection file. **Not solvable by the scripted reference agent**; intended for LLM agents. |
| `benchmarks/failures` | 5 | Deterministic scenarios, each engineered to trigger one detector (`looping`, `error_recovery`, `timeout`, `tool_misuse`, and a `partial_recovery` case that correctly comes back clean). |

The 23 generated tasks in the first five suites are all the same operation (sum the numbers in
`input.txt`, write the result to `output.txt`), because that is all the scripted agent implements.
See [`docs/benchmarks.md`](docs/benchmarks.md).

### Running LLM Agents (Anthropic / OpenAI-compatible)

`--agent llm` drives a real model through the same runtime, recorder and graders. The adapter has
only been exercised against fake transports in this repository's tests; treat the first live run
as the first real test. Transport failures (HTTP errors, timeouts) abort the run instead of being
recorded as agent failures.

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
arl benchmark run benchmarks/diverse \
  --agent llm --provider anthropic --model claude-sonnet-5-5

# Any OpenAI-compatible endpoint, e.g. a local Ollama server
arl benchmark run benchmarks/diverse \
  --agent llm --provider openai-compat \
  --base-url http://localhost:11434/v1 --model qwen2.5-coder:14b
```

Task text and tool output are sent to the provider you choose.

### Running the Fault-Injection Suite

To verify diagnostic accuracy across known failure topologies:

```bash
$ arl benchmark run benchmarks/failures
Task                      Run ID    Result  Steps  Failure category
failure-error-recovery…   fef6f0f5  FAIL    5      error_recovery_failure
failure-looping-001       2e1ed20c  FAIL    5      looping_failure
failure-partial-recove…   c87efe4e  PASS    5      none
failure-timeout-001       1b44f83a  FAIL    0      timeout_failure
failure-tool-misuse-001   88a77148  FAIL    5      tool_selection_failure

1/5 tasks passed

$ arl runs analyze fef6f0f5
Diagnosis: error_recovery_failure
Confidence: 0.95
Wasted tool calls: 3
Recovered: False
  - a tool error was followed by retrying the exact same failing call
```

Full details on benchmark design are available in [`docs/benchmarks.md`](docs/benchmarks.md), with a sample generated diagnostic report in [`docs/examples/failure_report_example.md`](docs/examples/failure_report_example.md).

---

## Failure Analysis Taxonomy

Failed runs are diagnosed by rule-based detectors that inspect the event sequence and cite the
event ids they matched on ([`docs/failure-analysis.md`](docs/failure-analysis.md)). Seven
categories have a detector:

- **`looping_failure`**: identical tool calls repeated with unchanged arguments.
- **`error_recovery_failure`**: a tool error followed by retrying the exact same failing call.
- **`tool_selection_failure`**: invoking an unknown or mismatched tool.
- **`tool_execution_failure`**: the same tool failing repeatedly in one run.
- **`premature_termination`**: the agent finishes before the required output exists.
- **`timeout_failure`**: the step budget or wall-clock limit was exceeded.
- **`incorrect_output`**: the run ended cleanly but the final workspace fails the grader.

`planning_failure`, `context_failure`, `reasoning_failure` and `safety_violation` exist in the
schema but have **no detector**. A failed run that matches no detector is reported as
`unknown_failure`.

---

## Early-Warning Reliability Prediction

The research question is whether features computed from a partial trajectory predict whether the
run will fail, before it finishes.

```bash
# Single run-grouped split; writes models/reliability_model.json and models/reliability_metrics.json
arl reliability train --seeds-per-profile 10

# Grouped cross-validation with cluster-bootstrap confidence intervals
arl reliability cv --seeds-per-profile 10
```

### Results (synthetic data)

Dataset: **6,630 trajectory prefixes** from **1,360 runs** (17 tasks × 8 fault profiles × 10
seeds; 41% of runs fail). Prefixes are cut at 20/40/60/80/100% of each run and labeled with that
run's final outcome. Nine features are computed from the event log
([`docs/research/methodology.md`](docs/research/methodology.md)).

**Early warning: prefixes up to 60% complete, 5-fold CV grouped by run:**

| Predictor | AUROC (95% CI) |
| :--- | :---: |
| Constant | `0.500` |
| Trajectory length only | `0.547` `[0.533, 0.560]` |
| Tool error count only | `0.549` `[0.521, 0.579]` |
| Logistic regression, 9 features | `0.771` `[0.756, 0.788]` |
| Gradient boosting, 9 features | `0.791` `[0.773, 0.808]` |

**By how much of the run has been seen (logistic regression):**

| 20% | 40% | 60% | 80% | 100% |
| :---: | :---: | :---: | :---: | :---: |
| 0.596 | 0.793 | 0.843 | 0.954 | 1.000 |

What these numbers do and do not show:

- The 9 features beat single-feature baselines on this data, so the signal is not just run length
  or error count.
- The faults are injected by the same code that defines the failure shapes the features detect.
  The 80% and 100% columns mostly reflect that the outcome is already visible in the log (1.000
  at 100% is a symptom of this), so only the early columns are evidence of early warning.
- Transfer to an unseen fault profile (leave-one-profile-out, prefixes ≤ 60%): 0.785 for logistic
  regression, 0.714 for gradient boosting. AUROC inside each held-out profile ranges from 0.59 to
  0.80 for logistic regression.
- Seeds of one (task, profile) pair are near-duplicates, so the confidence intervals are
  optimistic.
- Nothing here measures how well the predictor works on a real agent, or how many steps the
  stop-on-predicted-failure intervention would save one.

Full tables: [`docs/research/experiments.md`](docs/research/experiments.md) and
[`docs/research/cv_report.md`](docs/research/cv_report.md).

---

## Reproducible Experiments & Interventions

ARL includes a declarative experiment runner to compare agent configurations, model providers, and active runtime interventions (such as early termination on high predicted failure probability):

```bash
arl experiment run experiments/configs/baseline.yaml
arl experiment run experiments/configs/noisy_agent.yaml
arl experiment compare <EXPERIMENT_A_ID> <EXPERIMENT_B_ID>
```

Configuration schemas and intervention policies are documented in [`docs/configuration.md`](docs/configuration.md).

---

## Security & Sandboxing

- **`filesystem`** operates on an in-memory virtual filesystem and never touches the host disk.
- **`calculator`** is a restricted AST evaluator (arithmetic only, no `eval`).
- **`shell`** runs an allow-listed set of read-only commands, implemented in a fixed embedded
  Python script, in a scratch directory, with an empty environment, a timeout, and every path
  argument confined to that directory. This is process-level restriction, **not** container
  isolation.
- **Reliability models** are stored as JSON. Pickle files are refused because loading one executes
  code.
- **The dashboard** has no authentication; keep it bound to localhost. It renders run data (which
  can come from an agent or an imported file) with `textContent` only and sends a nonce-based CSP.
- **LLM adapters** send task text and tool output to the provider you configure.

See [`SECURITY.md`](SECURITY.md) for the threat model.

---

## Development & Quality Assurance

The package is checked with `mypy --strict` and `ruff`, and has 148 unit, integration and end-to-end tests.

```bash
make install      # Install package and development dependencies
make test         # Run the full pytest suite
make lint         # Run ruff static linter
make typecheck    # Run mypy --strict across the arl package
make check        # Execute lint, typecheck, and test suites sequentially
make benchmark    # Run the smoke benchmark suite
make dashboard    # Launch the local FastAPI telemetry dashboard
```

---

## Documentation Index

- [**Getting Started Guide**](docs/getting-started.md) — End-to-end installation and CLI tutorial.
- [**System Architecture**](docs/architecture.md) — Internal module design, schemas, and data flow.
- [**Evaluation Design**](docs/evaluation.md) — Deterministic grading principles and task assertions.
- [**Benchmarks Reference**](docs/benchmarks.md) — Breakdown of all task suites and fault scenarios.
- [**Failure Analysis**](docs/failure-analysis.md) — Detector rules, event matching, and taxonomy.
- [**Research Methodology**](docs/research/methodology.md) — Feature engineering and ML pipeline design.
- [**Experiments & CV Reports**](docs/research/experiments.md) — Detailed statistical tables and validation protocols ([`cv_report.md`](docs/research/cv_report.md) · [`limitations.md`](docs/research/limitations.md)).
- [**Configuration & Interventions**](docs/configuration.md) — YAML experiment specs and early-stop rules.

---

## Contributing & License

Contributions, bug reports, and new benchmark suites are welcome! Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a pull request. Structured issue templates for bugs, features, and research proposals are available under `.github/ISSUE_TEMPLATE/`.
