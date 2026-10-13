<div align="center">

# Agentic Reliability Lab (ARL)

**Benchmark, Telemetry, Fault-Injection & Early-Warning Failure Prediction Framework for Autonomous AI Agents**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](#installation)
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

---

## Key Capabilities

| Module | Description |
| :--- | :--- |
| **Sandboxed Execution Engine** | Runs agents against isolated benchmark suites inside a virtual filesystem with strict process-level allow-lists, clean environments, and execution timeouts. |
| **Controlled Fault Injection** | Includes a deterministic `FaultProfile` harness alongside live LLM adapters (`Anthropic`, `OpenAI-compatible`, `Ollama`) to systematically stress-test failure recovery and generate labeled reliability datasets. |
| **Deterministic Graders** | Evaluates final workspace states, file outputs, and side effects deterministically without relying on secondary LLM grading prompts ([`docs/evaluation.md`](docs/evaluation.md)). |
| **Rule-Based Diagnostics** | Classifies failed trajectories into actionable failure categories (`looping_failure`, `error_recovery_failure`, `tool_selection_failure`, etc.) with exact event-level citations. |
| **Predictive Reliability ML** | Trains and evaluates **Logistic Regression** and **Gradient Boosting** models over 9 structural trajectory features to trigger early-stop interventions on doomed runs. |
| **Local Web Dashboard & CLI** | Ships with a full-featured `arl` CLI and an interactive FastAPI/Uvicorn telemetry dashboard backed by a unified SQLite storage layer. |

---

## System Architecture

```text
+-----------------------------------------------------------------------+
|                         CLI  /  Web Dashboard                         |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                            Runtime Engine                             |
|        [ Agent Policy ]  <--->  [ Sandboxed Tools ]                   |
|                \                     /                                |
|                 v                   v                                 |
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
|                        SQLite Telemetry Storage                       |
+-----------------------------------------------------------------------+
```

For a deep dive into component boundaries, data schemas, and execution flow, see [`docs/architecture.md`](docs/architecture.md).

---

## Installation

ARL runs natively on **Linux**, **macOS**, and **Windows** (Python `3.11+`). We recommend using [**uv**](https://github.com/astral-sh/uv) for fast, reproducible environment management, though standard `venv` + `pip` is fully supported.

### 1. Install `uv` (Optional but Recommended)

```bash
# macOS / Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

### 2. Clone & Set Up Workspace

#### Linux & macOS
```bash
git clone [https://github.com/DushaneW/agentic-reliability-lab.git](https://github.com/DushaneW/agentic-reliability-lab.git)
cd agentic-reliability-lab

# Using uv:
uv sync
uv pip install -e .

# Or using standard Python venv:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
```

#### Windows (PowerShell)
```powershell
git clone [https://github.com/DushaneW/agentic-reliability-lab.git](https://github.com/DushaneW/agentic-reliability-lab.git)
cd agentic-reliability-lab

# Using uv:
uv sync
uv pip install -e .

# Or using standard Python venv:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

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
| `benchmarks/smoke` | 2 | Fast sanity check of the execution harness, virtual filesystem, and grading pipeline. |
| `benchmarks/coding` | 5 | Code manipulation, file generation, and structured transformation tasks. |
| `benchmarks/tool_use` | 6 | Multi-step tool chaining and argument formatting verification. |
| `benchmarks/recovery` | 5 | Transient error handling and state recovery scenarios. |
| `benchmarks/adversarial` | 5 | Edge-case inputs and hostile environment states. |
| `benchmarks/diverse` | 10 | Complex multi-file operations: word counting, log filtering, sorting, misleading paths, and prompt-injection resilience. |

### Running Live LLM Agents (Anthropic / OpenAI / Ollama)

Run any benchmark suite against cloud APIs or local open-weights models via the unified `--agent llm` adapter:

```bash
# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
arl benchmark run benchmarks/diverse \
  --agent llm \
  --provider anthropic \
  --model claude-3-5-sonnet-latest

# Local Models via Ollama / vLLM / LM Studio (OpenAI-Compatible API)
arl benchmark run benchmarks/diverse \
  --agent llm \
  --provider openai-compat \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:14b
```

Full details on benchmark design are available in [`docs/benchmarks.md`](docs/benchmarks.md), with a sample generated diagnostic report in [`docs/examples/failure_report_example.md`](docs/examples/failure_report_example.md).

---

## Failure Analysis Taxonomy

ARL diagnoses failed runs using deterministic, rule-based detectors that inspect event sequences and return structured evidence citations ([`docs/failure-analysis.md`](docs/failure-analysis.md)):

- **`looping_failure`** — Identical tool calls executed repeatedly with unchanged arguments and no state progression.
- **`error_recovery_failure`** — A tool execution error is immediately followed by retrying the exact same failing invocation without parameter adaptation.
- **`tool_selection_failure`** — Invocation of an invalid, disallowed, or contextually mismatched tool resulting in execution faults.
- **`tool_execution_failure`** — Unhandled runtime exceptions or malformed payloads during tool execution.
- **`premature_termination`** — Agent signals task completion prior to satisfying required output artifacts or state assertions.
- **`timeout_failure`** — Execution exceeds the configured step budget or wall-clock timeout threshold.
- **`incorrect_output`** — Agent completes execution cleanly, but final workspace artifacts fail deterministic grading checks.

---

## Early-Warning Reliability Prediction

A core research objective of ARL is **early failure detection**: predicting whether an active agent run will fail using only a partial trajectory prefix ($\le 60\%$ of steps completed), enabling automated early-stopping interventions to save compute and token costs.

### Training & Evaluating Predictors

```bash
# Train baseline predictors across fault profiles
arl reliability train --seeds-per-profile 10

# Evaluate predictive performance across completion horizons (20%..100%)
arl reliability evaluate

# Run 5-fold grouped cross-validation with bootstrap confidence intervals
arl reliability cv --seeds-per-profile 10
```

### Benchmark Evaluation Results

Evaluated across **6,630 trajectory prefixes** generated from **1,360 controlled runs** (17 tasks $\times$ 8 behavioral profiles $\times$ 10 seeds; 41% overall failure rate). Models extract 9 structural trajectory features—including repetition ratio, action diversity, and error recency ([`docs/research/methodology.md`](docs/research/methodology.md)).

**Early-Warning Performance (Prefixes $\le 60\%$ Complete, 5-Fold Run-Grouped CV):**

| Predictor Model | AUROC (95% CI) |
| :--- | :---: |
| Constant Baseline | `0.500` |
| Trajectory Length Only (Single Feature) | `0.547` `[0.533, 0.560]` |
| Tool Error Count Only (Single Feature) | `0.549` `[0.521, 0.579]` |
| **Logistic Regression (9 Structural Features)** | **`0.771`** `[0.756, 0.788]` |
| **Gradient Boosting (9 Structural Features)** | **`0.791`** `[0.773, 0.808]` |

**Key Analytical Findings:**
- **Structural Signal vs. Heuristics:** Multi-feature models (`0.771–0.791` AUROC) substantially outperform single-metric heuristics like trajectory length (`0.547`) or raw error counts (`0.549`), confirming that early failure manifests in action-diversity and repetition patterns rather than step count alone.
- **Horizon Progression:** Predictive power scales monotonically from the 20% prefix horizon (`0.596` AUROC) through the 60% early-warning window (`0.791` AUROC) and up to full-trajectory classification (`0.893` AUROC at 80–100%).
- **Profile Generalization:** Across held-out fault profiles, within-profile AUROC spans `0.59` to `0.80`, enabling reliable cost-saving early termination via ARL's intervention engine.

For full cross-validation tables, methodology, and experimental analysis, see [`docs/research/experiments.md`](docs/research/experiments.md) and [`docs/research/cv_report.md`](docs/research/cv_report.md).

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

All agent tool invocations operate inside a restricted execution boundary:
- **Virtual Filesystem Isolation:** File reads and writes are scoped to isolated temporary workspace directories per run.
- **Subprocess Hardening:** The `shell` tool executes commands under a strict binary allow-list, a sanitized environment (`env={}`), and hard process timeouts.

See [`SECURITY.md`](SECURITY.md) for the complete threat model and operational security guidelines.

---

## Development & Quality Assurance

ARL maintains strict static typing (`mypy --strict`), automated formatting (`ruff`), and a comprehensive 128-test suite covering unit, integration, and end-to-end execution pipelines.

```bash
make install      # Install package and development dependencies
make test         # Run the full pytest suite (128 unit, integration & E2E tests)
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
