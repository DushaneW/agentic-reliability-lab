<div align="center">

# Agentic Reliability Lab (ARL)

**Benchmark, Telemetry, Fault-Injection & Early-Warning Prediction Framework for Autonomous AI Agents**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](#installation)
[![Package Manager](https://img.shields.io/badge/uv-supported-purple.svg)](https://github.com/astral-sh/uv)
[![Typechecked](https://img.shields.io/badge/mypy-strict-success.svg)](#development--quality-assurance)
[![Code Style](https://img.shields.io/badge/linter-ruff-orange.svg)](#development--quality-assurance)
[![Platform](https://img.shields.io/badge/platform-Windows_%7C_macOS_%7C_Linux-lightgrey.svg)](#installation)

[Overview](#overview) · [Architecture](#system-architecture) · [Installation](#installation) · [CLI & Benchmarks](#benchmarks) · [ML Reliability](#early-warning-reliability-prediction)

</div>

---

## Overview

Autonomous agent runs often fail in structural ways before a task times out or finishes: tools get called repeatedly with identical arguments, execution errors are retried without changes, or the agent stops early without completing the task.

**Agentic Reliability Lab (ARL)** provides infrastructure to capture, evaluate, and predict these failure modes:

1. **Typed Trajectory Telemetry:** Captures model decisions, tool calls, arguments, and environment responses inside a sandboxed virtual filesystem.
2. **State-Based Evaluation:** Grades task outcomes directly against verifiable environment state without "LLM-as-a-judge" heuristics.
3. **Early-Warning Prediction:** Extracts trajectory features at `20%`, `40%`, `60%`, `80%`, and `100%` completion horizons to predict whether an active run will fail before it finishes.

---

## Key Capabilities

| Module | Description |
| :--- | :--- |
| **Sandboxed Execution Engine** | Runs agents against isolated benchmark suites in a virtual filesystem with process allow-lists, clean environments, and timeouts. |
| **Controlled Fault Injection** | Uses a `FaultProfile` harness to test recovery and generate labeled reliability datasets. |
| **State Graders** | Evaluates final workspace states, file outputs, and side effects ([`docs/evaluation.md`](docs/evaluation.md)). |
| **Predictive Reliability ML** | Trains **Logistic Regression** and **Gradient Boosting** models over 9 trajectory features to trigger early-stop interventions on failing runs. |
| **Local Web Dashboard & CLI** | Includes the `arl` CLI and a FastAPI/Uvicorn telemetry dashboard backed by SQLite storage. |

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
|                          Evaluation Pipeline                          |
|                           [ State Graders ]                           |
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

See [`docs/architecture.md`](docs/architecture.md) for component boundaries, data schemas, and execution flow.

---

## Installation

ARL runs on **Linux**, **macOS**, and **Windows** (Python `3.11+`). You can use [**uv**](https://github.com/astral-sh/uv) or standard `venv` + `pip`.

### 1. Install `uv` (Optional)

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
Task        Run ID    Result  Steps
smoke-001   97f1fd94  PASS    5
smoke-002   e1a3ad1b  PASS    5

2/2 tasks passed
```

### 2. Inspect Trajectories
```bash
$ arl runs list
$ arl runs inspect <RUN_ID>
```

### 3. Launch the Local Telemetry Dashboard
```bash
make dashboard
# Or directly via uvicorn:
uvicorn arl.dashboard.app:app --reload
```

See [`docs/getting-started.md`](docs/getting-started.md) for the full walkthrough.

---

## Benchmarks

ARL groups benchmarks into suites targeting specific operations:

| Suite | Tasks | Purpose |
| :--- | :---: | :--- |
| `benchmarks/smoke` | 2 | Sanity check of the execution harness, virtual filesystem, and grading pipeline. |
| `benchmarks/coding` | 5 | Code manipulation, file generation, and structured transformation tasks. |
| `benchmarks/tool_use` | 6 | Multi-step tool chaining and argument formatting verification. |
| `benchmarks/recovery` | 5 | Transient error handling and state recovery scenarios. |
| `benchmarks/adversarial` | 5 | Edge-case inputs and hostile environment states. |
| `benchmarks/diverse` | 10 | Multi-file operations: word counting, log filtering, sorting, misleading paths, and prompt-injection tests. |

See [`docs/benchmarks.md`](docs/benchmarks.md) for benchmark details.

---

## Early-Warning Reliability Prediction

ARL predicts whether an active agent run will fail using only a partial trajectory prefix ($\le 60\%$ of steps completed), allowing early-stop interventions to save compute and token costs.

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

Evaluated across **6,630 trajectory prefixes** from **1,360 controlled runs** (17 tasks $\times$ 8 behavioral profiles $\times$ 10 seeds; 41% overall failure rate). Models extract 9 trajectory features, including repetition ratio, action diversity, and error recency ([`docs/research/methodology.md`](docs/research/methodology.md)).

**Early-Warning Performance (Prefixes $\le 60\%$ Complete, 5-Fold Run-Grouped CV):**

| Predictor Model | AUROC (95% CI) |
| :--- | :---: |
| Constant Baseline | `0.500` |
| Trajectory Length Only (Single Feature) | `0.547` `[0.533, 0.560]` |
| Tool Error Count Only (Single Feature) | `0.549` `[0.521, 0.579]` |
| **Logistic Regression (9 Structural Features)** | **`0.771`** `[0.756, 0.788]` |
| **Gradient Boosting (9 Structural Features)** | **`0.791`** `[0.773, 0.808]` |

**Key Findings:**
- **Structural Signal vs. Heuristics:** Multi-feature models (`0.771-0.791` AUROC) outperform single-metric heuristics like trajectory length (`0.547`) or raw error counts (`0.549`).
- **Horizon Progression:** Predictive power scales from the 20% prefix horizon (`0.596` AUROC) through the 60% early-warning window (`0.791` AUROC) and up to full-trajectory classification (`0.893` AUROC at 80-100%).
- **Profile Generalization:** Across held-out fault profiles, within-profile AUROC spans `0.59` to `0.80`.

See [`docs/research/experiments.md`](docs/research/experiments.md) and [`docs/research/cv_report.md`](docs/research/cv_report.md) for full tables and analysis.

---

## Reproducible Experiments & Interventions

Compare agent configurations and runtime interventions:

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
- **Subprocess Hardening:** The `shell` tool executes commands under a binary allow-list, a sanitized environment (`env={}`), and process timeouts.

See [`SECURITY.md`](SECURITY.md) for the threat model and security guidelines.

---

## Development & Quality Assurance

ARL uses static typing (`mypy --strict`), `ruff` formatting, and a 128-test suite covering unit, integration, and end-to-end pipelines.

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

- [**Getting Started Guide**](docs/getting-started.md) - End-to-end installation and CLI tutorial.
- [**System Architecture**](docs/architecture.md) - Internal module design, schemas, and data flow.
- [**Evaluation Design**](docs/evaluation.md) - Grading principles and task assertions.
- [**Benchmarks Reference**](docs/benchmarks.md) - Breakdown of all task suites and scenarios.
- [**Research Methodology**](docs/research/methodology.md) - Feature engineering and ML pipeline design.
- [**Experiments & CV Reports**](docs/research/experiments.md) - Statistical tables and validation protocols ([`cv_report.md`](docs/research/cv_report.md) · [`limitations.md`](docs/research/limitations.md)).
- [**Configuration & Interventions**](docs/configuration.md) - YAML experiment specs and early-stop rules.
