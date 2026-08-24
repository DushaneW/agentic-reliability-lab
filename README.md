# Agentic Reliability Lab

A benchmark, telemetry, and failure-analysis framework for autonomous agents,
with a baseline model for predicting agent failure from partial trajectories.

> **Research question:** Can the shape of an agent's trajectory reveal that a
> run is likely to fail before the run has finished?

---

## Overview

Agentic Reliability Lab is an experimental framework for studying reliability
signals in autonomous-agent trajectories.

The project records structured agent activity, evaluates task outcomes
deterministically, diagnoses failure patterns, and trains a baseline model that
uses partial trajectory information to estimate the likelihood of eventual
failure.

The system is designed around a simple idea:

```text
agent behavior
      |
      v
trajectory telemetry
      |
      v
deterministic evaluation
      |
      v
failure diagnosis
      |
      v
early reliability prediction
Important limitation

The reference agent included in this repository is not an LLM.

It is a deterministic scripted policy. The original development environment did
not have an LLM API configured, so model calls are not simulated or fabricated.
Instead, controlled failure behavior is injected through FaultProfile
configurations.

This makes the experiments reproducible, but it also means that the reported
metrics describe this benchmark and its controlled fault profiles — not the
reliability of real-world LLM agents.

See docs/research/limitations.md for the
complete discussion.

What the project does
Structured trajectory capture

Every run records a typed trajectory containing:

agent decisions
tool calls
tool arguments
tool results
errors
termination information
evaluation results
Deterministic evaluation

Task outcomes are graded programmatically rather than by an LLM judge.

This keeps evaluation reproducible and avoids introducing another model into
the measurement pipeline.

See docs/evaluation.md.

Failure diagnosis

Failures are classified using explicit rule-based detectors.

Current categories include:

planning_failure
tool_selection_failure
tool_execution_failure
looping_failure
premature_termination
error_recovery_failure
timeout_failure
unknown_failure

Each diagnosis is backed by concrete trajectory evidence rather than a
free-form explanation.

Reliability prediction

The baseline predictor is a logistic-regression model using hand-built
trajectory features such as:

repetition ratio
action diversity
error recency
tool-call behavior
trajectory characteristics

The model is evaluated at different points in a trajectory to measure how
early useful warning signals appear.

Reproducible experiments

Experiments combine:

an agent configuration
a benchmark
an optional intervention
a random seed

This makes experiments repeatable and easier to compare.

Local dashboard

Run data is stored in SQLite and can also be exposed through the local
dashboard.

Architecture
                    CLI / Dashboard
                           |
                           v
                 Agent Runtime
                           |
                           v
          Agent <-> Tools <-> Recorder
                           |
                           v
              Deterministic Evaluation
                           |
                           v
                Failure Analysis
                           |
                           v
              Reliability Prediction
                           |
                           v
                    SQLite Storage

The main components are intentionally separated:

Component	Responsibility
Runtime	Executes the agent and records the trajectory
Tools	Provides controlled operations available to the agent
Evaluation	Determines whether the task actually succeeded
Failure analysis	Identifies observable failure patterns
Reliability	Extracts trajectory features and predicts failure
Storage	Persists runs and telemetry in SQLite
CLI / Dashboard	Provides access to the collected data

See docs/architecture.md for the detailed design.

Quickstart
Requirements
Python
Git
uv

Clone the repository:

git clone https://github.com/DushaneW/agentic-reliability-lab.git
cd agentic-reliability-lab

Install dependencies:

uv sync
uv pip install -e .

Create the local environment file:

cp .env.example .env

Run the smoke benchmark:

arl benchmark run benchmarks/smoke

List recorded runs:

arl runs list

Inspect a run:

arl runs inspect <RUN_ID>

Analyze a run:

arl runs analyze <RUN_ID>
Example

A normal benchmark run looks like:

$ arl benchmark run benchmarks/smoke

Task        Run ID    Result  Steps  Failure category
smoke-001   ...       PASS    5      none
smoke-002   ...       PASS    5      none

2/2 tasks passed

A failed run can be analyzed directly:

$ arl runs analyze <RUN_ID>

Run <RUN_ID>
Diagnosis: looping_failure
Confidence: 0.95
Wasted tool calls: 3
Recovered: False

  - repeated identical tool calls were detected

The actual run ID and evidence are generated from the local SQLite database.

Benchmarks

The repository currently contains 28 benchmark tasks across 6 suites.

Core benchmark suites

The main benchmark generator produces 23 tasks across:

smoke
coding
tool_use
recovery
adversarial

Tasks are generated by:

scripts/generate_benchmark_tasks.py

The current reference agent intentionally implements a narrow underlying
operation: summing numbers and writing the result.

This is a deliberate limitation rather than an attempt to make the benchmark
appear broader than the implementation actually is.

See docs/benchmarks.md.

Failure-mode benchmark

The repository also includes:

benchmarks/failures

This is a deterministic failure-analysis suite containing five hand-designed
scenarios:

Scenario	Expected diagnosis
Error recovery	error_recovery_failure
Repeated action	looping_failure
Timeout	timeout_failure
Tool misuse	tool_selection_failure
Partial recovery	successful recovery

Run the suite with:

arl benchmark run benchmarks/failures

The scenarios are designed to make failure detectors reproducible and easy to
inspect.

A generated Markdown example is included at:

docs/examples/failure_report_example.md

Failure analysis

The diagnostic system deliberately uses explicit rules instead of generating
free-form explanations.

For example, a looping failure can be detected when an agent repeatedly issues
the same tool call with identical arguments.

An error-recovery failure requires stronger evidence: an operation fails and the
agent subsequently retries the same failing operation instead of adapting.

This makes the diagnosis:

deterministic
inspectable
reproducible
tied to actual trajectory events

See docs/failure-analysis.md.

Reliability prediction

The baseline model uses logistic regression over nine hand-built trajectory
features.

Train the model:

arl reliability train --seeds-per-profile 10

Evaluate it:

arl reliability evaluate

The complete methodology and feature definitions are documented in:

docs/research/methodology.md

Results

The repository includes the reliability experiment results in:

models/reliability_metrics.json

The reported dataset contains:

6,630 trajectory-prefix examples
1,360 runs
17 tasks used by the reliability experiment
8 fault profiles
10 seeds per profile
41% failure rate
Overall performance
Metric	Score
AUROC	0.893
AUPRC	0.860
Precision	0.725
Recall	0.780
F1	0.752
False positive rate	0.211
Early warning

AUROC by trajectory completion:

Completion	AUROC
20%	0.599
40%	0.802
60%	0.849
80%	0.958
100%	1.000

The 100% result should be interpreted cautiously. At full trajectory
completion, the system already has access to the complete run.

The more relevant question is whether useful predictive signal appears earlier.

An ablation using only trajectory_length achieves 0.555 AUROC under the
same split, indicating that the reported signal is not simply a proxy for
trajectory length.

See docs/research/experiments.md for the
complete analysis.

Experiments

Run the baseline experiment:

arl experiment run experiments/configs/baseline.yaml

Run the noisy-agent configuration:

arl experiment run experiments/configs/noisy_agent.yaml

Compare experiments:

arl experiment compare <EXPERIMENT_A_ID> <EXPERIMENT_B_ID>

The currently implemented intervention strategy stops execution when failure
is predicted.

It is intentionally treated as a cost-control mechanism, not a recovery
mechanism.

See docs/configuration.md.

Limitations

The most important limitation is that the reference agent is a deterministic
scripted policy rather than a real LLM agent.

Other limitations include:

narrow task diversity
controlled fault injection
feature collinearity
no cross-validation in the baseline experiment
a single implemented intervention strategy
no production-scale evaluation
no detailed cost or latency measurements

The results should therefore be interpreted as a study of the measurement and
analysis pipeline, rather than a claim that these metrics transfer directly
to production agents.

See docs/research/limitations.md.

Security

The shell tool can execute real subprocesses.

Execution is restricted using:

an allow-list
an empty environment
a temporary working directory
a timeout

These are process-level restrictions, not container isolation.

The original development environment did not provide a Docker daemon for
verifying a containerized execution model.

See SECURITY.md for the complete threat model.

Development

Install development dependencies:

make install

Run tests:

make test

Run linting:

make lint

Run type checking:

make typecheck

Run the complete check suite:

make check

Run the benchmark:

make benchmark

Start the dashboard:

make dashboard
Verification

The current repository passes:

76 tests
ruff check .
mypy -p arl --strict

Latest verification:

76 passed
All checks passed!
Success: no issues found in 50 source files
Documentation
Document	Description
docs/getting-started.md	Getting started
docs/architecture.md	System architecture
docs/benchmarks.md	Benchmark design
docs/evaluation.md	Deterministic evaluation
docs/failure-analysis.md	Failure detection
docs/configuration.md	Configuration
docs/research/methodology.md	Research methodology
docs/research/experiments.md	Experiment results
docs/research/limitations.md	Limitations
SECURITY.md	Security model
Contributing

Contributions, bug reports, research ideas, and improvements are welcome.

See CONTRIBUTING.md for development guidelines.

Issue templates are available under:

.github/ISSUE_TEMPLATE/
License

Apache 2.0.

See LICENSE.
