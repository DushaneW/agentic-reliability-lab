# Getting Started

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Install

```bash
git clone https://github.com/<you>/agentic-reliability-lab
cd agentic-reliability-lab
uv sync
uv pip install -e .
cp .env.example .env
```

## Run the smoke benchmark

```bash
uv run arl benchmark run benchmarks/smoke
```

You should see 2 tasks, both passing, in a few milliseconds — the
reference agent is deterministic and the tasks are trivial by design (see
`docs/benchmarks.md`).

## Inspect a run

```bash
uv run arl runs list
uv run arl runs inspect <RUN_ID>   # accepts a prefix of the full run id
```

## Train the reliability model

```bash
uv run arl reliability train
uv run arl reliability evaluate
```

This regenerates a labeled trajectory dataset from the benchmark suite
(takes a few seconds) and trains a logistic regression classifier. See
`docs/research/experiments.md` for what the numbers mean.

## Run an experiment

```bash
uv run arl experiment run experiments/configs/baseline.yaml
uv run arl experiment run experiments/configs/noisy_agent.yaml
uv run arl experiment compare <EXPERIMENT_A_ID> <EXPERIMENT_B_ID>
```

## Launch the dashboard

```bash
make dashboard
# or: uv run uvicorn arl.dashboard.app:app --reload --port 8000
```

Open `http://localhost:8000`.

## Run the test suite

```bash
make test      # or: uv run pytest -q
make check     # lint + typecheck + test
```
