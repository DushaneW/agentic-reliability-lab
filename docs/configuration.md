# Configuration

## Environment variables

See `.env.example`. Currently one variable, `ARL_DB_PATH`, read by the
dashboard (`arl/dashboard/app.py`) to locate the SQLite database; CLI
commands take `--db` explicitly instead and default to `arl.db` in the
current directory.

## Experiment configs

YAML files under `experiments/configs/`, validated against
`arl.domain.experiment.ExperimentConfig`:

```yaml
name: baseline-clean
agent_type: reference
benchmark_path: benchmarks/coding
seed: 42
fault_profile: {}        # kwargs for arl.agents.reference.FaultProfile
intervention:
  enabled: false
  probability_threshold: 0.6
```

`fault_profile` keys map directly onto `FaultProfile`'s fields
(`wrong_tool_rate`, `repeat_action_rate`, `ignore_error_rate`,
`premature_termination_rate`); omit for a clean (fault-free) agent.

## Task grader config

Per-task grader configuration is documented in `docs/benchmarks.md`.

## Reliability model paths

`arl reliability train`/`evaluate` default to `models/reliability_model.pkl`
and `models/reliability_metrics.json`; override with `--model-path` /
`--metrics-path`.
