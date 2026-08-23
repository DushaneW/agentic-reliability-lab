# Reliability Prediction

See `docs/research/methodology.md` and `docs/research/experiments.md` for
the full research writeup. This page is the practical/API-level summary.

## Training

```bash
uv run arl reliability train --seeds-per-profile 10
```

Generates a labeled dataset (`arl/reliability/dataset.py`) from the
benchmark suite under 8 fault profiles, trains a logistic regression
(`arl/reliability/model.py`), and writes:

- `models/reliability_model.pkl` — the fitted scikit-learn classifier
- `models/reliability_metrics.json` — AUROC/AUPRC/precision/recall/F1,
  per-completion-fraction AUROC, and feature importances from this run

## Using the trained model

```python
from arl.reliability.model import ReliabilityModel
from arl.reliability.features import extract_features

model = ReliabilityModel.load("models/reliability_model.pkl")
probability = model.predict_proba(extract_features(trajectory_events).as_list())
```

## Feature set

See `arl/reliability/features.py` — 9 features computed purely from the
event log available at prediction time (no lookahead). Listed with
descriptions in `docs/research/methodology.md`.

## Intervention

`arl/experiments/intervention.py` implements one strategy: force the agent
to `finish` when predicted failure probability crosses a threshold,
instead of letting it keep burning steps. Read
`docs/research/limitations.md` before assuming this "fixes" failing
runs — it only reduces wasted steps on runs already headed for the label
it predicts; it cannot make a failing run succeed.
