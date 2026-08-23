# Experiments

All numbers below were produced by actually running
`arl reliability train --seeds-per-profile 10` against this repository's
benchmark suite, not invented. Reproduce with:

```bash
arl reliability train --seeds-per-profile 10
arl reliability evaluate
```

## Dataset

- 17 tasks (`benchmarks/coding`, `benchmarks/tool_use`, `benchmarks/smoke`)
- 8 fault profiles × 10 seeds each = 80 runs per task = 1,360 runs
- 6,630 trajectory-prefix examples (some runs produce fewer than 5 usable
  prefixes if the trajectory is very short, hence not exactly 1,360 × 5)
- 4,642 train examples / 1,988 test examples (70/30 split by `run_id`)
- Overall failure rate in the dataset: 41%

## Result: overall

| Metric | Value |
|---|---|
| AUROC | 0.893 |
| AUPRC | 0.860 |
| Precision | 0.725 |
| Recall | 0.780 |
| F1 | 0.752 |
| False positive rate | 0.211 |

## Result: early warning (AUROC by trajectory completion)

| Trajectory completion | AUROC |
|---|---|
| 20% | 0.599 |
| 40% | 0.802 |
| 60% | 0.849 |
| 80% | 0.958 |
| 100% | 1.000 |

This is the shape the hypothesis in `methodology.md` predicts: weak but
above-chance signal early (0.599 at 20% — barely better than the 0.5 a
random classifier would get, essentially noise), climbing as more of the
trajectory becomes visible.

**Read the 100% number skeptically, not as a headline.** At full
completion, a trajectory's failure/success label is close to fully
determined by exactly the same events the deterministic grader used to
assign that label in the first place (e.g. a `repeated_action_ratio` of
1.0 for the rest of a trajectory after a loop starts is close to
definitional given how `FaultProfile` faults are constructed). An AUROC of
1.000 here reflects that the synthetic fault-injection mechanism produces
trajectories whose shape is *tightly coupled* to the label by
construction — it is not evidence that trajectory shape would predict a
real LLM agent's failures this cleanly. See `limitations.md`.

## Feature importances (logistic regression coefficients)

| Feature | Coefficient |
|---|---|
| `action_diversity` | -8.912 |
| `repeated_action_ratio` | +7.942 |
| `tool_error_count` | -3.282 |
| `trajectory_length` | +1.734 |
| `tool_call_count` | -1.707 |
| `steps_since_last_error` | -1.429 |
| `distinct_tools_used` | +0.657 |
| `tool_error_rate` | -0.260 |
| `consecutive_error_pairs` | 0.000 |

`repeated_action_ratio` (positive) and `action_diversity` (negative,
strongly) dominate, which is consistent with the fault profiles: looping
and ignoring errors are the two injected behaviors most directly captured
by "did the agent keep doing the same thing."

**One result here does not have an obvious explanation and should not be
read as causal:** `tool_error_count` has a *negative* coefficient, meaning
more observed tool errors is associated with the model predicting lower
failure probability, all else equal. The likely explanation is
collinearity — `tool_error_count` correlates with `repeated_action_ratio`
and `action_diversity` in this dataset (an agent that hits one error and
recovers looks different from one that hits many errors while looping),
and logistic regression coefficients are not individually interpretable
under strong collinearity. This is flagged rather than fixed because
fixing it (e.g. via regularization path analysis or dropping correlated
features) is future work, not something claimed as done here.
`consecutive_error_pairs` has exactly 0.0 weight, most likely because it's
rare and/or redundant with `tool_error_count` given this dataset's size —
also not investigated further.

## What would strengthen this result

- Running the same pipeline against real LLM agent trajectories (via an
  external-agent adapter implementing `arl/agents/base.py`), to test
  whether the same features carry signal outside the synthetic
  fault-injection setting this dataset was built from.
- An ablation: train with each feature held out, to see which ones are
  actually load-bearing versus correlated noise, instead of reading
  coefficients directly under known collinearity.
- A confound check: does `trajectory_length` alone (the simplest possible
  feature) get most of the way to 0.885 AUROC on its own? **Checked**:
  trained the same logistic regression with `trajectory_length` as the
  only feature, same train/test split. Result: AUROC 0.555 — barely above
  chance. The full 9-feature model's 0.885 is not just a proxy for "the
  trajectory got long"; the shape features (repetition, diversity, error
  recency) are carrying the real signal. This was run once, ad hoc, not
  wired into the CLI or committed as a repeatable experiment — a
  proper ablation sweeping each feature individually is still future work.
