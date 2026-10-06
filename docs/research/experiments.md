# Experiments

Every number here was produced by running the commands shown against this
repository's benchmark suite. All data are **synthetic**: trajectories come
from the scripted reference agent with injected faults (see
`limitations.md`). The results say how well the pipeline finds the injected
fault shapes, not how well it would predict a real agent's failures.

Reproduce:

```bash
arl reliability cv --seeds-per-profile 10 --n-boot 300 --output docs/research/cv_report.md
arl reliability train --seeds-per-profile 10   # single split, writes models/
```

## Dataset

- 17 tasks (`benchmarks/coding`, `tool_use`, `smoke`), 8 named fault profiles,
  10 seeds each: 1,360 runs, 6,630 trajectory-prefix examples, 41% failures.
- Each run is cut at 20/40/60/80/100% of its events; every prefix is labeled
  with the final outcome of the run it came from.
- Seeds of one (task, profile) pair are near-duplicates because the policy is
  deterministic given the seed, so the effective sample size is well below
  1,360. Confidence intervals use a cluster bootstrap over runs, which
  accounts for prefixes of the same run but **not** for this seed similarity;
  treat the intervals as optimistic.

## Protocols

| Protocol | Test fold contains | Question it answers |
|---|---|---|
| `random_run` | unseen runs (5-fold grouped by run) | does it work on new runs from the same distribution |
| `leave_task_out` | one unseen task | does it transfer across tasks |
| `leave_profile_out` | one unseen fault profile | does it transfer to an unseen failure shape |

Headline numbers use prefixes at <= 60% completion only. At 80-100% the
outcome is largely visible in the log, so including those prefixes inflates
the score without measuring early warning.

An earlier version of `leave_*` used grouped k-fold and pooled the
out-of-fold scores. That made the 20% AUROC move between 0.76 and 0.86 as k
changed from 3 to 8, i.e. the number measured fold composition. It now uses
leave-one-group-out and also reports AUROC *within* each held-out group.

## Results: headline (`random_run`, prefixes <= 60%)

| Predictor | AUROC [95% CI] | AUPRC | Brier |
|---|---|---|---|
| constant | 0.500 | 0.403 | 0.250 |
| trajectory length only | 0.547 [0.533, 0.560] | 0.472 | 0.240 |
| tool error count only | 0.549 [0.521, 0.579] | 0.492 | 0.229 |
| logistic regression, 9 features | 0.771 [0.756, 0.788] | 0.742 | 0.181 |
| gradient boosting, 9 features | 0.791 [0.773, 0.808] | 0.776 | 0.160 |

By completion (logistic regression): 20%: 0.596 [0.558, 0.627], 40%: 0.793,
60%: 0.843, 80%: 0.954, 100%: 1.000. Full tables with intervals for every
predictor and protocol are in `cv_report.md`.

## Results: unseen tasks and unseen fault profiles

- `leave_task_out` is within 0.005 AUROC of `random_run` for logistic
  regression (0.772) and gradient boosting (0.791). **This is expected and not
  evidence of cross-task generalization**: all tasks are the same operation,
  so there is nothing task-specific to fail to transfer.
- `leave_profile_out`, AUROC inside each held-out profile (early prefixes):

| Predictor | early_stop | mixed_heavy | mixed_severe | repeat | wrong_tool | wrong_tool_ignore_error |
|---|---|---|---|---|---|---|
| length only | 0.758 | 0.500 | 0.602 | 0.500 | 0.500 | 0.500 |
| error count only | 0.500 | 0.533 | 0.559 | 0.500 | 0.531 | 0.559 |
| logistic | 0.708 | 0.738 | 0.801 | 0.775 | 0.588 | 0.617 |
| gradient boosting | 0.907 | 0.774 | 0.832 | 0.775 | 0.624 | 0.645 |

  (`clean` and `ignore_error` have no AUROC because neither ever fails: 0 of
  170 runs each. `ignore_error` only changes behavior after an error, and
  that profile injects none, so it is effectively a second clean profile.)
  The pooled
  `leave_profile_out` AUROC in `cv_report.md` mixes scores from different
  models and base rates; for weak baselines it falls well below 0.5, which is
  that artifact, not a real anti-signal. Use the per-profile table.

## Observations vs. hypotheses

Observed on this data:

- Single-feature baselines (length, error count) reach ~0.55 early AUROC; the
  9-feature models reach 0.77-0.79. The signal is not just "long trajectories
  fail".
- Gradient boosting beats logistic regression under `random_run` (0.791 vs
  0.771) but is worse pooled under `leave_profile_out` (0.714 vs 0.785),
  consistent with it fitting the training profiles more tightly. Per-profile it
  is better on most held-out profiles, so this comparison is not settled.
- The weakest held-out profiles for both models are the `wrong_tool` ones
  (0.59-0.65).
- At 20% completion the logistic model is near chance (0.596). Many profiles
  produce an identical short prefix at that point.

Hypotheses, not tested: that `early_stop` is easy because it shortens
trajectories (the length-only baseline also reaches 0.758 there); that the
`wrong_tool` profiles are hard because one injected error is followed by normal
behavior and the agent then often succeeds.

## Legacy single split (`arl reliability train`)

One 70/30 split by run (4,630 train / 2,000 test), all prefixes included:

| Metric | Value |
|---|---|
| AUROC | 0.883 |
| AUPRC | 0.849 |
| Precision | 0.718 |
| Recall | 0.764 |
| F1 | 0.740 |
| False positive rate | 0.213 |

AUROC by completion: 20%: 0.578, 40%: 0.782, 60%: 0.827, 80%: 0.948,
100%: 1.000. The overall figure averages in late prefixes and is not an early
warning number. The 1.000 at 100% reflects that synthetic faults are coupled to
the features by construction.

## Feature coefficients (logistic regression, unscaled features)

`action_diversity` -8.84, `repeated_action_ratio` +7.96, `tool_error_count`
-3.80, `trajectory_length` +1.89, `tool_call_count` -1.85,
`steps_since_last_error` -1.55, `consecutive_error_pairs` +1.28,
`distinct_tools_used` +0.74, `tool_error_rate` -0.22.

Coefficients are on raw feature scales and the features are collinear, so
neither magnitudes nor signs are causal readings (`tool_error_count` being
negative is a collinearity effect, not "errors protect against failure").

## Bug found while writing this: dead feature

`consecutive_error_pairs` had a coefficient of exactly 0.0, and an earlier
version of this document guessed that it was "rare or redundant". It was a
bug: the feature counted adjacent `tool_error` events, but every error is
preceded by its own `tool_call` event, so adjacency never occurs (0 pairs in
200 severe-fault runs). It now counts back-to-back failed tool *outcomes*.
Fixing it changed early AUROC by less than 0.002, so no earlier conclusion
depended on it.

## Not done

- No real-agent trajectories. `arl/agents/llm.py` exists and is tested against
  fake transports only; it has never been run against a live model here.
- No calibration beyond Brier score (no reliability diagram), although
  `intervention.py` treats probabilities as calibrated.
- No per-feature ablation sweep; no unweighted-vs-`balanced` comparison.
- No validation of the failure detectors against hand-labeled real
  trajectories.
