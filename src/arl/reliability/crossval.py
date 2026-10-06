"""Evaluation protocols for the reliability predictor.

The original single random run-level split answers "does the model work on
new runs drawn from the same tasks and fault profiles it trained on". That
is the easiest possible question. This module adds harder ones:

* ``random_run``      - GroupKFold by run (the old protocol, now k-fold).
* ``leave_task_out``  - every fold's test tasks are unseen in training.
* ``leave_profile_out`` - every fold's test fault profiles are unseen in
  training. This is the closest thing available to "does it generalise to a
  failure mode it was not trained on" without a real LLM agent.

``leave_task_out`` and ``leave_profile_out`` use true leave-one-group-out
folds. With grouped k-fold, pooled AUROC for these protocols depended on k
(for the 20% prefix it moved between 0.76 and 0.86 for k=3..8) because each
fold's held-out groups have different base rates and each fold has its own
model; leave-one-group-out removes the k dependence, and the per-group
AUROC reported alongside avoids pooling scores across models altogether.

All pooled metrics are computed on out-of-fold predictions. Confidence
intervals come from a cluster bootstrap over runs, because prefixes of one
run (and seeds of one task/profile) are not independent samples.

Headline numbers use only prefixes at <= ``EARLY_MAX_FRACTION`` completion.
At 100% the outcome is usually already visible in the trajectory, so
including it inflates the pooled score without measuring early warning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from arl.reliability.dataset import DatasetExample
from arl.reliability.features import FEATURE_NAMES

EARLY_MAX_FRACTION = 0.6

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Protocol(StrEnum):
    RANDOM_RUN = "random_run"
    LEAVE_TASK_OUT = "leave_task_out"
    LEAVE_PROFILE_OUT = "leave_profile_out"


class ConstantScore:
    """Always predicts 0.5. A fold-wise ``DummyClassifier(strategy="prior")``
    would emit a different constant per fold, and pooling those out-of-fold
    scores gives a spurious AUROC away from 0.5 (the training prior is
    anti-correlated with the held-out base rate)."""

    def fit(self, x: FloatArray, y: IntArray) -> ConstantScore:
        del x, y
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return np.full((len(x), 2), 0.5)


@dataclass(frozen=True)
class PredictorSpec:
    """A named model plus the subset of features it is allowed to see."""

    name: str
    columns: tuple[str, ...]
    make: Callable[[], Any]


def default_predictors() -> list[PredictorSpec]:
    all_cols = tuple(FEATURE_NAMES)
    return [
        PredictorSpec("constant (no features)", all_cols, ConstantScore),
        PredictorSpec(
            "length only",
            ("trajectory_length",),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
        PredictorSpec(
            "error count only",
            ("tool_error_count",),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
        PredictorSpec(
            "logistic (all features)",
            all_cols,
            lambda: make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
            ),
        ),
        PredictorSpec(
            "gradient boosting (all features)",
            all_cols,
            lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=0),
        ),
    ]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class ProtocolResult:
    predictor: str
    protocol: Protocol
    n_examples: int
    n_runs: int
    auroc_early: Interval
    auprc_early: float
    brier_early: float
    auroc_by_fraction: dict[float, Interval]
    # Early-prefix AUROC inside each held-out task/profile (groups with both
    # classes only). Empty for random_run.
    auroc_by_group_early: dict[str, float]


def _group_ids(examples: list[DatasetExample], protocol: Protocol) -> list[str]:
    if protocol is Protocol.RANDOM_RUN:
        return [e.run_id for e in examples]
    if protocol is Protocol.LEAVE_TASK_OUT:
        return [e.task_id for e in examples]
    return [e.profile_id for e in examples]


def grouped_splits(
    examples: list[DatasetExample], protocol: Protocol, n_splits: int = 5
) -> list[tuple[IntArray, IntArray]]:
    """Index splits. ``n_splits`` only applies to ``random_run``; the other
    protocols always leave exactly one task / profile out per fold."""
    groups = np.array(_group_ids(examples, protocol))
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise ValueError(f"{protocol.value} needs at least 2 groups, found {n_groups}")
    placeholder = np.zeros(len(examples))
    splitter = (
        GroupKFold(n_splits=min(n_splits, n_groups))
        if protocol is Protocol.RANDOM_RUN
        else LeaveOneGroupOut()
    )
    return [
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(placeholder, groups=groups)
    ]


def out_of_fold_predictions(
    examples: list[DatasetExample],
    spec: PredictorSpec,
    protocol: Protocol,
    n_splits: int = 5,
) -> FloatArray:
    """Probability of failure for every example, from a model that never saw
    that example's group (run / task / profile depending on ``protocol``)."""
    cols = [FEATURE_NAMES.index(c) for c in spec.columns]
    x = np.array([e.features.as_list() for e in examples], dtype=np.float64)[:, cols]
    y = np.array([e.label_failure for e in examples], dtype=np.int64)
    oof = np.full(len(examples), np.nan)
    for train_idx, test_idx in grouped_splits(examples, protocol, n_splits):
        model = spec.make()
        model.fit(x[train_idx], y[train_idx])
        proba = model.predict_proba(x[test_idx])
        # A fold whose training data has one class yields a single column.
        oof[test_idx] = proba[:, 1] if proba.shape[1] == 2 else float(y[train_idx][0])
    return oof


def _auroc(y: IntArray, p: FloatArray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _cluster_bootstrap_auroc(
    y: IntArray,
    p: FloatArray,
    runs: npt.NDArray[np.str_],
    n_boot: int,
    seed: int,
) -> Interval:
    point = _auroc(y, p)
    rng = np.random.default_rng(seed)
    unique_runs = np.unique(runs)
    index_by_run = {r: np.flatnonzero(runs == r) for r in unique_runs}
    stats: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_runs, size=len(unique_runs), replace=True)
        idx = np.concatenate([index_by_run[r] for r in sampled])
        stat = _auroc(y[idx], p[idx])
        if not np.isnan(stat):
            stats.append(stat)
    if not stats:
        return Interval(point, float("nan"), float("nan"))
    low, high = np.percentile(stats, [2.5, 97.5])
    return Interval(point, float(low), float(high))


def evaluate_protocol(
    examples: list[DatasetExample],
    spec: PredictorSpec,
    protocol: Protocol,
    n_boot: int = 300,
    seed: int = 0,
) -> ProtocolResult:
    oof = out_of_fold_predictions(examples, spec, protocol)
    y = np.array([e.label_failure for e in examples], dtype=np.int64)
    frac = np.array([e.fraction for e in examples])
    runs = np.array([e.run_id for e in examples])

    early = frac <= EARLY_MAX_FRACTION
    auroc_early = _cluster_bootstrap_auroc(y[early], oof[early], runs[early], n_boot, seed)
    has_both = len(set(y[early].tolist())) > 1
    auprc = float(average_precision_score(y[early], oof[early])) if has_both else float("nan")
    brier = float(brier_score_loss(y[early], oof[early]))

    by_fraction: dict[float, Interval] = {}
    for f in sorted(set(frac.tolist())):
        m = frac == f
        by_fraction[float(f)] = _cluster_bootstrap_auroc(y[m], oof[m], runs[m], n_boot, seed)

    by_group: dict[str, float] = {}
    if protocol is not Protocol.RANDOM_RUN:
        group_ids = np.array(_group_ids(examples, protocol))
        for g in sorted(set(group_ids.tolist())):
            m = early & (group_ids == g)
            value = _auroc(y[m], oof[m])
            if not np.isnan(value):
                by_group[g] = value

    return ProtocolResult(
        predictor=spec.name,
        protocol=protocol,
        n_examples=len(examples),
        n_runs=len(set(runs.tolist())),
        auroc_early=auroc_early,
        auprc_early=auprc,
        brier_early=brier,
        auroc_by_fraction=by_fraction,
        auroc_by_group_early=by_group,
    )


def evaluate_all(
    examples: list[DatasetExample],
    protocols: list[Protocol] | None = None,
    predictors: list[PredictorSpec] | None = None,
    n_boot: int = 300,
    seed: int = 0,
) -> list[ProtocolResult]:
    results: list[ProtocolResult] = []
    for protocol in protocols or list(Protocol):
        for spec in predictors or default_predictors():
            results.append(evaluate_protocol(examples, spec, protocol, n_boot, seed))
    return results


def render_markdown(results: list[ProtocolResult]) -> str:
    lines: list[str] = []
    for protocol in dict.fromkeys(r.protocol for r in results):
        subset = [r for r in results if r.protocol is protocol]
        lines.append(
            f"### {protocol.value}  ({subset[0].n_examples} examples, "
            f"{subset[0].n_runs} runs; headline = prefixes <= "
            f"{int(EARLY_MAX_FRACTION * 100)}%)"
        )
        lines.append("")
        if protocol is not Protocol.RANDOM_RUN:
            lines.append(
                "> Pooled numbers below mix out-of-fold scores from different models and "
                "different base rates. Values far from 0.5 for weak baselines (even < 0.5) "
                "are partly that artifact. Prefer the per-group table."
            )
            lines.append("")
        lines.append("| Predictor | AUROC early [95% CI] | AUPRC early | Brier early |")
        lines.append("| --- | --- | --- | --- |")
        for r in subset:
            a = r.auroc_early
            lines.append(
                f"| {r.predictor} | {a.point:.3f} [{a.low:.3f}, {a.high:.3f}] "
                f"| {r.auprc_early:.3f} | {r.brier_early:.3f} |"
            )
        lines.append("")
        fractions = sorted({f for r in subset for f in r.auroc_by_fraction})
        lines.append("AUROC by trajectory completion (point estimate [95% CI]):")
        lines.append("")
        lines.append("| Predictor | " + " | ".join(f"{int(f * 100)}%" for f in fractions) + " |")
        lines.append("| --- |" + " --- |" * len(fractions))
        for r in subset:
            cells = []
            for f in fractions:
                a = r.auroc_by_fraction[f]
                cells.append(f"{a.point:.3f} [{a.low:.3f}, {a.high:.3f}]")
            lines.append(f"| {r.predictor} | " + " | ".join(cells) + " |")
        lines.append("")
        groups = sorted({g for r in subset for g in r.auroc_by_group_early})
        if groups:
            lines.append(
                f"AUROC inside each held-out group (prefixes <= {int(EARLY_MAX_FRACTION * 100)}%; "
                "groups with a single outcome class are omitted; 0.500 = chance):"
            )
            lines.append("")
            lines.append("| Predictor | " + " | ".join(groups) + " |")
            lines.append("| --- |" + " --- |" * len(groups))
            for r in subset:
                row = [
                    f"{r.auroc_by_group_early[g]:.3f}" if g in r.auroc_by_group_early else "-"
                    for g in groups
                ]
                lines.append(f"| {r.predictor} | " + " | ".join(row) + " |")
            lines.append("")
    return "\n".join(lines)
