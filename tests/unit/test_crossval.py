from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from arl.reliability.crossval import (
    PredictorSpec,
    Protocol,
    default_predictors,
    evaluate_protocol,
    grouped_splits,
    out_of_fold_predictions,
)
from arl.reliability.dataset import DatasetExample
from arl.reliability.features import FEATURE_NAMES, FeatureVector


def _example(
    run: str, task: str, profile: str, fraction: float, label: int, signal: float
) -> DatasetExample:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values["tool_error_count"] = signal
    return DatasetExample(
        run_id=run,
        task_id=task,
        profile_id=profile,
        fraction=fraction,
        features=FeatureVector(values=values),
        label_failure=label,
    )


def _synthetic() -> list[DatasetExample]:
    rng = np.random.default_rng(0)
    out: list[DatasetExample] = []
    for t in range(4):
        for p in range(4):
            for s in range(6):
                label = int(rng.random() < 0.5)
                for f in (0.2, 0.6, 1.0):
                    out.append(
                        _example(
                            f"t{t}-p{p}-s{s}",
                            f"t{t}",
                            f"p{p}",
                            f,
                            label,
                            label + rng.normal(0, 0.1),
                        )
                    )
    return out


@pytest.mark.parametrize("protocol", list(Protocol))
def test_splits_never_share_a_group_between_train_and_test(protocol: Protocol) -> None:
    examples = _synthetic()
    key = {
        Protocol.RANDOM_RUN: lambda e: e.run_id,
        Protocol.LEAVE_TASK_OUT: lambda e: e.task_id,
        Protocol.LEAVE_PROFILE_OUT: lambda e: e.profile_id,
    }[protocol]
    all_test: set[int] = set()
    for train, test in grouped_splits(examples, protocol):
        train_groups = {key(examples[i]) for i in train}
        test_groups = {key(examples[i]) for i in test}
        assert train_groups.isdisjoint(test_groups)
        all_test.update(test.tolist())
    assert all_test == set(range(len(examples)))  # every example is tested exactly once


def test_informative_feature_beats_chance_and_ci_is_ordered() -> None:
    examples = _synthetic()
    spec = PredictorSpec("lr", ("tool_error_count",), lambda: LogisticRegression(max_iter=1000))
    r = evaluate_protocol(examples, spec, Protocol.LEAVE_TASK_OUT, n_boot=50)
    assert r.auroc_early.point > 0.95
    assert r.auroc_early.low <= r.auroc_early.point <= r.auroc_early.high


def test_constant_baseline_is_exactly_chance() -> None:
    examples = _synthetic()
    constant = default_predictors()[0]
    r = evaluate_protocol(examples, constant, Protocol.RANDOM_RUN, n_boot=20)
    assert r.auroc_early.point == pytest.approx(0.5)


def test_single_group_is_rejected() -> None:
    examples = [e for e in _synthetic() if e.task_id == "t0"]
    with pytest.raises(ValueError, match="at least 2 groups"):
        out_of_fold_predictions(examples, default_predictors()[0], Protocol.LEAVE_TASK_OUT)


def test_leave_group_out_has_one_fold_per_group_and_is_independent_of_n_splits() -> None:
    examples = _synthetic()
    for protocol, n_groups in ((Protocol.LEAVE_TASK_OUT, 4), (Protocol.LEAVE_PROFILE_OUT, 4)):
        assert len(grouped_splits(examples, protocol, n_splits=2)) == n_groups
        assert len(grouped_splits(examples, protocol, n_splits=99)) == n_groups


def test_per_group_auroc_is_reported_only_for_group_with_both_classes() -> None:
    examples = _synthetic()
    # make profile p0 single-class
    for e in examples:
        if e.profile_id == "p0":
            object.__setattr__(e, "label_failure", 0)
    spec = PredictorSpec("lr", ("tool_error_count",), lambda: LogisticRegression(max_iter=1000))
    r = evaluate_protocol(examples, spec, Protocol.LEAVE_PROFILE_OUT, n_boot=10)
    assert "p0" not in r.auroc_by_group_early
    assert set(r.auroc_by_group_early) <= {"p1", "p2", "p3"}
    assert (
        evaluate_protocol(examples, spec, Protocol.RANDOM_RUN, n_boot=10).auroc_by_group_early == {}
    )
