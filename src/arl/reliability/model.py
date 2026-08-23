"""Baseline reliability predictor.

Logistic regression over the hand-built features in features.py. Starting
here (not a neural network) is deliberate: the feature set is small,
interpretable, and the priority is measuring whether *any* signal in the
trajectory predicts failure before it happens — model capacity is not the
bottleneck at this dataset size.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from arl.reliability.dataset import DatasetExample
from arl.reliability.features import FEATURE_NAMES


@dataclass
class EvalMetrics:
    auroc: float
    auprc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    n_examples: int
    per_fraction_auroc: dict[float, float]


class ReliabilityModel:
    def __init__(self) -> None:
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.is_trained = False

    def fit(self, examples: list[DatasetExample]) -> None:
        x = np.array([e.features.as_list() for e in examples])
        y = np.array([e.label_failure for e in examples])
        self.classifier.fit(x, y)
        self.is_trained = True

    def predict_proba(self, feature_vector: list[float]) -> float:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained")
        return float(self.classifier.predict_proba([feature_vector])[0][1])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.classifier, f)

    @classmethod
    def load(cls, path: str | Path) -> ReliabilityModel:
        model = cls()
        with open(path, "rb") as f:
            model.classifier = pickle.load(f)  # noqa: S301
        model.is_trained = True
        return model


def train_test_split_examples(
    examples: list[DatasetExample], test_size: float = 0.3, random_state: int = 0
) -> tuple[list[DatasetExample], list[DatasetExample]]:
    run_ids = sorted({e.run_id for e in examples})
    train_runs, test_runs = train_test_split(
        run_ids, test_size=test_size, random_state=random_state
    )
    train_set = set(train_runs)
    train = [e for e in examples if e.run_id in train_set]
    test = [e for e in examples if e.run_id not in train_set]
    return train, test


def evaluate_model(model: ReliabilityModel, test_examples: list[DatasetExample]) -> EvalMetrics:
    x = np.array([e.features.as_list() for e in test_examples])
    y_true = np.array([e.label_failure for e in test_examples])
    y_proba = model.classifier.predict_proba(x)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    negatives = y_true == 0
    false_positives = int(((y_pred == 1) & negatives).sum())
    fpr = false_positives / negatives.sum() if negatives.sum() else 0.0

    per_fraction: dict[float, float] = {}
    for fraction in sorted({e.fraction for e in test_examples}):
        idx = [i for i, e in enumerate(test_examples) if e.fraction == fraction]
        if len(set(y_true[idx])) > 1:
            per_fraction[fraction] = float(roc_auc_score(y_true[idx], y_proba[idx]))

    return EvalMetrics(
        auroc=float(auroc),
        auprc=float(auprc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=float(fpr),
        n_examples=len(test_examples),
        per_fraction_auroc=per_fraction,
    )


def feature_importances(model: ReliabilityModel) -> dict[str, float]:
    coefs = model.classifier.coef_[0]
    return dict(zip(FEATURE_NAMES, [float(c) for c in coefs], strict=True))
