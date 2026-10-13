"""Baseline reliability predictor.

Logistic regression over the hand-built features in features.py. Starting
here (not a neural network) is deliberate: the feature set is small,
interpretable, and the priority is measuring whether *any* signal in the
trajectory predicts failure before it happens — model capacity is not the
bottleneck at this dataset size.
"""

from __future__ import annotations

import json
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


MODEL_FORMAT = "arl-logreg-v1"


class ModelFormatError(ValueError):
    """The model file is not a valid ARL model."""


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
        """Write the fitted coefficients as JSON.

        Not pickle: unpickling a file executes arbitrary code, and a model file is
        exactly the kind of artifact people download and share. The JSON also does
        not depend on the scikit-learn version that produced it.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained")
        payload = {
            "format": MODEL_FORMAT,
            "feature_names": list(FEATURE_NAMES),
            "coef": self.classifier.coef_.tolist(),
            "intercept": self.classifier.intercept_.tolist(),
            "classes": self.classifier.classes_.tolist(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ReliabilityModel:
        source = Path(path)
        if source.suffix in {".pkl", ".pickle"}:
            raise ModelFormatError(
                f"{source}: pickle models are no longer supported because loading one "
                "executes arbitrary code. Retrain with `arl reliability train` to "
                "produce a .json model."
            )
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelFormatError(f"{source}: not valid JSON: {exc.msg}") from exc
        except UnicodeDecodeError as exc:
            raise ModelFormatError(f"{source}: not a text file (binary or pickle?)") from exc
        if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT:
            raise ModelFormatError(f"{source}: unsupported model format")
        if data.get("feature_names") != FEATURE_NAMES:
            raise ModelFormatError(
                f"{source}: feature set differs from this version of arl; retrain the model"
            )
        try:
            coef = np.array(data["coef"], dtype=float)
            intercept = np.array(data["intercept"], dtype=float)
            classes = np.array(data["classes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelFormatError(f"{source}: malformed model data: {exc}") from exc
        expected_coef = (1, len(FEATURE_NAMES))
        if coef.shape != expected_coef or intercept.shape != (1,) or classes.shape != (2,):
            raise ModelFormatError(f"{source}: unexpected coefficient shapes")
        model = cls()
        model.classifier.coef_ = coef
        model.classifier.intercept_ = intercept
        model.classifier.classes_ = classes
        model.classifier.n_features_in_ = len(FEATURE_NAMES)
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
