"""Baseline modeling for the BRFSS 2015 diabetes dataset (P4: Baseline Modeling).

P4 establishes the trivial reference (`DummyClassifier`, US-0401) and the
first interpretable model (`LogisticRegression`, US-0402) on top of the P3
data contract in `src.data`. Splits come exclusively from
`src.data.prepare_data()` / `src.data.split_data()`; this module never
reloads or re-splits raw data. Models are fit on the train split only and
evaluated on train and test only; the calibration split is never read, so it
stays reserved for later probability calibration work (P8).

Preprocessing is intentionally minimal: Logistic Regression gets standard
scaling (BMI spans [12, 98] while binary indicators are 0/1), nothing else.
The P2 EDA's moderate feature-pair correlations (`GenHlth`/`PhysHlth`/
`DiffWalk` and `Education`/`Income`, Spearman ~0.42-0.45) were reviewed for
this design: all 21 features are kept, because moderate collinearity mainly
affects coefficient interpretability and the default L2 regularization keeps
the fit stable. No features are dropped on correlation grounds.

Results are returned as lightweight in-memory structures; no model artifact
or processed file is written in P4. Tree-based candidates, formal model
comparison/selection, and the serialization policy belong to P5 (Epic E8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS, RANDOM_SEED, TARGET, DataSplits

# Metric keys reported for every model/split evaluation. Accuracy is included
# as secondary context only: at ~13.9% positive prevalence, an always-negative
# model already scores ~86% accuracy.
METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "recall",
    "precision",
    "f1",
    "confusion_matrix",
    "accuracy",
)


@dataclass(frozen=True)
class TrainTestData:
    """Feature/target frames for P4: train and test only.

    The calibration split is deliberately absent so P4 code cannot consume
    it; it stays reserved for later probability calibration work.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass(frozen=True)
class BaselineResult:
    """A fitted baseline model with its train and test metrics."""

    name: str
    model: object
    train_metrics: dict
    test_metrics: dict


def to_train_test_data(splits: DataSplits) -> TrainTestData:
    """Convert P3 `DataSplits` into X/y train/test data.

    Rows are taken from the given splits as-is (no re-splitting); the
    calibration split is not read.
    """
    return TrainTestData(
        X_train=splits.train[FEATURE_COLUMNS].copy(),
        y_train=splits.train[TARGET].copy(),
        X_test=splits.test[FEATURE_COLUMNS].copy(),
        y_test=splits.test[TARGET].copy(),
    )


def build_dummy_baseline(random_state: int = RANDOM_SEED) -> DummyClassifier:
    """Build the trivial reference model (US-0401).

    `most_frequent` always predicts the majority class (negative, at ~13.9%
    positive prevalence), so recall/precision/F1 are 0 and accuracy equals
    the negative base rate -- the floor every real model must beat.
    """
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def build_logistic_regression_baseline(random_state: int = RANDOM_SEED) -> Pipeline:
    """Build the first interpretable model (US-0402).

    Standard scaling is the minimal preprocessing Logistic Regression needs
    here: the features share no common scale (BMI in [12, 98], binary
    indicators in {0, 1}) and scaling keeps lbfgs convergence
    well-conditioned. No feature engineering or class balancing in P4.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def predict_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of the positive class (target = 1)."""
    positive_column = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_column]


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute the P4 metric set for one split.

    Ranking metrics (ROC-AUC, PR-AUC) use `y_proba`; threshold metrics
    (recall, precision, F1, confusion matrix, accuracy) use `y_pred` at the
    model's default 0.5 threshold. `zero_division=0` keeps degenerate
    predictors (e.g. the always-negative dummy) well-defined.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_model(model, X: pd.DataFrame, y) -> dict:
    """Evaluate a fitted model on one split with the P4 metric set."""
    return compute_classification_metrics(
        y, model.predict(X), predict_positive_proba(model, X)
    )


def train_and_evaluate_baselines(
    splits: DataSplits, random_state: int = RANDOM_SEED
) -> dict[str, BaselineResult]:
    """Fit both P4 baselines on the train split; evaluate on train and test.

    Returns `{"dummy": ..., "logistic_regression": ...}` as in-memory
    `BaselineResult` structures. The calibration split is never read.
    """
    model_data = to_train_test_data(splits)
    results: dict[str, BaselineResult] = {}
    for name, builder in (
        ("dummy", build_dummy_baseline),
        ("logistic_regression", build_logistic_regression_baseline),
    ):
        model = builder(random_state=random_state)
        model.fit(model_data.X_train, model_data.y_train)
        results[name] = BaselineResult(
            name=name,
            model=model,
            train_metrics=evaluate_model(model, model_data.X_train, model_data.y_train),
            test_metrics=evaluate_model(model, model_data.X_test, model_data.y_test),
        )
    return results
