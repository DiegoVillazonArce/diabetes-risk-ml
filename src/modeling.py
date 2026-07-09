"""Modeling for the BRFSS 2015 diabetes dataset (P4 baselines, P5 comparison).

P4 (Baseline Modeling) establishes the trivial reference (`DummyClassifier`,
US-0401) and the first interpretable model (`LogisticRegression`, US-0402)
on top of the P3 data contract in `src.data`. P5 (Model Comparison and
Selection, Epic E8) extends the same module with a restrained tree-based
candidate (`HistGradientBoostingClassifier`, US-0801), a single
imbalance-aware Logistic Regression variant, an in-memory comparison table,
and deterministic primary-model selection (US-0802, D-016).

Splits come exclusively from `src.data.prepare_data()` /
`src.data.split_data()`; this module never reloads or re-splits raw data.
Models are fit on the train split only and evaluated on train and test only;
the calibration split is never read, so it stays reserved for later
probability calibration work (P8).

Preprocessing is intentionally minimal: Logistic Regression gets standard
scaling (BMI spans [12, 98] while binary indicators are 0/1) and the
tree-based candidate consumes the validated `uint8` features directly,
nothing else. The P2 EDA's moderate feature-pair correlations
(`GenHlth`/`PhysHlth`/`DiffWalk` and `Education`/`Income`, Spearman
~0.42-0.45) were reviewed for this design: all 21 features are kept, because
moderate collinearity mainly affects coefficient interpretability and the
default L2 regularization keeps the Logistic Regression fit stable. No
features are dropped on correlation grounds.

Results are returned as lightweight in-memory structures; this module never
writes a model artifact or processed file. Artifact serialization is
deferred to the Streamlit MVP phase (P6) per D-017, using the D-010 format,
and D-013 still governs how artifacts reach deployment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
    """Feature/target frames for modeling: train and test only.

    The calibration split is deliberately absent so P4/P5 modeling code
    cannot consume it; it stays reserved for later probability calibration
    work (P8).
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass(frozen=True)
class BaselineResult:
    """A fitted model with its train and test metrics.

    Named after its P4 origin; P5 comparison candidates reuse the same
    structure so every model reports metrics through one contract.
    """

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


def build_logistic_regression_balanced_variant(
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build the single P5 imbalance-aware variant (Epic E8).

    Identical to the P4 Logistic Regression baseline except for
    `class_weight="balanced"`, so the comparison isolates the effect of
    simple reweighting on the low default-threshold recall observed in P4.
    This is the only imbalance-aware variant in P5; SMOTE, other resampling,
    and threshold tuning stay out of scope.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_candidate(
    random_state: int = RANDOM_SEED,
) -> HistGradientBoostingClassifier:
    """Build the restrained tree-based candidate (US-0801).

    `HistGradientBoostingClassifier` with library defaults and a fixed seed:
    it is fast on the ~177k-row train split, supports `predict_proba`, needs
    no feature scaling, and is fully deterministic for a given seed. No
    hyperparameter tuning in P5.
    """
    return HistGradientBoostingClassifier(random_state=random_state)


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


def _train_and_evaluate(
    named_builders: Iterable[tuple[str, Callable[..., object]]],
    splits: DataSplits,
    random_state: int,
) -> dict[str, BaselineResult]:
    """Fit each model on the train split only; evaluate on train and test."""
    model_data = to_train_test_data(splits)
    results: dict[str, BaselineResult] = {}
    for name, builder in named_builders:
        model = builder(random_state=random_state)
        model.fit(model_data.X_train, model_data.y_train)
        results[name] = BaselineResult(
            name=name,
            model=model,
            train_metrics=evaluate_model(model, model_data.X_train, model_data.y_train),
            test_metrics=evaluate_model(model, model_data.X_test, model_data.y_test),
        )
    return results


def train_and_evaluate_baselines(
    splits: DataSplits, random_state: int = RANDOM_SEED
) -> dict[str, BaselineResult]:
    """Fit both P4 baselines on the train split; evaluate on train and test.

    Returns `{"dummy": ..., "logistic_regression": ...}` as in-memory
    `BaselineResult` structures. The calibration split is never read.
    """
    return _train_and_evaluate(
        (
            ("dummy", build_dummy_baseline),
            ("logistic_regression", build_logistic_regression_baseline),
        ),
        splits,
        random_state,
    )


def compare_models(
    splits: DataSplits, random_state: int = RANDOM_SEED
) -> dict[str, BaselineResult]:
    """Fit and evaluate every P5 comparison candidate (US-0801).

    Candidates: the two P4 baselines, the single imbalance-aware Logistic
    Regression variant, and the restrained tree-based candidate. All models
    fit on the train split only and are evaluated on train and test only
    with the shared metric protocol; the calibration split is never read.
    Results stay in memory; nothing is written to disk.
    """
    return _train_and_evaluate(
        (
            ("dummy", build_dummy_baseline),
            ("logistic_regression", build_logistic_regression_baseline),
            ("logistic_regression_balanced", build_logistic_regression_balanced_variant),
            ("hist_gradient_boosting", build_hist_gradient_boosting_candidate),
        ),
        splits,
        random_state,
    )


def comparison_table(results: dict[str, BaselineResult]) -> pd.DataFrame:
    """Flatten comparison results into a long-format in-memory table.

    One row per (model, split) with the documented metric protocol as
    columns; the confusion matrix is flattened to tn/fp/fn/tp columns.
    """
    rows = []
    for name, result in results.items():
        for split_name, metrics in (
            ("train", result.train_metrics),
            ("test", result.test_metrics),
        ):
            row: dict[str, object] = {"model": name, "split": split_name}
            for key in METRIC_KEYS:
                if key == "confusion_matrix":
                    row.update(metrics[key])
                else:
                    row[key] = metrics[key]
            rows.append(row)
    return pd.DataFrame(rows)


# Model-selection criteria (US-0802, D-016), fixed here before any comparison
# runs so selection cannot be tuned after seeing results:
# 1. Reference models (the dummy baseline) are never selectable.
# 2. Candidates with an obvious overfitting signal -- train PR-AUC exceeding
#    test PR-AUC by more than OVERFITTING_PR_AUC_GAP_LIMIT -- rank below every
#    well-generalizing candidate.
# 3. Remaining candidates rank by test-split metrics in
#    SELECTION_RANKING_METRICS order. Accuracy is deliberately excluded: at
#    ~13.9% positive prevalence an always-negative model already scores ~86%,
#    so accuracy cannot separate useful candidates.
# 4. The model name is the final tie-break, making selection deterministic.
SELECTION_RANKING_METRICS = ("pr_auc", "f1", "recall", "precision", "roc_auc")
OVERFITTING_PR_AUC_GAP_LIMIT = 0.10
REFERENCE_MODEL_NAMES = ("dummy",)


@dataclass(frozen=True)
class ModelSelection:
    """The selected primary model with the criteria that ranked it first."""

    name: str
    ranking: tuple[str, ...]
    criteria: dict[str, dict]
    rationale: str


def _selection_criteria(result: BaselineResult) -> dict:
    """Per-candidate values consumed by the selection ranking."""
    criteria: dict[str, object] = {
        metric: result.test_metrics[metric] for metric in SELECTION_RANKING_METRICS
    }
    gap = result.train_metrics["pr_auc"] - result.test_metrics["pr_auc"]
    criteria["train_test_pr_auc_gap"] = gap
    criteria["overfitting_risk"] = gap > OVERFITTING_PR_AUC_GAP_LIMIT
    return criteria


def select_primary_model(results: dict[str, BaselineResult]) -> ModelSelection:
    """Apply the documented deterministic selection criteria (US-0802, D-016).

    See the criteria comment above `SELECTION_RANKING_METRICS`. Raises
    ValueError when `results` contains only reference models, because the
    trivial baseline must never become the primary model.
    """
    candidates = {
        name: result
        for name, result in results.items()
        if name not in REFERENCE_MODEL_NAMES
    }
    if not candidates:
        raise ValueError(
            "No selectable candidates: results contain only reference models "
            f"({sorted(results)})."
        )

    criteria = {name: _selection_criteria(result) for name, result in candidates.items()}

    def sort_key(name: str) -> tuple:
        values = criteria[name]
        return (
            values["overfitting_risk"],
            *(-values[metric] for metric in SELECTION_RANKING_METRICS),
            name,
        )

    ranking = tuple(sorted(candidates, key=sort_key))
    selected = ranking[0]
    chosen = criteria[selected]
    rationale = (
        f"Selected '{selected}': test PR-AUC {chosen['pr_auc']:.4f}, "
        f"F1 {chosen['f1']:.4f}, recall {chosen['recall']:.4f}, "
        f"precision {chosen['precision']:.4f}, ROC-AUC {chosen['roc_auc']:.4f}; "
        f"train/test PR-AUC gap {chosen['train_test_pr_auc_gap']:.4f} "
        f"(overfitting limit {OVERFITTING_PR_AUC_GAP_LIMIT}). Ranking "
        "prioritizes test PR-AUC and positive-class F1/recall/precision over "
        "accuracy because the selected population is only ~13.9% positive."
    )
    return ModelSelection(
        name=selected, ranking=ranking, criteria=criteria, rationale=rationale
    )
