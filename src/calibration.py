"""Probability calibration and threshold analysis for the frozen D-016 model
(P8, Epic E6: US-0601, US-0606).

P8 evaluates whether sigmoid (Platt) or isotonic calibration makes the
selected `HistGradientBoostingClassifier`'s probabilities more honest, using
the leakage-safe protocol pre-declared in `docs/ml-analysis-plan.md`:

1. The base model is fitted on the train split only and then frozen: no
   refit, retuning, or reselection happens anywhere in P8.
2. The untouched calibration split provides the uncalibrated baseline and
   all calibrator fitting/evaluation data. Comparison and selection use
   stratified five-fold cross-fitting within the calibration split only:
   for each fold, a calibrator is fitted on the other four folds and
   predicts the held-out fold, assembling complete out-of-fold
   probabilities. Per-fold calibrators are discarded after prediction.
3. Selection (D-018) follows the operationalized criteria fixed before any
   out-of-fold result existed: a paired-bootstrap Brier adoption rule
   against the uncalibrated baseline, log loss under the same rule as the
   tie-break between two adoptable methods, a project-defined 0.005
   absolute ROC-AUC/PR-AUC ranking-preservation bound, and sigmoid on
   demonstrated equivalence; `calibration_method = "none"` when no method
   qualifies. Reliability diagrams are visual diagnostics only.
4. The test split participates in no P8 decision. This module deliberately
   exposes no helper that reads it: `CalibrationData` carries train and
   calibration rows only, and the one official P8 test evaluation (run
   after D-018 and D-019 are frozen) lives in `official_test_evaluation`,
   which is never called by the comparison, selection, or threshold code.

Calibrator API (decided against the pinned scikit-learn 1.7.1 source and
documentation before implementation): calibrators are public
`sklearn.calibration.CalibratedClassifierCV` instances wrapping the frozen
base model in `sklearn.frozen.FrozenEstimator` (the 1.6+ replacement for the
deprecated `cv="prefit"`, which 1.8 removes). With a frozen estimator and
`ensemble=False`, `fit(X, y)` never refits the base model (its `fit` is a
no-op) and fits exactly one calibrator on all provided rows; the internal
`cross_val_predict` over the frozen estimator reproduces the frozen model's
scores, so every provided row calibrates. The calibrated score
representation is the base model's `decision_function` output (log-odds):
scikit-learn prefers `decision_function` over `predict_proba` both when
fitting the calibrator and inside `CalibratedClassifierCV.predict_proba`,
so cross-fitting and final serving consume the identical representation by
construction. The quantity compared and served is always the positive-class
probability.

Per-row losses follow the pinned scikit-learn conventions so that their
means equal `sklearn.metrics.brier_score_loss` / `log_loss`: the log loss
clips probabilities to `[eps, 1 - eps]` with the float64 machine epsilon
(isotonic calibration can emit exact 0/1 probabilities).

This module never reads the raw CSV, never re-splits data, and never writes
model artifacts; it consumes the P3 `DataSplits` contract and returns
in-memory results. The offline evidence run (`python -m src.calibration`,
added with the threshold increment) writes only documentation tables and
figures under `docs/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data import FEATURE_COLUMNS, RANDOM_SEED, TARGET, DataSplits
from src.modeling import (
    build_hist_gradient_boosting_candidate,
    predict_positive_proba,
)

# Documentation evidence location for the offline P8 report
# (`python -m src.calibration` regenerates the tables and figures there).
P8_EVIDENCE_DIR_NAME = "p8-calibration"

# Calibration methods compared in P8. "none" is the explicit no-calibration
# outcome: the uncalibrated baseline is retained when no method qualifies.
CALIBRATION_METHODS = ("sigmoid", "isotonic")
NO_CALIBRATION = "none"

# D-018 resolution (2026-07-12, from the recorded out-of-fold evidence in
# docs/p8-calibration/ and docs/decisions.md): neither sigmoid nor isotonic
# passed the pre-declared paired-bootstrap Brier adoption rule against the
# uncalibrated baseline (the frozen D-016 model is already well calibrated
# on the calibration split), so the uncalibrated output is retained. The
# artifact generation path builds this contract; `select_calibration_method`
# remains the pre-declared rule set that produced (and reproduces) it.
CALIBRATION_SELECTION_DECISION = "D-018"
SELECTED_CALIBRATION_METHOD = NO_CALIBRATION

# Pre-declared protocol constants (docs/ml-analysis-plan.md, D-018). The
# bootstrap batch size only bounds memory (at most BOOTSTRAP_BATCH_SIZE x
# n_rows indices are materialized at once); it is part of the fixed protocol
# so the resample stream is reproducible byte for byte.
N_CALIBRATION_FOLDS = 5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_BATCH_SIZE = 500
RANKING_REGRESSION_LIMIT = 0.005  # project guardrail, not a clinical/statistical standard

# Equal-width bins for reliability diagrams and probability histograms.
# Reliability diagrams are visual diagnostics only and never a selection
# criterion; the bin count is fixed so the recorded evidence is reproducible.
RELIABILITY_BINS = 10

# The four probability-quality metrics reported for every contract: two
# proper scoring rules (selection criteria) and two ranking metrics
# (ranking-preservation context for the D-018 guard).
PROBABILITY_METRIC_KEYS = ("brier", "log_loss", "roc_auc", "pr_auc")


@dataclass(frozen=True)
class CalibrationData:
    """Feature/target frames for P8: train and calibration rows only.

    Train rows exist solely to fit the frozen base model; calibration rows
    are the only data calibrators ever see. The test split is deliberately
    absent so no comparison, selection, or threshold helper can consume it
    (mirroring how P4/P5's `TrainTestData` excluded calibration); test
    enters P8 exclusively through `official_test_evaluation`.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series


def to_calibration_data(splits: DataSplits) -> CalibrationData:
    """Convert P3 `DataSplits` into the P8 train/calibration frames.

    Rows are taken from the given splits as-is (no re-splitting); the test
    split is not read.
    """
    return CalibrationData(
        X_train=splits.train[FEATURE_COLUMNS].copy(),
        y_train=splits.train[TARGET].copy(),
        X_calibration=splits.calibration[FEATURE_COLUMNS].copy(),
        y_calibration=splits.calibration[TARGET].copy(),
    )


def train_frozen_base_model(
    data: CalibrationData, random_state: int = RANDOM_SEED
) -> object:
    """Fit the D-016 model on the train rows only; it is frozen afterwards.

    Uses the P5 builder so the configuration is exactly the selected D-016
    `HistGradientBoostingClassifier` (library defaults, fixed seed). Nothing
    in P8 may refit, retune, or reselect this model.
    """
    model = build_hist_gradient_boosting_candidate(random_state=random_state)
    model.fit(data.X_train, data.y_train)
    return model


# ---------------------------------------------------------------------------
# Probability validation and metrics
# ---------------------------------------------------------------------------


def validate_probabilities(probabilities) -> np.ndarray:
    """Require a non-empty 1-D array of finite probabilities in [0, 1].

    Every metric, comparison, and out-of-fold helper funnels probabilities
    through this check so invalid values (NaN, infinities, out-of-range)
    fail loudly instead of silently distorting the P8 evidence.
    """
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "Probabilities must be a non-empty 1-D array; got shape "
            f"{array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Probabilities contain non-finite values (NaN or inf).")
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(
            "Probabilities fall outside [0, 1]: "
            f"min {array.min()!r}, max {array.max()!r}."
        )
    return array


def _validate_binary_targets(y_true) -> np.ndarray:
    """Require a 1-D 0/1 target array with both classes present."""
    array = np.asarray(y_true, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"Targets must be a non-empty 1-D array; got shape {array.shape}."
        )
    values = set(np.unique(array))
    if not values <= {0.0, 1.0}:
        raise ValueError(f"Targets must be binary 0/1; observed values {sorted(values)}.")
    if values != {0.0, 1.0}:
        raise ValueError(
            "Targets must contain both classes; observed only "
            f"{sorted(values)}."
        )
    return array


def brier_losses(y_true, probabilities) -> np.ndarray:
    """Per-row squared error `(p_i - y_i)^2`; its mean is the Brier score."""
    y = _validate_binary_targets(y_true)
    p = validate_probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError(
            f"Targets ({len(y)}) and probabilities ({len(p)}) differ in length."
        )
    return (p - y) ** 2


def log_losses(y_true, probabilities) -> np.ndarray:
    """Per-row negative log-likelihood; its mean equals sklearn's `log_loss`.

    Mirrors the pinned scikit-learn 1.7.1 convention exactly: both the
    positive and the negative class probabilities are clipped to
    `[eps, 1 - eps]` with the float64 machine epsilon before taking logs,
    which keeps isotonic's exact 0/1 outputs finite.
    """
    y = _validate_binary_targets(y_true)
    p = validate_probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError(
            f"Targets ({len(y)}) and probabilities ({len(p)}) differ in length."
        )
    eps = np.finfo(np.float64).eps
    positive = np.clip(p, eps, 1.0 - eps)
    negative = np.clip(1.0 - p, eps, 1.0 - eps)
    return -(y * np.log(positive) + (1.0 - y) * np.log(negative))


def probability_metrics(y_true, probabilities) -> dict:
    """The P8 metric set for one probability contract on one split.

    Brier score and log loss are the proper scoring rules the D-018
    criteria act on; ROC-AUC and PR-AUC are the ranking-preservation
    context for the 0.005 regression guard.
    """
    y = _validate_binary_targets(y_true)
    p = validate_probabilities(probabilities)
    return {
        "brier": float(np.mean(brier_losses(y, p))),
        "log_loss": float(np.mean(log_losses(y, p))),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


# ---------------------------------------------------------------------------
# Reliability diagram and probability distribution evidence
# ---------------------------------------------------------------------------


def reliability_table(
    y_true, probabilities, n_bins: int = RELIABILITY_BINS
) -> pd.DataFrame:
    """Equal-width reliability-diagram data: predicted vs. observed per bin.

    Returns one row per bin with `bin_lower`, `bin_upper`, `n_rows`,
    `mean_predicted_probability`, and `observed_positive_rate` (the last two
    are NaN for empty bins, which plots simply skip). Reliability diagrams
    are visual diagnostics only; they never decide the D-018 selection.
    """
    y = _validate_binary_targets(y_true)
    p = validate_probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError(
            f"Targets ({len(y)}) and probabilities ({len(p)}) differ in length."
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(
        np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1
    )
    rows = []
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        n_rows = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(edges[bin_id]),
                "bin_upper": float(edges[bin_id + 1]),
                "n_rows": n_rows,
                "mean_predicted_probability": (
                    float(p[mask].mean()) if n_rows else float("nan")
                ),
                "observed_positive_rate": (
                    float(y[mask].mean()) if n_rows else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def probability_histogram_table(
    probabilities, n_bins: int = RELIABILITY_BINS
) -> pd.DataFrame:
    """Probability-distribution evidence: row counts per equal-width bin."""
    p = validate_probabilities(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(
        np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1
    )
    counts = np.bincount(bin_index, minlength=n_bins)
    return pd.DataFrame(
        {
            "bin_lower": edges[:-1].astype(float),
            "bin_upper": edges[1:].astype(float),
            "n_rows": counts.astype(int),
            "share": (counts / len(p)).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Stratified folds and out-of-fold assembly within the calibration split
# ---------------------------------------------------------------------------


def stratified_calibration_folds(
    y_calibration,
    n_splits: int = N_CALIBRATION_FOLDS,
    random_state: int = RANDOM_SEED,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Deterministic stratified folds over the calibration rows.

    Returns `(fit_positions, held_out_positions)` pairs of positional
    indices into the calibration split. The fixed seed makes fold
    assignment reproducible, and both classes must be present in every fit
    and held-out part so each per-fold calibrator sees a valid binary
    problem and each held-out evaluation is well-defined.
    """
    y = _validate_binary_targets(y_calibration)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    folds = tuple(
        (fit_positions, held_out_positions)
        for fit_positions, held_out_positions in splitter.split(
            np.zeros((len(y), 1)), y
        )
    )
    for fold_number, (fit_positions, held_out_positions) in enumerate(folds):
        for part_name, positions in (
            ("fit", fit_positions),
            ("held-out", held_out_positions),
        ):
            classes = set(np.unique(y[positions]))
            if classes != {0.0, 1.0}:
                raise ValueError(
                    f"Fold {fold_number} {part_name} part does not contain "
                    f"both classes (observed {sorted(classes)}); the "
                    "calibration split is too small or too imbalanced for "
                    f"{n_splits}-fold cross-fitting."
                )
    return folds


def assemble_out_of_fold(
    n_rows: int, fold_predictions: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Combine per-fold held-out predictions into one out-of-fold vector.

    `fold_predictions` holds `(held_out_positions, probabilities)` pairs.
    Validates the out-of-fold integrity contract: every calibration row
    receives exactly one prediction (no duplicates, no missing rows, no
    out-of-range positions, matching lengths) and all probabilities are
    finite values in [0, 1].
    """
    out_of_fold = np.full(n_rows, np.nan)
    seen = np.zeros(n_rows, dtype=bool)
    for held_out_positions, probabilities in fold_predictions:
        positions = np.asarray(held_out_positions)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError(
                "Held-out positions must be a non-empty 1-D array; got "
                f"shape {positions.shape}."
            )
        if positions.min() < 0 or positions.max() >= n_rows:
            raise ValueError(
                f"Held-out positions fall outside [0, {n_rows - 1}]."
            )
        values = validate_probabilities(probabilities)
        if len(values) != len(positions):
            raise ValueError(
                f"Fold predictions ({len(values)}) and held-out positions "
                f"({len(positions)}) differ in length."
            )
        if seen[positions].any():
            raise ValueError(
                "A calibration row received more than one out-of-fold "
                "prediction; folds must not overlap."
            )
        seen[positions] = True
        out_of_fold[positions] = values
    if not seen.all():
        raise ValueError(
            f"{int((~seen).sum())} calibration row(s) received no "
            "out-of-fold prediction; the folds must cover every row."
        )
    return validate_probabilities(out_of_fold)


# ---------------------------------------------------------------------------
# Uncalibrated baseline (recorded before any calibrator exists)
# ---------------------------------------------------------------------------


def uncalibrated_calibration_probabilities(
    base_model, data: CalibrationData
) -> np.ndarray:
    """The frozen model's positive-class probabilities on calibration rows.

    This is the uncalibrated baseline contract ("none"): the exact quantity
    the deployed schema-version-1 app serves today.
    """
    return validate_probabilities(
        predict_positive_proba(base_model, data.X_calibration)
    )


def uncalibrated_baseline_evaluation(base_model, data: CalibrationData) -> dict:
    """US-0601 baseline evidence, recorded before any calibrator is fitted.

    Returns the frozen model's calibration-split metrics (Brier, log loss,
    ROC-AUC, PR-AUC), the reliability-diagram table, and the probability
    histogram. Only train (already inside the frozen model) and calibration
    rows are involved.
    """
    probabilities = uncalibrated_calibration_probabilities(base_model, data)
    return {
        "metrics": probability_metrics(data.y_calibration, probabilities),
        "reliability": reliability_table(data.y_calibration, probabilities),
        "histogram": probability_histogram_table(probabilities),
    }


# ---------------------------------------------------------------------------
# Calibrators: sigmoid and isotonic over the frozen model's scores
# ---------------------------------------------------------------------------


def build_calibrator(base_model, method: str) -> CalibratedClassifierCV:
    """Build the calibrator contract shared by cross-fitting and serving.

    `CalibratedClassifierCV` over a `FrozenEstimator` with `ensemble=False`
    is the pinned scikit-learn 1.7.1 replacement for the deprecated
    `cv="prefit"`: `fit(X, y)` fits exactly one sigmoid/isotonic calibrator
    on the frozen model's `decision_function` scores for all provided rows
    and never refits the base model. The identical construct is used for
    every per-fold calibrator and for the final serving calibrator, so the
    out-of-fold evidence and the deployed artifact share one score
    representation by construction.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"Unknown calibration method {method!r}; expected one of "
            f"{CALIBRATION_METHODS}."
        )
    return CalibratedClassifierCV(
        FrozenEstimator(base_model), method=method, ensemble=False
    )


def cross_fit_out_of_fold(
    base_model,
    data: CalibrationData,
    method: str,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Stratified five-fold cross-fitting within the calibration split.

    For each fold, a fresh calibrator is fitted on the frozen model's
    scores for the other four folds and predicts the held-out fold; the
    per-fold calibrators are discarded once their held-out predictions
    exist. Returns the complete out-of-fold positive-class probabilities,
    validated so every calibration row has exactly one prediction. Train
    and test rows never reach any calibrator fit.
    """
    folds = stratified_calibration_folds(
        data.y_calibration, random_state=random_state
    )
    fold_predictions = []
    for fit_positions, held_out_positions in folds:
        calibrator = build_calibrator(base_model, method)
        calibrator.fit(
            data.X_calibration.iloc[fit_positions],
            data.y_calibration.iloc[fit_positions],
        )
        held_out_probabilities = predict_positive_proba(
            calibrator, data.X_calibration.iloc[held_out_positions]
        )
        fold_predictions.append((held_out_positions, held_out_probabilities))
        del calibrator  # per-fold calibrators are never reused (protocol)
    return assemble_out_of_fold(len(data.X_calibration), fold_predictions)


def fit_final_calibrator(
    base_model, data: CalibrationData, method: str
) -> CalibratedClassifierCV:
    """Fit the selected method's final calibrator on the full calibration split.

    Only called after D-018 selected a method and the D-019 threshold
    scenarios were frozen; never called when D-018 selected "none".
    """
    calibrator = build_calibrator(base_model, method)
    calibrator.fit(data.X_calibration, data.y_calibration)
    return calibrator


def out_of_fold_probabilities(
    base_model, data: CalibrationData, random_state: int = RANDOM_SEED
) -> dict[str, np.ndarray]:
    """Out-of-fold probabilities per contract, plus the "none" baseline.

    The "none" entry is the frozen model's uncalibrated probabilities: the
    reference every candidate must beat under the D-018 adoption rule.
    """
    probabilities = {
        NO_CALIBRATION: uncalibrated_calibration_probabilities(base_model, data)
    }
    for method in CALIBRATION_METHODS:
        probabilities[method] = cross_fit_out_of_fold(
            base_model, data, method, random_state=random_state
        )
    return probabilities


# ---------------------------------------------------------------------------
# Paired bootstrap (pre-declared D-018 convention)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """Percentile CI of the mean paired per-row loss difference.

    `delta_i = loss_candidate_i - loss_reference_i`; the candidate improves
    its reference only if the interval's upper limit is below zero, and the
    lower-loss side is superior in a pairwise comparison only if the
    interval excludes zero in its direction.
    """

    mean_delta: float
    ci_lower: float
    ci_upper: float
    n_resamples: int
    confidence: float
    random_seed: int

    @property
    def improves_reference(self) -> bool:
        return self.ci_upper < 0.0

    @property
    def excludes_zero(self) -> bool:
        return self.ci_upper < 0.0 or self.ci_lower > 0.0


def paired_bootstrap_interval(
    candidate_losses,
    reference_losses,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    random_state: int = RANDOM_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
) -> PairedBootstrapInterval:
    """The pre-declared paired bootstrap: 10,000 fixed-seed row resamples.

    Resamples the paired per-row differences with replacement and returns
    the percentile confidence interval of the mean difference. Resampling
    is generated in fixed-size batches so memory stays bounded (never the
    full `n_resamples x n_rows` index matrix at once) while the fixed seed
    and fixed batch size keep the resample stream fully deterministic.
    """
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    reference = np.asarray(reference_losses, dtype=np.float64)
    if candidate.ndim != 1 or candidate.size == 0:
        raise ValueError(
            f"Candidate losses must be a non-empty 1-D array; got shape "
            f"{candidate.shape}."
        )
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Candidate ({candidate.shape}) and reference ({reference.shape}) "
            "losses must be paired per row with identical shapes."
        )
    if not (np.all(np.isfinite(candidate)) and np.all(np.isfinite(reference))):
        raise ValueError("Losses contain non-finite values (NaN or inf).")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"Confidence must be in (0, 1); got {confidence!r}.")

    delta = candidate - reference
    n_rows = len(delta)
    rng = np.random.default_rng(random_state)
    resample_means = np.empty(n_resamples, dtype=np.float64)
    start = 0
    while start < n_resamples:
        size = min(batch_size, n_resamples - start)
        positions = rng.integers(0, n_rows, size=(size, n_rows))
        resample_means[start : start + size] = delta[positions].mean(axis=1)
        start += size
    alpha = 1.0 - confidence
    ci_lower, ci_upper = np.quantile(resample_means, [alpha / 2, 1.0 - alpha / 2])
    return PairedBootstrapInterval(
        mean_delta=float(delta.mean()),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_resamples=n_resamples,
        confidence=confidence,
        random_seed=random_state,
    )


# ---------------------------------------------------------------------------
# D-018 selection (operationalized criteria, pre-declared before results)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSelection:
    """The D-018 outcome with the complete evidence that produced it."""

    selected_method: str  # "sigmoid", "isotonic", or "none"
    metrics: dict  # per contract (incl. "none"): brier/log_loss/roc_auc/pr_auc
    brier_vs_none: dict  # per method: PairedBootstrapInterval (adoption rule)
    log_loss_vs_none: dict  # per method: PairedBootstrapInterval (recorded evidence)
    adoptable: dict  # per method: bool (Brier adoption rule outcome)
    ranking_guard: dict  # per method: roc_auc/pr_auc drops and pass/fail
    pairwise_brier: PairedBootstrapInterval | None  # sigmoid - isotonic
    pairwise_log_loss: PairedBootstrapInterval | None  # tie-break, if reached
    rationale: str


def _ranking_guard(baseline_metrics: dict, method_metrics: dict) -> dict:
    """Rule 3: the 0.005 absolute ROC-AUC/PR-AUC regression bound."""
    roc_drop = baseline_metrics["roc_auc"] - method_metrics["roc_auc"]
    pr_drop = baseline_metrics["pr_auc"] - method_metrics["pr_auc"]
    return {
        "roc_auc_drop": float(roc_drop),
        "pr_auc_drop": float(pr_drop),
        "passes": bool(
            roc_drop <= RANKING_REGRESSION_LIMIT
            and pr_drop <= RANKING_REGRESSION_LIMIT
        ),
    }


def select_calibration_method(
    y_calibration,
    probabilities: dict[str, np.ndarray],
    random_state: int = RANDOM_SEED,
) -> CalibrationSelection:
    """Apply the pre-declared D-018 criteria to the out-of-fold evidence.

    `probabilities` maps each contract ("none" plus every method in
    `CALIBRATION_METHODS`) to its calibration-split probabilities: the
    uncalibrated baseline for "none" and the complete out-of-fold
    predictions for the methods. The rules are exactly the ones fixed in
    `docs/ml-analysis-plan.md` before any result existed:

    1. Adoption: a method qualifies only if the upper limit of the 95%
       paired-bootstrap CI of its Brier delta against "none" is below zero.
    2. Choice between two adoptable methods: the paired-bootstrap Brier
       rule between them; log loss under the same rule as tie-break;
       sigmoid on demonstrated equivalence.
    3. Ranking guard: the chosen method must not reduce ROC-AUC or PR-AUC
       by more than 0.005 absolute versus "none"; a failing method is
       replaced by the other method if that one is adoptable and passes,
       otherwise by "none".

    No test rows are involved anywhere; the same fixed seed drives every
    bootstrap so the selection is deterministic.
    """
    expected_contracts = {NO_CALIBRATION, *CALIBRATION_METHODS}
    if set(probabilities) != expected_contracts:
        raise ValueError(
            f"Probabilities must cover exactly {sorted(expected_contracts)}; "
            f"got {sorted(probabilities)}."
        )
    y = _validate_binary_targets(y_calibration)
    contracts = {
        name: validate_probabilities(values)
        for name, values in probabilities.items()
    }
    for name, values in contracts.items():
        if len(values) != len(y):
            raise ValueError(
                f"Contract {name!r} has {len(values)} probabilities for "
                f"{len(y)} calibration rows."
            )

    metrics = {
        name: probability_metrics(y, values) for name, values in contracts.items()
    }
    baseline_brier = brier_losses(y, contracts[NO_CALIBRATION])
    baseline_log = log_losses(y, contracts[NO_CALIBRATION])
    brier_vs_none = {}
    log_loss_vs_none = {}
    for method in CALIBRATION_METHODS:
        brier_vs_none[method] = paired_bootstrap_interval(
            brier_losses(y, contracts[method]),
            baseline_brier,
            random_state=random_state,
        )
        log_loss_vs_none[method] = paired_bootstrap_interval(
            log_losses(y, contracts[method]),
            baseline_log,
            random_state=random_state,
        )
    adoptable = {
        method: brier_vs_none[method].improves_reference
        for method in CALIBRATION_METHODS
    }
    ranking_guard = {
        method: _ranking_guard(metrics[NO_CALIBRATION], metrics[method])
        for method in CALIBRATION_METHODS
    }

    pairwise_brier = None
    pairwise_log_loss = None
    reasons: list[str] = []

    adopted = [method for method in CALIBRATION_METHODS if adoptable[method]]
    if not adopted:
        preferred = None
        reasons.append(
            "Neither method's Brier 95% CI upper limit against the "
            "uncalibrated baseline is below zero, so no method is adoptable."
        )
    elif len(adopted) == 1:
        preferred = adopted[0]
        reasons.append(
            f"Only '{preferred}' passes the Brier adoption rule against the "
            "uncalibrated baseline."
        )
    else:
        sigmoid_brier = brier_losses(y, contracts["sigmoid"])
        isotonic_brier = brier_losses(y, contracts["isotonic"])
        pairwise_brier = paired_bootstrap_interval(
            sigmoid_brier, isotonic_brier, random_state=random_state
        )
        if pairwise_brier.ci_upper < 0.0:
            preferred = "sigmoid"
            reasons.append(
                "Both methods are adoptable; the pairwise Brier CI favors "
                "sigmoid (upper limit below zero)."
            )
        elif pairwise_brier.ci_lower > 0.0:
            preferred = "isotonic"
            reasons.append(
                "Both methods are adoptable; the pairwise Brier CI favors "
                "isotonic (lower limit above zero)."
            )
        else:
            pairwise_log_loss = paired_bootstrap_interval(
                log_losses(y, contracts["sigmoid"]),
                log_losses(y, contracts["isotonic"]),
                random_state=random_state,
            )
            if pairwise_log_loss.ci_upper < 0.0:
                preferred = "sigmoid"
                reasons.append(
                    "Both methods are adoptable and Brier-equivalent; the "
                    "log-loss tie-break favors sigmoid."
                )
            elif pairwise_log_loss.ci_lower > 0.0:
                preferred = "isotonic"
                reasons.append(
                    "Both methods are adoptable and Brier-equivalent; the "
                    "log-loss tie-break favors isotonic."
                )
            else:
                preferred = "sigmoid"
                reasons.append(
                    "Both methods are adoptable and practically equivalent "
                    "on Brier and log loss; sigmoid is selected for "
                    "simplicity and strict ranking preservation."
                )

    selected = NO_CALIBRATION
    if preferred is not None:
        if ranking_guard[preferred]["passes"]:
            selected = preferred
        else:
            reasons.append(
                f"'{preferred}' fails the {RANKING_REGRESSION_LIMIT} absolute "
                "ROC-AUC/PR-AUC ranking-preservation guard."
            )
            other = next(
                method for method in CALIBRATION_METHODS if method != preferred
            )
            if adoptable[other] and ranking_guard[other]["passes"]:
                selected = other
                reasons.append(
                    f"'{other}' is adoptable and passes the guard, so it is "
                    "selected instead."
                )
            else:
                reasons.append(
                    "No remaining method is adoptable and guard-passing, so "
                    "the uncalibrated output is retained."
                )
    reasons.append(f"Selected calibration method: '{selected}'.")

    return CalibrationSelection(
        selected_method=selected,
        metrics=metrics,
        brier_vs_none=brier_vs_none,
        log_loss_vs_none=log_loss_vs_none,
        adoptable=adoptable,
        ranking_guard=ranking_guard,
        pairwise_brier=pairwise_brier,
        pairwise_log_loss=pairwise_log_loss,
        rationale=" ".join(reasons),
    )


# ---------------------------------------------------------------------------
# Threshold analysis (US-0606, D-019) on the selected contract's
# calibration-split probabilities only
# ---------------------------------------------------------------------------

# Candidate threshold grid for the trade-off tables: 0.01 to 0.99 in steps
# of 0.01. The analysis explains model behavior at candidate cut-offs; it is
# not a served decision rule, a clinical claim, or a screening protocol.
THRESHOLD_GRID = tuple(np.round(np.arange(0.01, 1.0, 0.01), 2))

# Deterministic scenario rules, declared before the real threshold table was
# observed. Each scenario is picked from the grid using the selected
# contract's calibration-split probabilities only:
# - default_half: the 0.50 cut-off implied by `predict` defaults, recorded
#   because the D-016 selection documented its low recall as the motivating
#   concern for this analysis.
# - max_f1: the grid threshold maximizing positive-class F1 (ties resolve to
#   the lowest threshold, deterministically).
# - recall_floor_050 / recall_floor_075: the highest grid threshold whose
#   recall still reaches 0.50 / 0.75 -- precision-maximizing operating
#   points subject to product-neutral recall floors.
THRESHOLD_SCENARIO_RULES = (
    "default_half",
    "max_f1",
    "recall_floor_050",
    "recall_floor_075",
)

# D-019 resolution (2026-07-12): the scenario thresholds frozen from the
# rules above, applied to the D-018-selected contract's calibration-split
# probabilities (see docs/decisions.md and docs/p8-calibration/report.md).
# They were frozen before the official P8 test evaluation ran and never
# change afterwards. The product policy stays probability-only: these
# scenarios document trade-offs and are not served as decision labels,
# thresholds, or screening/diagnostic rules.
THRESHOLD_POLICY_DECISION = "D-019"
FROZEN_THRESHOLD_SCENARIOS = {
    "default_half": 0.50,
    "max_f1": 0.25,
    "recall_floor_050": 0.29,
    "recall_floor_075": 0.15,
}


def threshold_metrics(y_true, probabilities, threshold: float) -> dict:
    """Confusion-matrix metrics at one candidate threshold.

    Predictions are `probability >= threshold`. `zero_division=0` semantics
    are applied manually so degenerate cells stay well-defined.
    """
    y = _validate_binary_targets(y_true)
    p = validate_probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError(
            f"Targets ({len(y)}) and probabilities ({len(p)}) differ in length."
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold must be in (0, 1); got {threshold!r}.")
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1.0)))
    fp = int(np.sum(predicted & (y == 0.0)))
    fn = int(np.sum(~predicted & (y == 1.0)))
    tn = int(np.sum(~predicted & (y == 0.0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_table(
    y_true, probabilities, thresholds=THRESHOLD_GRID
) -> pd.DataFrame:
    """The US-0606 trade-off table: one row per candidate threshold."""
    return pd.DataFrame(
        [threshold_metrics(y_true, probabilities, t) for t in thresholds]
    )


def precision_recall_points(y_true, probabilities) -> pd.DataFrame:
    """Precision-recall curve points for the selected contract."""
    y = _validate_binary_targets(y_true)
    p = validate_probabilities(probabilities)
    precision, recall, thresholds = precision_recall_curve(y, p)
    return pd.DataFrame(
        {
            "precision": precision[:-1].astype(float),
            "recall": recall[:-1].astype(float),
            "threshold": thresholds.astype(float),
        }
    )


def select_threshold_scenarios(
    y_true, probabilities, thresholds=THRESHOLD_GRID
) -> dict[str, dict]:
    """Apply the pre-declared deterministic scenario rules to the grid.

    Returns `{scenario_name: threshold_metrics_dict}` computed exclusively
    from the selected contract's calibration-split probabilities. The
    resulting thresholds are frozen (D-019) before the official P8 test
    evaluation; they document trade-offs and are never served as a decision
    rule or presented as a validated screening/diagnostic cut-off.
    """
    table = threshold_table(y_true, probabilities, thresholds)
    scenarios: dict[str, dict] = {}

    default_row = table[np.isclose(table["threshold"], 0.5)]
    if default_row.empty:
        raise ValueError("The threshold grid must include the 0.50 default.")
    scenarios["default_half"] = default_row.iloc[0].to_dict()

    max_f1_row = table.iloc[int(table["f1"].idxmax())]  # first (lowest) on ties
    scenarios["max_f1"] = max_f1_row.to_dict()

    for name, floor in (("recall_floor_050", 0.50), ("recall_floor_075", 0.75)):
        reaching = table[table["recall"] >= floor]
        if reaching.empty:
            raise ValueError(
                f"No grid threshold reaches recall {floor}; the scenario "
                f"rule '{name}' cannot be evaluated."
            )
        scenarios[name] = reaching.iloc[-1].to_dict()  # highest such threshold

    for name, metrics in scenarios.items():
        metrics["scenario"] = name
        for key in ("tp", "fp", "fn", "tn"):
            metrics[key] = int(metrics[key])
    return scenarios


# ---------------------------------------------------------------------------
# Serving contract and the official P8 test evaluation
# ---------------------------------------------------------------------------


def contract_probabilities(base_model, final_calibrator, X) -> np.ndarray:
    """Positive-class probabilities of the frozen P8 serving contract.

    With a final calibrator (D-018 selected sigmoid/isotonic) the calibrated
    probabilities are served; with `None` (D-018 selected "none") the frozen
    model's own probabilities are served. This mirrors exactly what the
    schema-version-2 artifact serves.
    """
    scorer = base_model if final_calibrator is None else final_calibrator
    return validate_probabilities(predict_positive_proba(scorer, X))


def official_test_evaluation(
    splits: DataSplits,
    base_model,
    final_calibrator,
    scenario_thresholds: dict[str, float],
) -> dict:
    """The single P8 consumer of the test split.

    Run only after D-018 (method) and D-019 (threshold scenarios) are
    frozen: it evaluates the frozen serving contract on the test rows and
    records the contract metrics, the reliability table, the frozen
    scenarios' confusion-matrix metrics, and the uncalibrated schema-v1
    reference metrics for the descriptive comparison. Nothing may change
    after these results are observed; later calls only repeat the same
    deterministic evaluation as a regression check.
    """
    if not scenario_thresholds:
        raise ValueError(
            "Scenario thresholds must be frozen (D-019) before the official "
            "P8 test evaluation runs."
        )
    X_test = splits.test[FEATURE_COLUMNS]
    y_test = splits.test[TARGET]
    contract = contract_probabilities(base_model, final_calibrator, X_test)
    uncalibrated = validate_probabilities(
        predict_positive_proba(base_model, X_test)
    )
    return {
        "contract_metrics": probability_metrics(y_test, contract),
        "uncalibrated_reference_metrics": probability_metrics(
            y_test, uncalibrated
        ),
        "reliability": reliability_table(y_test, contract),
        "scenario_metrics": {
            name: threshold_metrics(y_test, contract, threshold)
            for name, threshold in scenario_thresholds.items()
        },
    }


# ---------------------------------------------------------------------------
# Offline evidence regeneration: `python -m src.calibration`
# ---------------------------------------------------------------------------


def _write_reliability_figure(directory, tables: dict[str, pd.DataFrame]) -> None:
    """One reliability-diagram panel per contract (visual diagnostics only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for axis, (title, table) in zip(axes.ravel(), tables.items()):
        populated = table.dropna(subset=["mean_predicted_probability"])
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.plot(
            populated["mean_predicted_probability"],
            populated["observed_positive_rate"],
            marker="o",
            linewidth=1.5,
        )
        axis.set_title(title)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed positive rate")
    figure.suptitle(
        "Reliability diagrams (equal-width bins; visual diagnostics only)"
    )
    figure.tight_layout()
    figure.savefig(directory / "reliability_diagrams.png", dpi=120)
    plt.close(figure)


def _write_histogram_figure(directory, histogram: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        histogram["bin_lower"],
        histogram["n_rows"],
        width=histogram["bin_upper"] - histogram["bin_lower"],
        align="edge",
        edgecolor="white",
    )
    axis.set_title("Served contract probabilities on the calibration split")
    axis.set_xlabel("Predicted positive-class probability")
    axis.set_ylabel("Rows")
    figure.tight_layout()
    figure.savefig(directory / "probability_histogram.png", dpi=120)
    plt.close(figure)


def _write_pr_curve_figure(
    directory, points: pd.DataFrame, scenarios: dict[str, dict]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(points["recall"], points["precision"], linewidth=1.5)
    for name, metrics in scenarios.items():
        axis.scatter(metrics["recall"], metrics["precision"], zorder=3)
        axis.annotate(
            f"{name} ({metrics['threshold']:.2f})",
            (metrics["recall"], metrics["precision"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    axis.set_title(
        "Precision-recall on the calibration split (frozen D-019 scenarios)"
    )
    axis.set_xlabel("Recall (positive class)")
    axis.set_ylabel("Precision (positive class)")
    figure.tight_layout()
    figure.savefig(directory / "precision_recall_curve.png", dpi=120)
    plt.close(figure)


def _interval_row(comparison: str, loss: str, interval: PairedBootstrapInterval) -> dict:
    return {
        "comparison": comparison,
        "loss": loss,
        "mean_delta": interval.mean_delta,
        "ci_lower": interval.ci_lower,
        "ci_upper": interval.ci_upper,
        "n_resamples": interval.n_resamples,
        "confidence": interval.confidence,
        "random_seed": interval.random_seed,
    }


def main() -> None:
    """Regenerate the recorded P8 evidence deterministically.

    Reruns the frozen pipeline end to end -- baseline, out-of-fold
    comparison, D-018 selection, threshold analysis, D-019 scenarios, and
    the official P8 test evaluation -- and fails loudly if any step stops
    reproducing the recorded decisions. Because D-018 and D-019 are frozen,
    the test evaluation here is exactly the recorded official evaluation
    repeated as a deterministic regression check; nothing is selected or
    modified from its results. Writes the evidence tables and figures under
    `docs/p8-calibration/`.
    """
    from src.data import PROJECT_ROOT, prepare_data

    directory = PROJECT_ROOT / "docs" / P8_EVIDENCE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)

    print("Preparing data through the P3 contract ...")
    splits, summary = prepare_data()
    cal_data = to_calibration_data(splits)
    print(
        f"Calibration split: {len(cal_data.X_calibration)} rows, "
        f"{int(cal_data.y_calibration.sum())} positives."
    )

    print("Training the frozen D-016 base model on the train split ...")
    base_model = train_frozen_base_model(cal_data)

    print("Recording the uncalibrated baseline ...")
    baseline = uncalibrated_baseline_evaluation(base_model, cal_data)

    print("Cross-fitting sigmoid and isotonic out-of-fold predictions ...")
    probabilities = out_of_fold_probabilities(base_model, cal_data)

    print("Applying the pre-declared D-018 selection criteria ...")
    selection = select_calibration_method(
        cal_data.y_calibration, probabilities
    )
    if selection.selected_method != SELECTED_CALIBRATION_METHOD:
        raise AssertionError(
            f"The pre-declared selection rules produced "
            f"{selection.selected_method!r}, which no longer reproduces the "
            f"recorded D-018 outcome {SELECTED_CALIBRATION_METHOD!r}; "
            "investigate before touching any decision."
        )

    print("Rebuilding the frozen D-019 threshold scenarios ...")
    contract_calibration = probabilities[SELECTED_CALIBRATION_METHOD]
    table = threshold_table(cal_data.y_calibration, contract_calibration)
    scenarios = select_threshold_scenarios(
        cal_data.y_calibration, contract_calibration
    )
    frozen = {name: metrics["threshold"] for name, metrics in scenarios.items()}
    if frozen != FROZEN_THRESHOLD_SCENARIOS:
        raise AssertionError(
            f"The pre-declared scenario rules produced {frozen!r}, which no "
            "longer reproduces the recorded D-019 scenarios "
            f"{FROZEN_THRESHOLD_SCENARIOS!r}; investigate before touching "
            "any decision."
        )

    final_calibrator = None
    if SELECTED_CALIBRATION_METHOD != NO_CALIBRATION:
        print("Refitting the final calibrator on the full calibration split ...")
        final_calibrator = fit_final_calibrator(
            base_model, cal_data, SELECTED_CALIBRATION_METHOD
        )

    print("Running the official P8 test evaluation (deterministic repeat) ...")
    official = official_test_evaluation(
        splits, base_model, final_calibrator, FROZEN_THRESHOLD_SCENARIOS
    )

    # --- evidence tables -------------------------------------------------
    contract_order = [NO_CALIBRATION, *CALIBRATION_METHODS]
    pd.DataFrame(
        [
            {"contract": contract, **selection.metrics[contract]}
            for contract in contract_order
        ]
    ).to_csv(directory / "oof_metrics.csv", index=False)

    comparisons = []
    for method in CALIBRATION_METHODS:
        comparisons.append(
            _interval_row(f"{method} - none", "brier", selection.brier_vs_none[method])
        )
        comparisons.append(
            _interval_row(
                f"{method} - none", "log_loss", selection.log_loss_vs_none[method]
            )
        )
    if selection.pairwise_brier is not None:
        comparisons.append(
            _interval_row("sigmoid - isotonic", "brier", selection.pairwise_brier)
        )
    if selection.pairwise_log_loss is not None:
        comparisons.append(
            _interval_row(
                "sigmoid - isotonic", "log_loss", selection.pairwise_log_loss
            )
        )
    pd.DataFrame(comparisons).to_csv(
        directory / "bootstrap_comparisons.csv", index=False
    )

    reliability_tables = {
        "none (baseline, calibration split)": baseline["reliability"],
    }
    for method in CALIBRATION_METHODS:
        reliability_tables[f"{method} (out-of-fold, calibration split)"] = (
            reliability_table(cal_data.y_calibration, probabilities[method])
        )
    reliability_tables["served contract (test split)"] = official["reliability"]
    for name, frame in (
        ("reliability_calibration_none.csv", baseline["reliability"]),
        (
            "reliability_calibration_sigmoid.csv",
            reliability_tables["sigmoid (out-of-fold, calibration split)"],
        ),
        (
            "reliability_calibration_isotonic.csv",
            reliability_tables["isotonic (out-of-fold, calibration split)"],
        ),
        ("reliability_test_contract.csv", official["reliability"]),
        ("probability_histogram_calibration.csv", baseline["histogram"]),
        ("threshold_table_calibration.csv", table),
    ):
        frame.to_csv(directory / name, index=False)

    scenario_rows = []
    for name in FROZEN_THRESHOLD_SCENARIOS:
        scenario_rows.append(
            {"scenario": name, "split": "calibration", **{
                key: value
                for key, value in scenarios[name].items()
                if key != "scenario"
            }}
        )
        scenario_rows.append(
            {"scenario": name, "split": "test", **official["scenario_metrics"][name]}
        )
    pd.DataFrame(scenario_rows).to_csv(
        directory / "threshold_scenarios.csv", index=False
    )

    pd.DataFrame(
        [
            {"split": "calibration", "contract": SELECTED_CALIBRATION_METHOD, **selection.metrics[SELECTED_CALIBRATION_METHOD]},
            {"split": "test", "contract": SELECTED_CALIBRATION_METHOD, **official["contract_metrics"]},
            {"split": "test", "contract": "schema-v1 reference (uncalibrated)", **official["uncalibrated_reference_metrics"]},
        ]
    ).to_csv(directory / "official_test_evaluation.csv", index=False)

    # --- figures ----------------------------------------------------------
    _write_reliability_figure(directory, reliability_tables)
    _write_histogram_figure(directory, baseline["histogram"])
    _write_pr_curve_figure(
        directory,
        precision_recall_points(cal_data.y_calibration, contract_calibration),
        scenarios,
    )

    # --- console summary ---------------------------------------------------
    print("\n=== D-018 (frozen) ===")
    print(f"Selected method: {selection.selected_method}")
    print(f"Rationale: {selection.rationale}")
    print("\n=== Out-of-fold metrics (calibration split) ===")
    for contract in contract_order:
        metrics = selection.metrics[contract]
        print(
            f"  {contract}: "
            + ", ".join(f"{key}={metrics[key]:.6f}" for key in PROBABILITY_METRIC_KEYS)
        )
    print("\n=== D-019 frozen scenarios (calibration split) ===")
    for name, metrics in scenarios.items():
        print(
            f"  {name}: threshold={metrics['threshold']:.2f}, "
            f"recall={metrics['recall']:.3f}, precision={metrics['precision']:.3f}, "
            f"f1={metrics['f1']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}"
        )
    print("\n=== Official P8 test evaluation (frozen contract) ===")
    contract_metrics = official["contract_metrics"]
    print(
        "  contract: "
        + ", ".join(
            f"{key}={contract_metrics[key]:.6f}" for key in PROBABILITY_METRIC_KEYS
        )
    )
    for name, metrics in official["scenario_metrics"].items():
        print(
            f"  {name}: threshold={metrics['threshold']:.2f}, "
            f"recall={metrics['recall']:.3f}, precision={metrics['precision']:.3f}, "
            f"f1={metrics['f1']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}"
        )
    print(f"\nEvidence written to {directory}")


if __name__ == "__main__":
    main()
