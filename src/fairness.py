"""Reproducible offline subgroup audit for the frozen P8 probability contract.

P12 is report-first and descriptive.  This module assigns the predeclared
D-029 cohorts, applies the accepted support rule, computes aggregate point
metrics and common-bin reliability data, and estimates percentile bootstrap
intervals under D-030.  It consumes explicit binary labels and positive-class
probabilities; it never fits, calibrates, selects a threshold, regenerates an
artifact, imports Streamlit, or publishes row-level data.

The command-line workflow deliberately separates the ordered gates::

    python -m src.fairness --calibration-support
    python -m src.fairness --benchmark
    python -m src.fairness --official-audit

Only the final command reads the test features and labels, and it refuses to
run until D-029, D-030, and D-031 are Accepted in the working tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ARTIFACT_PATH,
    load_artifact,
    predict_probability_frame,
    select_probability_scorer,
)
from src.calibration import FROZEN_THRESHOLD_SCENARIOS, NO_CALIBRATION
from src.data import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET,
    DataSplits,
    prepare_data,
)
from src.feature_labels import BINARY_VALUE_LABELS, ORDINAL_VALUE_LABELS


P12_EVIDENCE_DIR = PROJECT_ROOT / "docs" / "p12-fairness"
CALIBRATION_SUPPORT_PATH = P12_EVIDENCE_DIR / "calibration_support.csv"
BOOTSTRAP_BENCHMARK_PATH = P12_EVIDENCE_DIR / "bootstrap_benchmark.json"
MODEL_ARTIFACT_SHA256 = (
    "957c14ff5a490bbc60822121a889f92ee2a6a20f797eef741a710d887ecc9216"
)
SHAP_BACKGROUND_PATH = PROJECT_ROOT / "models" / "shap_background_v1.json"
SHAP_BACKGROUND_SHA256 = (
    "73d1ff21e3c98ee79fa7d72758517047f13e5f454d7ff95edb1ee93812cca120"
)

SUPPORT_MIN_ROWS = 500
SUPPORT_MIN_POSITIVES = 100
SUPPORT_MIN_NEGATIVES = 100

CALIBRATION_SUPPORT_COLUMNS = (
    "cohort_axis",
    "group_key",
    "group_label",
    "row_count",
    "positive_count",
    "negative_count",
    "prevalence",
    "meets_candidate_floor",
)
CALIBRATION_SUPPORT_COHORTS = 22

AGE_BANDS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("18_49", "18-49", tuple(range(1, 7))),
    ("50_64", "50-64", tuple(range(7, 10))),
    ("65_74", "65-74", (10, 11)),
    ("75_plus", "75+", (12, 13)),
)
COHORT_AXIS_ORDER = ("whole", "sex", "age", "income", "sex_x_age")

RELIABILITY_BIN_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)
RELIABILITY_BIN_LABELS = tuple(
    f"[{RELIABILITY_BIN_EDGES[index]:.1f}, "
    f"{RELIABILITY_BIN_EDGES[index + 1]:.1f}{']' if index == 9 else ')'}"
    for index in range(10)
)

PROBABILITY_METRICS = (
    "prevalence",
    "mean_probability",
    "brier_score",
    "log_loss",
    "roc_auc",
    "pr_auc",
    "calibration_gap",
)
THRESHOLD_NORMALIZED_METRICS = (
    "recall",
    "precision",
    "false_positive_rate",
)
THRESHOLD_COUNT_METRICS = (
    "false_positive_count",
    "false_negative_count",
)

BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_BATCH_SIZE = 128
BOOTSTRAP_RUNTIME_LIMIT_SECONDS = 600.0
BOOTSTRAP_MEMORY_LIMIT_MIB = 512.0
BOOTSTRAP_INTERVAL_ROWS = 855
BOOTSTRAP_METHOD = "ordinary nonparametric whole-split row bootstrap with replacement"
BOOTSTRAP_RNG_ORDER = "numpy Generator.integers in resample-major row order"


@dataclass(frozen=True)
class SupportRule:
    """D-029 full-metric support rule."""

    min_rows: int = SUPPORT_MIN_ROWS
    min_positives: int = SUPPORT_MIN_POSITIVES
    min_negatives: int = SUPPORT_MIN_NEGATIVES

    def is_supported(self, rows: int, positives: int, negatives: int) -> bool:
        return (
            rows > 0
            and rows >= self.min_rows
            and positives >= self.min_positives
            and negatives >= self.min_negatives
        )


@dataclass(frozen=True)
class CohortSlice:
    """One deterministic aggregate cohort mask."""

    cohort_axis: str
    group_key: str
    group_label: str
    mask: np.ndarray


@dataclass(frozen=True)
class AuditTables:
    """Aggregate P12 point evidence; no row-level values are retained."""

    support: pd.DataFrame
    probability_metrics: pd.DataFrame
    metric_gaps: pd.DataFrame
    reliability: pd.DataFrame
    threshold_metrics: pd.DataFrame


@dataclass(frozen=True)
class _BootstrapPlan:
    cohort: CohortSlice
    positions: np.ndarray
    values: np.ndarray
    descending_order: np.ndarray
    tie_starts: np.ndarray
    supported: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_artifacts() -> dict[str, str]:
    """Fail before audit work if either reviewed artifact changed."""
    observed = {
        "model": _sha256(DEFAULT_ARTIFACT_PATH),
        "shap_background": _sha256(SHAP_BACKGROUND_PATH),
    }
    expected = {
        "model": MODEL_ARTIFACT_SHA256,
        "shap_background": SHAP_BACKGROUND_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            "Frozen artifact hash mismatch; P12 must stop without regenerating "
            f"either artifact (observed={observed}, expected={expected})."
        )
    return observed


def _validate_features(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise ValueError("Audit features must be a pandas DataFrame.")
    if list(features.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Audit feature columns must match FEATURE_COLUMNS in exact order."
        )
    if features.empty:
        raise ValueError("Audit features must contain at least one row.")
    for feature, allowed in (
        ("Sex", {0, 1}),
        ("Age", set(range(1, 14))),
        ("Income", set(range(1, 9))),
    ):
        observed = set(pd.unique(features[feature]))
        if not observed <= allowed:
            raise ValueError(
                f"Feature {feature!r} contains values outside the P3 contract: "
                f"{sorted(observed - allowed, key=repr)}."
            )


def _validate_labels(labels, expected_rows: int) -> np.ndarray:
    array = np.asarray(labels, dtype=np.float64)
    if array.ndim != 1 or len(array) != expected_rows:
        raise ValueError(
            "Audit labels must be a one-dimensional vector aligned to features."
        )
    if not np.all(np.isfinite(array)) or not set(np.unique(array)) <= {0.0, 1.0}:
        raise ValueError("Audit labels must contain only finite binary 0/1 values.")
    return array


def _validate_probabilities(probabilities, expected_rows: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 1 or len(array) != expected_rows:
        raise ValueError(
            "Audit probabilities must be a one-dimensional vector aligned to features."
        )
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Audit probabilities must be finite values in [0, 1].")
    return array


def validate_audit_inputs(
    features: pd.DataFrame, labels, probabilities
) -> tuple[np.ndarray, np.ndarray]:
    """Validate explicit labels and served positive-class probabilities."""
    _validate_features(features)
    y = _validate_labels(labels, len(features))
    p = _validate_probabilities(probabilities, len(features))
    return y, p


def cohort_slices(
    features: pd.DataFrame, *, include_whole: bool = True
) -> tuple[CohortSlice, ...]:
    """Assign exhaustive, mutually exclusive D-029 cohorts in fixed order."""
    _validate_features(features)
    n_rows = len(features)
    cohorts: list[CohortSlice] = []
    if include_whole:
        cohorts.append(
            CohortSlice("whole", "whole", "Whole test cohort", np.ones(n_rows, bool))
        )

    for code in (0, 1):
        cohorts.append(
            CohortSlice(
                "sex",
                f"sex_{code}",
                BINARY_VALUE_LABELS["Sex"][code],
                features["Sex"].to_numpy() == code,
            )
        )
    for slug, label, codes in AGE_BANDS:
        cohorts.append(
            CohortSlice(
                "age",
                f"age_{slug}",
                label,
                features["Age"].isin(codes).to_numpy(),
            )
        )
    for code in range(1, 9):
        cohorts.append(
            CohortSlice(
                "income",
                f"income_{code}",
                ORDINAL_VALUE_LABELS["Income"][code],
                features["Income"].to_numpy() == code,
            )
        )
    for sex_code in (0, 1):
        sex_label = BINARY_VALUE_LABELS["Sex"][sex_code]
        for slug, age_label, codes in AGE_BANDS:
            cohorts.append(
                CohortSlice(
                    "sex_x_age",
                    f"sex_{sex_code}__age_{slug}",
                    f"{sex_label} | {age_label}",
                    (
                        (features["Sex"].to_numpy() == sex_code)
                        & features["Age"].isin(codes).to_numpy()
                    ),
                )
            )

    for axis in ("sex", "age", "income", "sex_x_age"):
        masks = [cohort.mask.astype(np.int8) for cohort in cohorts if cohort.cohort_axis == axis]
        assignments = np.sum(np.vstack(masks), axis=0)
        if not np.all(assignments == 1):
            raise ValueError(
                f"Cohort axis {axis!r} is not exhaustive and mutually exclusive."
            )
    return tuple(cohorts)


def _support_record(
    cohort: CohortSlice, labels: np.ndarray, support_rule: SupportRule
) -> dict[str, object]:
    group_labels = labels[cohort.mask]
    rows = int(len(group_labels))
    positives = int(group_labels.sum())
    negatives = rows - positives
    return {
        "cohort_axis": cohort.cohort_axis,
        "group_key": cohort.group_key,
        "group_label": cohort.group_label,
        "row_count": rows,
        "positive_count": positives,
        "negative_count": negatives,
        "prevalence": positives / rows if rows else None,
        "meets_support_floor": support_rule.is_supported(rows, positives, negatives),
    }


def support_table(
    features: pd.DataFrame,
    labels,
    *,
    include_whole: bool,
    support_rule: SupportRule = SupportRule(),
) -> pd.DataFrame:
    """Return support/prevalence for every declared cohort."""
    _validate_features(features)
    y = _validate_labels(labels, len(features))
    return pd.DataFrame(
        [
            _support_record(cohort, y, support_rule)
            for cohort in cohort_slices(features, include_whole=include_whole)
        ]
    )


def calibration_support_table(
    splits: DataSplits, support_rule: SupportRule = SupportRule()
) -> pd.DataFrame:
    """D-029 support evidence from calibration only; no probability is read."""
    frame = support_table(
        splits.calibration[FEATURE_COLUMNS],
        splits.calibration[TARGET],
        include_whole=False,
        support_rule=support_rule,
    )
    return frame.rename(columns={"meets_support_floor": "meets_candidate_floor"})


def _available(value: float) -> tuple[str, float, str]:
    if not math.isfinite(float(value)):
        raise ValueError("A metric marked available must be finite.")
    return "available", float(value), ""


def _unavailable(reason: str) -> tuple[str, None, str]:
    return "unavailable", None, reason


def _point_probability_values(
    labels: np.ndarray, probabilities: np.ndarray, supported: bool
) -> dict[str, tuple[str, float | None, str]]:
    rows = len(labels)
    if not rows:
        return {
            metric: _unavailable("empty_group") for metric in PROBABILITY_METRICS
        }
    prevalence = float(labels.mean())
    results: dict[str, tuple[str, float | None, str]] = {
        "prevalence": _available(prevalence)
    }
    if not supported:
        for metric in PROBABILITY_METRICS[1:]:
            results[metric] = _unavailable("support_floor_not_met")
        return results

    mean_probability = float(probabilities.mean())
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    results.update(
        {
            "mean_probability": _available(mean_probability),
            "brier_score": _available(float(np.mean((probabilities - labels) ** 2))),
            "log_loss": _available(
                float(
                    np.mean(
                        -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
                    )
                )
            ),
            "calibration_gap": _available(mean_probability - prevalence),
        }
    )
    if set(np.unique(labels)) == {0.0, 1.0}:
        results["roc_auc"] = _available(float(roc_auc_score(labels, probabilities)))
        results["pr_auc"] = _available(
            float(average_precision_score(labels, probabilities))
        )
    else:
        results["roc_auc"] = _unavailable("both_classes_required")
        results["pr_auc"] = _unavailable("both_classes_required")
    return {metric: results[metric] for metric in PROBABILITY_METRICS}


def _point_threshold_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    supported: bool,
) -> dict[str, tuple[str, float | None, str]]:
    if not supported:
        return {
            metric: _unavailable("support_floor_not_met")
            for metric in (*THRESHOLD_NORMALIZED_METRICS, *THRESHOLD_COUNT_METRICS)
        }
    predicted = probabilities >= threshold
    positive = labels == 1.0
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    positives = int(positive.sum())
    negatives = int(negative.sum())
    predicted_positives = tp + fp
    return {
        "recall": (
            _available(tp / positives)
            if positives
            else _unavailable("positive_class_required")
        ),
        "precision": (
            _available(tp / predicted_positives)
            if predicted_positives
            else _unavailable("predicted_positive_required")
        ),
        "false_positive_rate": (
            _available(fp / negatives)
            if negatives
            else _unavailable("negative_class_required")
        ),
        "false_positive_count": _available(float(fp)),
        "false_negative_count": _available(float(fn)),
    }


def audit_point_estimates(
    features: pd.DataFrame,
    labels,
    probabilities,
    *,
    support_rule: SupportRule = SupportRule(),
    thresholds: Mapping[str, float] = FROZEN_THRESHOLD_SCENARIOS,
) -> AuditTables:
    """Compute all aggregate point evidence in deterministic contract order."""
    y, p = validate_audit_inputs(features, labels, probabilities)
    cohorts = cohort_slices(features, include_whole=True)
    support_records = [_support_record(cohort, y, support_rule) for cohort in cohorts]
    support = pd.DataFrame(support_records)
    support_lookup = {
        record["group_key"]: bool(record["meets_support_floor"])
        for record in support_records
    }
    support_lookup["whole"] = True

    probability_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    for cohort in cohorts:
        group_y = y[cohort.mask]
        group_p = p[cohort.mask]
        supported = support_lookup[cohort.group_key]
        for metric, (status, value, reason) in _point_probability_values(
            group_y, group_p, supported
        ).items():
            probability_rows.append(
                {
                    "cohort_axis": cohort.cohort_axis,
                    "group_key": cohort.group_key,
                    "group_label": cohort.group_label,
                    "metric": metric,
                    "value": value,
                    "status": status,
                    "unavailable_reason": reason,
                }
            )
        for scenario, threshold in thresholds.items():
            for metric, (status, value, reason) in _point_threshold_values(
                group_y, group_p, threshold, supported
            ).items():
                threshold_rows.append(
                    {
                        "cohort_axis": cohort.cohort_axis,
                        "group_key": cohort.group_key,
                        "group_label": cohort.group_label,
                        "scenario": scenario,
                        "threshold": float(threshold),
                        "metric": metric,
                        "value": value,
                        "status": status,
                        "unavailable_reason": reason,
                    }
                )

        bin_index = np.clip(
            np.searchsorted(RELIABILITY_BIN_EDGES, group_p, side="right") - 1,
            0,
            9,
        )
        for bin_id in range(10):
            mask = bin_index == bin_id
            bin_rows = int(mask.sum())
            record: dict[str, object] = {
                "cohort_axis": cohort.cohort_axis,
                "group_key": cohort.group_key,
                "group_label": cohort.group_label,
                "bin_id": bin_id,
                "bin_label": RELIABILITY_BIN_LABELS[bin_id],
                "bin_lower": float(RELIABILITY_BIN_EDGES[bin_id]),
                "bin_upper": float(RELIABILITY_BIN_EDGES[bin_id + 1]),
            }
            if not supported:
                record.update(
                    {
                        "row_count": None,
                        "positive_count": None,
                        "mean_probability": None,
                        "observed_prevalence": None,
                        "calibration_gap": None,
                        "status": "unavailable",
                        "unavailable_reason": "support_floor_not_met",
                    }
                )
            elif not bin_rows:
                record.update(
                    {
                        "row_count": 0,
                        "positive_count": 0,
                        "mean_probability": None,
                        "observed_prevalence": None,
                        "calibration_gap": None,
                        "status": "unavailable",
                        "unavailable_reason": "empty_bin",
                    }
                )
            else:
                observed = float(group_y[mask].mean())
                mean_probability = float(group_p[mask].mean())
                record.update(
                    {
                        "row_count": bin_rows,
                        "positive_count": int(group_y[mask].sum()),
                        "mean_probability": mean_probability,
                        "observed_prevalence": observed,
                        "calibration_gap": mean_probability - observed,
                        "status": "available",
                        "unavailable_reason": "",
                    }
                )
            reliability_rows.append(record)

    probability = pd.DataFrame(probability_rows)
    threshold = pd.DataFrame(threshold_rows)
    probability_lookup = {
        (row.group_key, row.metric): row
        for row in probability.itertuples(index=False)
    }
    threshold_lookup = {
        (row.group_key, row.scenario, row.metric): row
        for row in threshold.itertuples(index=False)
    }
    gap_rows: list[dict[str, object]] = []
    for cohort in cohorts:
        if cohort.group_key == "whole":
            continue
        for metric in PROBABILITY_METRICS:
            group = probability_lookup[(cohort.group_key, metric)]
            whole = probability_lookup[("whole", metric)]
            if group.status == whole.status == "available":
                status, value, reason = _available(group.value - whole.value)
            else:
                status, value, reason = _unavailable(
                    group.unavailable_reason or whole.unavailable_reason
                )
            gap_rows.append(
                {
                    "cohort_axis": cohort.cohort_axis,
                    "group_key": cohort.group_key,
                    "group_label": cohort.group_label,
                    "metric_family": "probability",
                    "scenario": "",
                    "metric": metric,
                    "gap_direction": "group_minus_whole_cohort",
                    "gap": value,
                    "status": status,
                    "unavailable_reason": reason,
                }
            )
        for scenario in thresholds:
            for metric in THRESHOLD_NORMALIZED_METRICS:
                group = threshold_lookup[(cohort.group_key, scenario, metric)]
                whole = threshold_lookup[("whole", scenario, metric)]
                if group.status == whole.status == "available":
                    status, value, reason = _available(group.value - whole.value)
                else:
                    status, value, reason = _unavailable(
                        group.unavailable_reason or whole.unavailable_reason
                    )
                gap_rows.append(
                    {
                        "cohort_axis": cohort.cohort_axis,
                        "group_key": cohort.group_key,
                        "group_label": cohort.group_label,
                        "metric_family": "threshold",
                        "scenario": scenario,
                        "metric": metric,
                        "gap_direction": "group_minus_whole_cohort",
                        "gap": value,
                        "status": status,
                        "unavailable_reason": reason,
                    }
                )

    return AuditTables(
        support=support,
        probability_metrics=probability,
        metric_gaps=pd.DataFrame(gap_rows),
        reliability=pd.DataFrame(reliability_rows),
        threshold_metrics=threshold,
    )


def _bootstrap_plans(
    features: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    support_rule: SupportRule,
    thresholds: Mapping[str, float],
) -> tuple[_BootstrapPlan, ...]:
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    row_log_loss = -(
        labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
    )
    plans = []
    for cohort in cohort_slices(features, include_whole=True):
        positions = np.flatnonzero(cohort.mask)
        group_y = labels[positions]
        group_p = probabilities[positions]
        columns = [
            np.ones(len(positions), dtype=np.float64),
            group_y,
            group_p,
            (group_p - group_y) ** 2,
            row_log_loss[positions],
        ]
        for threshold in thresholds.values():
            predicted = group_p >= threshold
            columns.extend(
                [predicted * group_y, predicted * (1.0 - group_y)]
            )
        values = np.column_stack(columns)
        descending = np.argsort(-group_p, kind="stable")
        sorted_probabilities = group_p[descending]
        tie_starts = np.r_[
            0, np.flatnonzero(sorted_probabilities[1:] != sorted_probabilities[:-1]) + 1
        ].astype(np.int64)
        rows = len(positions)
        positives = int(group_y.sum())
        supported = cohort.group_key == "whole" or support_rule.is_supported(
            rows, positives, rows - positives
        )
        plans.append(
            _BootstrapPlan(
                cohort=cohort,
                positions=positions,
                values=values,
                descending_order=descending,
                tie_starts=tie_starts,
                supported=supported,
            )
        )
    return tuple(plans)


def _bootstrap_weight_batches(
    n_rows: int,
    n_resamples: int,
    random_state: int,
    batch_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield deterministic ordinary-bootstrap multiplicity weights."""
    if n_resamples <= 0 or batch_size <= 0:
        raise ValueError("Bootstrap resamples and batch size must be positive.")
    rng = np.random.default_rng(random_state)
    start = 0
    while start < n_resamples:
        size = min(batch_size, n_resamples - start)
        sampled = rng.integers(0, n_rows, size=(size, n_rows), dtype=np.int32)
        weights = np.empty((size, n_rows), dtype=np.int32)
        for row in range(size):
            weights[row] = np.bincount(sampled[row], minlength=n_rows)
        yield start, weights
        start += size


def _weighted_ranking_metrics(
    group_weights: np.ndarray,
    plan: _BootstrapPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact weighted ROC-AUC and average precision with score ties."""
    ordered_weights = group_weights[:, plan.descending_order]
    ordered_positive = plan.values[plan.descending_order, 1]
    total_by_tie = np.add.reduceat(ordered_weights, plan.tie_starts, axis=1)
    positive_by_tie = np.add.reduceat(
        ordered_weights * ordered_positive, plan.tie_starts, axis=1
    )
    negative_by_tie = total_by_tie - positive_by_tie
    total_positive = positive_by_tie.sum(axis=1)
    total_negative = negative_by_tie.sum(axis=1)

    cumulative_positive = np.cumsum(positive_by_tie, axis=1)
    cumulative_total = np.cumsum(total_by_tie, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.full_like(cumulative_positive, np.nan),
        where=cumulative_total > 0,
    )
    pr_numerator = np.nansum(positive_by_tie * precision, axis=1)
    pr_auc = np.divide(
        pr_numerator,
        total_positive,
        out=np.full_like(total_positive, np.nan),
        where=total_positive > 0,
    )

    cumulative_negative = np.cumsum(negative_by_tie, axis=1)
    negative_below = total_negative[:, None] - cumulative_negative
    roc_numerator = np.sum(
        positive_by_tie * (negative_below + 0.5 * negative_by_tie), axis=1
    )
    denominator = total_positive * total_negative
    roc_auc = np.divide(
        roc_numerator,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    return roc_auc, pr_auc


def _metric_sample_keys(
    cohorts: Iterable[CohortSlice], thresholds: Mapping[str, float]
) -> tuple[tuple[str, str, str, str], ...]:
    keys: list[tuple[str, str, str, str]] = []
    for cohort in cohorts:
        for metric in PROBABILITY_METRICS:
            keys.append((cohort.group_key, "probability", "", metric))
        for scenario in thresholds:
            for metric in THRESHOLD_NORMALIZED_METRICS:
                keys.append((cohort.group_key, "threshold", scenario, metric))
    return tuple(keys)


def bootstrap_metric_samples(
    features: pd.DataFrame,
    labels,
    probabilities,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = RANDOM_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
    support_rule: SupportRule = SupportRule(),
    thresholds: Mapping[str, float] = FROZEN_THRESHOLD_SCENARIOS,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    """Return deterministic normalized-metric samples for D-030 intervals."""
    y, p = validate_audit_inputs(features, labels, probabilities)
    plans = _bootstrap_plans(features, y, p, support_rule, thresholds)
    samples = {
        key: np.full(n_resamples, np.nan, dtype=np.float64)
        for key in _metric_sample_keys((plan.cohort for plan in plans), thresholds)
    }
    scenario_names = tuple(thresholds)
    executed = 0
    for start, integer_weights in _bootstrap_weight_batches(
        len(features), n_resamples, random_state, batch_size
    ):
        size = len(integer_weights)
        stop = start + size
        weights = integer_weights.astype(np.float64)
        for plan in plans:
            if not plan.supported:
                continue
            group_weights = (
                weights
                if plan.cohort.group_key == "whole"
                else weights[:, plan.positions]
            )
            aggregates = group_weights @ plan.values
            rows = aggregates[:, 0]
            positives = aggregates[:, 1]
            negatives = rows - positives
            mean_probability = np.divide(
                aggregates[:, 2], rows, out=np.full(size, np.nan), where=rows > 0
            )
            prevalence = np.divide(
                positives, rows, out=np.full(size, np.nan), where=rows > 0
            )
            samples[(plan.cohort.group_key, "probability", "", "prevalence")][
                start:stop
            ] = prevalence
            samples[(plan.cohort.group_key, "probability", "", "mean_probability")][
                start:stop
            ] = mean_probability
            samples[(plan.cohort.group_key, "probability", "", "brier_score")][
                start:stop
            ] = np.divide(
                aggregates[:, 3], rows, out=np.full(size, np.nan), where=rows > 0
            )
            samples[(plan.cohort.group_key, "probability", "", "log_loss")][
                start:stop
            ] = np.divide(
                aggregates[:, 4], rows, out=np.full(size, np.nan), where=rows > 0
            )
            samples[(plan.cohort.group_key, "probability", "", "calibration_gap")][
                start:stop
            ] = mean_probability - prevalence
            roc_auc, pr_auc = _weighted_ranking_metrics(group_weights, plan)
            samples[(plan.cohort.group_key, "probability", "", "roc_auc")][
                start:stop
            ] = roc_auc
            samples[(plan.cohort.group_key, "probability", "", "pr_auc")][
                start:stop
            ] = pr_auc

            offset = 5
            for scenario in scenario_names:
                tp = aggregates[:, offset]
                fp = aggregates[:, offset + 1]
                offset += 2
                predicted_positive = tp + fp
                samples[(plan.cohort.group_key, "threshold", scenario, "recall")][
                    start:stop
                ] = np.divide(
                    tp, positives, out=np.full(size, np.nan), where=positives > 0
                )
                samples[(plan.cohort.group_key, "threshold", scenario, "precision")][
                    start:stop
                ] = np.divide(
                    tp,
                    predicted_positive,
                    out=np.full(size, np.nan),
                    where=predicted_positive > 0,
                )
                samples[
                    (
                        plan.cohort.group_key,
                        "threshold",
                        scenario,
                        "false_positive_rate",
                    )
                ][start:stop] = np.divide(
                    fp, negatives, out=np.full(size, np.nan), where=negatives > 0
                )
        executed += size
    if executed != n_resamples:
        raise RuntimeError(
            f"Bootstrap executed {executed} resamples; expected {n_resamples}."
        )
    return samples


def bootstrap_intervals(
    features: pd.DataFrame,
    labels,
    probabilities,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    random_state: int = RANDOM_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
    support_rule: SupportRule = SupportRule(),
    thresholds: Mapping[str, float] = FROZEN_THRESHOLD_SCENARIOS,
) -> pd.DataFrame:
    """Percentile intervals for normalized group metrics and directional gaps."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1).")
    point = audit_point_estimates(
        features,
        labels,
        probabilities,
        support_rule=support_rule,
        thresholds=thresholds,
    )
    samples = bootstrap_metric_samples(
        features,
        labels,
        probabilities,
        n_resamples=n_resamples,
        random_state=random_state,
        batch_size=batch_size,
        support_rule=support_rule,
        thresholds=thresholds,
    )
    alpha = 1.0 - confidence
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    cohorts = cohort_slices(features, include_whole=True)
    probability_lookup = {
        (row.group_key, row.metric): row
        for row in point.probability_metrics.itertuples(index=False)
    }
    threshold_lookup = {
        (row.group_key, row.scenario, row.metric): row
        for row in point.threshold_metrics.itertuples(index=False)
    }
    gap_lookup = {
        (row.group_key, row.metric_family, row.scenario, row.metric): row
        for row in point.metric_gaps.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []

    def add_interval(
        cohort: CohortSlice,
        estimate_type: str,
        family: str,
        scenario: str,
        metric: str,
        point_estimate,
        point_status: str,
        point_reason: str,
        values: np.ndarray,
    ) -> None:
        finite = values[np.isfinite(values)]
        if point_status != "available" or not len(finite):
            status = "unavailable"
            lower = upper = None
            reason = point_reason or "no_valid_bootstrap_resamples"
        else:
            status = "available"
            lower, upper = (float(value) for value in np.quantile(finite, quantiles))
            reason = ""
        rows.append(
            {
                "cohort_axis": cohort.cohort_axis,
                "group_key": cohort.group_key,
                "group_label": cohort.group_label,
                "estimate_type": estimate_type,
                "metric_family": family,
                "scenario": scenario,
                "metric": metric,
                "gap_direction": (
                    "group_minus_whole_cohort" if estimate_type == "gap" else ""
                ),
                "point_estimate": point_estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence": confidence,
                "n_resamples": n_resamples,
                "valid_resamples": int(len(finite)),
                "random_seed": random_state,
                "status": status,
                "unavailable_reason": reason,
            }
        )

    for cohort in cohorts:
        for metric in PROBABILITY_METRICS:
            point_row = probability_lookup[(cohort.group_key, metric)]
            values = samples[(cohort.group_key, "probability", "", metric)]
            add_interval(
                cohort,
                "group_metric",
                "probability",
                "",
                metric,
                point_row.value,
                point_row.status,
                point_row.unavailable_reason,
                values,
            )
        for scenario in thresholds:
            for metric in THRESHOLD_NORMALIZED_METRICS:
                point_row = threshold_lookup[(cohort.group_key, scenario, metric)]
                values = samples[(cohort.group_key, "threshold", scenario, metric)]
                add_interval(
                    cohort,
                    "group_metric",
                    "threshold",
                    scenario,
                    metric,
                    point_row.value,
                    point_row.status,
                    point_row.unavailable_reason,
                    values,
                )

    whole_key = "whole"
    for cohort in cohorts:
        if cohort.group_key == whole_key:
            continue
        for metric in PROBABILITY_METRICS:
            point_row = gap_lookup[(cohort.group_key, "probability", "", metric)]
            values = (
                samples[(cohort.group_key, "probability", "", metric)]
                - samples[(whole_key, "probability", "", metric)]
            )
            add_interval(
                cohort,
                "gap",
                "probability",
                "",
                metric,
                point_row.gap,
                point_row.status,
                point_row.unavailable_reason,
                values,
            )
        for scenario in thresholds:
            for metric in THRESHOLD_NORMALIZED_METRICS:
                point_row = gap_lookup[
                    (cohort.group_key, "threshold", scenario, metric)
                ]
                values = (
                    samples[(cohort.group_key, "threshold", scenario, metric)]
                    - samples[(whole_key, "threshold", scenario, metric)]
                )
                add_interval(
                    cohort,
                    "gap",
                    "threshold",
                    scenario,
                    metric,
                    point_row.gap,
                    point_row.status,
                    point_row.unavailable_reason,
                    values,
                )
    return pd.DataFrame(rows)


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize an aggregate frame as deterministic UTF-8/LF CSV bytes."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(frame.columns)
    for values in frame.itertuples(index=False, name=None):
        writer.writerow(
            [
                ""
                if value is None or (isinstance(value, float) and math.isnan(value))
                else value
                for value in values
            ]
        )
    return buffer.getvalue().encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_calibration_support(
    splits: DataSplits,
    path: Path = CALIBRATION_SUPPORT_PATH,
    support_rule: SupportRule = SupportRule(),
) -> pd.DataFrame:
    """Generate and write the calibration-only D-029 evidence."""
    if len(splits.calibration) != 25_368:
        raise RuntimeError(
            f"Calibration split has {len(splits.calibration)} rows; expected 25,368."
        )
    if len(splits.test) != 50_736 or len(splits.test) != 2 * len(splits.calibration):
        raise RuntimeError(
            "Test split must contain 50,736 rows and be exactly twice calibration."
        )
    frame = calibration_support_table(splits, support_rule)
    if not frame["meets_candidate_floor"].all():
        failed = frame.loc[~frame["meets_candidate_floor"], "group_key"].tolist()
        raise RuntimeError(
            "Calibration support contradicts the predeclared D-029 candidates; "
            f"stop before test (failed={failed})."
        )
    _write_bytes(path, dataframe_csv_bytes(frame))
    return frame


def _load_and_validate_calibration_support(splits: DataSplits) -> pd.DataFrame:
    """Read-only D-029 gate for the frozen calibration support evidence."""
    if not CALIBRATION_SUPPORT_PATH.is_file():
        raise RuntimeError(
            "D-029 calibration support evidence is missing; stop before test scoring."
        )

    expected = calibration_support_table(splits)
    if list(expected.columns) != list(CALIBRATION_SUPPORT_COLUMNS):
        raise RuntimeError(
            "Generated D-029 calibration support columns no longer match the "
            "frozen contract; stop before test scoring."
        )
    if len(expected) != CALIBRATION_SUPPORT_COHORTS:
        raise RuntimeError(
            "Generated D-029 calibration support does not contain exactly "
            f"{CALIBRATION_SUPPORT_COHORTS} cohorts; stop before test scoring."
        )
    expected_order = [
        (cohort.cohort_axis, cohort.group_key, cohort.group_label)
        for cohort in cohort_slices(
            splits.calibration[FEATURE_COLUMNS], include_whole=False
        )
    ]
    observed_order = list(
        expected[["cohort_axis", "group_key", "group_label"]].itertuples(
            index=False, name=None
        )
    )
    if observed_order != expected_order:
        raise RuntimeError(
            "Generated D-029 calibration cohort order no longer matches the "
            "frozen contract; stop before test scoring."
        )
    if not expected["meets_candidate_floor"].eq(True).all():
        failed = expected.loc[
            ~expected["meets_candidate_floor"], "group_key"
        ].tolist()
        raise RuntimeError(
            "Generated calibration support contradicts D-029; stop before test "
            f"scoring (failed={failed})."
        )

    expected_bytes = dataframe_csv_bytes(expected)
    try:
        observed_bytes = CALIBRATION_SUPPORT_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "D-029 calibration support evidence could not be read; stop before "
            "test scoring."
        ) from exc
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            "D-029 calibration support evidence was altered or is not the exact "
            "prepare_data() result; stop before test scoring."
        )
    return expected


def score_official_test(bundle: dict, splits: DataSplits) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read and score exactly the frozen P3 test split; no other split is read."""
    test_features = splits.test[FEATURE_COLUMNS].copy()
    test_labels = splits.test[TARGET].to_numpy(dtype=np.float64, copy=True)
    if len(test_features) != 50_736:
        raise RuntimeError(
            f"Official P12 test split has {len(test_features)} rows; expected 50,736."
        )
    scorer = select_probability_scorer(bundle)
    probabilities = predict_probability_frame(scorer, test_features)
    validate_audit_inputs(test_features, test_labels, probabilities)
    return test_features, test_labels, probabilities


def _score_calibration_for_benchmark(
    bundle: dict, splits: DataSplits
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Score calibration solely as unreported computational benchmark input."""
    features = splits.calibration[FEATURE_COLUMNS].copy()
    labels = splits.calibration[TARGET].to_numpy(dtype=np.float64, copy=True)
    scorer = select_probability_scorer(bundle)
    probabilities = predict_probability_frame(scorer, features)
    validate_audit_inputs(features, labels, probabilities)
    return features, labels, probabilities


def _package_versions() -> dict[str, str]:
    import matplotlib as matplotlib_package

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib_package.__version__,
    }


def benchmark_bootstrap(
    bundle: dict,
    splits: DataSplits,
    *,
    path: Path = BOOTSTRAP_BENCHMARK_PATH,
) -> dict[str, object]:
    """Measure the exact 5,000-resample D-030 candidate on calibration."""
    artifact_hashes = verify_frozen_artifacts()
    features, labels, probabilities = _score_calibration_for_benchmark(bundle, splits)

    # A tiny untimed run warms imports and array kernels without inspecting or
    # publishing any subgroup result from calibration.
    bootstrap_intervals(
        features,
        labels,
        probabilities,
        n_resamples=8,
        random_state=RANDOM_SEED,
        batch_size=4,
    )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    intervals = bootstrap_intervals(
        features,
        labels,
        probabilities,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        random_state=RANDOM_SEED,
        batch_size=BOOTSTRAP_BATCH_SIZE,
    )
    runtime_seconds = time.perf_counter() - started
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    incremental_current = max(0, after_current - before_current)
    incremental_peak = max(0, peak - before_current)
    memory_mib = incremental_peak / (1024.0 * 1024.0)
    all_requested_resamples_recorded = bool(
        set(intervals["n_resamples"]) == {BOOTSTRAP_RESAMPLES}
    )
    passes = (
        runtime_seconds <= BOOTSTRAP_RUNTIME_LIMIT_SECONDS
        and memory_mib <= BOOTSTRAP_MEMORY_LIMIT_MIB
        and all_requested_resamples_recorded
    )
    evidence: dict[str, object] = {
        "schema_version": 1,
        "purpose": "D-030 computational feasibility only; no calibration subgroup result is published or interpreted",
        "source_split": "calibration",
        "source_rows": len(features),
        "artifacts": {
            "model_sha256": artifact_hashes["model"],
            "shap_background_sha256": artifact_hashes["shap_background"],
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "random_seed": RANDOM_SEED,
            "batch_size": BOOTSTRAP_BATCH_SIZE,
            "rng_order": BOOTSTRAP_RNG_ORDER,
        },
        "project_operational_guardrails": {
            "warm_runtime_seconds_max": BOOTSTRAP_RUNTIME_LIMIT_SECONDS,
            "incremental_python_memory_mib_max": BOOTSTRAP_MEMORY_LIMIT_MIB,
            "statement": "Project operational limits, not statistical standards",
        },
        "measurement": {
            "warm_runtime_seconds": runtime_seconds,
            "incremental_python_current_bytes": incremental_current,
            "incremental_python_peak_bytes": incremental_peak,
            "incremental_python_peak_mib": memory_mib,
            "interval_rows": len(intervals),
            "result_sha256": hashlib.sha256(dataframe_csv_bytes(intervals)).hexdigest(),
            "all_requested_resamples_recorded": all_requested_resamples_recorded,
            "passes": passes,
        },
        "determinism": {
            "fixed_seed": RANDOM_SEED,
            "algorithm_and_order_frozen": True,
            "repeatability_covered_by_focused_tests": True,
            "point_estimates_independent_of_seed": True,
        },
        "environment": {
            **_package_versions(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    if not passes:
        raise RuntimeError(
            "The 5,000-resample calibration benchmark exceeded a frozen "
            f"D-030 guardrail (runtime={runtime_seconds:.3f}s, "
            f"memory={memory_mib:.3f} MiB); stop before the official audit."
        )
    _write_bytes(path, _json_bytes(evidence))
    return evidence


def _assert_decisions_accepted() -> None:
    decisions = (PROJECT_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    for decision in ("D-029", "D-030", "D-031"):
        prefix = f"| {decision} |"
        row = next((line for line in decisions.splitlines() if line.startswith(prefix)), None)
        if row is None or "| Accepted |" not in row:
            raise RuntimeError(
                f"{decision} must be Accepted in the working tree before official test scoring."
            )


def _load_and_validate_benchmark(
    *, artifact_hashes: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Load and fully validate the frozen D-030 benchmark evidence."""
    if not BOOTSTRAP_BENCHMARK_PATH.is_file():
        raise RuntimeError("D-030 benchmark evidence is missing; stop before test.")
    try:
        payload = json.loads(BOOTSTRAP_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "D-030 benchmark evidence is unreadable or invalid JSON; stop before test."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "D-030 benchmark evidence must be a JSON object; stop before test."
        )

    def invalid(detail: str) -> None:
        raise RuntimeError(
            f"D-030 benchmark evidence {detail}; stop before test."
        )

    def required_mapping(name: str) -> Mapping[str, object]:
        value = payload.get(name)
        if not isinstance(value, dict):
            invalid(f"has a missing or invalid {name!r} object")
        return value

    def finite_nonnegative(value: object, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            invalid(f"has an invalid non-negative finite value for {name!r}")
        return float(value)

    def nonnegative_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            invalid(f"has an invalid non-negative integer for {name!r}")
        return value

    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 1:
        invalid("has an unexpected schema_version")
    if payload.get("purpose") != (
        "D-030 computational feasibility only; no calibration subgroup result "
        "is published or interpreted"
    ):
        invalid("has an unexpected purpose")
    if payload.get("source_split") != "calibration":
        invalid("has an unexpected source_split")
    if type(payload.get("source_rows")) is not int or payload.get(
        "source_rows"
    ) != 25_368:
        invalid("has an unexpected source_rows value")

    bootstrap = required_mapping("bootstrap")
    expected_bootstrap = {
        "method": BOOTSTRAP_METHOD,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "interval": "percentile",
        "random_seed": RANDOM_SEED,
        "batch_size": BOOTSTRAP_BATCH_SIZE,
        "rng_order": BOOTSTRAP_RNG_ORDER,
    }
    if any(bootstrap.get(key) != value for key, value in expected_bootstrap.items()):
        invalid("does not match the frozen bootstrap contract")
    for key in ("n_resamples", "random_seed", "batch_size"):
        if isinstance(bootstrap.get(key), bool) or not isinstance(
            bootstrap.get(key), int
        ):
            invalid(f"has an invalid integer type for bootstrap.{key}")
    if isinstance(bootstrap.get("confidence"), bool) or not isinstance(
        bootstrap.get("confidence"), (int, float)
    ):
        invalid("has an invalid numeric type for bootstrap.confidence")

    guardrails = required_mapping("project_operational_guardrails")
    runtime_limit = finite_nonnegative(
        guardrails.get("warm_runtime_seconds_max"),
        "project_operational_guardrails.warm_runtime_seconds_max",
    )
    memory_limit = finite_nonnegative(
        guardrails.get("incremental_python_memory_mib_max"),
        "project_operational_guardrails.incremental_python_memory_mib_max",
    )
    if (
        runtime_limit != BOOTSTRAP_RUNTIME_LIMIT_SECONDS
        or memory_limit != BOOTSTRAP_MEMORY_LIMIT_MIB
        or guardrails.get("statement")
        != "Project operational limits, not statistical standards"
    ):
        invalid("does not match the frozen project operational guardrails")

    measurement = required_mapping("measurement")
    runtime_seconds = finite_nonnegative(
        measurement.get("warm_runtime_seconds"), "measurement.warm_runtime_seconds"
    )
    current_bytes = nonnegative_integer(
        measurement.get("incremental_python_current_bytes"),
        "measurement.incremental_python_current_bytes",
    )
    peak_bytes = nonnegative_integer(
        measurement.get("incremental_python_peak_bytes"),
        "measurement.incremental_python_peak_bytes",
    )
    peak_mib = finite_nonnegative(
        measurement.get("incremental_python_peak_mib"),
        "measurement.incremental_python_peak_mib",
    )
    if current_bytes > peak_bytes or not math.isclose(
        peak_mib,
        peak_bytes / (1024.0 * 1024.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        invalid("has inconsistent incremental Python memory measurements")
    if type(measurement.get("interval_rows")) is not int or measurement.get(
        "interval_rows"
    ) != BOOTSTRAP_INTERVAL_ROWS:
        invalid("has an unexpected measurement.interval_rows value")

    result_sha256 = measurement.get("result_sha256")
    if (
        not isinstance(result_sha256, str)
        or len(result_sha256) != 64
        or result_sha256 != result_sha256.lower()
        or any(character not in "0123456789abcdef" for character in result_sha256)
    ):
        invalid("has an invalid measurement.result_sha256")

    all_requested_resamples_recorded = measurement.get(
        "all_requested_resamples_recorded"
    )
    if not isinstance(all_requested_resamples_recorded, bool):
        invalid("has an invalid all_requested_resamples_recorded flag")
    recomputed_passes = (
        runtime_seconds <= BOOTSTRAP_RUNTIME_LIMIT_SECONDS
        and peak_mib <= BOOTSTRAP_MEMORY_LIMIT_MIB
        and all_requested_resamples_recorded
    )
    if not (measurement.get("passes") is recomputed_passes is True):
        invalid("does not pass the recomputed operational guardrails")

    determinism = required_mapping("determinism")
    if determinism != {
        "fixed_seed": RANDOM_SEED,
        "algorithm_and_order_frozen": True,
        "repeatability_covered_by_focused_tests": True,
        "point_estimates_independent_of_seed": True,
    }:
        invalid("does not match the frozen determinism contract")

    observed_hashes = (
        dict(artifact_hashes)
        if artifact_hashes is not None
        else verify_frozen_artifacts()
    )
    artifact_evidence = required_mapping("artifacts")
    if artifact_evidence != {
        "model_sha256": observed_hashes.get("model"),
        "shap_background_sha256": observed_hashes.get("shap_background"),
    }:
        invalid("is not bound to the current frozen artifact hashes")

    environment = required_mapping("environment")
    required_versions = {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "joblib",
        "matplotlib",
        "platform",
        "machine",
        "processor",
    }
    if set(environment) != required_versions or any(
        not isinstance(environment[key], str) for key in required_versions
    ):
        invalid("has an incomplete or invalid environment record")
    return payload


def _reference_profile_contract(bundle: dict) -> list[dict[str, object]]:
    """Recheck the four public synthetic profiles without duplicating them."""
    from tests.reference_profiles import REFERENCE_PROFILES, format_display

    scorer = select_probability_scorer(bundle)
    rows = pd.DataFrame(
        [profile.features for profile in REFERENCE_PROFILES],
        columns=FEATURE_COLUMNS,
    ).astype("uint8")
    observed = predict_probability_frame(scorer, rows)
    records = []
    for profile, probability in zip(REFERENCE_PROFILES, observed):
        if probability != profile.expected_probability:
            raise RuntimeError(
                f"Reference profile {profile.name!r} changed from "
                f"{profile.expected_probability!r} to {float(probability)!r}."
            )
        if format_display(float(probability)) != profile.expected_display:
            raise RuntimeError(f"Reference display changed for {profile.name!r}.")
        records.append(
            {
                "name": profile.name,
                "probability": float(probability),
                "display": profile.expected_display,
            }
        )
    return records


def _write_metric_intervals_plot(
    intervals: pd.DataFrame, support: pd.DataFrame, path: Path
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_metrics = (
        ("brier_score", "Brier score"),
        ("calibration_gap", "Calibration gap"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    )
    group_intervals = intervals[
        (intervals["estimate_type"] == "group_metric")
        & (intervals["metric_family"] == "probability")
        & (intervals["cohort_axis"] != "whole")
    ]
    whole = intervals[
        (intervals["estimate_type"] == "group_metric")
        & (intervals["group_key"] == "whole")
        & (intervals["metric_family"] == "probability")
    ].set_index("metric")
    ordered_labels = support.loc[support["cohort_axis"] != "whole", "group_label"].tolist()
    rendered_labels = [label.replace("$", r"\$") for label in ordered_labels]
    figure, axes = plt.subplots(1, 4, figsize=(18, 11), sharey=True)
    y_positions = np.arange(len(ordered_labels))
    for axis, (metric, title) in zip(axes, selected_metrics):
        subset = group_intervals[group_intervals["metric"] == metric].set_index(
            "group_label"
        ).loc[ordered_labels]
        values = subset["point_estimate"].to_numpy(float)
        lower = subset["ci_lower"].to_numpy(float)
        upper = subset["ci_upper"].to_numpy(float)
        axis.errorbar(
            values,
            y_positions,
            xerr=np.vstack([values - lower, upper - values]),
            fmt="o",
            markersize=4,
            linewidth=1,
            capsize=2,
            color="#3f6388",
            ecolor="#7892ad",
        )
        axis.axvline(
            float(whole.loc[metric, "point_estimate"]),
            color="#5f6368",
            linestyle="--",
            linewidth=1,
            label="Whole cohort",
        )
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Metric value (95% percentile interval)")
    axes[0].set_yticks(y_positions, rendered_labels)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="lower right", frameon=False)
    figure.suptitle("P12 subgroup metrics with bootstrap uncertainty", fontsize=15)
    figure.text(
        0.5,
        0.01,
        "Intervals describe sampling uncertainty. D-019 scenarios are descriptive points, not clinical decisions.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "diabetes-risk-ml P12"},
    )
    plt.close(figure)


def _write_calibration_plot(
    intervals: pd.DataFrame, support: pd.DataFrame, path: Path
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    point = intervals[
        (intervals["estimate_type"] == "group_metric")
        & (intervals["metric_family"] == "probability")
    ]
    prevalence = point[point["metric"] == "prevalence"].set_index("group_key")
    mean_probability = point[point["metric"] == "mean_probability"].set_index(
        "group_key"
    )
    axes_order = ("sex", "age", "income", "sex_x_age")
    figure, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    marker_by_axis = {"sex": "o", "age": "s", "income": "^", "sex_x_age": "D"}
    axis_titles = {
        "sex": "Sex",
        "age": "Age",
        "income": "Income",
        "sex_x_age": "Sex × Age",
    }
    intersection_offsets = {
        "sex_0__age_18_49": (-82, 9),
        "sex_0__age_50_64": (-78, -13),
        "sex_0__age_65_74": (-86, 16),
        "sex_0__age_75_plus": (-78, -20),
        "sex_1__age_18_49": (6, -13),
        "sex_1__age_50_64": (8, -16),
        "sex_1__age_65_74": (8, 16),
        "sex_1__age_75_plus": (8, -20),
    }
    age_offsets = {
        "age_18_49": (6, 9),
        "age_50_64": (6, -13),
        "age_65_74": (8, 16),
        "age_75_plus": (8, -20),
    }
    for axis, cohort_axis in zip(axes.ravel(), axes_order):
        rows = support[support["cohort_axis"] == cohort_axis]
        for row in rows.itertuples(index=False):
            x = prevalence.loc[row.group_key]
            y = mean_probability.loc[row.group_key]
            axis.errorbar(
                float(x.point_estimate),
                float(y.point_estimate),
                xerr=np.array(
                    [
                        [float(x.point_estimate - x.ci_lower)],
                        [float(x.ci_upper - x.point_estimate)],
                    ]
                ),
                yerr=np.array(
                    [
                        [float(y.point_estimate - y.ci_lower)],
                        [float(y.ci_upper - y.point_estimate)],
                    ]
                ),
                fmt=marker_by_axis[cohort_axis],
                markersize=5,
                color="#4f6f8f",
                ecolor="#8ca0b4",
                capsize=2,
            )
            if cohort_axis == "sex_x_age":
                offset = intersection_offsets[row.group_key]
                horizontal_alignment = "left"
            elif cohort_axis == "income":
                offset = (6, 4)
                horizontal_alignment = "left"
            elif cohort_axis == "age":
                offset = age_offsets[row.group_key]
                horizontal_alignment = "left"
            else:
                offset = (6, 4)
                horizontal_alignment = "left"
            axis.annotate(
                (
                    row.group_key.removeprefix("income_")
                    if cohort_axis == "income"
                    else row.group_label.replace("$", r"\$")
                ),
                (float(x.point_estimate), float(y.point_estimate)),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                ha=horizontal_alignment,
            )
        if cohort_axis == "income":
            handles = [
                Line2D(
                    [],
                    [],
                    color="#4f6f8f",
                    marker="^",
                    linestyle="None",
                    markersize=5,
                    label=(
                        f"{row.group_key.removeprefix('income_')}  "
                        f"{row.group_label.replace('$', chr(92) + '$')}"
                    ),
                )
                for row in rows.itertuples(index=False)
            ]
            axis.legend(
                handles=handles,
                title="Income code",
                loc="upper left",
                frameon=False,
                fontsize=7,
                title_fontsize=8,
                handletextpad=0.3,
                labelspacing=0.35,
            )
        axis.plot([0, 0.3], [0, 0.3], color="#666666", linestyle="--", linewidth=1)
        axis.set_title(axis_titles[cohort_axis])
        axis.grid(alpha=0.2)
        axis.set_xlabel("Observed prevalence (95% interval)")
        axis.set_ylabel("Mean served probability (95% interval)")
    figure.suptitle("Observed prevalence and mean served probability by cohort", fontsize=15)
    figure.text(
        0.5,
        0.015,
        "The diagonal is descriptive agreement, not a fairness or clinical pass/fail boundary.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "diabetes-risk-ml P12"},
    )
    plt.close(figure)


def _markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    def render(value: object) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _build_report(
    audit: AuditTables,
    intervals: pd.DataFrame,
    configuration: Mapping[str, object],
    benchmark: Mapping[str, object],
) -> str:
    probability_available = audit.probability_metrics[
        audit.probability_metrics["status"] == "available"
    ]
    probability_wide = probability_available.pivot(
        index=["cohort_axis", "group_key", "group_label"],
        columns="metric",
        values="value",
    ).reset_index()
    probability_wide = probability_wide[
        ["cohort_axis", "group_key", "group_label", *PROBABILITY_METRICS]
    ]
    threshold_available = audit.threshold_metrics[
        audit.threshold_metrics["status"] == "available"
    ]
    threshold_wide = threshold_available.pivot(
        index=["cohort_axis", "group_key", "group_label", "scenario", "threshold"],
        columns="metric",
        values="value",
    ).reset_index()
    support_rows = [
        (
            row.cohort_axis,
            row.group_label,
            row.row_count,
            row.positive_count,
            row.negative_count,
            row.prevalence,
            row.meets_support_floor,
        )
        for row in audit.support.itertuples(index=False)
    ]
    metric_rows = [
        tuple(row[column] for column in probability_wide.columns)
        for _, row in probability_wide.iterrows()
    ]
    threshold_rows = [
        (
            row.cohort_axis,
            row.group_label,
            row.scenario,
            row.threshold,
            row.recall,
            row.precision,
            row.false_positive_rate,
            int(row.false_positive_count),
            int(row.false_negative_count),
        )
        for row in threshold_wide.itertuples(index=False)
    ]
    whole = probability_wide[probability_wide["group_key"] == "whole"].iloc[0]
    gap_available = audit.metric_gaps[audit.metric_gaps["status"] == "available"]
    calibration_gaps = gap_available[
        (gap_available["metric_family"] == "probability")
        & (gap_available["metric"] == "calibration_gap")
    ].copy()
    largest_calibration = calibration_gaps.iloc[
        calibration_gaps["gap"].abs().to_numpy().argmax()
    ]
    measurement = benchmark["measurement"]
    profiles = configuration["reference_profiles"]
    return f"""# P12 Fairness Audit Technical Report

**Local evidence date:** 2026-07-17

**Decisions:** D-029, D-030, and D-031 (Accepted before official test scoring)

**Reproduction:** `python -m src.fairness --official-audit`

## Objective and scope

P12 audits the behavior of the frozen P8 positive-class probability contract across predeclared demographic and socioeconomic cohorts. It is a descriptive, uncertainty-aware audit, not a fairness certification. The model is not retrained, recalibrated, reweighted, compared, or mitigated; no threshold is optimized and no group-specific decision rule is created.

The audited contract remains the D-016 `HistGradientBoostingClassifier`, artifact schema version 2, D-018 `calibration_method = none`, and D-019 probability-only product. P9 explanations, P10 scenarios, P11 batch behavior, and `app/streamlit_app.py` are unchanged.

## Ordered execution and decision gates

1. The official model and SHAP-background hashes, schema, D-018 outcome, four D-019 scenarios, and four synthetic reference probabilities were recorded and verified.
2. `calibration_support.csv` was generated through `prepare_data()` from calibration only. It contains no subgroup model-performance result. Calibration has 25,368 rows; test has 50,736, exactly twice calibration.
3. D-029 accepted the predeclared cohorts and the 500-row/100-positive/100-negative reporting floor. The calibration table and acceptance are unstaged working-tree changes intended to be versioned together after human review; the table was not already versioned during execution.
4. The pure engine and synthetic tests froze formulas, common bins, unavailable states, ordering, seed, resampling, and `group - whole cohort` direction.
5. The exact 5,000-resample calibration benchmark passed the project operational limits before D-030 was accepted. Calibration arrays were used only for computational feasibility; their subgroup metrics are neither published nor interpreted.
6. D-031 accepted report-first publication: complete aggregate evidence in GitHub documentation, a concise README summary, no Streamlit fairness section, and no P12 deployment gate.
7. Only after all three decisions were Accepted did the workflow score the unchanged P3 test split and generate the official audit.

## Cohorts and semantics

- **Sex:** dataset codes 0/1, labeled Female/Male by the existing feature-label contract. This historical binary field does not cover all sex characteristics or gender identities.
- **Age:** BRFSS ordinal codes grouped as 18-49 (1-6), 50-64 (7-9), 65-74 (10-11), and 75+ (12-13).
- **Income:** all eight original ordinal household-income categories, without regrouping.
- **Intersection:** Sex × Age only, in the fixed Sex-then-Age order.

Every row belongs to exactly one group on each declared axis. No cohort was selected, merged, hidden, renamed, or removed after official results were observed.

## Support

{_markdown_table(['Axis', 'Group', 'Rows', 'Positive', 'Negative', 'Prevalence', 'Supported'], support_rows)}

All official test groups meet D-029. Groups below the floor would retain support and prevalence while every other field remained explicitly unavailable; one-class ranking metrics also have an explicit unavailable state.

## Exact metric definitions

For labels `y_i in {0, 1}`, served probabilities `p_i`, and group size `n`:

- Prevalence: `(1 / n) * sum_i(y_i)`.
- Mean served probability: `(1 / n) * sum_i(p_i)`.
- Brier score: `(1 / n) * sum_i((p_i - y_i)^2)`.
- Log loss: `-(1 / n) * sum_i(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))`, with the same float64 machine-epsilon clipping convention as P8.
- ROC-AUC: within-group positive-versus-negative ranking probability with score ties receiving half credit; both classes are required.
- PR-AUC: scikit-learn average precision on the within-group ranking; both classes are required.
- Signed calibration gap: mean served probability minus observed prevalence. Positive values mean the mean estimate is higher than observed prevalence; negative values mean it is lower.
- Recall: $TP/(TP+FN)$; precision: $TP/(TP+FP)$; false-positive rate: $FP/(FP+TN)$.
- Every gap is exactly `group metric - whole-cohort metric`; its sign is directional, not an approval category.

False-positive and false-negative counts are published as descriptive counts without bootstrap intervals or gap calculations. Intervals cover the normalized probability metrics (including prevalence) plus recall, precision, and false-positive rate at the four D-019 scenarios. Reliability-bin statistics and counts do not receive intervals.

## Reliability data

All groups use the same ten equal-width bins: `[0.0, 0.1)`, `[0.1, 0.2)`, ..., `[0.9, 1.0]`; exactly 1.0 belongs to the last bin. Empty bins remain explicit with `status = unavailable` and `unavailable_reason = empty_bin`. The fixed-bin reliability data are published in `official_test_reliability.csv`. The separate `calibration_by_group.png` visualizes calibration-in-the-large by comparing each cohort's observed prevalence with its mean served probability.

## Bootstrap and computational benchmark

The uncertainty contract uses 5,000 ordinary nonparametric resamples of the complete audited split with replacement, NumPy seed 42, and percentile 95% intervals. Whole-cohort and subgroup metrics are recomputed from the same multiplicity weights in each resample, so gap samples preserve their dependence. Undefined degenerate resamples are excluded metric-by-metric and counted in `valid_resamples`; all official eligible results retained all 5,000 resamples.

The calibration benchmark measured {measurement['warm_runtime_seconds']:.3f} warm seconds and {measurement['incremental_python_peak_mib']:.3f} MiB incremental Python peak memory. It passed the frozen project limits of 600 seconds and 512 MiB. These are project operational guardrails, not statistical standards. The intervals are descriptive; no significance test, multiple-comparison claim, universal tolerance, or fairness pass/fail threshold is used.

## Split provenance and environment

- P3 selected population: 253,680 BRFSS 2015 rows with exact duplicates retained (D-014).
- Train: 177,576 rows; calibration: 25,368; test: 50,736.
- Calibration selected only cohort feasibility and benchmark viability. It supplied no published P12 subgroup-performance claim.
- Official P12 evidence uses test only after D-029 through D-031 were frozen. Test was already used in P5 model selection and P8 evaluation, so it is not described as pristine or once-only.
- Feature order: exact 21-column `FEATURE_COLUMNS` contract.
- Packages: {', '.join(f'{name} {version}' for name, version in configuration['package_versions'].items())}.

## Artifact identity and serving regression

- `models/diabetes_risk_model.joblib`: `{configuration['artifacts']['model']['sha256']}`.
- `models/shap_background_v1.json`: `{configuration['artifacts']['shap_background']['sha256']}`.
- Artifact schema: {configuration['artifacts']['model']['schema_version']}.
- Calibration method: `{configuration['artifacts']['model']['calibration_method']}`.
- D-019 scenarios: {', '.join(f'{name}={value:.2f}' for name, value in FROZEN_THRESHOLD_SCENARIOS.items())}.
- Synthetic reference probabilities/displays: {', '.join(f"{profile['probability']!r} ({profile['display']})" for profile in profiles)}.

Neither artifact was regenerated. All four probabilities and displays are unchanged.

## Complete probability, ranking, and calibration results

Whole-cohort test values are prevalence {whole.prevalence:.6f}, mean probability {whole.mean_probability:.6f}, Brier {whole.brier_score:.6f}, log loss {whole.log_loss:.6f}, ROC-AUC {whole.roc_auc:.6f}, PR-AUC {whole.pr_auc:.6f}, and signed calibration gap {whole.calibration_gap:+.6f}.

The largest absolute group-minus-whole calibration-gap difference is {largest_calibration.gap:+.6f} for {largest_calibration.group_label}. This is a descriptive model-behavior difference, not evidence of a cause or discriminatory mechanism.

{_markdown_table(list(probability_wide.columns), metric_rows)}

`official_test_bootstrap_intervals.csv` gives the point interval and directional gap interval for every normalized metric. `official_test_metric_gaps.csv` gives all point gaps. `metric_intervals.png` provides an accessible view of selected uncertainty intervals.

![P12 metric intervals](metric_intervals.png)

![P12 calibration by group](calibration_by_group.png)

## Complete D-019 scenario results

The four thresholds are frozen documentation points from P8. They are not served decisions, clinical cutoffs, screening rules, or recommendations, and P12 selects no new threshold.

{_markdown_table(['Axis', 'Group', 'Scenario', 'Threshold', 'Recall', 'Precision', 'FPR', 'FP', 'FN'], threshold_rows)}

## Responsible interpretation and limitations

BRFSS 2015 is historical and self-reported. `Diabetes_binary` combines self-reported prediabetes/diabetes and may reflect access to diagnosis, survey measurement, and reporting bias rather than an independently verified clinical state. The fitted model and every subgroup result inherit those limitations.

Precision, PR-AUC, and threshold-conditioned errors depend on prevalence, so differences across groups with different observed base rates cannot be read as a single intrinsic model property. Age and Income are ordinal survey groups, not exact continuous values. Sex is binary in this processed dataset and does not cover all identities. Race and ethnicity are absent from the processed model dataset, so P12 cannot audit them or claim complete demographic coverage.

Observed differences do not prove causality, biological mechanisms, discrimination, clinical validity, demographic parity, or equalized odds. Small differences or overlapping intervals do not prove fairness. Conversely, an unfavorable difference is not hidden and does not by itself identify its cause. Group averages and intervals cannot determine whether one individual's prediction is fair.

P12 performs no mitigation. It does not retrain, reweight, recalibrate, alter a threshold, create group-specific thresholds, or change the product. Mitigation, broader identity coverage, and product response require separately scoped work after review.

## Privacy and product boundary

All published P12 files are aggregates. No real feature row, target vector, individual probability, source/split index, SHAP value, identifier, or uploaded user record is published. False-positive/false-negative values are cohort counts only.

P12 is report-first under D-031. `app/streamlit_app.py` was not modified, Streamlit was not run, and no deployment, restart, localhost review, or public smoke test was performed. The public application therefore remains functionally unchanged.

## Generated files and exact reproduction

- `configuration.json`
- `calibration_support.csv`
- `bootstrap_benchmark.json`
- `official_test_group_support.csv`
- `official_test_probability_metrics.csv`
- `official_test_metric_gaps.csv`
- `official_test_bootstrap_intervals.csv`
- `official_test_reliability.csv`
- `official_test_threshold_metrics.csv`
- `metric_intervals.png`
- `calibration_by_group.png`
- `report.md`

From the pinned Python 3.12 environment with the documented raw CSV present:

```powershell
.\\.venv\\Scripts\\python.exe -m src.fairness --official-audit
```

Repeated official-audit runs may reproduce this evidence only. They cannot modify D-029 through D-031 or any serving contract.
"""


def official_audit(
    *, evidence_dir: Path = P12_EVIDENCE_DIR
) -> tuple[AuditTables, pd.DataFrame, dict[str, object]]:
    """Run and write the single frozen official P12 aggregate audit."""
    _assert_decisions_accepted()
    hashes = verify_frozen_artifacts()
    splits, summary = prepare_data(random_state=RANDOM_SEED)
    _load_and_validate_calibration_support(splits)
    if (
        len(splits.train) != 177_576
        or len(splits.calibration) != 25_368
        or len(splits.test) != 50_736
    ):
        raise RuntimeError("P3 split sizes no longer match the frozen contract.")
    benchmark = _load_and_validate_benchmark(artifact_hashes=hashes)
    bundle = load_artifact()
    metadata = bundle["metadata"]
    if (
        metadata["schema_version"] != ARTIFACT_SCHEMA_VERSION
        or metadata["calibration_method"] != NO_CALIBRATION
        or metadata["threshold_scenarios"] != FROZEN_THRESHOLD_SCENARIOS
    ):
        raise RuntimeError("The loaded artifact no longer matches the frozen P8 contract.")
    reference_profiles = _reference_profile_contract(bundle)
    features, labels, probabilities = score_official_test(bundle, splits)
    audit = audit_point_estimates(features, labels, probabilities)
    unsupported = audit.support.loc[
        ~audit.support["meets_support_floor"], "group_key"
    ].tolist()
    if unsupported:
        # D-029 still defines output for this state, but the planning evidence
        # expected every actual test group to be eligible. Stop rather than
        # silently continue under a contradicted implementation assumption.
        raise RuntimeError(
            f"Official test support contradicted the expected eligible cohorts: {unsupported}."
        )
    intervals = bootstrap_intervals(
        features,
        labels,
        probabilities,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        random_state=RANDOM_SEED,
        batch_size=BOOTSTRAP_BATCH_SIZE,
    )
    available = intervals[intervals["status"] == "available"]
    if set(available["valid_resamples"]) != {BOOTSTRAP_RESAMPLES}:
        raise RuntimeError(
            "An official eligible interval lost bootstrap resamples; stop for review."
        )

    configuration: dict[str, object] = {
        "schema_version": 1,
        "phase": "P12 Fairness Audit",
        "run_date": "2026-07-17",
        "decisions": {"D-029": "Accepted", "D-030": "Accepted", "D-031": "Accepted"},
        "artifacts": {
            "model": {
                "path": str(DEFAULT_ARTIFACT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": hashes["model"],
                "schema_version": metadata["schema_version"],
                "model_class": metadata["model_class"],
                "calibration_method": metadata["calibration_method"],
            },
            "shap_background": {
                "path": str(SHAP_BACKGROUND_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": hashes["shap_background"],
            },
        },
        "reference_profiles": reference_profiles,
        "data": {
            "source": "BRFSS 2015 processed binary dataset",
            "prepared_by": "src.data.prepare_data",
            "dataset_rows": summary["n_rows"],
            "train_rows": len(splits.train),
            "calibration_rows": len(splits.calibration),
            "test_rows": len(splits.test),
            "official_audit_split": "test",
            "feature_columns": list(FEATURE_COLUMNS),
            "target": TARGET,
        },
        "cohorts": {
            "axis_order": list(COHORT_AXIS_ORDER),
            "age_bands": [
                {"key": slug, "label": label, "codes": list(codes)}
                for slug, label, codes in AGE_BANDS
            ],
            "income_codes": list(range(1, 9)),
            "intersection": "sex_x_age",
            "support_rule": {
                "minimum_rows": SUPPORT_MIN_ROWS,
                "minimum_positives": SUPPORT_MIN_POSITIVES,
                "minimum_negatives": SUPPORT_MIN_NEGATIVES,
            },
        },
        "metrics": {
            "probability": list(PROBABILITY_METRICS),
            "threshold_normalized": list(THRESHOLD_NORMALIZED_METRICS),
            "threshold_counts_without_intervals": list(THRESHOLD_COUNT_METRICS),
            "gap_direction": "group_minus_whole_cohort",
            "reliability_bin_edges": RELIABILITY_BIN_EDGES.tolist(),
            "reliability_last_bin_includes_one": True,
        },
        "threshold_scenarios": dict(FROZEN_THRESHOLD_SCENARIOS),
        "bootstrap": {
            "method": "ordinary nonparametric whole-test-split row bootstrap with replacement",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "random_seed": RANDOM_SEED,
            "batch_size": BOOTSTRAP_BATCH_SIZE,
            "valid_resample_policy": "metric-specific finite resamples; all official eligible metrics require all 5,000",
        },
        "package_versions": _package_versions(),
        "reproduction_command": ".\\.venv\\Scripts\\python.exe -m src.fairness --official-audit",
        "publication": {
            "aggregate_only": True,
            "report_first": True,
            "streamlit_changed": False,
            "deployment_required": False,
            "fairness_certification": False,
            "mitigation_performed": False,
        },
    }

    evidence_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "configuration.json": _json_bytes(configuration),
        "official_test_group_support.csv": dataframe_csv_bytes(audit.support),
        "official_test_probability_metrics.csv": dataframe_csv_bytes(
            audit.probability_metrics
        ),
        "official_test_metric_gaps.csv": dataframe_csv_bytes(audit.metric_gaps),
        "official_test_bootstrap_intervals.csv": dataframe_csv_bytes(intervals),
        "official_test_reliability.csv": dataframe_csv_bytes(audit.reliability),
        "official_test_threshold_metrics.csv": dataframe_csv_bytes(
            audit.threshold_metrics
        ),
    }
    for name, payload in outputs.items():
        _write_bytes(evidence_dir / name, payload)
    _write_metric_intervals_plot(
        intervals, audit.support, evidence_dir / "metric_intervals.png"
    )
    _write_calibration_plot(
        intervals, audit.support, evidence_dir / "calibration_by_group.png"
    )
    report = _build_report(audit, intervals, configuration, benchmark)
    _write_bytes(evidence_dir / "report.md", report.encode("utf-8"))
    return audit, intervals, configuration


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--calibration-support",
        action="store_true",
        help="Generate only calibration_support.csv before accepting D-029.",
    )
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run only the 5,000-resample calibration benchmark for D-030.",
    )
    group.add_argument(
        "--official-audit",
        action="store_true",
        help="Run the frozen official test audit after D-029 through D-031.",
    )
    args = parser.parse_args(argv)

    if args.calibration_support:
        splits, _ = prepare_data(random_state=RANDOM_SEED)
        frame = write_calibration_support(splits)
        print(
            f"Calibration support written: {len(frame)} aggregate groups; "
            "all meet the candidate floor."
        )
        return

    if args.benchmark:
        verify_frozen_artifacts()
        splits, _ = prepare_data(random_state=RANDOM_SEED)
        bundle = load_artifact()
        evidence = benchmark_bootstrap(bundle, splits)
        measurement = evidence["measurement"]
        print(
            "Bootstrap benchmark passed: "
            f"{measurement['warm_runtime_seconds']:.3f} seconds, "
            f"{measurement['incremental_python_peak_mib']:.3f} MiB peak."
        )
        return

    audit, intervals, _ = official_audit()
    print(
        f"Official P12 audit written: {len(audit.support)} support rows, "
        f"{len(intervals)} bootstrap interval rows."
    )


if __name__ == "__main__":
    main()
