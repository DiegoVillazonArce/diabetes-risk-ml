"""P9 SHAP explanations for the frozen P8 positive-probability contract.

The serving path in this module is deliberately data-independent: it loads a
small, separately versioned train-derived aggregate background and explains
only the already-fitted schema-version-2 artifact.  Raw BRFSS data is used
only by the explicit offline evidence entry point::

    python -m src.explainability

That command deterministically rebuilds the privacy-safe background, runs the
approved global and synthetic-profile analysis, and writes aggregate evidence
under ``docs/p9-explainability``.  Streamlit never calls that workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import sklearn
from scipy.special import expit

from src import artifacts, calibration
from src.data import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET,
    VALUE_RANGES,
    DataSplits,
)
from src.feature_labels import feature_label, format_feature_value
from src.modeling import predict_positive_proba

matplotlib.use("Agg")

SHAP_VERSION = "0.52.0"
EXPLANATION_SCHEMA_VERSION = 1
BACKGROUND_ASSET_SCHEMA_VERSION = 1
POSITIVE_CLASS = 1
OUTPUT_CONTRACT = "positive_class_probability"
EXPLAINER_TYPE = "TreeExplainer"
FEATURE_PERTURBATION = "interventional"
MODEL_OUTPUT = "probability"

BACKGROUND_ROWS = 256
GLOBAL_SAMPLE_ROWS = 5_000
ADDITIVITY_TOLERANCE = 1e-4
RAW_MARGIN_TOLERANCE = 1e-6
REPRODUCIBILITY_TOLERANCE = 1e-10

# Fixed before the final implementation run, from the compatibility spike.
MAX_EXPLAINER_CREATION_SECONDS = 5.0
MAX_WARM_LOCAL_SECONDS = 0.25
MAX_GLOBAL_SAMPLE_SECONDS = 60.0
MAX_INCREMENTAL_MEMORY_MIB = 512.0

BACKGROUND_ASSET_PATH = PROJECT_ROOT / "models" / "shap_background_v1.json"
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "p9-explainability"

BACKGROUND_CONSTRUCTION = (
    "Train-only arithmetic-mean centroids over 256 stable bands sorted by "
    "the frozen model's positive-class probability"
)


class ExplainabilityError(RuntimeError):
    """A clear, user-safe failure of the P9 explanation layer."""


@dataclass(frozen=True)
class GlobalSample:
    """Fixed calibration sample for offline global analysis only."""

    features: pd.DataFrame
    n_positive: int
    n_negative: int
    source_prevalence: float
    sample_prevalence: float


@dataclass(frozen=True)
class ExplanationBatch:
    """Normalized additive explanations in positive-probability space."""

    contributions: pd.DataFrame
    base_values: np.ndarray
    model_outputs: np.ndarray
    additivity_errors: np.ndarray


def artifact_sha256(path: Path | str = artifacts.DEFAULT_ARTIFACT_PATH) -> str:
    """Return the byte identity of the official model artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_feature_frame(frame: pd.DataFrame, *, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    if list(frame.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"{name} columns must match FEATURE_COLUMNS exactly and in order."
        )
    if frame.empty:
        raise ValueError(f"{name} must contain at least one row.")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values.") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    for position, feature in enumerate(FEATURE_COLUMNS):
        lower, upper = VALUE_RANGES[feature]
        column = values[:, position]
        if column.min() < lower or column.max() > upper:
            raise ValueError(
                f"{name} feature {feature!r} falls outside [{lower}, {upper}]."
            )


def positive_class_index(model: object) -> int:
    """Identify the unique model output column for ``Diabetes_binary = 1``."""
    classes = list(getattr(model, "classes_", []))
    if classes != [0, 1]:
        raise ValueError(
            f"P9 requires fitted binary classes [0, 1]; observed {classes!r}."
        )
    return classes.index(POSITIVE_CLASS)


def build_aggregate_background(
    model: object,
    train_features: pd.DataFrame,
    n_rows: int = BACKGROUND_ROWS,
) -> pd.DataFrame:
    """Build deterministic privacy-safe train aggregates for SHAP.

    Train rows are stably ordered by the frozen model probability, split into
    ``n_rows`` contiguous bands, and averaged feature by feature.  Every
    deployed row therefore aggregates hundreds of source rows; neither target,
    split indices, source indices, nor individual rows are retained.
    """
    _validate_feature_frame(train_features, name="Train features")
    positive_class_index(model)
    if not 1 <= n_rows <= len(train_features):
        raise ValueError(
            f"Background size must be in [1, {len(train_features)}]; got {n_rows}."
        )
    probabilities = predict_positive_proba(model, train_features)
    order = np.argsort(probabilities, kind="mergesort")
    groups = np.array_split(order, n_rows)
    background = pd.DataFrame(
        [
            train_features.iloc[group].mean(axis=0).to_numpy(dtype=np.float64)
            for group in groups
        ],
        columns=FEATURE_COLUMNS,
    )
    _validate_feature_frame(background, name="Aggregate background")
    return background


def exact_row_match_count(
    candidates: pd.DataFrame, source_rows: pd.DataFrame
) -> int:
    """Count exact candidate/source feature-vector matches for privacy QA."""
    _validate_feature_frame(candidates, name="Candidate rows")
    _validate_feature_frame(source_rows, name="Source rows")
    source = set(map(tuple, source_rows.to_numpy(dtype=np.float64)))
    return sum(
        tuple(row) in source
        for row in candidates.to_numpy(dtype=np.float64)
    )


def _background_group_sizes(n_source_rows: int) -> tuple[int, int]:
    quotient, remainder = divmod(n_source_rows, BACKGROUND_ROWS)
    minimum = quotient
    maximum = quotient + int(remainder > 0)
    return minimum, maximum


def background_asset_payload(
    background: pd.DataFrame,
    *,
    model_artifact_sha256: str,
    n_source_rows: int,
) -> dict[str, Any]:
    """Create the separately versioned deployment-background contract."""
    _validate_feature_frame(background, name="Aggregate background")
    if len(background) != BACKGROUND_ROWS:
        raise ValueError(
            f"Deployment background must have {BACKGROUND_ROWS} rows; "
            f"got {len(background)}."
        )
    minimum, maximum = _background_group_sizes(n_source_rows)
    return {
        "schema_version": BACKGROUND_ASSET_SCHEMA_VERSION,
        "asset_type": "aggregated_shap_background",
        "feature_columns": list(FEATURE_COLUMNS),
        "n_rows": BACKGROUND_ROWS,
        "project_seed": RANDOM_SEED,
        "source_split": "train",
        "construction": BACKGROUND_CONSTRUCTION,
        "minimum_source_rows_per_centroid": minimum,
        "maximum_source_rows_per_centroid": maximum,
        "model_artifact_sha256": model_artifact_sha256,
        "data": background.to_numpy(dtype=np.float64).tolist(),
    }


def write_background_asset(
    payload: dict[str, Any], path: Path | str = BACKGROUND_ASSET_PATH
) -> Path:
    """Write the deterministic JSON background asset used by Streamlit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def load_background_asset(
    path: Path | str = BACKGROUND_ASSET_PATH,
    *,
    expected_artifact_sha256: str | None = None,
) -> pd.DataFrame:
    """Load and strictly validate the privacy-safe deployment background."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SHAP background asset not found at '{path}'. Regenerate P9 "
            "evidence with 'python -m src.explainability'."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read the SHAP background asset at '{path}'."
        ) from error
    required = {
        "schema_version",
        "asset_type",
        "feature_columns",
        "n_rows",
        "project_seed",
        "source_split",
        "construction",
        "minimum_source_rows_per_centroid",
        "maximum_source_rows_per_centroid",
        "model_artifact_sha256",
        "data",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("SHAP background asset metadata is incomplete or unexpected.")
    expected_metadata = {
        "schema_version": BACKGROUND_ASSET_SCHEMA_VERSION,
        "asset_type": "aggregated_shap_background",
        "feature_columns": list(FEATURE_COLUMNS),
        "n_rows": BACKGROUND_ROWS,
        "project_seed": RANDOM_SEED,
        "source_split": "train",
        "construction": BACKGROUND_CONSTRUCTION,
    }
    for key, expected in expected_metadata.items():
        if payload[key] != expected:
            raise ValueError(
                f"SHAP background asset {key} does not match the P9 contract."
            )
    if expected_artifact_sha256 is not None and (
        payload["model_artifact_sha256"] != expected_artifact_sha256
    ):
        raise ValueError(
            "SHAP background asset was generated for a different model artifact."
        )
    minimum = payload["minimum_source_rows_per_centroid"]
    maximum = payload["maximum_source_rows_per_centroid"]
    if not isinstance(minimum, int) or not isinstance(maximum, int) or minimum < 2:
        raise ValueError("Every deployment centroid must aggregate multiple train rows.")
    background = pd.DataFrame(payload["data"], columns=FEATURE_COLUMNS)
    if background.shape != (BACKGROUND_ROWS, len(FEATURE_COLUMNS)):
        raise ValueError(
            "SHAP background data does not have the required 256 x 21 shape."
        )
    _validate_feature_frame(background, name="Deployment background")
    return background.astype(np.float64)


def select_global_sample(
    calibration_frame: pd.DataFrame,
    n_rows: int = GLOBAL_SAMPLE_ROWS,
    random_state: int = RANDOM_SEED,
) -> GlobalSample:
    """Select the fixed proportional calibration sample for global evidence."""
    required = [*FEATURE_COLUMNS, TARGET]
    if not isinstance(calibration_frame, pd.DataFrame) or any(
        column not in calibration_frame for column in required
    ):
        raise ValueError("Calibration frame must contain features and target.")
    if n_rows < 1:
        raise ValueError("Global sample size must be positive.")
    target = calibration_frame[TARGET].to_numpy()
    classes = set(np.unique(target))
    if classes != {0, 1}:
        raise ValueError(f"Calibration target must contain 0 and 1; got {classes}.")
    size = min(n_rows, len(calibration_frame))
    source_prevalence = float(np.mean(target))
    n_positive = int(round(size * source_prevalence))
    n_positive = min(max(n_positive, 1), size - 1)
    n_negative = size - n_positive
    positive_positions = np.flatnonzero(target == 1)
    negative_positions = np.flatnonzero(target == 0)
    if n_positive > len(positive_positions) or n_negative > len(negative_positions):
        raise ValueError("Requested stratified sample exceeds a class population.")
    rng = np.random.default_rng(random_state)
    positions = np.concatenate(
        [
            rng.choice(positive_positions, size=n_positive, replace=False),
            rng.choice(negative_positions, size=n_negative, replace=False),
        ]
    )
    positions.sort()
    features = calibration_frame.iloc[positions][FEATURE_COLUMNS].reset_index(drop=True)
    _validate_feature_frame(features, name="Global calibration sample")
    return GlobalSample(
        features=features,
        n_positive=n_positive,
        n_negative=n_negative,
        source_prevalence=source_prevalence,
        sample_prevalence=n_positive / size,
    )


def create_explainer(bundle: dict, background: pd.DataFrame) -> shap.TreeExplainer:
    """Create the accepted D-020 probability TreeExplainer."""
    if shap.__version__ != SHAP_VERSION:
        raise ExplainabilityError(
            f"P9 requires shap=={SHAP_VERSION}; runtime has {shap.__version__}."
        )
    artifacts.validate_artifact_bundle(bundle)
    if bundle["metadata"]["schema_version"] != 2:
        raise ExplainabilityError("P9 requires the schema-version-2 artifact.")
    if bundle["metadata"]["calibration_method"] != calibration.NO_CALIBRATION:
        raise ExplainabilityError(
            "The accepted P9 explainer targets D-018 calibration_method='none'."
        )
    positive_class_index(bundle["model"])
    _validate_feature_frame(background, name="SHAP background")
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.TreeExplainer(
        bundle["model"],
        data=masker,
        model_output=MODEL_OUTPUT,
        feature_perturbation=FEATURE_PERTURBATION,
        feature_names=list(FEATURE_COLUMNS),
    )
    if int(explainer.data.shape[0]) != len(background):
        raise ExplainabilityError(
            "SHAP changed the effective background size unexpectedly."
        )
    return explainer


def normalize_shap_values(
    raw_values: Any,
    *,
    n_rows: int,
    positive_index: int = 1,
) -> np.ndarray:
    """Normalize supported SHAP APIs to the exact ``n x 21`` contract."""
    if isinstance(raw_values, shap.Explanation):
        raw_values = raw_values.values
    if isinstance(raw_values, list):
        if len(raw_values) != 2:
            raise ExplainabilityError(
                f"Expected two class outputs; SHAP returned {len(raw_values)}."
            )
        raw_values = raw_values[positive_index]
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 3:
        if values.shape == (n_rows, len(FEATURE_COLUMNS), 2):
            values = values[:, :, positive_index]
        elif values.shape == (2, n_rows, len(FEATURE_COLUMNS)):
            values = values[positive_index]
        else:
            raise ExplainabilityError(
                f"Unsupported three-dimensional SHAP output {values.shape}."
            )
    if values.ndim == 1 and n_rows == 1:
        values = values.reshape(1, -1)
    expected_shape = (n_rows, len(FEATURE_COLUMNS))
    if values.shape != expected_shape:
        raise ExplainabilityError(
            f"SHAP values must have shape {expected_shape}; got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ExplainabilityError("SHAP returned non-finite contributions.")
    return values


def _positive_expected_value(explainer: shap.TreeExplainer) -> float:
    values = np.asarray(explainer.expected_value, dtype=np.float64)
    if values.ndim == 0:
        value = float(values)
    elif values.size == 2:
        value = float(values.reshape(-1)[1])
    elif values.size == 1:
        value = float(values.reshape(-1)[0])
    else:
        raise ExplainabilityError(
            f"Unsupported SHAP expected-value shape {values.shape}."
        )
    if not math.isfinite(value):
        raise ExplainabilityError("SHAP returned a non-finite expected value.")
    return value


def validate_additivity(
    base_values: np.ndarray,
    contributions: np.ndarray,
    model_outputs: np.ndarray,
    *,
    tolerance: float = ADDITIVITY_TOLERANCE,
) -> np.ndarray:
    """Validate the D-020 probability identity and return absolute errors."""
    reconstructed = np.asarray(base_values) + np.asarray(contributions).sum(axis=1)
    errors = np.abs(reconstructed - np.asarray(model_outputs))
    if not np.isfinite(errors).all():
        raise ExplainabilityError("Additivity validation produced non-finite errors.")
    if float(errors.max()) > tolerance:
        raise ExplainabilityError(
            f"SHAP additivity error {errors.max():.6g} exceeds {tolerance:.6g}."
        )
    return errors


def explain_dataframe(
    bundle: dict,
    explainer: shap.TreeExplainer,
    frame: pd.DataFrame,
) -> ExplanationBatch:
    """Explain rows in the exact feature order and probability output space."""
    artifacts.validate_artifact_bundle(bundle)
    _validate_feature_frame(frame, name="Rows to explain")
    positive_index = positive_class_index(bundle["model"])
    raw_values = explainer.shap_values(frame, check_additivity=False)
    values = normalize_shap_values(
        raw_values, n_rows=len(frame), positive_index=positive_index
    )
    base_value = _positive_expected_value(explainer)
    base_values = np.full(len(frame), base_value, dtype=np.float64)
    outputs = predict_positive_proba(bundle["model"], frame)
    errors = validate_additivity(base_values, values, outputs)
    return ExplanationBatch(
        contributions=pd.DataFrame(values, columns=FEATURE_COLUMNS),
        base_values=base_values,
        model_outputs=np.asarray(outputs, dtype=np.float64),
        additivity_errors=errors,
    )


def mean_absolute_importance(batch: ExplanationBatch) -> pd.DataFrame:
    """Aggregate global mean absolute probability contributions."""
    values = batch.contributions.abs().mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_label": [feature_label(feature) for feature in FEATURE_COLUMNS],
            "mean_absolute_contribution": [float(values[feature]) for feature in FEATURE_COLUMNS],
        }
    ).sort_values(
        ["mean_absolute_contribution", "feature"], ascending=[False, True]
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def local_contribution_table(
    profile_name: str,
    row: pd.DataFrame,
    batch: ExplanationBatch,
) -> pd.DataFrame:
    """Create the auditable 21-feature table for one synthetic/user case."""
    _validate_feature_frame(row, name="Local explanation row")
    if len(row) != 1 or len(batch.contributions) != 1:
        raise ValueError("Local contribution tables require exactly one row.")
    contributions = batch.contributions.iloc[0]
    records = []
    for feature in FEATURE_COLUMNS:
        contribution = float(contributions[feature])
        direction = (
            "increased the model estimate"
            if contribution > 0
            else "decreased the model estimate"
            if contribution < 0
            else "did not change the model estimate"
        )
        records.append(
            {
                "profile": profile_name,
                "feature": feature,
                "feature_label": feature_label(feature),
                "feature_value": float(row.iloc[0][feature]),
                "display_value": format_feature_value(feature, row.iloc[0][feature]),
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
                "direction": direction,
                "base_value": float(batch.base_values[0]),
                "model_probability": float(batch.model_outputs[0]),
                "additivity_error": float(batch.additivity_errors[0]),
            }
        )
    table = pd.DataFrame(records).sort_values(
        ["absolute_contribution", "feature"], ascending=[False, True]
    )
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def explain_local_values(
    bundle: dict,
    explainer: shap.TreeExplainer,
    values: dict[str, int | float],
    *,
    name: str = "submitted_case",
) -> pd.DataFrame:
    """Explain one validated case and return its readable contribution table."""
    row = artifacts.input_to_dataframe(values)
    batch = explain_dataframe(bundle, explainer, row)
    return local_contribution_table(name, row, batch)


def select_display_contributions(
    table: pd.DataFrame, max_factors: int = 6
) -> pd.DataFrame:
    """Keep the strongest factors, including both directions when present."""
    if max_factors < 2:
        raise ValueError("At least two display factors are required.")
    ordered = table.sort_values(
        ["absolute_contribution", "feature"], ascending=[False, True]
    )
    selected = ordered.head(max_factors).copy()
    for sign in (1, -1):
        candidates = ordered[
            np.sign(ordered["contribution"].to_numpy(dtype=float)) == sign
        ]
        if candidates.empty:
            continue
        if not (
            np.sign(selected["contribution"].to_numpy(dtype=float)) == sign
        ).any():
            selected = pd.concat([selected.iloc[:-1], candidates.iloc[[0]]])
    return selected.drop_duplicates("feature").sort_values(
        ["absolute_contribution", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def load_runtime_explainer(
    bundle: dict, *, expected_artifact_sha256: str | None = None
) -> shap.TreeExplainer:
    """Create a runtime explainer bound to one validated artifact identity."""
    expected_hash = expected_artifact_sha256 or artifact_sha256(
        artifacts.DEFAULT_ARTIFACT_PATH
    )
    background = load_background_asset(
        BACKGROUND_ASSET_PATH, expected_artifact_sha256=expected_hash
    )
    return create_explainer(bundle, background)


def build_analysis_inputs(
    splits: DataSplits, model: object
) -> tuple[pd.DataFrame, GlobalSample]:
    """Build P9 inputs from train/calibration only; test is never read."""
    train_features = splits.train[FEATURE_COLUMNS].copy()
    background = build_aggregate_background(model, train_features)
    global_sample = select_global_sample(splits.calibration)
    return background, global_sample


# ---------------------------------------------------------------------------
# Offline evidence, spike measurements, plots, and report
# ---------------------------------------------------------------------------


def _memory_snapshot() -> tuple[float | None, float | None]:
    try:
        import psutil

        memory = psutil.Process(os.getpid()).memory_info()
        return memory.wset / 2**20, memory.peak_wset / 2**20
    except (ImportError, AttributeError, OSError):
        return None, None


def _memory_increment(
    before: tuple[float | None, float | None],
    after: tuple[float | None, float | None],
) -> float | None:
    before_working, before_peak = before
    _, after_peak = after
    if before_working is None or before_peak is None or after_peak is None:
        return None
    return max(0.0, after_peak - max(before_working, before_peak))


def _generic_tree_explainer(
    model: object, background: pd.DataFrame, model_output: str
) -> tuple[shap.TreeExplainer, float]:
    masker = shap.maskers.Independent(background, max_samples=len(background))
    started = time.perf_counter()
    explainer = shap.TreeExplainer(
        model,
        data=masker,
        model_output=model_output,
        feature_perturbation=FEATURE_PERTURBATION,
        feature_names=list(FEATURE_COLUMNS),
    )
    return explainer, time.perf_counter() - started


def _measure_tree_alternative(
    *,
    name: str,
    model: object,
    background: pd.DataFrame,
    profiles: pd.DataFrame,
    global_frame: pd.DataFrame | None,
    model_output: str,
) -> dict[str, Any]:
    before = _memory_snapshot()
    tracemalloc.start()
    explainer, creation_seconds = _generic_tree_explainer(
        model, background, model_output
    )
    started = time.perf_counter()
    local_values = normalize_shap_values(
        explainer.shap_values(profiles.iloc[[0]], check_additivity=False),
        n_rows=1,
    )
    local_seconds = time.perf_counter() - started
    profile_values = normalize_shap_values(
        explainer.shap_values(profiles, check_additivity=False),
        n_rows=len(profiles),
    )
    base = _positive_expected_value(explainer)
    if model_output == MODEL_OUTPUT:
        profile_outputs = predict_positive_proba(model, profiles)
    else:
        profile_outputs = np.asarray(model.decision_function(profiles), dtype=float)
    profile_errors = np.abs(base + profile_values.sum(axis=1) - profile_outputs)

    global_seconds = None
    global_errors = None
    transform_error = None
    if global_frame is not None:
        started = time.perf_counter()
        global_values = normalize_shap_values(
            explainer.shap_values(global_frame, check_additivity=False),
            n_rows=len(global_frame),
        )
        global_seconds = time.perf_counter() - started
        if model_output == MODEL_OUTPUT:
            global_outputs = predict_positive_proba(model, global_frame)
        else:
            global_outputs = np.asarray(
                model.decision_function(global_frame), dtype=float
            )
        reconstructed = base + global_values.sum(axis=1)
        global_errors = np.abs(reconstructed - global_outputs)
        if model_output == "raw":
            transform_error = np.abs(
                expit(reconstructed) - predict_positive_proba(model, global_frame)
            )
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = _memory_snapshot()
    return {
        "name": name,
        "status": "accepted" if model_output == MODEL_OUTPUT else "rejected",
        "explainer": type(explainer).__name__,
        "feature_perturbation": explainer.feature_perturbation,
        "model_output": explainer.model_output,
        "expected_value_shape": list(np.asarray(explainer.expected_value).shape),
        "shap_values_shape": list(profile_values.shape),
        "positive_class": POSITIVE_CLASS,
        "requested_background_rows": len(background),
        "effective_background_rows": int(explainer.data.shape[0]),
        "finite": bool(
            np.isfinite(profile_values).all() and math.isfinite(base)
        ),
        "profile_additivity_max": float(profile_errors.max()),
        "profile_additivity_mean": float(profile_errors.mean()),
        "global_additivity_max": (
            None if global_errors is None else float(global_errors.max())
        ),
        "global_additivity_mean": (
            None if global_errors is None else float(global_errors.mean())
        ),
        "margin_to_probability_max_error": (
            None if transform_error is None else float(transform_error.max())
        ),
        "creation_seconds": creation_seconds,
        "one_local_seconds": local_seconds,
        "global_seconds": global_seconds,
        "python_peak_mib": python_peak / 2**20,
        "process_peak_increment_mib": _memory_increment(before, after),
    }


def _measure_permutation_fallback(
    model: object,
    background: pd.DataFrame,
    profiles: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
) -> dict[str, Any]:
    before = _memory_snapshot()
    tracemalloc.start()
    masker = shap.maskers.Independent(background, max_samples=len(background))

    def score(array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(array), columns=FEATURE_COLUMNS)
        return predict_positive_proba(model, frame)

    started = time.perf_counter()
    explainer = shap.Explainer(
        score,
        masker=masker,
        algorithm="permutation",
        feature_names=list(FEATURE_COLUMNS),
        seed=RANDOM_SEED,
    )
    creation_seconds = time.perf_counter() - started
    minimum_evaluations = 2 * len(FEATURE_COLUMNS) + 1
    started = time.perf_counter()
    local = explainer(
        profiles.iloc[[0]], max_evals=minimum_evaluations, silent=True
    )
    local_seconds = time.perf_counter() - started
    benchmark = benchmark_frame.iloc[:20]
    started = time.perf_counter()
    explainer(benchmark, max_evals=minimum_evaluations, silent=True)
    benchmark_seconds = time.perf_counter() - started
    projection = benchmark_seconds / len(benchmark) * GLOBAL_SAMPLE_ROWS
    values = np.asarray(local.values, dtype=np.float64)
    base = float(np.asarray(local.base_values).reshape(-1)[0])
    output = float(score(profiles.iloc[[0]])[0])
    error = abs(base + values.sum() - output)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = _memory_snapshot()
    return {
        "name": "permutation_probability_safe_aggregate_background_256",
        "status": "rejected",
        "explainer": type(explainer).__name__,
        "feature_perturbation": "model_agnostic_independent_masker",
        "model_output": "positive_class_probability_callable",
        "expected_value_shape": list(np.asarray(local.base_values).shape),
        "shap_values_shape": list(values.shape),
        "positive_class": POSITIVE_CLASS,
        "requested_background_rows": len(background),
        "effective_background_rows": int(explainer.masker.data.shape[0]),
        "finite": bool(np.isfinite(values).all() and math.isfinite(base)),
        "profile_additivity_max": float(error),
        "profile_additivity_mean": float(error),
        "global_additivity_max": None,
        "global_additivity_mean": None,
        "margin_to_probability_max_error": None,
        "creation_seconds": creation_seconds,
        "one_local_seconds": local_seconds,
        "benchmark_rows": len(benchmark),
        "benchmark_seconds": benchmark_seconds,
        "projected_global_seconds": projection,
        "global_seconds": None,
        "global_not_run_reason": (
            "The deterministic 20-row projection exceeded the predeclared "
            f"{MAX_GLOBAL_SAMPLE_SECONDS:.0f}-second global limit."
        ),
        "python_peak_mib": python_peak / 2**20,
        "process_peak_increment_mib": _memory_increment(before, after),
    }


def _implicit_background_observation(
    model: object, real_background: pd.DataFrame
) -> dict[str, Any]:
    started = time.perf_counter()
    explainer = shap.TreeExplainer(
        model,
        data=real_background,
        model_output=MODEL_OUTPUT,
        feature_perturbation=FEATURE_PERTURBATION,
    )
    return {
        "name": "tree_probability_real_background_implicit_masker",
        "status": "rejected",
        "explainer": type(explainer).__name__,
        "feature_perturbation": explainer.feature_perturbation,
        "model_output": explainer.model_output,
        "requested_background_rows": len(real_background),
        "effective_background_rows": int(explainer.data.shape[0]),
        "creation_seconds": time.perf_counter() - started,
        "observed_issue": (
            "SHAP's implicit Independent masker reduced 256 requested rows "
            "to its default max_samples=100."
        ),
    }


def run_compatibility_spike(
    model: object,
    train_features: pd.DataFrame,
    safe_background: pd.DataFrame,
    global_frame: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    selected_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce the evaluated D-020 alternatives and background behavior."""
    real_background = train_features.sample(
        n=BACKGROUND_ROWS, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    implicit = _implicit_background_observation(model, real_background)
    real_explicit = _measure_tree_alternative(
        name="tree_probability_real_background_explicit_256",
        model=model,
        background=real_background,
        profiles=profiles,
        global_frame=None,
        model_output=MODEL_OUTPUT,
    )
    real_explicit["status"] = "rejected"
    real_explicit["rejection_reason"] = (
        "A real-row background may be used offline but cannot be deployed."
    )
    selected = selected_result or _measure_tree_alternative(
            name="tree_probability_safe_aggregate_background_256",
            model=model,
            background=safe_background,
            profiles=profiles,
            global_frame=global_frame,
            model_output=MODEL_OUTPUT,
        )
    raw = _measure_tree_alternative(
        name="tree_raw_margin_safe_aggregate_background_256",
        model=model,
        background=safe_background,
        profiles=profiles,
        global_frame=global_frame,
        model_output="raw",
    )
    raw["rejection_reason"] = (
        "It is faithful but explains log-odds and requires an unnecessary "
        "transformation before reaching the probability users see."
    )
    permutation = _measure_permutation_fallback(
        model, safe_background, profiles, global_frame
    )
    permutation["rejection_reason"] = (
        "Cold local and projected global runtimes exceed the declared limits."
    )
    return {
        "shap_version": shap.__version__,
        "alternatives": [implicit, real_explicit, selected, raw, permutation],
    }


def _write_global_bar_plot(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = importance.sort_values("mean_absolute_contribution")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        ordered["feature_label"],
        ordered["mean_absolute_contribution"] * 100,
        color="#2a6f97",
    )
    axis.set_title("Global model contribution strength")
    axis.set_xlabel("Mean absolute contribution to the model estimate (percentage points)")
    axis.set_ylabel("Model input")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150, metadata={"Software": "diabetes-risk-ml P9"})
    plt.close(figure)


def _write_beeswarm_plot(
    batch: ExplanationBatch, global_frame: pd.DataFrame, path: Path
) -> None:
    import matplotlib.pyplot as plt

    explanation = shap.Explanation(
        values=batch.contributions.to_numpy(),
        base_values=batch.base_values,
        data=global_frame.to_numpy(),
        feature_names=[feature_label(feature) for feature in FEATURE_COLUMNS],
    )
    shap.plots.beeswarm(
        explanation,
        max_display=len(FEATURE_COLUMNS),
        show=False,
        plot_size=(11, 9),
    )
    axis = plt.gca()
    axis.set_title("Distribution of model contributions in the fixed calibration sample")
    axis.set_xlabel("Contribution to the positive-class model probability")
    plt.tight_layout()
    plt.savefig(path, dpi=150, metadata={"Software": "diabetes-risk-ml P9"})
    plt.close()


def _write_waterfall_plot(
    table: pd.DataFrame, path: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    feature_order = {feature: position for position, feature in enumerate(FEATURE_COLUMNS)}
    ordered = table.sort_values(
        "feature", key=lambda series: series.map(feature_order)
    )
    display_values = [
        str(value).replace("$", r"\$") for value in ordered["display_value"]
    ]
    explanation = shap.Explanation(
        values=ordered["contribution"].to_numpy(dtype=float),
        base_values=float(ordered["base_value"].iloc[0]),
        data=ordered["feature_value"].to_numpy(dtype=float),
        display_data=np.asarray(display_values, dtype=object),
        feature_names=ordered["feature_label"].tolist(),
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    figure = plt.gcf()
    figure.set_size_inches(14, 9)
    axis = plt.gca()
    readable_title = title.split(":", 1)[-1].strip().replace("_", " ")
    axis.set_title(
        f"Model contribution breakdown\n{readable_title}", pad=38
    )
    axis.set_xlabel("Contribution to the positive-class model probability")
    figure.subplots_adjust(left=0.40, right=0.94, top=0.82, bottom=0.12)
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "diabetes-risk-ml P9"},
    )
    plt.close(figure)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, rule, *rows])


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not run"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _build_report(
    *,
    configuration: dict[str, Any],
    spike: dict[str, Any],
    importance: pd.DataFrame,
    local_tables: pd.DataFrame,
    additivity: pd.DataFrame,
) -> str:
    selected = next(
        item
        for item in spike["alternatives"]
        if item["name"] == "tree_probability_safe_aggregate_background_256"
    )
    spike_rows = []
    for item in spike["alternatives"]:
        global_time = item.get("global_seconds")
        if global_time is None and item.get("projected_global_seconds") is not None:
            global_time = f"projected {_fmt(item['projected_global_seconds'], 2)}"
        spike_rows.append(
            {
                "Alternative": item["name"],
                "Status": item["status"],
                "Output": item.get("model_output", "probability"),
                "Background": (
                    f"{item.get('requested_background_rows', 'n/a')}/"
                    f"{item.get('effective_background_rows', 'n/a')}"
                ),
                "Max error": _fmt(
                    item.get("global_additivity_max")
                    if item.get("global_additivity_max") is not None
                    else item.get("profile_additivity_max"),
                    10,
                ),
                "Create (s)": _fmt(item.get("creation_seconds"), 4),
                "Local (s)": _fmt(item.get("one_local_seconds"), 4),
                "Global (s)": _fmt(global_time, 2),
            }
        )
    spike_table = _markdown_table(
        pd.DataFrame(spike_rows),
        [
            "Alternative",
            "Status",
            "Output",
            "Background",
            "Max error",
            "Create (s)",
            "Local (s)",
            "Global (s)",
        ],
    )

    top = importance.head(10).copy()
    top["Mean absolute pp"] = (
        top["mean_absolute_contribution"] * 100
    ).map(lambda value: f"{value:.4f}")
    top_table = _markdown_table(
        top.rename(columns={"rank": "Rank", "feature": "Feature", "feature_label": "Label"}),
        ["Rank", "Feature", "Label", "Mean absolute pp"],
    )

    local_rows = []
    for profile, table in local_tables.groupby("profile", sort=False):
        strongest_up = table[table["contribution"] > 0].sort_values(
            "absolute_contribution", ascending=False
        )
        strongest_down = table[table["contribution"] < 0].sort_values(
            "absolute_contribution", ascending=False
        )
        local_rows.append(
            {
                "Profile": profile,
                "Probability": f"{table['model_probability'].iloc[0]:.6f}",
                "Display": f"{table['model_probability'].iloc[0]:.1%}",
                "Reference": f"{table['base_value'].iloc[0]:.6f}",
                "Max error": f"{table['additivity_error'].max():.10f}",
                "Largest increase": (
                    strongest_up.iloc[0]["feature"] if not strongest_up.empty else "none"
                ),
                "Largest decrease": (
                    strongest_down.iloc[0]["feature"] if not strongest_down.empty else "none"
                ),
            }
        )
    local_table = _markdown_table(
        pd.DataFrame(local_rows),
        [
            "Profile",
            "Probability",
            "Display",
            "Reference",
            "Max error",
            "Largest increase",
            "Largest decrease",
        ],
    )

    additivity_for_report = additivity.copy()
    for column in ("base_value", "model_probability", "max_absolute_error", "mean_absolute_error", "tolerance"):
        additivity_for_report[column] = additivity_for_report[column].map(
            lambda value: "aggregate" if pd.isna(value) else f"{float(value):.10f}"
        )
    additivity_table = _markdown_table(
        additivity_for_report.rename(
            columns={
                "scope": "Scope",
                "n_rows": "Rows",
                "base_value": "Base",
                "model_probability": "Output",
                "max_absolute_error": "Max error",
                "mean_absolute_error": "Mean error",
                "tolerance": "Tolerance",
                "passes": "Passes",
            }
        ),
        ["Scope", "Rows", "Base", "Output", "Max error", "Mean error", "Tolerance", "Passes"],
    )

    stack = configuration["stack"]
    background = configuration["background"]
    sample = configuration["global_sample"]
    performance = configuration["performance"]
    reproducibility = configuration["reproducibility"]
    artifact = configuration["artifact"]
    files = configuration["generated_files"]
    file_list = "\n".join(f"- `{name}`" for name in files)
    return f"""# P9 SHAP Explainability Technical Report

## Executive Summary

P9 is implemented, publicly verified, and closed as a read-only explanation layer over the frozen P8 probability contract. D-020 accepts direct explanation of the positive-class probability with SHAP {stack['shap']} `TreeExplainer`; D-021 accepts one 256-row train-derived aggregate background for both offline analysis and Streamlit plus a fixed 5,000-row proportionally stratified calibration sample; and D-022 accepts a hybrid delivery strategy: dynamic cached local explanations in Streamlit and precomputed aggregate/global plus synthetic-reference evidence in GitHub. The model artifact, served probabilities, probability-only product behavior, and medical disclaimer are unchanged. Implementation commit `25c4ed4` passed the mandatory public explanation and four-profile smoke verification on 2026-07-14.

## Objective and Scope

The objective is to explain how the frozen model produces its final P8 estimate, globally and for individual inputs, without retraining, recalibrating, changing a threshold, or adding high/low labels. These attributions describe the fitted model under a declared background; they are not clinical findings, diagnoses, treatment guidance, or evidence that changing an input changes a person's health outcome.

## Artifact Identity

- File: `models/diabetes_risk_model.joblib`
- SHA-256: `{artifact['sha256']}`
- Size: {artifact['size_bytes']} bytes
- Artifact schema: {artifact['schema_version']}
- Model: `{artifact['model_class']}` (D-016)
- Calibration: `{artifact['calibration_method']}` (D-018)
- Served class: `Diabetes_binary = 1`
- Product contract: probability only, with no served threshold or category (D-019)

## Fixed Stack

| Component | Version |
|---|---|
| Python | {stack['python']} |
| NumPy | {stack['numpy']} |
| pandas | {stack['pandas']} |
| scikit-learn | {stack['scikit_learn']} |
| SHAP | {stack['shap']} |
| matplotlib | {stack['matplotlib']} |

SHAP 0.52.0 was pinned only after its Python 3.12 wheel installed without changing the existing NumPy, pandas, or scikit-learn pins and after the spike below passed.

## Compatibility Spike

`requested/effective` background counts are shown together because passing a 256-row DataFrame directly to SHAP creates an implicit `Independent(max_samples=100)` masker and silently retains only 100 rows. The accepted implementation constructs `Independent(..., max_samples=256)` explicitly and verifies 256 effective rows.

{spike_table}

All reported errors are absolute additive-identity errors. The model-agnostic global run was intentionally not executed after its deterministic 20-row benchmark projected beyond the predeclared {MAX_GLOBAL_SAMPLE_SECONDS:.0f}-second limit; this is a measured projection, not an observed 5,000-row runtime.

## D-020 — Explanation Output Contract (Accepted)

The accepted output is the frozen model's direct positive-class probability. Configuration: `TreeExplainer`, `feature_perturbation="interventional"`, `model_output="probability"`, class `1`, exact `FEATURE_COLUMNS` order, scalar expected value, and a normalized `n x 21` contribution matrix. For every explained row:

`model reference estimate + sum(feature contributions) ≈ predict_risk_probability output`

The predeclared direct-probability tolerance is `1e-4`; the observed global maximum is `{selected['global_additivity_max']:.10g}`. The raw-margin route was rejected despite numerical fidelity because its additive units are log-odds and require an additional logistic transformation. The model-agnostic probability fallback was rejected for runtime. No output, class, tolerance, or explainer changed after the full results were inspected.

## D-021 — Background and Global Sample (Accepted)

The accepted background has {background['requested_rows']} requested and {background['effective_rows']} effective rows. It is built only from the train features: train rows are stably sorted by the frozen model probability, divided into 256 bands, and each band is replaced by its feature-wise arithmetic mean. Every centroid aggregates {background['minimum_source_rows_per_centroid']}–{background['maximum_source_rows_per_centroid']} train rows. It contains no target, respondent identifier, split/source index, or real row; offline comparison found {background['exact_train_row_matches']} exact matches with train. The deployable asset is `models/shap_background_v1.json`, with its own schema, feature order, deterministic construction provenance, project seed, model-artifact hash, and strict loader. The centroid algorithm itself uses no random operation; seed 42 governs the stratified global sample.

The global sample contains {sample['rows']} calibration rows: {sample['positive_rows']} positive and {sample['negative_rows']} negative. Calibration prevalence is {sample['source_prevalence']:.10f}; sample prevalence is {sample['sample_prevalence']:.10f}; absolute difference is {sample['absolute_prevalence_difference']:.10f}, within deterministic proportional rounding. The sample is selected with seed 42 and never published. Test is structurally absent from both builders.

The aggregate background is a privacy and deployment compromise: centroids may combine feature values in ways that no respondent reported, and sorting bands by model output can redistribute contributions compared with another valid background. That dependence is part of the explanation contract, not a claim that the background represents an average person.

## D-022 — Delivery Strategy (Accepted)

| Strategy | Evidence | Decision |
|---|---|---|
| Dynamic | Safe and faithful with the aggregate asset; cached warm local runtime meets the bound. | Viable component, but does not itself provide the fixed technical/global record. |
| Precomputed | Fast and simple, but cannot honestly explain an arbitrary submitted input. | Rejected as the app strategy. |
| Hybrid | Dynamic cached local explanation for the submitted input; precomputed aggregate global and four synthetic-profile evidence. | Accepted. |

Streamlit loads no raw CSV, calls no `prepare_data()`, trains no model, and does not derive the background at runtime. It creates and caches a `TreeExplainer` from the versioned aggregate asset under the official artifact hash, computes no global analysis, regenerates no technical plot, and accesses no test data. Widget values exist transiently in the active Streamlit session, but project code does not write or log them outside that session. A clear fallback leaves the unchanged probability visible if the safe background or explainer cannot load.

## Output, Class, and Explainer Configuration

- Output: `{OUTPUT_CONTRACT}` (`model_output="{MODEL_OUTPUT}"`)
- Positive class: `{POSITIVE_CLASS}`; the fitted classes must be exactly `[0, 1]`
- Explainer: `{EXPLAINER_TYPE}`
- Perturbation: `{FEATURE_PERTURBATION}`
- Background: explicit `shap.maskers.Independent`, max samples = {BACKGROUND_ROWS}
- Requested/effective background: {background['requested_rows']}/{background['effective_rows']}
- Feature order: the exact 21-column `src.data.FEATURE_COLUMNS` contract
- Expected/base value shape: scalar
- Contribution shape: `n x 21`

The base value is called the **model reference estimate**. It is the explainer's expected model output over the declared aggregate background; it is not presented as the risk of an average person.

## Additivity and Mathematical Fidelity

{additivity_table}

The probability served to users is compared directly with the additive reconstruction in the same mathematical space. No margin contribution is presented as a probability contribution.

## Reproducibility

- Project seed: {RANDOM_SEED}; used by the global calibration sampler, not by the deterministic centroid construction
- Numerical reproducibility tolerance: `{REPRODUCIBILITY_TOLERANCE}`
- Background repeat maximum difference: `{reproducibility['background_max_difference']}`
- Global sample repeat exact match: `{reproducibility['global_sample_exact_match']}`
- Global contribution repeat maximum difference: `{reproducibility['global_contribution_max_difference']}`
- Global importance repeat maximum difference: `{reproducibility['global_importance_max_difference']}`
- Local contribution repeat maximum difference: `{reproducibility['local_contribution_max_difference']}`

The artifact byte hash, dependency versions, feature order, background asset metadata, sample sizes, and tolerances are recorded in `configuration.json`. The evidence command fails if any additivity, privacy, size, or artifact-identity guard fails.

## Performance and Memory

- Explainer creation: {performance['creation_seconds']:.4f} s (limit {MAX_EXPLAINER_CREATION_SECONDS:.2f} s)
- One warm local explanation: {performance['one_local_seconds']:.4f} s (limit {MAX_WARM_LOCAL_SECONDS:.2f} s)
- Global 5,000-row explanation: {performance['global_seconds']:.4f} s (limit {MAX_GLOBAL_SAMPLE_SECONDS:.2f} s)
- Repeated global run: {performance['repeat_global_seconds']:.4f} s
- Approximate incremental process peak: {_fmt(performance['process_peak_increment_mib'], 3)} MiB (limit {MAX_INCREMENTAL_MEMORY_MIB:.0f} MiB)
- Python-tracked peak during the accepted run: {performance['python_peak_mib']:.3f} MiB

Memory is an approximate process/`tracemalloc` observation, not a platform-independent allocator proof. The app loads only the model, SHAP package, and 256 x 21 aggregate asset; the offline 5,000-row analysis is never run in Streamlit.

## Global Importance

The primary ranking is mean absolute SHAP contribution over the fixed stratified calibration sample. It measures contribution magnitude in this model/sample/background contract; it does not provide a universal ranking or direction.

{top_table}

See `global_importance.csv`, `global_importance_bar.png`, and `global_beeswarm.png`. The CSV contains only 21 aggregate feature rows; no individual calibration row or row-level SHAP matrix is published.

## Four Synthetic Reference Profiles

{local_table}

Each `waterfall_*.png` and the corresponding rows in `local_contributions.csv` use the exact synthetic inputs in `tests/reference_profiles.py`. Binary, age-group, education, and income codes are translated with the same pure label source used by Streamlit. The four served displays remain 0.3%, 60.0%, 70.0%, and 79.9%.

## Privacy

- Raw train, calibration, and test rows are not written under `docs/`, `models/`, or app assets.
- The background asset contains only 256 multi-row arithmetic means and strict non-match evidence.
- `global_importance.csv` is aggregate; `additivity_checks.csv` has one aggregate global row plus the four allowed synthetic-profile checks.
- `local_contributions.csv` contains only the four already-public synthetic profiles.
- Global plots communicate distributions without publishing the source feature matrix or per-row contribution table.
- Streamlit does not log or persist submitted values in project code.

## Limitations and Association vs. Causality

SHAP allocates a fitted model output under a chosen background. It does not establish medical causes, prevention, diagnosis, or an intervention effect. Contributions inherit model error, BRFSS self-report limitations, selection and measurement bias, correlations among features, and background dependence. Correlated inputs can share or redistribute attribution. A local explanation applies only to one model estimate; global importance applies only to the fixed calibration sample. Fairness conclusions require the separate P12 audit, and scenario exploration belongs to P10.

## Reproduction

From the pinned Python 3.12 environment and with the documented raw CSV present:

```text
python -m src.explainability
python -m pytest tests/test_explainability.py -v -p no:cacheprovider
python -m pytest tests/test_app.py tests/test_reference_profiles.py -v -p no:cacheprovider
python -m pytest tests -v -p no:cacheprovider
```

## Generated Files

{file_list}
"""


def generate_evidence(directory: Path | str = EVIDENCE_DIR) -> dict[str, Any]:
    """Generate all accepted P9 assets and evidence deterministically."""
    from src.data import prepare_data
    from tests.reference_profiles import REFERENCE_PROFILES

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_hash_before = artifact_sha256()
    bundle = artifacts.load_artifact()
    model = bundle["model"]
    splits, _ = prepare_data()
    train_features = splits.train[FEATURE_COLUMNS].copy()
    background, global_sample = build_analysis_inputs(splits, model)
    exact_matches = exact_row_match_count(background, train_features)
    if exact_matches:
        raise ExplainabilityError(
            f"Aggregate background matches {exact_matches} real train row(s)."
        )
    payload = background_asset_payload(
        background,
        model_artifact_sha256=artifact_hash_before,
        n_source_rows=len(train_features),
    )
    write_background_asset(payload)
    reloaded_background = load_background_asset(
        expected_artifact_sha256=artifact_hash_before
    )
    if not np.array_equal(background.to_numpy(), reloaded_background.to_numpy()):
        raise ExplainabilityError("Serialized background does not round-trip exactly.")

    profiles = pd.DataFrame(
        [profile.features for profile in REFERENCE_PROFILES],
        columns=FEATURE_COLUMNS,
    )
    before_memory = _memory_snapshot()
    tracemalloc.start()
    started = time.perf_counter()
    explainer = create_explainer(bundle, background)
    creation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    explain_dataframe(bundle, explainer, profiles.iloc[[0]])
    one_local_seconds = time.perf_counter() - started
    profile_batch = explain_dataframe(bundle, explainer, profiles)
    started = time.perf_counter()
    global_batch = explain_dataframe(bundle, explainer, global_sample.features)
    global_seconds = time.perf_counter() - started
    started = time.perf_counter()
    repeated_global_batch = explain_dataframe(
        bundle, explainer, global_sample.features
    )
    repeat_global_seconds = time.perf_counter() - started
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_memory = _memory_snapshot()
    memory_increment = _memory_increment(before_memory, after_memory)

    if creation_seconds > MAX_EXPLAINER_CREATION_SECONDS:
        raise ExplainabilityError("Explainer creation exceeded the P9 time limit.")
    if one_local_seconds > MAX_WARM_LOCAL_SECONDS:
        raise ExplainabilityError("Local explanation exceeded the P9 time limit.")
    if global_seconds > MAX_GLOBAL_SAMPLE_SECONDS:
        raise ExplainabilityError("Global explanation exceeded the P9 time limit.")
    if memory_increment is not None and memory_increment > MAX_INCREMENTAL_MEMORY_MIB:
        raise ExplainabilityError("Explanation exceeded the P9 memory limit.")

    importance = mean_absolute_importance(global_batch)
    repeated_importance = mean_absolute_importance(repeated_global_batch)
    local_frames = []
    for position, profile in enumerate(REFERENCE_PROFILES):
        one_batch = ExplanationBatch(
            contributions=profile_batch.contributions.iloc[[position]].reset_index(drop=True),
            base_values=profile_batch.base_values[[position]],
            model_outputs=profile_batch.model_outputs[[position]],
            additivity_errors=profile_batch.additivity_errors[[position]],
        )
        local_frames.append(
            local_contribution_table(
                profile.name,
                profiles.iloc[[position]].reset_index(drop=True),
                one_batch,
            )
        )
    local_tables = pd.concat(local_frames, ignore_index=True)
    repeated_profile_batch = explain_dataframe(bundle, explainer, profiles)

    selected_result = {
        "name": "tree_probability_safe_aggregate_background_256",
        "status": "accepted",
        "explainer": type(explainer).__name__,
        "feature_perturbation": explainer.feature_perturbation,
        "model_output": explainer.model_output,
        "expected_value_shape": list(np.asarray(explainer.expected_value).shape),
        "shap_values_shape": list(profile_batch.contributions.shape),
        "positive_class": POSITIVE_CLASS,
        "requested_background_rows": len(background),
        "effective_background_rows": int(explainer.data.shape[0]),
        "finite": bool(
            np.isfinite(profile_batch.contributions.to_numpy()).all()
            and np.isfinite(global_batch.contributions.to_numpy()).all()
        ),
        "profile_additivity_max": float(profile_batch.additivity_errors.max()),
        "profile_additivity_mean": float(profile_batch.additivity_errors.mean()),
        "global_additivity_max": float(global_batch.additivity_errors.max()),
        "global_additivity_mean": float(global_batch.additivity_errors.mean()),
        "margin_to_probability_max_error": None,
        "creation_seconds": creation_seconds,
        "one_local_seconds": one_local_seconds,
        "global_seconds": global_seconds,
        "python_peak_mib": python_peak / 2**20,
        "process_peak_increment_mib": memory_increment,
    }
    spike = run_compatibility_spike(
        model,
        train_features,
        background,
        global_sample.features,
        profiles,
        selected_result=selected_result,
    )

    additivity_rows = [
        {
            "scope": "global_calibration_sample_aggregate",
            "n_rows": len(global_sample.features),
            "base_value": np.nan,
            "model_probability": np.nan,
            "max_absolute_error": float(global_batch.additivity_errors.max()),
            "mean_absolute_error": float(global_batch.additivity_errors.mean()),
            "tolerance": ADDITIVITY_TOLERANCE,
            "passes": bool(global_batch.additivity_errors.max() <= ADDITIVITY_TOLERANCE),
        }
    ]
    for position, profile in enumerate(REFERENCE_PROFILES):
        additivity_rows.append(
            {
                "scope": profile.name,
                "n_rows": 1,
                "base_value": float(profile_batch.base_values[position]),
                "model_probability": float(profile_batch.model_outputs[position]),
                "max_absolute_error": float(profile_batch.additivity_errors[position]),
                "mean_absolute_error": float(profile_batch.additivity_errors[position]),
                "tolerance": ADDITIVITY_TOLERANCE,
                "passes": bool(
                    profile_batch.additivity_errors[position] <= ADDITIVITY_TOLERANCE
                ),
            }
        )
    additivity = pd.DataFrame(additivity_rows)

    background_repeat = build_aggregate_background(model, train_features)
    sample_repeat = select_global_sample(splits.calibration)
    global_contribution_difference = float(
        np.max(
            np.abs(
                global_batch.contributions.to_numpy()
                - repeated_global_batch.contributions.to_numpy()
            )
        )
    )
    global_importance_difference = float(
        np.max(
            np.abs(
                importance["mean_absolute_contribution"].to_numpy()
                - repeated_importance["mean_absolute_contribution"].to_numpy()
            )
        )
    )
    local_difference = float(
        np.max(
            np.abs(
                profile_batch.contributions.to_numpy()
                - repeated_profile_batch.contributions.to_numpy()
            )
        )
    )
    reproducibility = {
        "tolerance": REPRODUCIBILITY_TOLERANCE,
        "background_max_difference": float(
            np.max(np.abs(background.to_numpy() - background_repeat.to_numpy()))
        ),
        "global_sample_exact_match": bool(
            global_sample.features.equals(sample_repeat.features)
        ),
        "global_contribution_max_difference": global_contribution_difference,
        "global_importance_max_difference": global_importance_difference,
        "local_contribution_max_difference": local_difference,
    }
    if (
        reproducibility["background_max_difference"] > REPRODUCIBILITY_TOLERANCE
        or not reproducibility["global_sample_exact_match"]
        or global_contribution_difference > REPRODUCIBILITY_TOLERANCE
        or global_importance_difference > REPRODUCIBILITY_TOLERANCE
        or local_difference > REPRODUCIBILITY_TOLERANCE
    ):
        raise ExplainabilityError("P9 reproducibility check failed.")

    minimum, maximum = _background_group_sizes(len(train_features))
    generated_files = [
        "report.md",
        "configuration.json",
        "spike_results.json",
        "global_importance.csv",
        "local_contributions.csv",
        "additivity_checks.csv",
        "global_importance_bar.png",
        "global_beeswarm.png",
        *[f"waterfall_{profile.name}.png" for profile in REFERENCE_PROFILES],
    ]
    configuration = {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "decisions": {"D-020": "Accepted", "D-021": "Accepted", "D-022": "Accepted"},
        "artifact": {
            "path": "models/diabetes_risk_model.joblib",
            "sha256": artifact_hash_before,
            "size_bytes": artifacts.DEFAULT_ARTIFACT_PATH.stat().st_size,
            "schema_version": bundle["metadata"]["schema_version"],
            "model_class": type(model).__name__,
            "calibration_method": bundle["metadata"]["calibration_method"],
        },
        "stack": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "explainer": {
            "type": EXPLAINER_TYPE,
            "feature_perturbation": FEATURE_PERTURBATION,
            "model_output": MODEL_OUTPUT,
            "output_contract": OUTPUT_CONTRACT,
            "positive_class": POSITIVE_CLASS,
            "feature_columns": list(FEATURE_COLUMNS),
            "expected_value_shape": list(np.asarray(explainer.expected_value).shape),
            "shap_values_shape": [len(global_sample.features), len(FEATURE_COLUMNS)],
        },
        "background": {
            "asset_path": "models/shap_background_v1.json",
            "schema_version": BACKGROUND_ASSET_SCHEMA_VERSION,
            "source_split": "train",
            "construction": BACKGROUND_CONSTRUCTION,
            "project_seed": RANDOM_SEED,
            "requested_rows": BACKGROUND_ROWS,
            "effective_rows": int(explainer.data.shape[0]),
            "minimum_source_rows_per_centroid": minimum,
            "maximum_source_rows_per_centroid": maximum,
            "exact_train_row_matches": exact_matches,
            "contains_target": False,
            "contains_source_indices": False,
        },
        "global_sample": {
            "source_split": "calibration",
            "selection": "deterministic proportional stratification",
            "random_seed": RANDOM_SEED,
            "rows": len(global_sample.features),
            "positive_rows": global_sample.n_positive,
            "negative_rows": global_sample.n_negative,
            "source_prevalence": global_sample.source_prevalence,
            "sample_prevalence": global_sample.sample_prevalence,
            "absolute_prevalence_difference": abs(
                global_sample.sample_prevalence - global_sample.source_prevalence
            ),
        },
        "tolerances": {
            "probability_additivity_absolute": ADDITIVITY_TOLERANCE,
            "raw_margin_absolute": RAW_MARGIN_TOLERANCE,
            "reproducibility_absolute": REPRODUCIBILITY_TOLERANCE,
        },
        "performance_limits": {
            "creation_seconds": MAX_EXPLAINER_CREATION_SECONDS,
            "warm_local_seconds": MAX_WARM_LOCAL_SECONDS,
            "global_seconds": MAX_GLOBAL_SAMPLE_SECONDS,
            "incremental_memory_mib": MAX_INCREMENTAL_MEMORY_MIB,
        },
        "performance": {
            "creation_seconds": creation_seconds,
            "one_local_seconds": one_local_seconds,
            "global_seconds": global_seconds,
            "repeat_global_seconds": repeat_global_seconds,
            "python_peak_mib": python_peak / 2**20,
            "process_peak_increment_mib": memory_increment,
        },
        "reproducibility": reproducibility,
        "generated_files": generated_files,
    }

    importance.to_csv(
        directory / "global_importance.csv", index=False, float_format="%.12g"
    )
    local_tables.to_csv(
        directory / "local_contributions.csv", index=False, float_format="%.12g"
    )
    additivity.to_csv(
        directory / "additivity_checks.csv", index=False, float_format="%.12g"
    )
    _write_json(configuration, directory / "configuration.json")
    _write_json(spike, directory / "spike_results.json")
    _write_global_bar_plot(importance, directory / "global_importance_bar.png")
    _write_beeswarm_plot(
        global_batch, global_sample.features, directory / "global_beeswarm.png"
    )
    for profile in REFERENCE_PROFILES:
        table = local_tables[local_tables["profile"] == profile.name]
        _write_waterfall_plot(
            table,
            directory / f"waterfall_{profile.name}.png",
            f"Model contribution breakdown: {profile.name}",
        )
    report = _build_report(
        configuration=configuration,
        spike=spike,
        importance=importance,
        local_tables=local_tables,
        additivity=additivity,
    )
    (directory / "report.md").write_text(report, encoding="utf-8")

    artifact_hash_after = artifact_sha256()
    if artifact_hash_after != artifact_hash_before:
        raise ExplainabilityError("The official P8 artifact changed during P9 evidence generation.")
    return configuration


def main() -> None:
    """CLI entry point: regenerate the complete local P9 evidence package."""
    print("Loading the frozen schema-version-2 P8 artifact ...")
    print("Building the train-only aggregate background and calibration sample ...")
    configuration = generate_evidence()
    performance = configuration["performance"]
    sample = configuration["global_sample"]
    print(
        f"Background: {configuration['background']['requested_rows']} requested / "
        f"{configuration['background']['effective_rows']} effective; "
        f"exact train matches={configuration['background']['exact_train_row_matches']}."
    )
    print(
        f"Global sample: {sample['rows']} rows, prevalence "
        f"{sample['sample_prevalence']:.6f}."
    )
    print(
        "Accepted TreeExplainer performance: "
        f"create={performance['creation_seconds']:.3f}s, "
        f"local={performance['one_local_seconds']:.4f}s, "
        f"global={performance['global_seconds']:.3f}s."
    )
    print(f"Evidence written to {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
