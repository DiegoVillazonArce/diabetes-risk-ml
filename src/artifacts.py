"""Model artifact contract and app-facing serving helpers (P6, Epic E5).

P6 (Streamlit MVP) is the first phase that persists a trained model: per
D-017 the D-016 `HistGradientBoostingClassifier` is trained once at the
start of the phase through the existing P3/P5 contracts and serialized with
`joblib` (D-010) as a single bundle -- the fitted model plus the metadata
the app needs for safe, reproducible inference: exact feature order, target
name, model identity, random seed, the P5-protocol metrics that back the
D-016 selection, and package versions.

Training goes through `src.data.prepare_data()` and the P4/P5 helpers in
`src.modeling` only: the model fits on the train split, metrics are
recomputed on train and test with the shared P5 metric protocol (the same
fixed seed reproduces the D-016 evidence; this verifies the already selected
model, it is not a new comparison round), and the calibration split is never
read, staying reserved for P8 probability calibration.

This module also owns the app-facing serving path (US-0501, US-0503): input
validation against the P3 `VALUE_RANGES` contract, assembly of a one-row
frame in exact `FEATURE_COLUMNS` order, and positive-class probability
prediction. It never imports Streamlit, so the whole serving path is
testable without the Streamlit runtime. No custom decision threshold,
calibration, or probability post-processing happens here; those belong to
P8.

Artifacts default to `models/diabetes_risk_model.joblib`. Per D-013
(P7), that official artifact is version-controlled as a controlled Git
exception so deployment ships it with the repository; every other artifact
under `models/` (temporary or alternative files) stays git-ignored.

Generate the local artifact from the project root with:

    python -m src.artifacts
"""

from __future__ import annotations

import math
import numbers
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET,
    VALUE_RANGES,
    DataSplits,
    prepare_data,
)
from src.modeling import (
    build_hist_gradient_boosting_candidate,
    evaluate_model,
    predict_positive_proba,
    to_train_test_data,
)

# Identity of the D-016 selected model, matching its P5 comparison key.
SELECTED_MODEL_NAME = "hist_gradient_boosting"
SELECTED_MODEL_CLASS = "HistGradientBoostingClassifier"
SELECTION_DECISION = "D-016"

# Bump when the bundle layout changes, so stale local artifacts fail loudly
# at load time instead of serving through an outdated contract.
ARTIFACT_SCHEMA_VERSION = 1

# The persisted estimator is a Python object whose compatibility depends on
# the training/runtime environment. D-013 therefore makes these versions part
# of the serving contract, not merely informational metadata. Python patch
# versions may differ (the clean-environment check deliberately used 3.12.1
# for an artifact produced on 3.12.7), while the model-facing libraries stay
# on the exact requirements.txt pins.
ARTIFACT_PYTHON_MAJOR_MINOR = (3, 12)
ARTIFACT_PACKAGE_VERSION_PINS = {
    "numpy": "2.2.6",
    "pandas": "2.3.1",
    "scikit-learn": "1.7.1",
    "joblib": "1.5.1",
}

# Default artifact location. This exact file is the D-013 controlled Git
# exception (version-controlled for deployment); all other `models/*.joblib`
# files remain git-ignored.
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "diabetes_risk_model.joblib"

REQUIRED_METADATA_KEYS = (
    "schema_version",
    "model_name",
    "model_class",
    "selection_decision",
    "feature_columns",
    "target",
    "random_seed",
    "metrics",
    "package_versions",
    "created_at",
)


def _package_versions() -> dict[str, str]:
    """Minimal reproducibility metadata: the interpreter plus the packages
    whose versions can change unpickling or prediction behavior."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _python_major_minor(version: object, *, source: str) -> tuple[int, int]:
    """Parse a recorded/runtime Python version for the 3.12 contract."""
    if not isinstance(version, str):
        raise ValueError(
            f"{source} Python version must be a dotted string; got {version!r}."
        )
    parts = version.split(".")
    try:
        if len(parts) < 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source} Python version must be a dotted string; got {version!r}."
        ) from error


def _validate_package_versions(package_versions: object) -> None:
    """Enforce the D-013 artifact provenance and runtime compatibility."""
    if not isinstance(package_versions, dict):
        raise ValueError(
            "Artifact package_versions metadata must be a dict; "
            f"got {type(package_versions).__name__}."
        )

    expected_keys = {"python", *ARTIFACT_PACKAGE_VERSION_PINS}
    actual_keys = set(package_versions)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        unexpected = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            "Artifact package_versions keys do not match the D-013 "
            f"provenance contract (missing={missing}, unexpected={unexpected}); "
            "regenerate the artifact with 'python -m src.artifacts'."
        )

    recorded_python = _python_major_minor(
        package_versions["python"], source="Artifact"
    )
    if recorded_python != ARTIFACT_PYTHON_MAJOR_MINOR:
        required = ".".join(map(str, ARTIFACT_PYTHON_MAJOR_MINOR))
        raise ValueError(
            f"Artifact Python version {package_versions['python']!r} is not "
            f"compatible with the required Python {required}.x environment; "
            "regenerate the artifact with 'python -m src.artifacts'."
        )

    for package, expected in ARTIFACT_PACKAGE_VERSION_PINS.items():
        recorded = package_versions[package]
        if recorded != expected:
            raise ValueError(
                f"Artifact {package} version {recorded!r} does not match the "
                f"required requirements.txt pin {expected!r}; regenerate the "
                "artifact with 'python -m src.artifacts'."
            )

    runtime_versions = _package_versions()
    runtime_python = _python_major_minor(
        runtime_versions["python"], source="Runtime"
    )
    if runtime_python != ARTIFACT_PYTHON_MAJOR_MINOR:
        required = ".".join(map(str, ARTIFACT_PYTHON_MAJOR_MINOR))
        raise ValueError(
            f"Runtime Python version {runtime_versions['python']!r} is not "
            f"compatible with the required Python {required}.x environment; "
            "recreate the environment from requirements.txt."
        )
    for package, expected in ARTIFACT_PACKAGE_VERSION_PINS.items():
        runtime = runtime_versions[package]
        if runtime != expected:
            raise ValueError(
                f"Runtime {package} version {runtime!r} does not match the "
                f"required requirements.txt pin {expected!r}; recreate the "
                "environment from requirements.txt."
            )


def build_artifact_bundle(
    splits: DataSplits,
    random_state: int = RANDOM_SEED,
    dataset_summary: dict | None = None,
) -> dict:
    """Train the D-016 model and assemble the serializable artifact bundle.

    Fits `HistGradientBoostingClassifier` on the train split only, through
    the P4/P5 `to_train_test_data` conversion so the calibration split is
    never read, and records train/test metrics with the shared P5 metric
    protocol as the artifact's selection evidence.
    """
    model_data = to_train_test_data(splits)
    model = build_hist_gradient_boosting_candidate(random_state=random_state)
    model.fit(model_data.X_train, model_data.y_train)

    metadata: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_name": SELECTED_MODEL_NAME,
        "model_class": type(model).__name__,
        "selection_decision": SELECTION_DECISION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": TARGET,
        "random_seed": random_state,
        "metrics": {
            "train": evaluate_model(model, model_data.X_train, model_data.y_train),
            "test": evaluate_model(model, model_data.X_test, model_data.y_test),
        },
        "package_versions": _package_versions(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if dataset_summary is not None:
        metadata["dataset_summary"] = dict(dataset_summary)

    bundle = {"model": model, "metadata": metadata}
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle: object) -> None:
    """Validate bundle structure, metadata completeness, and compatibility.

    Rejects anything the app could not serve safely. Structure checks: dict
    layout with 'model' and 'metadata', required metadata keys, schema
    version, model name, feature-column contract (exact names and order),
    target name, and D-013 package provenance/runtime compatibility. The
    model object is then checked directly rather than
    trusting self-declared metadata: it must actually be a fitted D-016
    `HistGradientBoostingClassifier` whose fitted feature order matches the
    current contract, whose classes are exactly [0, 1], and whose
    hyperparameters match the D-016 configuration (library defaults with
    the recorded seed); the declared model class and selection decision
    must match the accepted D-016 values. Raises ValueError on the first
    violation.
    """
    if not isinstance(bundle, dict):
        raise ValueError(
            "Artifact bundle must be a dict with 'model' and 'metadata' "
            f"entries; got {type(bundle).__name__}."
        )
    missing_entries = [key for key in ("model", "metadata") if key not in bundle]
    if missing_entries:
        raise ValueError(f"Artifact bundle is missing required entries: {missing_entries}.")

    metadata = bundle["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Artifact metadata must be a dict; got {type(metadata).__name__}."
        )
    missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_keys:
        raise ValueError(f"Artifact metadata is missing required keys: {missing_keys}.")

    _validate_package_versions(metadata["package_versions"])

    if metadata["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Artifact schema version {metadata['schema_version']!r} is not "
            f"compatible with the expected version {ARTIFACT_SCHEMA_VERSION}; "
            "regenerate the artifact with 'python -m src.artifacts'."
        )
    if metadata["model_name"] != SELECTED_MODEL_NAME:
        raise ValueError(
            f"Artifact model '{metadata['model_name']}' is not the D-016 "
            f"selected model '{SELECTED_MODEL_NAME}'."
        )
    if metadata["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError(
            "Artifact feature columns do not match the current "
            "src.data.FEATURE_COLUMNS contract (exact names and order "
            "required); regenerate the artifact with 'python -m src.artifacts'."
        )
    if metadata["target"] != TARGET:
        raise ValueError(
            f"Artifact target '{metadata['target']}' does not match the "
            f"expected target '{TARGET}'."
        )
    if metadata["selection_decision"] != SELECTION_DECISION:
        raise ValueError(
            f"Artifact selection decision {metadata['selection_decision']!r} "
            f"does not match the accepted decision '{SELECTION_DECISION}'."
        )
    if metadata["model_class"] != SELECTED_MODEL_CLASS:
        raise ValueError(
            f"Artifact metadata declares model class "
            f"{metadata['model_class']!r}; the D-016 selection requires "
            f"'{SELECTED_MODEL_CLASS}'."
        )

    # The model object itself is checked directly; self-declared metadata is
    # not trusted to describe it.
    model = bundle["model"]
    if not isinstance(model, HistGradientBoostingClassifier):
        raise ValueError(
            f"Artifact model is a {type(model).__name__}, not the D-016 "
            f"{SELECTED_MODEL_CLASS}."
        )
    fitted_feature_names = getattr(model, "feature_names_in_", None)
    if fitted_feature_names is None:
        raise ValueError(
            "Artifact model is not fitted (it has no feature_names_in_); "
            "regenerate the artifact with 'python -m src.artifacts'."
        )
    if list(fitted_feature_names) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Artifact model was fitted with a feature order different from "
            "the current src.data.FEATURE_COLUMNS contract; regenerate the "
            "artifact with 'python -m src.artifacts'."
        )
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError(
            f"Artifact model classes {getattr(model, 'classes_', None)!r} do "
            "not match the expected binary target classes [0, 1]."
        )
    expected_params = build_hist_gradient_boosting_candidate(
        random_state=metadata["random_seed"]
    ).get_params()
    if model.get_params() != expected_params:
        raise ValueError(
            "Artifact model hyperparameters do not match the D-016 "
            "configuration (library defaults with the recorded seed); "
            "regenerate the artifact with 'python -m src.artifacts'."
        )


def save_artifact(bundle: dict, path: Path | str | None = None) -> Path:
    """Serialize a validated bundle with joblib (D-010) and return the path.

    Every artifact must use the '.joblib' extension, and a path inside the
    repository must point directly into `models/`: that is the only
    directory where artifacts are managed by policy -- the official
    artifact is the D-013 version-controlled exception and everything else
    there is git-ignored, so a save anywhere else in the repository would
    create an unmanaged file. Locations outside the repository (for example
    pytest tmp dirs) are the caller's responsibility. The write is atomic:
    the bundle is dumped to a temporary sibling first and moved into place,
    so an interrupted save cannot leave a truncated artifact behind.
    """
    validate_artifact_bundle(bundle)
    path = DEFAULT_ARTIFACT_PATH if path is None else Path(path)
    if path.suffix != ".joblib":
        raise ValueError(
            "Artifact path must use the '.joblib' extension (D-010); "
            f"got '{path.name}'."
        )
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT) and resolved.parent != PROJECT_ROOT / "models":
        raise ValueError(
            "Artifact paths inside the repository must point directly into "
            "'models/', the only directory where artifacts are managed "
            "(git-ignored by default, with the official artifact as the "
            f"D-013 version-controlled exception); got '{path}'."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary sibling keeps the .joblib suffix so a crash leftover
    # under models/ would still be git-ignored.
    temp_path = path.with_name(path.stem + ".tmp.joblib")
    joblib.dump(bundle, temp_path)
    temp_path.replace(path)
    return path


def load_artifact(path: Path | str | None = None) -> dict:
    """Load and validate a local artifact bundle.

    Fails clearly in every unusable state: FileNotFoundError when the file
    is absent, and ValueError with regeneration or environment-rebuild
    guidance when the file cannot be deserialized or fails bundle validation.
    """
    path = DEFAULT_ARTIFACT_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Generate it from the "
            "project root with 'python -m src.artifacts' (requires the raw "
            "dataset; see data/README.md)."
        )
    try:
        bundle = joblib.load(path)
    except Exception as error:
        raise ValueError(
            f"Could not deserialize the model artifact at '{path}' "
            f"({type(error).__name__}). The file may be truncated, corrupt, "
            "or from an incompatible environment; regenerate it with "
            "'python -m src.artifacts'."
        ) from error
    validate_artifact_bundle(bundle)
    return bundle


def validate_input_values(values: dict) -> None:
    """Validate one user-entered case against the P3 feature contract.

    Requires exactly the 21 `FEATURE_COLUMNS` keys with numeric,
    integer-like, finite values inside the documented `VALUE_RANGES`.
    Booleans are accepted for binary features because UI toggles produce
    them. Raises ValueError on the first violation.
    """
    missing = [feature for feature in FEATURE_COLUMNS if feature not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}.")
    unexpected = sorted(set(values) - set(FEATURE_COLUMNS))
    if unexpected:
        raise ValueError(f"Unexpected features: {unexpected}.")

    for feature in FEATURE_COLUMNS:
        value = values[feature]
        if not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(
                f"Feature '{feature}' must be a finite number; got {value!r}."
            )
        if not float(value).is_integer():
            raise ValueError(
                f"Feature '{feature}' must be a whole number; got {value!r}."
            )
        lower, upper = VALUE_RANGES[feature]
        if not lower <= value <= upper:
            raise ValueError(
                f"Feature '{feature}' value {value!r} is outside the valid "
                f"range [{lower}, {upper}]."
            )


def input_to_dataframe(values: dict) -> pd.DataFrame:
    """Assemble one validated case as a single-row frame for the model.

    Columns follow the exact training feature order (`FEATURE_COLUMNS`) and
    use the same `uint8` dtype the model was trained on.
    """
    validate_input_values(values)
    row = [[int(values[feature]) for feature in FEATURE_COLUMNS]]
    return pd.DataFrame(row, columns=FEATURE_COLUMNS).astype("uint8")


def predict_risk_probability(bundle: dict, values: dict) -> float:
    """Score one case: the positive-class probability from `predict_proba`.

    The raw probability is returned without a custom decision threshold or
    calibration; threshold and calibration work belong to P8.
    """
    validate_artifact_bundle(bundle)
    row = input_to_dataframe(values)
    probability = float(predict_positive_proba(bundle["model"], row)[0])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Model returned an invalid probability: {probability!r}.")
    return probability


def example_input() -> dict[str, int]:
    """A fixed contract-valid single case for the load/predict smoke check."""
    values = {feature: 0 for feature in BINARY_FEATURES}
    values.update(
        {
            "GenHlth": 3,
            "Age": 9,
            "Education": 4,
            "Income": 6,
            "BMI": 28,
            "MentHlth": 0,
            "PhysHlth": 2,
        }
    )
    return {feature: values[feature] for feature in FEATURE_COLUMNS}


def load_predict_smoke_check(path: Path | str | None = None) -> float:
    """Local load/predict verification (US-0504, D-017).

    Reloads the saved artifact and scores the fixed example case;
    `predict_risk_probability` raises unless the result is a finite
    probability in [0, 1]. Returns that probability.
    """
    bundle = load_artifact(path)
    return predict_risk_probability(bundle, example_input())


def create_default_artifact(
    path: Path | str | None = None,
    data_path: Path | str = RAW_DATA_PATH,
    random_state: int = RANDOM_SEED,
) -> tuple[Path, float]:
    """Full local artifact generation: prepare, build, save, smoke-check.

    Data comes exclusively through the P3 `prepare_data()` contract. Returns
    the saved artifact path and the smoke-check probability.
    """
    splits, summary = prepare_data(data_path, random_state=random_state)
    bundle = build_artifact_bundle(
        splits, random_state=random_state, dataset_summary=summary
    )
    saved_path = save_artifact(bundle, path)
    return saved_path, load_predict_smoke_check(saved_path)


def main() -> None:
    """Command-line entry point: `python -m src.artifacts`."""
    print(f"Training the {SELECTED_MODEL_CLASS} ({SELECTION_DECISION}) on the P3 train split ...")
    path, probability = create_default_artifact()
    test_metrics = load_artifact(path)["metadata"]["metrics"]["test"]
    print(f"Artifact saved to: {path}")
    print(
        "Test metrics (P5 protocol): "
        f"PR-AUC {test_metrics['pr_auc']:.3f}, ROC-AUC {test_metrics['roc_auc']:.3f}, "
        f"recall {test_metrics['recall']:.3f}, precision {test_metrics['precision']:.3f}, "
        f"F1 {test_metrics['f1']:.3f}."
    )
    print(f"Load/predict smoke check passed: example-case probability {probability:.4f}.")


if __name__ == "__main__":
    main()
