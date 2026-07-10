"""Tests for src/artifacts.py: P6 artifact contract and serving helpers.

Most tests run on small synthetic splits (reusing the P3/P4 fixtures), so
the suite stays fast and does not require the raw CSV; the real-data
integration test is skipped when the git-ignored raw file is absent. Every
artifact write goes to a pytest temporary directory -- the repository's
`models/` and `data/processed/` directories must stay untouched.
"""

import inspect
import math

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src import artifacts, data, modeling
from tests.test_data import requires_raw_data
from tests.test_modeling import make_splits


@pytest.fixture(scope="module")
def bundle():
    """One shared artifact bundle built on small synthetic splits."""
    return artifacts.build_artifact_bundle(make_splits())


def copy_bundle(bundle: dict) -> dict:
    """Shallow copy safe for per-test mutation of top-level entries."""
    return {"model": bundle["model"], "metadata": dict(bundle["metadata"])}


# ---------------------------------------------------------------------------
# Bundle contents: selected model and required metadata
# ---------------------------------------------------------------------------


def test_bundle_contains_fitted_selected_model(bundle):
    model = bundle["model"]

    assert type(model).__name__ == artifacts.SELECTED_MODEL_CLASS
    assert callable(model.predict_proba)
    # Fitted: it can score the fixed example case without refitting.
    probability = artifacts.predict_risk_probability(bundle, artifacts.example_input())
    assert 0.0 <= probability <= 1.0


def test_bundle_metadata_records_the_serving_contract(bundle):
    metadata = bundle["metadata"]

    assert metadata["schema_version"] == artifacts.ARTIFACT_SCHEMA_VERSION
    assert metadata["model_name"] == artifacts.SELECTED_MODEL_NAME
    assert metadata["model_class"] == artifacts.SELECTED_MODEL_CLASS
    assert metadata["selection_decision"] == "D-016"
    assert metadata["feature_columns"] == data.FEATURE_COLUMNS
    assert metadata["target"] == data.TARGET
    assert metadata["random_seed"] == data.RANDOM_SEED
    # Selection evidence follows the shared P5 metric protocol on train/test
    # only; the calibration split has no entry by construction.
    assert set(metadata["metrics"]) == {"train", "test"}
    for split in ("train", "test"):
        assert set(metadata["metrics"][split]) == set(modeling.METRIC_KEYS)
    for package in ("python", "numpy", "pandas", "scikit-learn", "joblib"):
        assert package in metadata["package_versions"]
    assert metadata["created_at"]


# ---------------------------------------------------------------------------
# Training scope: train rows only, calibration untouched
# ---------------------------------------------------------------------------


def test_build_fits_only_on_train_rows(monkeypatch):
    splits = make_splits()
    fitted_indices = {}
    original_builder = artifacts.build_hist_gradient_boosting_candidate

    def spying_builder(random_state=data.RANDOM_SEED):
        model = original_builder(random_state=random_state)
        original_fit = model.fit

        def recording_fit(X, y, **kwargs):
            fitted_indices["rows"] = list(X.index)
            return original_fit(X, y, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(
        artifacts, "build_hist_gradient_boosting_candidate", spying_builder
    )

    artifacts.build_artifact_bundle(splits)

    assert fitted_indices["rows"] == list(splits.train.index)


def test_build_does_not_consume_calibration_split():
    # Poison the calibration split: any fit/predict/metric touching it would
    # propagate NaN or raise, so a clean run proves it was never consumed.
    splits = make_splits()
    poisoned = data.DataSplits(
        train=splits.train,
        calibration=splits.calibration * np.nan,
        test=splits.test,
    )

    bundle = artifacts.build_artifact_bundle(poisoned)

    assert not np.isnan(bundle["metadata"]["metrics"]["test"]["roc_auc"])


# ---------------------------------------------------------------------------
# Save/load round trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip_preserves_model_and_metadata(bundle, tmp_path):
    path = artifacts.save_artifact(bundle, tmp_path / "artifact.joblib")

    assert path.is_file()
    loaded = artifacts.load_artifact(path)

    assert loaded["metadata"] == bundle["metadata"]
    example = artifacts.example_input()
    assert artifacts.predict_risk_probability(loaded, example) == pytest.approx(
        artifacts.predict_risk_probability(bundle, example)
    )


def test_loaded_metadata_preserves_exact_feature_order(bundle, tmp_path):
    loaded = artifacts.load_artifact(
        artifacts.save_artifact(bundle, tmp_path / "artifact.joblib")
    )

    assert loaded["metadata"]["feature_columns"] == data.FEATURE_COLUMNS


def test_loaded_model_returns_probability_in_unit_interval(bundle, tmp_path):
    loaded = artifacts.load_artifact(
        artifacts.save_artifact(bundle, tmp_path / "artifact.joblib")
    )

    probability = artifacts.predict_risk_probability(loaded, artifacts.example_input())

    assert isinstance(probability, float)
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_load_missing_artifact_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.artifacts"):
        artifacts.load_artifact(tmp_path / "missing.joblib")


@pytest.mark.parametrize("payload", [b"", b"not a joblib payload"])
def test_load_corrupt_artifact_raises_clear_error(tmp_path, payload):
    # Deserialization failures (EOFError, unpickling errors, ...) must be
    # wrapped so the app's error path always shows the regeneration command.
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="python -m src.artifacts"):
        artifacts.load_artifact(corrupt)


def test_save_requires_the_gitignored_joblib_extension(bundle, tmp_path):
    with pytest.raises(ValueError, match="joblib"):
        artifacts.save_artifact(bundle, tmp_path / "artifact.pkl")


def test_smoke_check_returns_valid_probability(bundle, tmp_path):
    path = artifacts.save_artifact(bundle, tmp_path / "artifact.joblib")

    probability = artifacts.load_predict_smoke_check(path)

    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


# ---------------------------------------------------------------------------
# Bundle validation rejects incompatible or incomplete bundles
# ---------------------------------------------------------------------------


def break_not_a_dict(bundle):
    return ["not", "a", "bundle"]


def break_missing_model(bundle):
    del bundle["model"]
    return bundle


def break_missing_metadata(bundle):
    del bundle["metadata"]
    return bundle


def break_metadata_not_a_dict(bundle):
    bundle["metadata"] = "metadata"
    return bundle


def break_missing_required_key(bundle):
    del bundle["metadata"]["feature_columns"]
    return bundle


def break_wrong_schema_version(bundle):
    bundle["metadata"]["schema_version"] = artifacts.ARTIFACT_SCHEMA_VERSION + 1
    return bundle


def break_wrong_model_name(bundle):
    bundle["metadata"]["model_name"] = "logistic_regression"
    return bundle


def break_wrong_selection_decision(bundle):
    bundle["metadata"]["selection_decision"] = "D-999"
    return bundle


def break_reordered_features(bundle):
    bundle["metadata"]["feature_columns"] = list(reversed(data.FEATURE_COLUMNS))
    return bundle


def break_wrong_target(bundle):
    bundle["metadata"]["target"] = "NotTheTarget"
    return bundle


def break_model_without_predict_proba(bundle):
    bundle["model"] = object()
    return bundle


def break_model_class_mismatch(bundle):
    bundle["metadata"]["model_class"] = "RandomForestClassifier"
    return bundle


@pytest.mark.parametrize(
    "breaker",
    [
        break_not_a_dict,
        break_missing_model,
        break_missing_metadata,
        break_metadata_not_a_dict,
        break_missing_required_key,
        break_wrong_schema_version,
        break_wrong_model_name,
        break_wrong_selection_decision,
        break_reordered_features,
        break_wrong_target,
        break_model_without_predict_proba,
        break_model_class_mismatch,
    ],
)
def test_validation_rejects_incompatible_bundles(bundle, breaker):
    broken = breaker(copy_bundle(bundle))

    with pytest.raises(ValueError):
        artifacts.validate_artifact_bundle(broken)


def test_save_rejects_invalid_bundles(bundle, tmp_path):
    broken = break_missing_required_key(copy_bundle(bundle))

    with pytest.raises(ValueError):
        artifacts.save_artifact(broken, tmp_path / "artifact.joblib")
    assert not (tmp_path / "artifact.joblib").exists()


# The model object itself is verified, not just its self-declared metadata
# (P6 review finding): impostor estimators, unfitted models, drifted fitted
# feature order, non-binary classes, and tuned hyperparameters are rejected.


def test_validation_rejects_an_impostor_model_despite_matching_metadata(bundle):
    model_data = modeling.to_train_test_data(make_splits())
    impostor = DummyClassifier(strategy="prior").fit(
        model_data.X_train, model_data.y_train
    )

    disguised = copy_bundle(bundle)
    disguised["model"] = impostor
    with pytest.raises(ValueError, match="D-016"):
        artifacts.validate_artifact_bundle(disguised)

    # Editing model_class to match the impostor must not help either.
    self_consistent = copy_bundle(bundle)
    self_consistent["model"] = impostor
    self_consistent["metadata"]["model_class"] = type(impostor).__name__
    with pytest.raises(ValueError):
        artifacts.validate_artifact_bundle(self_consistent)


def test_validation_rejects_an_unfitted_model(bundle):
    broken = copy_bundle(bundle)
    broken["model"] = modeling.build_hist_gradient_boosting_candidate()

    with pytest.raises(ValueError, match="not fitted"):
        artifacts.validate_artifact_bundle(broken)


def test_validation_rejects_a_model_fitted_with_different_feature_order(bundle):
    model_data = modeling.to_train_test_data(make_splits())
    reordered = model_data.X_train[list(reversed(data.FEATURE_COLUMNS))]
    model = modeling.build_hist_gradient_boosting_candidate().fit(
        reordered, model_data.y_train
    )
    broken = copy_bundle(bundle)
    broken["model"] = model

    with pytest.raises(ValueError, match="feature order"):
        artifacts.validate_artifact_bundle(broken)


def test_validation_rejects_non_binary_target_classes(bundle):
    model_data = modeling.to_train_test_data(make_splits())
    X = model_data.X_train.head(30)
    multiclass = modeling.build_hist_gradient_boosting_candidate().fit(
        X, np.tile([0, 1, 2], 10)
    )
    broken = copy_bundle(bundle)
    broken["model"] = multiclass

    with pytest.raises(ValueError, match="classes"):
        artifacts.validate_artifact_bundle(broken)


def test_validation_rejects_non_default_hyperparameters(bundle):
    model_data = modeling.to_train_test_data(make_splits())
    tuned = HistGradientBoostingClassifier(
        random_state=data.RANDOM_SEED, max_depth=3
    ).fit(model_data.X_train, model_data.y_train)
    broken = copy_bundle(bundle)
    broken["model"] = tuned

    with pytest.raises(ValueError, match="hyperparameters"):
        artifacts.validate_artifact_bundle(broken)


# ---------------------------------------------------------------------------
# App-facing input assembly and validation
# ---------------------------------------------------------------------------


def test_example_input_satisfies_the_contract():
    artifacts.validate_input_values(artifacts.example_input())


def test_input_to_dataframe_preserves_feature_order():
    scrambled = dict(reversed(list(artifacts.example_input().items())))

    row = artifacts.input_to_dataframe(scrambled)

    assert list(row.columns) == data.FEATURE_COLUMNS
    assert len(row) == 1
    assert (row.dtypes == "uint8").all()
    for feature in data.FEATURE_COLUMNS:
        assert row.iloc[0][feature] == scrambled[feature]


def test_input_validation_rejects_missing_feature():
    values = artifacts.example_input()
    del values["BMI"]

    with pytest.raises(ValueError, match="Missing features.*BMI"):
        artifacts.validate_input_values(values)


def test_input_validation_rejects_unexpected_feature():
    values = artifacts.example_input()
    values["ExtraFeature"] = 1

    with pytest.raises(ValueError, match="Unexpected features.*ExtraFeature"):
        artifacts.validate_input_values(values)


@pytest.mark.parametrize("bad_value", [25.5, "high", None, float("nan")])
def test_input_validation_rejects_non_integer_like_values(bad_value):
    values = artifacts.example_input()
    values["BMI"] = bad_value

    with pytest.raises(ValueError, match="'BMI'"):
        artifacts.validate_input_values(values)


@pytest.mark.parametrize(
    "feature, out_of_range",
    [("BMI", 11), ("BMI", 99), ("GenHlth", 6), ("HighBP", 2), ("MentHlth", -1)],
)
def test_input_validation_rejects_out_of_range_values(feature, out_of_range):
    values = artifacts.example_input()
    values[feature] = out_of_range

    with pytest.raises(ValueError, match=f"'{feature}'.*outside the valid range"):
        artifacts.validate_input_values(values)


def test_input_validation_accepts_integer_like_floats_and_booleans():
    values = artifacts.example_input()
    values["BMI"] = 25.0
    values["HighBP"] = True

    row = artifacts.input_to_dataframe(values)

    assert row.iloc[0]["BMI"] == 25
    assert row.iloc[0]["HighBP"] == 1


# ---------------------------------------------------------------------------
# Serving path works without Streamlit; scope guards
# ---------------------------------------------------------------------------


def test_serving_helpers_do_not_require_streamlit(bundle):
    # The whole app-facing path must be exercisable without the Streamlit
    # runtime (US-0503): the module never imports it.
    assert "streamlit" not in inspect.getsource(artifacts)

    probability = artifacts.predict_risk_probability(bundle, artifacts.example_input())

    assert 0.0 <= probability <= 1.0


def test_artifacts_module_does_not_reload_or_resplit_data():
    # P6 consumes the P3 contract through prepare_data(); it must never
    # reload or re-split raw data ad hoc.
    source = inspect.getsource(artifacts)
    for forbidden in ("train_test_split", "read_csv", "load_raw_data"):
        assert forbidden not in source


def test_default_artifact_path_is_gitignored_under_models():
    assert artifacts.DEFAULT_ARTIFACT_PATH.parent == data.PROJECT_ROOT / "models"
    assert artifacts.DEFAULT_ARTIFACT_PATH.suffix == ".joblib"
    gitignore = (data.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "models/*.joblib" in gitignore


def test_save_rejects_repository_paths_outside_models(bundle):
    # Only models/*.joblib is git-ignored, so a save into any other
    # repository location must be refused instead of silently unignored
    # (P6 review finding). Paths outside the repository stay allowed --
    # tests rely on pytest tmp dirs.
    for bad_path in (
        data.PROJECT_ROOT / "artifact.joblib",
        data.PROJECT_ROOT / "app" / "artifact.joblib",
        data.PROJECT_ROOT / "models" / "nested" / "artifact.joblib",
    ):
        with pytest.raises(ValueError, match="models/"):
            artifacts.save_artifact(bundle, bad_path)
        assert not bad_path.exists()


def test_artifact_tests_write_only_under_tmp_path(bundle, tmp_path):
    models_dir = data.PROJECT_ROOT / "models"
    processed_dir = data.PROJECT_ROOT / "data" / "processed"

    def listing(directory):
        return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []

    before = (listing(models_dir), listing(processed_dir))

    artifacts.save_artifact(bundle, tmp_path / "artifact.joblib")
    artifacts.load_predict_smoke_check(tmp_path / "artifact.joblib")

    assert (listing(models_dir), listing(processed_dir)) == before


# ---------------------------------------------------------------------------
# Real-data integration (skipped when the raw CSV is unavailable)
# ---------------------------------------------------------------------------


@requires_raw_data
def test_create_default_artifact_end_to_end_on_real_dataset(tmp_path):
    path, probability = artifacts.create_default_artifact(
        path=tmp_path / "diabetes_risk_model.joblib"
    )

    assert path.is_file()
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0

    metadata = artifacts.load_artifact(path)["metadata"]
    assert metadata["dataset_summary"]["n_rows"] == data.EXPECTED_RAW_ROW_COUNT
    # The artifact must reproduce the D-016 selection evidence (fixed seed,
    # same splits, same protocol), not re-open the selection.
    assert metadata["metrics"]["test"]["pr_auc"] == pytest.approx(0.423, abs=0.01)
    assert metadata["metrics"]["test"]["roc_auc"] == pytest.approx(0.827, abs=0.01)
