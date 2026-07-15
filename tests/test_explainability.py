"""P9 SHAP contract, privacy, reproducibility, and evidence tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap

from src import artifacts, explainability
from src.data import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET,
    DataSplits,
    prepare_data,
)
from src.feature_labels import FEATURE_LABELS, format_feature_value
from tests.reference_profiles import (
    PROBABILITY_TOLERANCE,
    REFERENCE_PROFILES,
    format_display,
)

EXPECTED_P8_ARTIFACT_SHA256 = (
    "957c14ff5a490bbc60822121a889f92ee2a6a20f797eef741a710d887ecc9216"
)


@pytest.fixture(scope="module")
def official_bundle():
    return artifacts.load_artifact()


@pytest.fixture(scope="module")
def real_inputs(official_bundle):
    if not RAW_DATA_PATH.is_file():
        pytest.skip(f"Raw P9 reproduction data missing at {RAW_DATA_PATH}")
    splits, _ = prepare_data()
    background, sample = explainability.build_analysis_inputs(
        splits, official_bundle["model"]
    )
    return splits, background, sample


@pytest.fixture(scope="module")
def explainer(official_bundle, real_inputs):
    _, background, _ = real_inputs
    return explainability.create_explainer(official_bundle, background)


@pytest.fixture(scope="module")
def profile_frame():
    return pd.DataFrame(
        [profile.features for profile in REFERENCE_PROFILES],
        columns=FEATURE_COLUMNS,
    )


@pytest.fixture(scope="module")
def profile_batch(official_bundle, explainer, profile_frame):
    return explainability.explain_dataframe(
        official_bundle, explainer, profile_frame
    )


@pytest.fixture(scope="module")
def global_batch(official_bundle, explainer, real_inputs):
    _, _, sample = real_inputs
    return explainability.explain_dataframe(
        official_bundle, explainer, sample.features
    )


def test_shap_imports_at_the_pinned_compatible_version():
    assert shap.__version__ == explainability.SHAP_VERSION == "0.52.0"
    requirements = (artifacts.PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "shap==0.52.0" in requirements.splitlines()


def test_official_artifact_identity_and_p8_contract_are_unchanged(official_bundle):
    assert explainability.artifact_sha256() == EXPECTED_P8_ARTIFACT_SHA256
    assert official_bundle["metadata"]["schema_version"] == 2
    assert official_bundle["metadata"]["calibration_method"] == "none"
    assert official_bundle["calibrator"] is None


def test_positive_class_is_identified_unambiguously(official_bundle):
    assert explainability.positive_class_index(official_bundle["model"]) == 1


def test_background_is_exactly_256_by_21_finite_and_ordered(real_inputs):
    _, background, _ = real_inputs
    assert background.shape == (256, 21)
    assert list(background) == FEATURE_COLUMNS
    assert np.isfinite(background.to_numpy()).all()


def test_explicit_masker_retains_all_requested_background_rows(explainer):
    assert explainer.feature_perturbation == "interventional"
    assert explainer.model_output == "probability"
    assert explainer.data.shape == (256, 21)


def test_implicit_masker_reduction_is_understood(official_bundle, real_inputs):
    splits, _, _ = real_inputs
    requested = splits.train[FEATURE_COLUMNS].sample(
        n=256, random_state=RANDOM_SEED
    )
    implicit = shap.TreeExplainer(
        official_bundle["model"],
        data=requested,
        model_output="probability",
        feature_perturbation="interventional",
    )
    assert len(requested) == 256
    assert implicit.data.shape[0] == 100


def test_normalization_selects_positive_class_and_exact_shape():
    values = np.zeros((2, 21, 2), dtype=float)
    values[:, :, 1] = 3.0
    normalized = explainability.normalize_shap_values(values, n_rows=2)
    assert normalized.shape == (2, 21)
    assert np.all(normalized == 3.0)


def test_explanations_have_exact_shape_order_and_finite_values(profile_batch):
    assert profile_batch.contributions.shape == (4, 21)
    assert list(profile_batch.contributions) == FEATURE_COLUMNS
    assert np.isfinite(profile_batch.contributions.to_numpy()).all()
    assert np.isfinite(profile_batch.base_values).all()


def test_direct_probability_additivity_passes_fixed_tolerance(profile_batch):
    assert profile_batch.additivity_errors.max() <= 1e-4
    reconstructed = (
        profile_batch.base_values
        + profile_batch.contributions.to_numpy().sum(axis=1)
    )
    assert reconstructed == pytest.approx(
        profile_batch.model_outputs, abs=explainability.ADDITIVITY_TOLERANCE
    )


def test_explained_output_is_the_served_probability(
    official_bundle, profile_batch
):
    served = np.array(
        [
            artifacts.predict_risk_probability(official_bundle, profile.features)
            for profile in REFERENCE_PROFILES
        ]
    )
    assert profile_batch.model_outputs == pytest.approx(served, abs=1e-15)


def test_background_and_global_sample_reproduce_with_seed_42(
    official_bundle, real_inputs
):
    splits, background, sample = real_inputs
    repeated_background, repeated_sample = explainability.build_analysis_inputs(
        splits, official_bundle["model"]
    )
    pd.testing.assert_frame_equal(background, repeated_background, check_exact=True)
    pd.testing.assert_frame_equal(
        sample.features, repeated_sample.features, check_exact=True
    )
    assert repeated_sample.n_positive == sample.n_positive
    assert repeated_sample.n_negative == sample.n_negative
    assert repeated_sample.source_prevalence == sample.source_prevalence
    assert repeated_sample.sample_prevalence == sample.sample_prevalence


def test_global_sample_preserves_calibration_prevalence_by_rounding(real_inputs):
    _, _, sample = real_inputs
    assert len(sample.features) == 5_000
    assert sample.n_positive == 697
    assert sample.n_negative == 4_303
    assert abs(sample.sample_prevalence - sample.source_prevalence) <= 0.5 / 5_000


def test_global_matrix_is_finite_additive_and_5000_by_21(global_batch):
    assert global_batch.contributions.shape == (5_000, 21)
    assert np.isfinite(global_batch.contributions.to_numpy()).all()
    assert global_batch.additivity_errors.max() <= explainability.ADDITIVITY_TOLERANCE


def test_global_importance_is_reproducible(global_batch):
    first = explainability.mean_absolute_importance(global_batch)
    second = explainability.mean_absolute_importance(global_batch)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_test_split_cannot_affect_background_or_global_sample(
    official_bundle, real_inputs
):
    splits, background, sample = real_inputs

    class PoisonTest:
        def __getattribute__(self, name):
            raise AssertionError("test must not be read by P9 configuration")

    poisoned = DataSplits(
        train=splits.train,
        calibration=splits.calibration,
        test=PoisonTest(),
    )
    poisoned_background, poisoned_sample = explainability.build_analysis_inputs(
        poisoned, official_bundle["model"]
    )
    pd.testing.assert_frame_equal(background, poisoned_background, check_exact=True)
    pd.testing.assert_frame_equal(
        sample.features, poisoned_sample.features, check_exact=True
    )


def test_deployment_background_has_no_exact_train_row(real_inputs):
    splits, background, _ = real_inputs
    matches = explainability.exact_row_match_count(
        background, splits.train[FEATURE_COLUMNS]
    )
    assert matches == 0


def test_deployed_background_asset_has_strict_privacy_metadata(real_inputs):
    _, expected_background, _ = real_inputs
    path = explainability.BACKGROUND_ASSET_PATH
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert "target" not in raw.lower()
    assert "source_index" not in raw.lower()
    assert "split_index" not in raw.lower()
    assert payload["source_split"] == "train"
    assert payload["project_seed"] == 42
    assert "random_seed" not in payload
    assert payload["minimum_source_rows_per_centroid"] >= 2
    loaded = explainability.load_background_asset(
        expected_artifact_sha256=EXPECTED_P8_ARTIFACT_SHA256
    )
    pd.testing.assert_frame_equal(loaded, expected_background, check_exact=True)


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_all_four_reference_probabilities_and_displays_are_unchanged(
    official_bundle, profile
):
    probability = artifacts.predict_risk_probability(
        official_bundle, profile.features
    )
    assert probability == pytest.approx(
        profile.expected_probability, abs=PROBABILITY_TOLERANCE
    )
    assert format_display(probability) == profile.expected_display


def test_local_tables_cover_every_profile_and_feature(
    official_bundle, explainer
):
    tables = []
    for profile in REFERENCE_PROFILES:
        table = explainability.explain_local_values(
            official_bundle, explainer, profile.features, name=profile.name
        )
        tables.append(table)
        assert len(table) == 21
        assert set(table["feature"]) == set(FEATURE_COLUMNS)
        assert table["model_probability"].iloc[0] == pytest.approx(
            profile.expected_probability, abs=PROBABILITY_TOLERANCE
        )
    assert {table["profile"].iloc[0] for table in tables} == {
        profile.name for profile in REFERENCE_PROFILES
    }


def test_user_labels_translate_binary_age_education_and_income_codes():
    assert set(FEATURE_LABELS) == set(FEATURE_COLUMNS)
    assert format_feature_value("HighBP", 1) == "Yes"
    assert format_feature_value("Sex", 0) == "Female"
    assert format_feature_value("Age", 10) == "65-69"
    assert "High school" in format_feature_value("Education", 4)
    assert "$35,000" in format_feature_value("Income", 6)


def test_display_selection_includes_both_directions_when_available(
    official_bundle, explainer
):
    table = explainability.explain_local_values(
        official_bundle, explainer, REFERENCE_PROFILES[1].features
    )
    display = explainability.select_display_contributions(table)
    assert len(display) <= 6
    if (table["contribution"] > 0).any():
        assert (display["contribution"] > 0).any()
    if (table["contribution"] < 0).any():
        assert (display["contribution"] < 0).any()


def test_runtime_performance_respects_predeclared_limits(
    official_bundle, real_inputs
):
    _, background, _ = real_inputs
    started = time.perf_counter()
    runtime_explainer = explainability.create_explainer(
        official_bundle, background
    )
    creation = time.perf_counter() - started
    started = time.perf_counter()
    explainability.explain_local_values(
        official_bundle, runtime_explainer, REFERENCE_PROFILES[0].features
    )
    local = time.perf_counter() - started
    assert creation <= explainability.MAX_EXPLAINER_CREATION_SECONDS
    assert local <= explainability.MAX_WARM_LOCAL_SECONDS


def test_background_loader_rejects_wrong_artifact_hash(tmp_path):
    payload = json.loads(
        explainability.BACKGROUND_ASSET_PATH.read_text(encoding="utf-8")
    )
    path = tmp_path / "background.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different model artifact"):
        explainability.load_background_asset(
            path, expected_artifact_sha256="0" * 64
        )


def test_published_csvs_expose_only_aggregates_or_synthetic_profiles():
    directory = explainability.EVIDENCE_DIR
    importance = pd.read_csv(directory / "global_importance.csv")
    local = pd.read_csv(directory / "local_contributions.csv")
    additivity = pd.read_csv(directory / "additivity_checks.csv")
    assert len(importance) == 21
    assert set(importance) == {
        "rank",
        "feature",
        "feature_label",
        "mean_absolute_contribution",
    }
    assert set(local["profile"]) == {
        profile.name for profile in REFERENCE_PROFILES
    }
    assert len(local) == 4 * 21
    assert len(additivity) == 5
    assert (additivity["scope"] == "global_calibration_sample_aggregate").sum() == 1
    for frame in (importance, local, additivity):
        lowered = {column.lower() for column in frame.columns}
        assert "target" not in lowered
        assert "source_index" not in lowered
        assert "split_index" not in lowered


def test_generated_plots_cover_global_and_all_reference_profiles():
    directory = explainability.EVIDENCE_DIR
    expected = {
        "global_importance_bar.png",
        "global_beeswarm.png",
        *{
            f"waterfall_{profile.name}.png"
            for profile in REFERENCE_PROFILES
        },
    }
    for filename in expected:
        path = directory / filename
        assert path.is_file()
        assert path.stat().st_size > 1_000


def test_published_importance_and_local_contributions_match_recomputation(
    global_batch, profile_batch, profile_frame
):
    recorded_importance = pd.read_csv(
        explainability.EVIDENCE_DIR / "global_importance.csv"
    )
    recomputed_importance = explainability.mean_absolute_importance(global_batch)
    assert recorded_importance["feature"].tolist() == recomputed_importance[
        "feature"
    ].tolist()
    np.testing.assert_allclose(
        recorded_importance["mean_absolute_contribution"].to_numpy(),
        recomputed_importance["mean_absolute_contribution"].to_numpy(),
        atol=1e-12,
        rtol=0,
    )
    recorded_local = pd.read_csv(
        explainability.EVIDENCE_DIR / "local_contributions.csv"
    )
    first = REFERENCE_PROFILES[0]
    one_batch = explainability.ExplanationBatch(
        contributions=profile_batch.contributions.iloc[[0]].reset_index(drop=True),
        base_values=profile_batch.base_values[[0]],
        model_outputs=profile_batch.model_outputs[[0]],
        additivity_errors=profile_batch.additivity_errors[[0]],
    )
    recomputed_local = explainability.local_contribution_table(
        first.name, profile_frame.iloc[[0]].reset_index(drop=True), one_batch
    )
    recorded_first = recorded_local[recorded_local["profile"] == first.name]
    assert recorded_first["feature"].tolist() == recomputed_local["feature"].tolist()
    np.testing.assert_allclose(
        recorded_first["contribution"].to_numpy(),
        recomputed_local["contribution"].to_numpy(),
        atol=1e-12,
        rtol=0,
    )
