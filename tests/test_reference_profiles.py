"""Tests for the P7 deployment-smoke reference profiles (US-0703).

Static checks (no artifact needed) pin the recorded expectations to the
serving contract: exactly the 21 contract features, valid values, correct
exact-age-to-BRFSS conversion, display strings consistent with the recorded
probabilities, and safe distance from display-rounding boundaries.
Artifact-dependent checks require the official version-controlled artifact
(D-013), recompute every profile against it, and verify that it is the
untampered D-016 model. Its absence is a test failure: otherwise a commit
that accidentally omitted this deployment dependency could stay green.
"""

import math

import pytest
import streamlit as st

from src import artifacts, data, modeling
from tests.reference_profiles import (
    DISPLAY_ROUNDING_MARGIN,
    PROBABILITY_TOLERANCE,
    REFERENCE_PROFILES,
    display_boundary_distance,
    format_display,
)
from tests.test_app import APP_PATH, import_app

@pytest.fixture(scope="module")
def official_bundle():
    """The official version-controlled artifact (D-013), fully validated."""
    assert artifacts.DEFAULT_ARTIFACT_PATH.is_file(), (
        "D-013 requires the official version-controlled artifact at "
        f"'{artifacts.DEFAULT_ARTIFACT_PATH}'. Do not omit it from the P7 "
        "commit; regenerate it with 'python -m src.artifacts' if needed."
    )
    return artifacts.load_artifact()


# ---------------------------------------------------------------------------
# Static contract checks on the recorded profiles
# ---------------------------------------------------------------------------


def test_profiles_cover_low_and_high_outputs_with_unique_names():
    names = [profile.name for profile in REFERENCE_PROFILES]
    assert len(names) == len(set(names))
    probabilities = [profile.expected_probability for profile in REFERENCE_PROFILES]
    # US-0703: at least one low-output profile and several high-output ones.
    assert sum(probability < 0.05 for probability in probabilities) >= 1
    assert sum(probability > 0.5 for probability in probabilities) >= 2


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_profile_has_exactly_the_21_contract_features(profile):
    assert list(profile.features) == data.FEATURE_COLUMNS


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_profile_satisfies_the_input_contract(profile):
    artifacts.validate_input_values(profile.features)


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_profile_ui_age_converts_to_the_recorded_brfss_code(profile):
    module = import_app("streamlit_app_profile_age")
    assert module.age_to_group_code(profile.ui_age) == profile.age_group_code
    assert profile.features["Age"] == profile.age_group_code


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_recorded_display_matches_recorded_probability(profile):
    assert format_display(profile.expected_probability) == profile.expected_display


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_recorded_probability_keeps_the_display_rounding_margin(profile):
    # A recorded value close to a 0.05-percentage-point rounding boundary
    # could render a different display under benign float variation; the
    # profiles are chosen to stay clear of every boundary.
    assert display_boundary_distance(profile.expected_probability) >= (
        DISPLAY_ROUNDING_MARGIN
    )


def test_tolerance_cannot_hide_display_drift():
    # The tolerance must stay below the boundary margin (so an accepted
    # probability always renders the recorded display) and far below one
    # display step (so a mismatch cannot be "fixed" by widening it).
    assert PROBABILITY_TOLERANCE < DISPLAY_ROUNDING_MARGIN
    assert PROBABILITY_TOLERANCE <= 5e-4


# ---------------------------------------------------------------------------
# Recomputation against the official artifact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_official_artifact_reproduces_the_recorded_expectation(
    official_bundle, profile
):
    probability = artifacts.predict_risk_probability(
        official_bundle, profile.features
    )

    assert math.isfinite(probability)
    assert probability == pytest.approx(
        profile.expected_probability, abs=PROBABILITY_TOLERANCE
    )
    assert format_display(probability) == profile.expected_display


def test_expectations_come_from_the_untampered_d016_artifact(official_bundle):
    # The profiles must never be made to pass by swapping the model or its
    # configuration: the official artifact has to remain the D-016
    # HistGradientBoostingClassifier (library defaults, recorded seed, exact
    # feature order, binary classes -- all enforced by the bundle validator)
    # and reproduce the D-016 selection evidence on the P5 metric protocol.
    artifacts.validate_artifact_bundle(official_bundle)
    metadata = official_bundle["metadata"]
    assert metadata["selection_decision"] == "D-016"
    assert metadata["random_seed"] == data.RANDOM_SEED
    package_versions = metadata["package_versions"]
    assert tuple(map(int, package_versions["python"].split(".")[:2])) == (
        artifacts.ARTIFACT_PYTHON_MAJOR_MINOR
    )
    assert {
        package: package_versions[package]
        for package in artifacts.ARTIFACT_PACKAGE_VERSION_PINS
    } == artifacts.ARTIFACT_PACKAGE_VERSION_PINS
    model_test = metadata["metrics"]["model_selection"]["test"]
    assert model_test["pr_auc"] == pytest.approx(0.423, abs=0.01)
    assert model_test["roc_auc"] == pytest.approx(0.827, abs=0.01)
    assert metadata["schema_version"] == 2
    assert metadata["calibration_decision"] == "D-018"
    assert metadata["calibration_method"] == "none"
    assert official_bundle["calibrator"] is None
    assert metadata["threshold_policy_decision"] == "D-019"
    # The selected P8 serving path applies no decision threshold and D-018
    # selected no post-hoc calibrator, so the expectation remains the raw
    # positive-class predict_proba output.
    row = artifacts.input_to_dataframe(REFERENCE_PROFILES[0].features)
    raw = float(
        modeling.predict_positive_proba(official_bundle["model"], row)[0]
    )
    assert raw == pytest.approx(
        REFERENCE_PROFILES[0].expected_probability, abs=PROBABILITY_TOLERANCE
    )


# ---------------------------------------------------------------------------
# End-to-end: the app form renders the recorded display for each profile
# ---------------------------------------------------------------------------


def fill_form_with_profile(app, module, profile) -> None:
    """Drive every form widget from the profile's recorded inputs."""
    checkbox_features = {
        label: feature for feature, label in module.BINARY_CHECKBOX_LABELS.items()
    }
    for checkbox in app.checkbox:
        feature = checkbox_features[checkbox.label]
        checkbox.set_value(bool(profile.features[feature]))

    ordinal_features = {
        label: feature
        for feature, (label, _, _) in module.ORDINAL_INPUTS.items()
    }
    for selectbox in app.selectbox:
        # select() takes the raw option code; the app's format_func only
        # affects what the user sees, not the underlying widget value.
        if selectbox.label == "Sex":
            selectbox.select(profile.features["Sex"])
        else:
            selectbox.select(profile.features[ordinal_features[selectbox.label]])

    numeric_features = {
        label: feature for feature, (label, _) in module.NUMERIC_INPUTS.items()
    }
    for number_input in app.number_input:
        if number_input.label == "Age":
            number_input.set_value(profile.ui_age)
        else:
            number_input.set_value(profile.features[numeric_features[number_input.label]])


@pytest.mark.parametrize("profile", REFERENCE_PROFILES, ids=lambda p: p.name)
def test_app_displays_the_recorded_profile_expectation(official_bundle, profile):
    from streamlit.testing.v1 import AppTest

    module = import_app("streamlit_app_profile_form")
    st.cache_resource.clear()

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    assert not app.exception

    fill_form_with_profile(app, module, profile)
    app.button[0].set_value(True).run()

    assert not app.exception
    assert len(app.metric) == 1
    assert app.metric[0].value == profile.expected_display
    age_label = module.AGE_GROUP_LABELS[profile.age_group_code]
    assert any(
        f"{age_label} model age group" in caption.value for caption in app.caption
    )
    assert len(app.warning) >= 1  # disclaimer stays visible with the result
