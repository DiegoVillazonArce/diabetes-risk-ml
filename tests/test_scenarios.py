"""P10 deterministic scenario-engine contract tests."""

from __future__ import annotations

import copy
import inspect
import math

import pytest

from src import artifacts, scenarios
from src.data import FEATURE_COLUMNS
from tests.test_modeling import make_splits


@pytest.fixture(scope="module")
def bundle():
    return artifacts.build_artifact_bundle(make_splits())


def baseline() -> dict[str, int]:
    values = artifacts.example_input()
    values.update({"PhysActivity": 0, "Fruits": 0, "Veggies": 0})
    return {feature: values[feature] for feature in FEATURE_COLUMNS}


def test_d023_whitelist_is_exact_and_in_feature_order():
    assert scenarios.EDITABLE_SCENARIO_FEATURES == (
        "PhysActivity",
        "Fruits",
        "Veggies",
    )
    positions = [
        FEATURE_COLUMNS.index(feature)
        for feature in scenarios.EDITABLE_SCENARIO_FEATURES
    ]
    assert positions == sorted(positions)


def test_original_profile_is_not_mutated(bundle):
    original = baseline()
    snapshot = copy.deepcopy(original)

    result = scenarios.compare_scenario(bundle, original, {"Fruits": 1})

    assert original == snapshot
    assert original is not result.original_profile
    assert original is not result.hypothetical_profile
    assert result.original_profile == snapshot


def test_returned_comparison_mappings_are_read_only(bundle):
    result = scenarios.compare_scenario(
        bundle, baseline(), {"Fruits": 1}
    )

    with pytest.raises(TypeError):
        result.original_profile["BMI"] = 99
    with pytest.raises(TypeError):
        result.hypothetical_profile["Fruits"] = 0
    with pytest.raises(TypeError):
        result.effective_changes["Veggies"] = scenarios.ScenarioChange(0, 1)


def test_no_change_returns_exact_identity_and_zero_delta(bundle):
    original = baseline()

    result = scenarios.compare_scenario(bundle, original, {})

    assert result.original_profile == result.hypothetical_profile == original
    assert result.effective_changes == {}
    assert result.hypothetical_probability == result.original_probability
    assert result.delta_percentage_points == 0.0


def test_equal_supplied_value_is_an_effective_no_change(bundle):
    original = baseline()

    result = scenarios.compare_scenario(
        bundle, original, {"PhysActivity": original["PhysActivity"]}
    )

    assert result.effective_changes == {}
    assert result.hypothetical_probability == result.original_probability
    assert result.delta_percentage_points == 0.0


def test_reset_is_an_exact_ordered_copy_of_all_21_fields():
    original = dict(reversed(list(baseline().items())))

    reset = scenarios.reset_scenario_profile(original)

    assert reset == baseline()
    assert reset is not original
    assert len(reset) == 21
    assert list(reset) == FEATURE_COLUMNS


def test_profiles_preserve_exact_feature_order_and_unedited_values(bundle):
    original = dict(reversed(list(baseline().items())))

    result = scenarios.compare_scenario(bundle, original, {"Veggies": 1})

    assert list(result.original_profile) == FEATURE_COLUMNS
    assert list(result.hypothetical_profile) == FEATURE_COLUMNS
    assert len(result.original_profile) == len(result.hypothetical_profile) == 21
    for feature in FEATURE_COLUMNS:
        expected = 1 if feature == "Veggies" else original[feature]
        assert result.hypothetical_profile[feature] == expected


@pytest.mark.parametrize("feature", scenarios.EDITABLE_SCENARIO_FEATURES)
def test_every_and_only_approved_field_is_editable(bundle, feature):
    original = baseline()

    result = scenarios.compare_scenario(bundle, original, {feature: 1})

    assert list(result.effective_changes) == [feature]
    assert result.effective_changes[feature] == scenarios.ScenarioChange(0, 1)


@pytest.mark.parametrize(
    "feature",
    [
        "BMI",
        "Smoker",
        "HighBP",
        "Stroke",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "GenHlth",
        "MentHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    ],
)
def test_excluded_fields_are_rejected(bundle, feature):
    with pytest.raises(ValueError, match="not editable"):
        scenarios.compare_scenario(bundle, baseline(), {feature: baseline()[feature]})


def test_unknown_fields_are_rejected(bundle):
    with pytest.raises(ValueError, match="Unknown scenario fields.*NotAFeature"):
        scenarios.compare_scenario(bundle, baseline(), {"NotAFeature": 1})


def test_more_than_one_field_is_rejected(bundle):
    with pytest.raises(ValueError, match="at most 1 changed field"):
        scenarios.compare_scenario(
            bundle, baseline(), {"PhysActivity": 1, "Fruits": 1}
        )


@pytest.mark.parametrize("changes", [None, [], "PhysActivity"])
def test_changes_must_be_a_mapping(bundle, changes):
    with pytest.raises(ValueError, match="must be a mapping"):
        scenarios.compare_scenario(bundle, baseline(), changes)


@pytest.mark.parametrize(
    "bad_value",
    [None, "yes", object(), float("nan"), float("inf"), float("-inf"), 0.5],
)
def test_missing_nonfinite_and_incorrectly_typed_values_are_rejected(
    bundle, bad_value
):
    with pytest.raises(ValueError, match="'PhysActivity'"):
        scenarios.compare_scenario(
            bundle, baseline(), {"PhysActivity": bad_value}
        )


@pytest.mark.parametrize("bad_value", [-1, 2])
def test_out_of_domain_values_are_rejected(bundle, bad_value):
    with pytest.raises(ValueError, match="'Fruits'.*outside the valid range"):
        scenarios.compare_scenario(bundle, baseline(), {"Fruits": bad_value})


def test_invalid_original_profile_reuses_existing_validation(bundle):
    original = baseline()
    del original["BMI"]

    with pytest.raises(ValueError, match="Missing features.*BMI"):
        scenarios.compare_scenario(bundle, original, {})


def test_scenario_probability_equals_direct_production_scoring(bundle):
    original = baseline()
    result = scenarios.compare_scenario(bundle, original, {"Fruits": 1})
    direct = artifacts.predict_risk_probability(
        bundle, result.hypothetical_profile
    )

    assert result.hypothetical_probability == direct


def test_both_probabilities_use_predict_risk_probability(monkeypatch):
    calls = []

    def fake_predict(bundle, values):
        calls.append(dict(values))
        return 0.2 + 0.1 * values["Fruits"]

    monkeypatch.setattr(scenarios, "predict_risk_probability", fake_predict)
    result = scenarios.compare_scenario(object(), baseline(), {"Fruits": 1})

    assert calls == [result.original_profile, result.hypothetical_profile]
    assert result.original_probability == 0.2
    assert result.hypothetical_probability == pytest.approx(0.3)


def test_delta_formula_and_signs_with_absolute_1e12_tolerance(monkeypatch):
    def fake_predict(bundle, values):
        return 0.25 + 0.125 * values["PhysActivity"]

    monkeypatch.setattr(scenarios, "predict_risk_probability", fake_predict)
    inactive = baseline()
    active = dict(inactive, PhysActivity=1)

    positive = scenarios.compare_scenario(
        object(), inactive, {"PhysActivity": 1}
    )
    negative = scenarios.compare_scenario(
        object(), active, {"PhysActivity": 0}
    )
    zero = scenarios.compare_scenario(object(), inactive, {})

    for result in (positive, negative, zero):
        expected = 100.0 * (
            result.hypothetical_probability - result.original_probability
        )
        assert result.delta_percentage_points == pytest.approx(
            expected, abs=scenarios.DELTA_ABSOLUTE_TOLERANCE
        )
    assert positive.delta_percentage_points > 0
    assert negative.delta_percentage_points < 0
    assert zero.delta_percentage_points == 0
    assert scenarios.DELTA_ABSOLUTE_TOLERANCE == 1e-12


def test_engine_is_deterministic(bundle):
    first = scenarios.compare_scenario(bundle, baseline(), {"Veggies": 1})
    second = scenarios.compare_scenario(bundle, baseline(), {"Veggies": 1})

    assert first == second


def test_engine_has_no_data_io_fitting_explanation_or_persistence_paths():
    source = inspect.getsource(scenarios).lower()
    for forbidden in (
        "prepare_data",
        "load_raw_data",
        "read_csv",
        ".fit(",
        "calibrat",
        "shap",
        "explainability",
        "joblib",
        "to_csv",
        "write_text",
        "open(",
        "logging",
        "requests",
        "optimization",
        "ranking",
        "preset",
        "threshold",
        "high_risk",
        "low_risk",
    ):
        assert forbidden not in source


def test_returned_probabilities_and_delta_are_finite(bundle):
    result = scenarios.compare_scenario(bundle, baseline(), {"Fruits": 1})

    assert math.isfinite(result.original_probability)
    assert math.isfinite(result.hypothetical_probability)
    assert math.isfinite(result.delta_percentage_points)
