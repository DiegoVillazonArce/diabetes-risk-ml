"""Pure P10 comparison of one profile with one hypothetical input variant.

The module reuses the validated P8 serving contract and the P3 feature
contract. It contains no data loading, fitting, explanation, persistence, or
Streamlit code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from src.artifacts import predict_risk_probability, validate_input_values
from src.data import FEATURE_COLUMNS

ScenarioValue: TypeAlias = int | float | bool

# D-023: the complete semantic audit is recorded in
# docs/p10-scenarios/report.md. Keep this tuple in FEATURE_COLUMNS order.
EDITABLE_SCENARIO_FEATURES = (
    "PhysActivity",
    "Fruits",
    "Veggies",
)

# D-024 deliberately permits a single changed field in the single scenario.
MAX_SCENARIO_CHANGES = 1
DELTA_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ScenarioChange:
    """One effective difference between the original and hypothetical input."""

    original_value: ScenarioValue
    hypothetical_value: ScenarioValue


@dataclass(frozen=True)
class ScenarioComparison:
    """Complete, ordered, read-only result of one model comparison."""

    original_profile: Mapping[str, ScenarioValue]
    hypothetical_profile: Mapping[str, ScenarioValue]
    effective_changes: Mapping[str, ScenarioChange]
    original_probability: float
    hypothetical_probability: float
    delta_percentage_points: float


def reset_scenario_profile(
    original_profile: Mapping[str, ScenarioValue],
) -> dict[str, ScenarioValue]:
    """Return a validated, ordered copy of all 21 original feature values."""
    validate_input_values(original_profile)
    return {
        feature: original_profile[feature]
        for feature in FEATURE_COLUMNS
    }


def _validate_change_keys(changes: Mapping[str, ScenarioValue]) -> None:
    unknown = sorted(
        (feature for feature in changes if feature not in FEATURE_COLUMNS),
        key=repr,
    )
    if unknown:
        raise ValueError(f"Unknown scenario fields: {unknown}.")

    excluded = [
        feature
        for feature in FEATURE_COLUMNS
        if feature in changes and feature not in EDITABLE_SCENARIO_FEATURES
    ]
    if excluded:
        raise ValueError(
            "Scenario fields are not editable under D-023: "
            f"{excluded}."
        )

    if len(changes) > MAX_SCENARIO_CHANGES:
        raise ValueError(
            f"A scenario accepts at most {MAX_SCENARIO_CHANGES} changed "
            f"field; got {len(changes)}."
        )


def compare_scenario(
    bundle: dict,
    original_profile: Mapping[str, ScenarioValue],
    changes: Mapping[str, ScenarioValue],
) -> ScenarioComparison:
    """Score the original profile and one validated hypothetical variant.

    Both probabilities use ``predict_risk_probability``. The caller's profile
    and change mapping are never modified. An empty mapping, or a supplied
    value equal to the original, returns the exact original probability for
    both sides and a signed delta of exactly zero.
    """
    original = reset_scenario_profile(original_profile)
    if not isinstance(changes, Mapping):
        raise ValueError("Scenario changes must be a mapping.")
    _validate_change_keys(changes)

    hypothetical = dict(original)
    for feature, value in changes.items():
        hypothetical[feature] = value
    validate_input_values(hypothetical)
    hypothetical = {
        feature: hypothetical[feature]
        for feature in FEATURE_COLUMNS
    }

    effective_changes = {
        feature: ScenarioChange(
            original_value=original[feature],
            hypothetical_value=hypothetical[feature],
        )
        for feature in EDITABLE_SCENARIO_FEATURES
        if hypothetical[feature] != original[feature]
    }

    original_probability = predict_risk_probability(bundle, original)
    hypothetical_probability = predict_risk_probability(bundle, hypothetical)
    if not effective_changes:
        hypothetical_probability = original_probability
        delta_percentage_points = 0.0
    else:
        delta_percentage_points = 100.0 * (
            hypothetical_probability - original_probability
        )

    return ScenarioComparison(
        original_profile=MappingProxyType(original),
        hypothetical_profile=MappingProxyType(hypothetical),
        effective_changes=MappingProxyType(effective_changes),
        original_probability=original_probability,
        hypothetical_probability=hypothetical_probability,
        delta_percentage_points=delta_percentage_points,
    )


if not set(EDITABLE_SCENARIO_FEATURES).issubset(FEATURE_COLUMNS):
    raise RuntimeError("Scenario fields must be a subset of FEATURE_COLUMNS.")
