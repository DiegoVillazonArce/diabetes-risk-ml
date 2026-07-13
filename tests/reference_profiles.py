"""Deployment-smoke reference profiles (P7/P8, Epics E7/E6).

Each profile records the exact 21-feature model input for one synthetic
reference case, the exact age a user types into the app's Age field together
with the BRFSS age-group code the app must derive from it, and the expected
model output computed locally from the official artifact
(`models/diabetes_risk_model.joblib`, D-013/D-016/D-018) under the pinned
environment (Python 3.12, `requirements.txt`).

The profiles are the single source of truth shared by the local pytest
checks (`tests/test_reference_profiles.py`) and the manual smoke tests
against the publicly deployed app: enter each profile's inputs in the form
and compare the displayed percentage with `expected_display`. Print a
human-readable checklist for that manual verification with:

    python -m tests.reference_profiles

Provenance: the inputs were reconstructed and verified in P7 against the
official artifact. The P6 manual observations near 0.3%, 60%, 70%, and
79.9% guided the coverage targets, but they were never fixtures themselves;
these recorded inputs and expectations replace them. Every expected
probability is produced through the schema-version-2 serving contract
(`src.artifacts.predict_risk_probability`). D-018 selected `none`, so this
contract still returns the raw positive-class `predict_proba` output with no
post-hoc calibration; D-019 adds no served threshold or decision layer. The
tests verify both the untampered D-016 model and the accepted P8 contract.
"""

from __future__ import annotations

from dataclasses import dataclass

# Comparison tolerance for a recomputed probability against the recorded
# expectation. Recomputing with the same artifact and pinned environment is
# deterministic (difference 0.0); the tolerance only absorbs benign float
# variation across OS or math-library builds (observed magnitudes are far
# below 1e-6) while still failing loudly on any real input, model, or
# contract mismatch, which shifts probabilities by whole percentage points.
PROBABILITY_TOLERANCE = 2e-4

# Every recorded probability keeps at least this distance from the nearest
# 0.1-percentage-point display-rounding boundary (odd multiples of 5e-4 in
# probability space). Because this margin exceeds PROBABILITY_TOLERANCE, any
# probability accepted by the tolerance also renders exactly the recorded
# display string, so the two expectations can never disagree.
DISPLAY_ROUNDING_MARGIN = 2.5e-4


def format_display(probability: float) -> str:
    """Render a probability exactly the way the app's result metric does."""
    return f"{probability:.1%}"


def display_boundary_distance(probability: float) -> float:
    """Distance to the nearest 0.1-percentage-point rounding boundary."""
    return abs(probability % 1e-3 - 5e-4)


@dataclass(frozen=True)
class ReferenceProfile:
    """One recorded single-case input with its locally verified expectation."""

    name: str
    description: str
    ui_age: int  # exact age typed into the app's Age input
    age_group_code: int  # BRFSS code the app must derive; equals features["Age"]
    features: dict[str, int]  # exact 21-feature model input, training order
    expected_probability: float  # official-artifact predict_proba, positive class
    expected_display: str  # app rendering of expected_probability


REFERENCE_PROFILES: tuple[ReferenceProfile, ...] = (
    ReferenceProfile(
        name="low_risk_young_healthy",
        description=(
            "24-year-old woman, healthy on every indicator: active, daily "
            "fruit and vegetables, normal BMI, excellent self-rated health, "
            "college graduate, $50,000-$74,999 income, insured, recent "
            "cholesterol check."
        ),
        ui_age=24,  # upper boundary of BRFSS group 1 (18-24)
        age_group_code=1,
        features={
            "HighBP": 0,
            "HighChol": 0,
            "CholCheck": 1,
            "BMI": 22,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 1,
            "MentHlth": 0,
            "PhysHlth": 0,
            "DiffWalk": 0,
            "Sex": 0,
            "Age": 1,
            "Education": 6,
            "Income": 7,
        },
        expected_probability=0.0030013847190188967,
        expected_display="0.3%",
    ),
    ReferenceProfile(
        name="high_risk_cardiac_smoker",
        description=(
            "80-year-old man with coronary heart disease, smoker, high blood "
            "pressure and cholesterol, BMI 42, inactive, no daily fruit or "
            "vegetables, difficulty walking, 20 poor-physical-health days, "
            "good self-rated health, high-school graduate, under $10,000 "
            "income."
        ),
        ui_age=80,  # lower boundary of the open-ended BRFSS group 13 (80+)
        age_group_code=13,
        features={
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 42,
            "Smoker": 1,
            "Stroke": 0,
            "HeartDiseaseorAttack": 1,
            "PhysActivity": 0,
            "Fruits": 0,
            "Veggies": 0,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 3,
            "MentHlth": 0,
            "PhysHlth": 20,
            "DiffWalk": 1,
            "Sex": 1,
            "Age": 13,
            "Education": 4,
            "Income": 1,
        },
        expected_probability=0.6000009431177805,
        expected_display="60.0%",
    ),
    ReferenceProfile(
        name="high_risk_poor_health",
        description=(
            "70-year-old man in poor self-rated health with high blood "
            "pressure and cholesterol, BMI 38, inactive, no daily fruit or "
            "vegetables, difficulty walking, 10 poor-physical-health days, "
            "some high school, under $10,000 income."
        ),
        ui_age=70,  # lower boundary of BRFSS group 11 (70-74)
        age_group_code=11,
        features={
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 38,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 0,
            "Fruits": 0,
            "Veggies": 0,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 5,
            "MentHlth": 0,
            "PhysHlth": 10,
            "DiffWalk": 1,
            "Sex": 1,
            "Age": 11,
            "Education": 3,
            "Income": 1,
        },
        expected_probability=0.699987950051215,
        expected_display="70.0%",
    ),
    ReferenceProfile(
        name="high_risk_severe_obesity_cardiac",
        description=(
            "65-year-old man with coronary heart disease, high blood "
            "pressure and cholesterol, BMI 45, inactive, no daily fruit or "
            "vegetables, difficulty walking, 10 poor-physical-health days, "
            "poor self-rated health, high-school graduate, $35,000-$49,999 "
            "income."
        ),
        ui_age=65,  # lower boundary of BRFSS group 10 (65-69)
        age_group_code=10,
        features={
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 45,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 1,
            "PhysActivity": 0,
            "Fruits": 0,
            "Veggies": 0,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 5,
            "MentHlth": 0,
            "PhysHlth": 10,
            "DiffWalk": 1,
            "Sex": 1,
            "Age": 10,
            "Education": 4,
            "Income": 6,
        },
        expected_probability=0.799000716697458,
        expected_display="79.9%",
    ),
)


def main() -> None:
    """Print the manual smoke-test checklist for the deployed app."""
    print("Deployment smoke-test reference profiles (P7, US-0703)")
    print(
        "Enter each profile in the deployed app and compare the displayed "
        "risk percentage.\n"
        f"Expected agreement: exact display match; recomputed probabilities "
        f"must stay within {PROBABILITY_TOLERANCE} of the recorded value.\n"
    )
    for profile in REFERENCE_PROFILES:
        print(f"--- {profile.name} ---")
        print(f"  {profile.description}")
        print(
            f"  Age input: {profile.ui_age} "
            f"(must map to BRFSS group code {profile.age_group_code})"
        )
        print("  Model features (training order):")
        for feature, value in profile.features.items():
            print(f"    {feature} = {value}")
        print(f"  Expected probability: {profile.expected_probability!r}")
        print(f"  Expected displayed risk: {profile.expected_display}\n")


if __name__ == "__main__":
    main()
