"""Pure user-facing labels for BRFSS model features and encoded values.

P9 uses this module as the single source of truth shared by Streamlit,
offline SHAP evidence, and tests.  It deliberately contains no Streamlit,
model, artifact, or data-loading code.
"""

from __future__ import annotations

import math

from src.data import BINARY_FEATURES, FEATURE_COLUMNS

FEATURE_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check in the past 5 years",
    "BMI": "Body mass index (BMI)",
    "Smoker": "Smoked at least 100 cigarettes in lifetime",
    "Stroke": "Ever had a stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack",
    "PhysActivity": "Physical activity in the past 30 days",
    "Fruits": "Eats fruit at least once per day",
    "Veggies": "Eats vegetables at least once per day",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has any kind of health care coverage",
    "NoDocbcCost": "Skipped a doctor visit due to cost in the past year",
    "GenHlth": "General health (self-rated)",
    "MentHlth": "Days of poor mental health in the past 30 days",
    "PhysHlth": "Days of poor physical health in the past 30 days",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "Sex": "Sex",
    "Age": "Age group (BRFSS)",
    "Education": "Education level",
    "Income": "Annual household income",
}

AGE_GROUP_LABELS = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80 or older",
}

ORDINAL_VALUE_LABELS = {
    "GenHlth": {
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor",
    },
    "Age": AGE_GROUP_LABELS,
    "Education": {
        1: "Never attended school or only kindergarten",
        2: "Elementary (grades 1-8)",
        3: "Some high school (grades 9-11)",
        4: "High school graduate or GED",
        5: "Some college or technical school",
        6: "College graduate (4 years or more)",
    },
    "Income": {
        1: "Less than $10,000",
        2: "$10,000 to $14,999",
        3: "$15,000 to $19,999",
        4: "$20,000 to $24,999",
        5: "$25,000 to $34,999",
        6: "$35,000 to $49,999",
        7: "$50,000 to $74,999",
        8: "$75,000 or more",
    },
}

BINARY_VALUE_LABELS = {
    feature: {0: "No", 1: "Yes"} for feature in BINARY_FEATURES if feature != "Sex"
}
BINARY_VALUE_LABELS["Sex"] = {0: "Female", 1: "Male"}


def feature_label(feature: str) -> str:
    """Return the audited user-facing label for one model feature."""
    if feature not in FEATURE_LABELS:
        raise ValueError(f"Unknown model feature {feature!r}.")
    return FEATURE_LABELS[feature]


def format_feature_value(feature: str, value: float | int) -> str:
    """Translate one encoded model value without exposing avoidable codes."""
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown model feature {feature!r}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Feature {feature!r} has a non-finite value {value!r}.")
    if feature in BINARY_VALUE_LABELS:
        code = int(numeric)
        if numeric != code or code not in BINARY_VALUE_LABELS[feature]:
            raise ValueError(f"Invalid encoded value {value!r} for {feature!r}.")
        return BINARY_VALUE_LABELS[feature][code]
    if feature in ORDINAL_VALUE_LABELS:
        code = int(numeric)
        if numeric != code or code not in ORDINAL_VALUE_LABELS[feature]:
            raise ValueError(f"Invalid encoded value {value!r} for {feature!r}.")
        return ORDINAL_VALUE_LABELS[feature][code]
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}"


if set(FEATURE_LABELS) != set(FEATURE_COLUMNS):
    raise RuntimeError("Feature labels must cover exactly FEATURE_COLUMNS.")
