"""Streamlit app: single-case diabetes risk estimation (P6/P8, Epics E5/E6).

The app is a thin UI over the P6 serving contract in `src.artifacts`: it
loads the local D-016 model artifact once per server process (cached; the
app never trains, per D-007 and D-017), collects the 21
`src.data.FEATURE_COLUMNS` inputs in one form grouped by the documented
binary/ordinal/numeric feature groups, and shows the positive-class
probability as an educational risk percentage with a visible medical
disclaimer (US-0501, US-0502). Model-feature bounds come from the P3
`VALUE_RANGES` contract; an exact adult age is validated in the UI and mapped
to the model's BRFSS age-group code before `src.artifacts` re-validates and
scores the case through the schema-version-2 P8 probability contract.

Run from the project root (the artifact ships with the repository per
D-013; regenerate it only to reproduce training):

    python -m src.artifacts                       # optional: retrain the artifact
    python -m streamlit run app/streamlit_app.py
"""

import numbers
import sys
from pathlib import Path

# `streamlit run` executes this file directly, so the project root (the
# parent of app/) must be on sys.path before importing from src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import artifacts
from src.data import VALUE_RANGES

def medical_disclaimer(bundle: dict) -> str:
    """Medical warning whose calibration wording matches the served contract."""
    calibrated = artifacts.probability_is_calibrated(bundle)
    method = bundle["metadata"]["calibration_method"]
    estimate = (
        f"a post-hoc {method}-calibrated statistical estimate"
        if calibrated
        else "an uncalibrated statistical estimate"
    )
    return (
        "**Medical disclaimer:** this tool is an educational portfolio project "
        "built on self-reported BRFSS 2015 survey data. The percentage shown is "
        f"{estimate} from a machine-learning model -- it is not a diagnosis, "
        "it cannot detect or rule out diabetes, and it does not replace "
        "professional medical advice. Consult a healthcare professional for "
        "any medical concern."
    )


def probability_contract_caption(bundle: dict) -> str:
    """Explain the validated P8 contract without introducing a decision layer."""
    calibrated = artifacts.probability_is_calibrated(bundle)
    method = bundle["metadata"]["calibration_method"]
    if calibrated:
        return (
            f"Positive-class probability with post-hoc {method} calibration "
            "selected by D-018; no custom decision threshold is applied."
        )
    return (
        "Uncalibrated positive-class probability from the model's "
        "`predict_proba`. D-018 retained this contract because neither "
        "post-hoc method met the pre-declared adoption criteria; no custom "
        "decision threshold is applied."
    )

# Yes/no indicators rendered as checkboxes (checked = 1). `Sex` is rendered
# separately because its 0/1 codes mean female/male, not no/yes.
BINARY_CHECKBOX_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check in the past 5 years",
    "Smoker": "Smoked at least 100 cigarettes in lifetime",
    "Stroke": "Ever had a stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack",
    "PhysActivity": "Physical activity in the past 30 days",
    "Fruits": "Eats fruit at least once per day",
    "Veggies": "Eats vegetables at least once per day",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has any kind of health care coverage",
    "NoDocbcCost": "Skipped a doctor visit due to cost in the past year",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
}

# Ordinal scales: label, the BRFSS meaning of each code, and a neutral
# default index for the form. The internal codes are never shown to users.
ORDINAL_INPUTS = {
    "GenHlth": (
        "General health (self-rated)",
        {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
        2,
    ),
    "Education": (
        "Education level",
        {
            1: "Never attended school or only kindergarten",
            2: "Elementary (grades 1-8)",
            3: "Some high school (grades 9-11)",
            4: "High school graduate or GED",
            5: "Some college or technical school",
            6: "College graduate (4 years or more)",
        },
        3,
    ),
    "Income": (
        "Annual household income",
        {
            1: "Less than $10,000",
            2: "$10,000 to $14,999",
            3: "$15,000 to $19,999",
            4: "$20,000 to $24,999",
            5: "$25,000 to $34,999",
            6: "$35,000 to $49,999",
            7: "$50,000 to $74,999",
            8: "$75,000 or more",
        },
        5,
    ),
}

# The model was trained on BRFSS age-group codes, but the UI accepts an exact
# adult age and translates it before inference. The upper bound is a
# conservative UI validation limit; every age from 80 onward maps to the
# dataset's open-ended final group.
MIN_AGE = 18
MAX_AGE = 120
DEFAULT_AGE = 40
AGE_GROUPS = (
    (18, 24, 1, "18-24"),
    (25, 29, 2, "25-29"),
    (30, 34, 3, "30-34"),
    (35, 39, 4, "35-39"),
    (40, 44, 5, "40-44"),
    (45, 49, 6, "45-49"),
    (50, 54, 7, "50-54"),
    (55, 59, 8, "55-59"),
    (60, 64, 9, "60-64"),
    (65, 69, 10, "65-69"),
    (70, 74, 11, "70-74"),
    (75, 79, 12, "75-79"),
    (80, MAX_AGE, 13, "80 or older"),
)
AGE_GROUP_LABELS = {code: label for _, _, code, label in AGE_GROUPS}

# Numeric measures: label and form default; bounds come from VALUE_RANGES.
NUMERIC_INPUTS = {
    "BMI": ("Body mass index (BMI)", 25),
    "MentHlth": ("Days of poor mental health in the past 30 days", 0),
    "PhysHlth": ("Days of poor physical health in the past 30 days", 0),
}


def age_to_group_code(age: int) -> int:
    """Translate an exact adult age into the BRFSS code expected by the model."""
    if isinstance(age, bool) or not isinstance(age, numbers.Integral):
        raise ValueError(f"Age must be a whole number; got {age!r}.")
    if not MIN_AGE <= age <= MAX_AGE:
        raise ValueError(
            f"Age must be between {MIN_AGE} and {MAX_AGE}; got {age!r}."
        )
    for lower, upper, code, _ in AGE_GROUPS:
        if lower <= age <= upper:
            return code
    raise ValueError(f"Age {age!r} does not map to a BRFSS age group.")


@st.cache_resource(show_spinner="Loading the local model artifact ...")
def load_model_bundle() -> dict:
    """Load and validate the local artifact once per server process."""
    return artifacts.load_artifact()


def render_input_form() -> tuple[dict, int] | None:
    """Collect all 21 features in one submission form.

    Returns the raw feature-code values keyed by exact feature name together
    with the entered age, or None before the form is submitted.
    """
    with st.form("single_case"):
        values: dict[str, int] = {}

        st.subheader("Yes / no health indicators")
        checkbox_features = list(BINARY_CHECKBOX_LABELS)
        midpoint = (len(checkbox_features) + 1) // 2
        left, right = st.columns(2)
        for column, group in (
            (left, checkbox_features[:midpoint]),
            (right, checkbox_features[midpoint:]),
        ):
            with column:
                for feature in group:
                    values[feature] = int(st.checkbox(BINARY_CHECKBOX_LABELS[feature]))
        values["Sex"] = int(
            st.selectbox("Sex", options=(0, 1), format_func=lambda code: ("Female", "Male")[code])
        )

        st.subheader("Age and rated scales")
        exact_age = int(
            st.number_input(
                "Age",
                min_value=MIN_AGE,
                max_value=MAX_AGE,
                value=DEFAULT_AGE,
                step=1,
                help="Your age is converted to the age group used by the model.",
            )
        )
        values["Age"] = age_to_group_code(exact_age)

        for feature, (label, code_labels, default_index) in ORDINAL_INPUTS.items():
            values[feature] = int(
                st.selectbox(
                    label,
                    options=list(code_labels),
                    index=default_index,
                    format_func=lambda code, labels=code_labels: labels[code],
                )
            )

        st.subheader("Counts and measurements")
        for feature, (label, default) in NUMERIC_INPUTS.items():
            lower, upper = VALUE_RANGES[feature]
            values[feature] = int(
                st.number_input(
                    label,
                    min_value=lower,
                    max_value=upper,
                    value=default,
                    step=1,
                    help=f"Valid range: {lower} to {upper}.",
                )
            )

        submitted = st.form_submit_button("Estimate risk", type="primary")

    return (values, exact_age) if submitted else None


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Estimator", page_icon="🩺")
    st.title("Diabetes Risk Estimator")
    st.caption(
        "Educational MVP that estimates self-reported diabetes/prediabetes "
        "risk with the project's selected model (D-016). Not medical advice."
    )

    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info(
            "Generate a valid local artifact from the project root with "
            "`python -m src.artifacts`, then reload this page."
        )
        st.stop()

    submission = render_input_form()

    if submission is None:
        st.info("Fill in the form and press **Estimate risk** to get an educational risk estimate.")
    else:
        values, exact_age = submission
        try:
            probability = artifacts.predict_risk_probability(bundle, values)
        except ValueError as error:
            st.error(f"Could not score this case: {error}")
        else:
            st.subheader("Estimated risk")
            st.metric(
                "Model-estimated probability of diabetes or prediabetes",
                f"{probability:.1%}",
            )
            st.caption(
                f"Age {exact_age} was evaluated in the "
                f"{AGE_GROUP_LABELS[values['Age']]} model age group."
            )
            st.caption(probability_contract_caption(bundle))
    st.warning(medical_disclaimer(bundle))


if __name__ == "__main__":
    main()
