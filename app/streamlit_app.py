"""Streamlit app: single-case risk estimation with P9 explanation.

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

from src import artifacts, explainability
from src.data import VALUE_RANGES
from src.feature_labels import (
    AGE_GROUP_LABELS,
    FEATURE_LABELS,
    ORDINAL_VALUE_LABELS,
)

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
    feature: FEATURE_LABELS[feature]
    for feature in (
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
    )
}

# Ordinal scales: label, the BRFSS meaning of each code, and a neutral
# default index for the form. The internal codes are never shown to users.
ORDINAL_INPUTS = {
    "GenHlth": (
        FEATURE_LABELS["GenHlth"],
        ORDINAL_VALUE_LABELS["GenHlth"],
        2,
    ),
    "Education": (
        FEATURE_LABELS["Education"],
        ORDINAL_VALUE_LABELS["Education"],
        3,
    ),
    "Income": (
        FEATURE_LABELS["Income"],
        ORDINAL_VALUE_LABELS["Income"],
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
# Numeric measures: label and form default; bounds come from VALUE_RANGES.
NUMERIC_INPUTS = {
    "BMI": (FEATURE_LABELS["BMI"], 25),
    "MentHlth": (FEATURE_LABELS["MentHlth"], 0),
    "PhysHlth": (FEATURE_LABELS["PhysHlth"], 0),
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
def load_model_bundle(artifact_hash: str) -> dict:
    """Load the local artifact under a cache key bound to its byte hash."""
    return artifacts.load_artifact()


@st.cache_resource(show_spinner="Preparing the model explanation ...")
def load_shap_explainer(_bundle: dict, artifact_hash: str):
    """Cache one local explainer under the model artifact's byte hash."""
    return explainability.load_runtime_explainer(
        _bundle, expected_artifact_sha256=artifact_hash
    )


def build_local_contribution_chart(display):
    """Build a two-direction Vega-Lite chart with an explicit zero reference."""
    chart = display.copy()
    chart["Factor"] = chart.apply(
        lambda row: f"{row['feature_label']}: {row['display_value']}", axis=1
    )
    chart["Model-estimate change (percentage points)"] = (
        chart["contribution"] * 100
    )
    chart["Direction"] = chart["contribution"].map(
        lambda value: "Increased" if value >= 0 else "Decreased"
    )
    chart["Display order"] = range(len(chart))
    contribution_points = chart[
        "Model-estimate change (percentage points)"
    ].astype(float)
    domain_min = min(0.0, float(contribution_points.min()))
    domain_max = max(0.0, float(contribution_points.max()))
    if domain_min == domain_max:
        domain_min, domain_max = -0.01, 0.01
    spec = {
        "height": 300,
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusEnd": 3},
                "encoding": {
                    "x": {
                        "field": "Model-estimate change (percentage points)",
                        "type": "quantitative",
                        "title": "Change in model estimate (percentage points)",
                        "stack": None,
                        "scale": {
                            "domain": [domain_min, domain_max],
                            "nice": True,
                        },
                        "axis": {"format": ".2f"},
                    },
                    "y": {
                        "field": "Factor",
                        "type": "nominal",
                        "title": None,
                        "sort": {"field": "Display order", "order": "ascending"},
                        "axis": {"labelLimit": 420, "labelFontSize": 12},
                    },
                    "color": {
                        "field": "Direction",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Increased", "Decreased"],
                            "range": ["#D95F02", "#1B77B8"],
                        },
                        "legend": {"title": "Effect on model estimate"},
                    },
                    "tooltip": [
                        {"field": "Factor", "type": "nominal"},
                        {"field": "Direction", "type": "nominal"},
                        {
                            "field": "Model-estimate change (percentage points)",
                            "type": "quantitative",
                            "format": ".2f",
                        },
                    ],
                },
            },
            {
                "mark": {"type": "rule", "color": "#555555", "strokeWidth": 1},
                "encoding": {"x": {"datum": 0}},
            },
        ],
    }
    return chart, spec


def escape_markdown_dollar_signs(value: object) -> str:
    """Render currency labels literally instead of as inline math."""
    return str(value).replace("$", r"\$")


def render_local_explanation(
    bundle: dict,
    values: dict,
    probability: float,
    artifact_hash: str,
) -> None:
    """Render a compact everyday-language explanation for the submitted case."""
    try:
        explainer = load_shap_explainer(bundle, artifact_hash)
        full_table = explainability.explain_local_values(
            bundle, explainer, values
        )
        explained_probability = float(full_table["model_probability"].iloc[0])
        if abs(explained_probability - probability) > explainability.ADDITIVITY_TOLERANCE:
            raise explainability.ExplainabilityError(
                "The explanation does not match the displayed probability."
            )
        display = explainability.select_display_contributions(full_table)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.warning(
            "A detailed model explanation is temporarily unavailable. "
            "The probability shown above is unchanged."
        )
        with st.expander("Explanation error details"):
            st.code(str(error))
        return

    st.subheader("How the model interprets this estimate")
    st.write(
        "The model starts from a reference estimate, then the entered values "
        "contribute upward or downward until they reach this estimate."
    )
    st.metric(
        "Model reference estimate",
        f"{float(full_table['base_value'].iloc[0]):.1%}",
        help=(
            "The model output averaged over the privacy-safe aggregate "
            "background used by this explanation."
        ),
    )

    chart, chart_spec = build_local_contribution_chart(display)
    st.vega_lite_chart(
        chart,
        chart_spec,
        use_container_width=True,
        theme=None,
    )

    st.markdown("**Largest contributions for this estimate**")
    for row in display.itertuples(index=False):
        display_value = escape_markdown_dollar_signs(row.display_value)
        st.markdown(
            f"- **{row.feature_label}: {display_value}** {row.direction} "
            f"by {abs(row.contribution) * 100:.2f} model-estimate percentage points."
        )

    with st.expander("What this explanation means"):
        st.markdown(
            "- The **model reference estimate** is the model output averaged "
            "over a privacy-safe aggregate background. It is not the risk of "
            "an average person.\n"
            "- The contributions describe how this fitted model combined the "
            "submitted values. They add to the displayed model estimate.\n"
            "- The explanation depends on this model, its training data, and "
            "the selected aggregate background. A different model or "
            "background can allocate contributions differently.\n"
            "- These contributions explain model behavior, not medical causes. "
            "They are not a diagnosis or a recommendation."
        )


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
        artifact_hash = explainability.artifact_sha256(
            artifacts.DEFAULT_ARTIFACT_PATH
        )
        bundle = load_model_bundle(artifact_hash)
    except (FileNotFoundError, ValueError) as error:
        message = str(error)
        if "python -m src.artifacts" not in message:
            message += " Generate it with 'python -m src.artifacts'."
        st.error(message)
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
            render_local_explanation(
                bundle, values, probability, artifact_hash
            )
    st.warning(medical_disclaimer(bundle))


if __name__ == "__main__":
    main()
