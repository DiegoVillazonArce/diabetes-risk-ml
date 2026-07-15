"""App-level tests for app/streamlit_app.py (P6, Epic E5).

These tests never launch a Streamlit server. They confirm the app can be
imported without side effects (no artifact load, no training, no data
access), that it stays a pure local consumer of the project's own artifact
(no training code, no remote artifact sources or deployment secrets --
D-013 resolved artifact distribution as a controlled Git exception, so the
deployed app loads the same version-controlled local file), and -- via
Streamlit's built-in headless `AppTest` harness -- that the prediction
workflow renders and produces a valid educational risk percentage from a
temporary artifact, with clear errors for missing or corrupt artifacts.
"""

import importlib.util
import inspect

import pandas as pd
import pytest
import streamlit as st

from src import artifacts, calibration, data, explainability, scenarios
from tests.test_modeling import make_splits

APP_PATH = data.PROJECT_ROOT / "app" / "streamlit_app.py"


def import_app(name: str):
    """Import the app as a plain module, outside any Streamlit runtime."""
    spec = importlib.util.spec_from_file_location(name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tmp_artifact(tmp_path, monkeypatch):
    """A valid artifact in a pytest tmp dir, wired in as the app's default."""
    splits = make_splits()
    bundle = artifacts.build_artifact_bundle(splits)
    path = artifacts.save_artifact(bundle, tmp_path / "diabetes_risk_model.joblib")
    monkeypatch.setattr(artifacts, "DEFAULT_ARTIFACT_PATH", path)
    # The small shared modeling fixture has only 210 train rows; repeat its
    # feature frame solely to exercise the fixed 256-row asset contract.
    test_background_source = pd.concat(
        [splits.train[data.FEATURE_COLUMNS]] * 3, ignore_index=True
    )
    background = explainability.build_aggregate_background(
        bundle["model"], test_background_source
    )
    payload = explainability.background_asset_payload(
        background,
        model_artifact_sha256=explainability.artifact_sha256(path),
        n_source_rows=len(test_background_source),
    )
    background_path = explainability.write_background_asset(
        payload, tmp_path / "shap_background_v1.json"
    )
    monkeypatch.setattr(
        explainability, "BACKGROUND_ASSET_PATH", background_path
    )
    return path


# ---------------------------------------------------------------------------
# Import safety: no artifact load, no training, no data access at import
# ---------------------------------------------------------------------------


def test_app_import_is_side_effect_free(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "importing the app must not load the artifact, train, or touch data"
        )

    monkeypatch.setattr(artifacts, "load_artifact", forbidden)
    monkeypatch.setattr(artifacts, "build_artifact_bundle", forbidden)
    monkeypatch.setattr(data, "prepare_data", forbidden)
    monkeypatch.setattr(explainability, "generate_evidence", forbidden)
    monkeypatch.setattr(explainability, "build_aggregate_background", forbidden)
    monkeypatch.setattr(explainability, "select_global_sample", forbidden)

    module = import_app("streamlit_app_import_check")

    assert callable(module.main)
    assert callable(module.render_input_form)


def test_app_form_covers_all_21_features():
    module = import_app("streamlit_app_feature_contract")
    form_features = (
        set(module.BINARY_CHECKBOX_LABELS)
        | {"Sex", "Age"}
        | set(module.ORDINAL_INPUTS)
        | set(module.NUMERIC_INPUTS)
    )

    assert form_features == set(data.FEATURE_COLUMNS)


def test_exact_age_maps_to_the_documented_brfss_groups():
    module = import_app("streamlit_app_age_mapping")
    expected = {
        18: 1,
        24: 1,
        25: 2,
        29: 2,
        30: 3,
        34: 3,
        35: 4,
        39: 4,
        40: 5,
        44: 5,
        45: 6,
        49: 6,
        50: 7,
        54: 7,
        55: 8,
        59: 8,
        60: 9,
        64: 9,
        65: 10,
        69: 10,
        70: 11,
        74: 11,
        75: 12,
        79: 12,
        80: 13,
        120: 13,
    }

    for age, code in expected.items():
        assert module.age_to_group_code(age) == code


def test_exact_age_rejects_invalid_values():
    module = import_app("streamlit_app_age_validation")

    for invalid in (17, 121, 18.5, True, None):
        with pytest.raises(ValueError, match="Age"):
            module.age_to_group_code(invalid)


def test_app_source_never_trains_or_reloads_data():
    # The app is a pure consumer of the local artifact (D-007, D-017): no
    # training, no raw-data access, no re-splitting, no serialization.
    source = APP_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "prepare_data",
        "load_raw_data",
        "read_csv",
        "train_test_split",
        ".fit(",
        "compare_models",
        "build_hist_gradient_boosting",
        "build_artifact_bundle",
        "create_default_artifact",
        "joblib",
    ):
        assert forbidden not in source


def test_app_source_never_runs_global_shap_or_persists_inputs_externally():
    source = APP_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "generate_evidence",
        "select_global_sample",
        "build_aggregate_background",
        "global_importance",
        "to_csv",
        "write_text",
        "to_json",
        "sqlite",
        "analytics",
    ):
        assert forbidden not in source
    assert "session_state" in source


def test_explainer_cache_is_keyed_by_artifact_hash(monkeypatch):
    module = import_app("streamlit_app_explainer_cache_key")
    calls = []

    def fake_runtime_loader(_bundle, *, expected_artifact_sha256):
        calls.append(expected_artifact_sha256)
        return object()

    monkeypatch.setattr(
        explainability, "load_runtime_explainer", fake_runtime_loader
    )
    st.cache_resource.clear()
    first_hash = "a" * 64
    second_hash = "b" * 64

    first = module.load_shap_explainer({}, first_hash)
    repeated = module.load_shap_explainer({}, first_hash)
    second = module.load_shap_explainer({}, second_hash)

    assert first is repeated
    assert second is not first
    assert calls == [first_hash, second_hash]


def test_local_chart_has_two_directions_full_labels_and_zero_rule():
    module = import_app("streamlit_app_local_chart_contract")
    display = pd.DataFrame(
        {
            "feature_label": [
                "General health (self-rated)",
                "Serious difficulty walking or climbing stairs",
            ],
            "display_value": ["Poor", "No"],
            "contribution": [0.12, -0.08],
        }
    )

    chart, spec = module.build_local_contribution_chart(display)

    assert set(chart["Direction"]) == {"Increased", "Decreased"}
    assert chart["Factor"].str.contains(
        "Serious difficulty walking or climbing stairs", regex=False
    ).any()
    bar, zero_rule = spec["layer"]
    color = bar["encoding"]["color"]
    assert color["scale"]["domain"] == ["Increased", "Decreased"]
    assert len(set(color["scale"]["range"])) == 2
    assert bar["encoding"]["x"]["stack"] is None
    x_domain = bar["encoding"]["x"]["scale"]["domain"]
    assert x_domain[0] <= chart[
        "Model-estimate change (percentage points)"
    ].min()
    assert x_domain[1] >= chart[
        "Model-estimate change (percentage points)"
    ].max()
    assert x_domain[0] <= 0 <= x_domain[1]
    assert x_domain[0] < x_domain[1]
    assert bar["encoding"]["y"]["axis"]["labelLimit"] >= 400
    assert zero_rule["mark"]["type"] == "rule"
    assert zero_rule["encoding"]["x"]["datum"] == 0


def test_currency_labels_escape_streamlit_inline_math_delimiters():
    module = import_app("streamlit_app_currency_markdown")

    assert module.escape_markdown_dollar_signs("$35,000 to $49,999") == (
        r"\$35,000 to \$49,999"
    )
    assert module.escape_markdown_dollar_signs("Less than $10,000") == (
        r"Less than \$10,000"
    )
    assert module.escape_markdown_dollar_signs("Poor") == "Poor"


def test_app_shows_disclaimer_and_uses_the_artifact_helpers():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "load_artifact" in source
    assert "predict_risk_probability" in source
    lowered = source.lower()
    assert "not a diagnosis" in lowered
    assert "educational" in lowered
    assert "medical advice" in lowered


def test_app_wording_tracks_the_conditional_probability_contract():
    module = import_app("streamlit_app_probability_contract")
    splits = make_splits()
    uncalibrated = artifacts.build_artifact_bundle(splits)

    uncalibrated_caption = module.probability_contract_caption(uncalibrated)
    uncalibrated_disclaimer = module.medical_disclaimer(uncalibrated)
    assert "Uncalibrated" in uncalibrated_caption
    assert "D-018 retained" in uncalibrated_caption
    assert "uncalibrated statistical estimate" in uncalibrated_disclaimer

    calibrated = {
        "model": uncalibrated["model"],
        "calibrator": calibration.fit_final_calibrator(
            uncalibrated["model"], calibration.to_calibration_data(splits), "sigmoid"
        ),
        "metadata": dict(uncalibrated["metadata"]),
    }
    calibrated["metadata"]["calibration_method"] = "sigmoid"

    calibrated_caption = module.probability_contract_caption(calibrated)
    calibrated_disclaimer = module.medical_disclaimer(calibrated)
    assert "post-hoc sigmoid calibration" in calibrated_caption
    assert "Uncalibrated" not in calibrated_caption
    assert "post-hoc sigmoid-calibrated" in calibrated_disclaimer
    assert "uncalibrated" not in calibrated_disclaimer
    for disclaimer in (uncalibrated_disclaimer, calibrated_disclaimer):
        assert "not a diagnosis" in disclaimer
        assert "medical advice" in disclaimer


def test_artifact_loading_is_local_only():
    # Durable policy from the ML analysis plan (independent of the P6/P7
    # phase boundary): the app and the artifact helpers only load artifacts
    # produced by this project's offline pipeline -- never from remote
    # sources -- and read no deployment secrets. D-013 resolved distribution
    # with a controlled Git exception (the official artifact ships in the
    # repository), so local-only loading holds in deployment as well.
    app_source = APP_PATH.read_text(encoding="utf-8")
    artifacts_source = inspect.getsource(artifacts)
    for source in (app_source, artifacts_source):
        for forbidden in ("requests", "urllib", "http://", "https://", "st.secrets", "boto3"):
            assert forbidden not in source


# ---------------------------------------------------------------------------
# Headless render/predict workflow via Streamlit's AppTest harness
# ---------------------------------------------------------------------------


def test_app_renders_and_predicts_headlessly(tmp_artifact):
    from streamlit.testing.v1 import AppTest

    # Isolate the cached artifact loader from any previous in-process state.
    st.cache_resource.clear()

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.warning) >= 1  # disclaimer visible before any prediction
    assert any("uncalibrated" in warning.value.lower() for warning in app.warning)
    assert len(app.button) == 1  # the single form submit button
    assert len(app.selectbox) == 4  # Sex plus three coded ordinal scales
    assert len(app.number_input) == 4  # Exact age plus three numeric measures
    assert app.number_input[0].label == "Age"
    assert app.number_input[0].min == 18
    assert app.number_input[0].max == 120
    for selectbox in app.selectbox[1:]:
        assert all(
            not option.split(" ", 1)[0].rstrip("-").isdigit()
            for option in selectbox.options
        )

    app.button[0].set_value(True).run()

    assert not app.exception
    assert len(app.metric) == 5
    displayed = app.metric[0].value
    assert displayed.endswith("%")
    probability = float(displayed.rstrip("%")) / 100
    assert 0.0 <= probability <= 1.0
    assert any("40-44 model age group" in caption.value for caption in app.caption)
    assert any(
        subheader.value == "How the model interprets this estimate"
        for subheader in app.subheader
    )
    assert any(
        subheader.value == "Model scenario explorer"
        for subheader in app.subheader
    )
    metrics_by_label = {metric.label: metric.value for metric in app.metric}
    assert metrics_by_label["Original estimate"] == displayed
    assert metrics_by_label["Hypothetical scenario"] == displayed
    assert metrics_by_label["Difference"] == "0.00 pp"
    assert len(app.button) == 2
    assert app.button[1].label == "Reset to original"
    rendered_text = " ".join(
        element.value for element in [*app.markdown, *app.text]
    ).lower()
    assert "increased the model estimate" in rendered_text
    assert "decreased the model estimate" in rendered_text
    assert len(app.warning) >= 1  # disclaimer visible alongside the prediction


def test_explanation_is_hidden_until_a_valid_submission(tmp_artifact):
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    assert not any(
        subheader.value == "How the model interprets this estimate"
        for subheader in app.subheader
    )
    assert not any(
        subheader.value == "Model scenario explorer"
        for subheader in app.subheader
    )
    app.button[0].set_value(True).run()
    assert any(
        subheader.value == "How the model interprets this estimate"
        for subheader in app.subheader
    )
    assert any(
        subheader.value == "Model scenario explorer"
        for subheader in app.subheader
    )


def test_app_keeps_probability_visible_when_explainer_fails(
    tmp_artifact, tmp_path, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        explainability, "BACKGROUND_ASSET_PATH", tmp_path / "missing-background.json"
    )
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()

    assert not app.exception
    assert len(app.metric) == 4
    assert app.metric[0].value.endswith("%")
    assert any(
        "explanation is temporarily unavailable" in warning.value.lower()
        for warning in app.warning
    )
    assert any("medical disclaimer" in warning.value.lower() for warning in app.warning)


def test_app_explanation_wording_is_non_causal_and_probability_only():
    source = APP_PATH.read_text(encoding="utf-8").lower()
    for forbidden_claim in (
        "causes diabetes",
        "prevents diabetes",
        "will reduce your risk",
        "recommended intervention",
        "high-risk label",
        "low-risk label",
    ):
        assert forbidden_claim not in source


def test_scenario_language_is_neutral_for_positive_negative_and_zero_deltas():
    module = import_app("streamlit_app_scenario_language")

    assert module.format_scenario_delta(1.234) == "+1.23 pp"
    assert module.format_scenario_delta(-1.234) == "-1.23 pp"
    assert module.format_scenario_delta(0.0) == "0.00 pp"
    statements = [
        module.scenario_delta_statement(delta)
        for delta in (1.234, -1.234, 0.0)
    ]
    assert all(
        statement.startswith("The model estimate changed by")
        for statement in statements
    )
    source = APP_PATH.read_text(encoding="utf-8").lower()
    for forbidden_claim in (
        "reduce your risk",
        "improve your health",
        "best scenario",
        "you should",
        "this change will prevent diabetes",
        "recommended",
        "high risk",
        "low risk",
        "high-risk",
        "low-risk",
    ):
        assert forbidden_claim not in source


def test_scenario_ui_exposes_only_the_d023_whitelist(tmp_artifact):
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()

    selector = next(
        item for item in app.selectbox if item.label == "Input to vary"
    )
    assert selector.options == [
        "No input change",
        "Physical activity in the past 30 days",
        "Eats fruit at least once per day",
        "Eats vegetables at least once per day",
    ]
    source = APP_PATH.read_text(encoding="utf-8")
    assert "EDITABLE_SCENARIO_FEATURES" in source
    assert "predict_risk_probability(bundle, hypothetical" not in source


def test_scenario_reset_restores_the_exact_original_comparison(tmp_artifact):
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()
    original_display = app.metric[0].value

    feature_selector = next(
        item for item in app.selectbox if item.label == "Input to vary"
    )
    feature_selector.select("PhysActivity").run()
    value_selector = next(
        item for item in app.selectbox if item.label == "Hypothetical value"
    )
    value_selector.select(1).run()
    app.button[1].click().run()

    assert not app.exception
    metrics_by_label = {metric.label: metric.value for metric in app.metric}
    assert metrics_by_label["Original estimate"] == original_display
    assert metrics_by_label["Hypothetical scenario"] == original_display
    assert metrics_by_label["Difference"] == "0.00 pp"
    feature_selector = next(
        item for item in app.selectbox if item.label == "Input to vary"
    )
    assert feature_selector.value is None
    assert any(
        "all 21 original inputs are preserved" in caption.value
        for caption in app.caption
    )


def test_new_valid_submission_resets_scenario_widgets_and_comparison(tmp_artifact):
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()

    feature_selector = next(
        item for item in app.selectbox if item.label == "Input to vary"
    )
    feature_selector.select("PhysActivity").run()
    value_selector = next(
        item for item in app.selectbox if item.label == "Hypothetical value"
    )
    value_selector.select(1).run()
    app.button[0].set_value(True).run()

    assert not app.exception
    feature_selector = next(
        item for item in app.selectbox if item.label == "Input to vary"
    )
    assert feature_selector.value is None
    assert not any(
        item.label == "Hypothetical value" for item in app.selectbox
    )
    metrics_by_label = {metric.label: metric.value for metric in app.metric}
    assert metrics_by_label["Original estimate"] == (
        metrics_by_label["Hypothetical scenario"]
    )
    assert metrics_by_label["Difference"] == "0.00 pp"


def test_failed_new_prediction_clears_the_previous_result(
    tmp_artifact, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()
    assert app.metric

    def fail_prediction(*args, **kwargs):
        raise ValueError("controlled original scoring failure")

    monkeypatch.setattr(
        artifacts, "predict_risk_probability", fail_prediction
    )
    app.button[0].set_value(True).run()

    assert not app.exception
    assert any(
        "could not score this case" in error.value.lower()
        for error in app.error
    )
    assert not app.metric
    assert not any(
        subheader.value == "How the model interprets this estimate"
        for subheader in app.subheader
    )
    assert not any(
        subheader.value == "Model scenario explorer"
        for subheader in app.subheader
    )
    assert any("medical disclaimer" in warning.value.lower() for warning in app.warning)


def test_artifact_hash_change_invalidates_the_saved_result(
    tmp_artifact, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()
    assert app.metric

    monkeypatch.setattr(
        explainability,
        "artifact_sha256",
        lambda path: "f" * 64,
    )
    app.run()

    assert not app.exception
    assert not app.metric
    assert not any(
        subheader.value == "How the model interprets this estimate"
        for subheader in app.subheader
    )
    assert not any(
        subheader.value == "Model scenario explorer"
        for subheader in app.subheader
    )
    assert any(
        "artifact changed during this session" in info.value.lower()
        for info in app.info
    )
    assert any("medical disclaimer" in warning.value.lower() for warning in app.warning)


def test_scenario_failure_preserves_original_and_p9_explanation(
    tmp_artifact, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    def fail_scenario(*args, **kwargs):
        raise ValueError("controlled scenario failure")

    monkeypatch.setattr(scenarios, "compare_scenario", fail_scenario)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.button[0].set_value(True).run()

    assert not app.exception
    assert app.metric[0].label == (
        "Model-estimated probability of diabetes or prediabetes"
    )
    assert app.metric[0].value.endswith("%")
    assert any(
        subheader.value == "How the model interprets this estimate"
        for subheader in app.subheader
    )
    assert any(
        "hypothetical scenario is temporarily unavailable" in warning.value.lower()
        for warning in app.warning
    )
    assert any("medical disclaimer" in warning.value.lower() for warning in app.warning)


def test_scenario_ui_has_no_search_ranking_presets_or_scenario_shap():
    source = APP_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "optimize",
        "optimal scenario",
        "scenario ranking",
        "rank scenarios",
        "recommended preset",
        "scenario preset",
        "explain_scenario",
    ):
        assert forbidden not in source
    assert source.count("render_local_explanation(") == 2


def test_app_shows_clear_error_when_artifact_is_missing(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        artifacts, "DEFAULT_ARTIFACT_PATH", tmp_path / "absent.joblib"
    )
    st.cache_resource.clear()

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.error) == 1
    assert "python -m src.artifacts" in app.error[0].value
    assert len(app.metric) == 0


def test_app_shows_clear_error_when_artifact_is_corrupt(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"")
    monkeypatch.setattr(artifacts, "DEFAULT_ARTIFACT_PATH", corrupt)
    st.cache_resource.clear()

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.error) == 1
    assert "python -m src.artifacts" in app.error[0].value
    assert len(app.metric) == 0
