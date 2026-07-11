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

import pytest
import streamlit as st

from src import artifacts, data
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
    bundle = artifacts.build_artifact_bundle(make_splits())
    path = artifacts.save_artifact(bundle, tmp_path / "diabetes_risk_model.joblib")
    monkeypatch.setattr(artifacts, "DEFAULT_ARTIFACT_PATH", path)
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


def test_app_shows_disclaimer_and_uses_the_artifact_helpers():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "load_artifact" in source
    assert "predict_risk_probability" in source
    lowered = source.lower()
    assert "not a diagnosis" in lowered
    assert "educational" in lowered
    assert "medical advice" in lowered


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
    assert len(app.metric) == 1
    displayed = app.metric[0].value
    assert displayed.endswith("%")
    probability = float(displayed.rstrip("%")) / 100
    assert 0.0 <= probability <= 1.0
    assert any("40-44 model age group" in caption.value for caption in app.caption)
    assert len(app.warning) >= 1  # disclaimer visible alongside the prediction


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
