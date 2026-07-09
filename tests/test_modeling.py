"""Tests for src/modeling.py: P4 baseline modeling (Dummy + Logistic Regression).

All tests run on small synthetic frames that satisfy the P3 data contract
(reusing `make_valid_df` from tests/test_data.py), so the suite stays fast and
does not require the raw CSV. Splits are always produced through
`src.data.split_data`, matching the P4 rule that modeling consumes the P3
split contract instead of creating its own splits.
"""

import inspect

import numpy as np
import pytest

from src import data, modeling
from tests.test_data import make_valid_df


def make_splits(n_rows: int = 300, positive_rate: float = 0.2, seed: int = 0) -> data.DataSplits:
    """Small stratified splits built through the P3 split contract."""
    df = make_valid_df(n_rows=n_rows, positive_rate=positive_rate, seed=seed)
    return data.split_data(df)


# ---------------------------------------------------------------------------
# DataSplits -> X/y conversion
# ---------------------------------------------------------------------------


def test_to_train_test_data_takes_rows_exactly_from_p3_splits():
    splits = make_splits()

    model_data = modeling.to_train_test_data(splits)

    assert list(model_data.X_train.columns) == data.FEATURE_COLUMNS
    assert data.TARGET not in model_data.X_train.columns
    assert list(model_data.X_train.index) == list(splits.train.index)
    assert list(model_data.X_test.index) == list(splits.test.index)
    assert model_data.y_train.equals(splits.train[data.TARGET])
    assert model_data.y_test.equals(splits.test[data.TARGET])


def test_to_train_test_data_excludes_calibration_split():
    # P4 must leave the calibration split untouched for later calibration work.
    splits = make_splits()

    model_data = modeling.to_train_test_data(splits)

    used_indices = set(model_data.X_train.index) | set(model_data.X_test.index)
    assert used_indices.isdisjoint(splits.calibration.index)
    assert not hasattr(model_data, "X_calibration")
    assert not hasattr(model_data, "y_calibration")


# ---------------------------------------------------------------------------
# Baseline builders: small-sample fit and evaluation
# ---------------------------------------------------------------------------


def test_dummy_baseline_fits_and_evaluates_on_small_sample():
    splits = make_splits()
    model_data = modeling.to_train_test_data(splits)

    model = modeling.build_dummy_baseline()
    model.fit(model_data.X_train, model_data.y_train)
    metrics = modeling.evaluate_model(model, model_data.X_test, model_data.y_test)

    assert set(metrics) == set(modeling.METRIC_KEYS)
    # most_frequent always predicts the negative majority class.
    assert metrics["recall"] == 0.0
    assert metrics["confusion_matrix"]["tp"] == 0
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["roc_auc"] == pytest.approx(0.5)
    # With constant scores, average precision collapses to the prevalence.
    assert metrics["pr_auc"] == pytest.approx(model_data.y_test.mean())


def test_logistic_regression_baseline_fits_and_evaluates_on_small_sample():
    splits = make_splits()
    model_data = modeling.to_train_test_data(splits)

    model = modeling.build_logistic_regression_baseline()
    model.fit(model_data.X_train, model_data.y_train)
    metrics = modeling.evaluate_model(model, model_data.X_test, model_data.y_test)

    assert set(metrics) == set(modeling.METRIC_KEYS)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


@pytest.mark.parametrize(
    "builder",
    [modeling.build_dummy_baseline, modeling.build_logistic_regression_baseline],
)
def test_predict_proba_returns_valid_probabilities(builder):
    splits = make_splits()
    model_data = modeling.to_train_test_data(splits)
    model = builder()
    model.fit(model_data.X_train, model_data.y_train)

    proba = modeling.predict_positive_proba(model, model_data.X_test)

    assert proba.shape == (len(model_data.X_test),)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_match_hand_computed_values():
    y_true = np.array([0, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.05])

    metrics = modeling.compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["roc_auc"] == pytest.approx(7 / 8)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)


def test_metrics_handle_degenerate_all_negative_predictions():
    y_true = np.array([0, 0, 1, 0])
    y_pred = np.zeros(4)
    y_proba = np.zeros(4)

    metrics = modeling.compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 1, "tp": 0}
    assert metrics["accuracy"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Orchestration: both baselines, train/test only, deterministic
# ---------------------------------------------------------------------------


def test_train_and_evaluate_baselines_returns_both_models():
    splits = make_splits()

    results = modeling.train_and_evaluate_baselines(splits)

    assert set(results) == {"dummy", "logistic_regression"}
    for result in results.values():
        assert set(result.train_metrics) == set(modeling.METRIC_KEYS)
        assert set(result.test_metrics) == set(modeling.METRIC_KEYS)
        train_cm = result.train_metrics["confusion_matrix"]
        test_cm = result.test_metrics["confusion_matrix"]
        assert sum(train_cm.values()) == len(splits.train)
        assert sum(test_cm.values()) == len(splits.test)


def test_train_and_evaluate_baselines_is_deterministic():
    splits = make_splits()

    first = modeling.train_and_evaluate_baselines(splits)
    second = modeling.train_and_evaluate_baselines(splits)

    for name in first:
        assert first[name].train_metrics == second[name].train_metrics
        assert first[name].test_metrics == second[name].test_metrics


def test_training_uses_only_train_rows(monkeypatch):
    splits = make_splits()
    fitted_indices = {}

    def spying(name, builder):
        def build(random_state=data.RANDOM_SEED):
            model = builder(random_state=random_state)
            original_fit = model.fit

            def recording_fit(X, y, **kwargs):
                fitted_indices[name] = list(X.index)
                return original_fit(X, y, **kwargs)

            model.fit = recording_fit
            return model

        return build

    monkeypatch.setattr(
        modeling, "build_dummy_baseline", spying("dummy", modeling.build_dummy_baseline)
    )
    monkeypatch.setattr(
        modeling,
        "build_logistic_regression_baseline",
        spying("logistic_regression", modeling.build_logistic_regression_baseline),
    )

    modeling.train_and_evaluate_baselines(splits)

    for name in ("dummy", "logistic_regression"):
        assert fitted_indices[name] == list(splits.train.index)


def test_calibration_split_is_not_consumed_in_p4():
    # Poison the calibration split: any fit/predict/metric touching it would
    # propagate NaN or raise, so a clean run proves it was never consumed.
    splits = make_splits()
    poisoned = data.DataSplits(
        train=splits.train,
        calibration=splits.calibration * np.nan,
        test=splits.test,
    )

    results = modeling.train_and_evaluate_baselines(poisoned)

    assert set(results) == {"dummy", "logistic_regression"}
    for result in results.values():
        assert not np.isnan(result.test_metrics["roc_auc"])


def test_modeling_module_does_not_reload_or_resplit_data():
    # P4 consumes the P3 split contract; it must never create new splits or
    # reload raw data ad hoc.
    source = inspect.getsource(modeling)
    for forbidden in ("train_test_split", "read_csv", "load_raw_data"):
        assert forbidden not in source
