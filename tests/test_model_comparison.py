"""Tests for P5 model comparison and selection in src/modeling.py (Epic E8).

All tests run on small synthetic frames that satisfy the P3 data contract
(reusing the fixtures from the P3/P4 test modules), so the suite stays fast
and does not require the raw CSV. Splits always come from
`src.data.split_data`; P5 code must consume the P3 split contract without
re-splitting, reloading raw data, touching the calibration split, or writing
model artifacts.
"""

import inspect

import numpy as np
import pytest

from src import data, modeling
from tests.test_modeling import make_splits

EXPECTED_COMPARISON_MODELS = {
    "dummy",
    "logistic_regression",
    "logistic_regression_balanced",
    "hist_gradient_boosting",
}


@pytest.fixture(scope="module")
def comparison_results():
    """One shared comparison run on small synthetic splits."""
    return modeling.compare_models(make_splits())


def fake_result(
    name: str,
    *,
    test_pr_auc: float,
    test_f1: float = 0.5,
    test_recall: float = 0.5,
    test_precision: float = 0.5,
    test_roc_auc: float = 0.5,
    test_accuracy: float = 0.5,
    train_pr_auc: float | None = None,
) -> modeling.BaselineResult:
    """Hand-crafted result for exercising the selection logic in isolation."""

    def metrics(pr_auc: float) -> dict:
        return {
            "roc_auc": test_roc_auc,
            "pr_auc": pr_auc,
            "recall": test_recall,
            "precision": test_precision,
            "f1": test_f1,
            "confusion_matrix": {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
            "accuracy": test_accuracy,
        }

    if train_pr_auc is None:
        train_pr_auc = test_pr_auc
    return modeling.BaselineResult(
        name=name,
        model=None,
        train_metrics=metrics(train_pr_auc),
        test_metrics=metrics(test_pr_auc),
    )


# ---------------------------------------------------------------------------
# Tree-based candidate and imbalance-aware variant: fit and predict_proba
# ---------------------------------------------------------------------------


def test_hist_gradient_boosting_fits_and_evaluates_on_small_sample():
    splits = make_splits()
    model_data = modeling.to_train_test_data(splits)

    model = modeling.build_hist_gradient_boosting_candidate()
    model.fit(model_data.X_train, model_data.y_train)
    metrics = modeling.evaluate_model(model, model_data.X_test, model_data.y_test)

    assert set(metrics) == set(modeling.METRIC_KEYS)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert sum(metrics["confusion_matrix"].values()) == len(model_data.X_test)


@pytest.mark.parametrize(
    "builder",
    [
        modeling.build_hist_gradient_boosting_candidate,
        modeling.build_logistic_regression_balanced_variant,
    ],
)
def test_p5_candidates_predict_valid_probabilities(builder):
    splits = make_splits()
    model_data = modeling.to_train_test_data(splits)
    model = builder()
    model.fit(model_data.X_train, model_data.y_train)

    proba = modeling.predict_positive_proba(model, model_data.X_test)

    assert proba.shape == (len(model_data.X_test),)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


# ---------------------------------------------------------------------------
# Comparison output: all models, metrics by model and split
# ---------------------------------------------------------------------------


def test_compare_models_includes_all_expected_models_and_metrics(comparison_results):
    assert set(comparison_results) == EXPECTED_COMPARISON_MODELS
    for result in comparison_results.values():
        assert set(result.train_metrics) == set(modeling.METRIC_KEYS)
        assert set(result.test_metrics) == set(modeling.METRIC_KEYS)


def test_comparison_table_reports_metrics_by_model_and_split(comparison_results):
    table = modeling.comparison_table(comparison_results)

    assert set(table["model"]) == EXPECTED_COMPARISON_MODELS
    assert set(table["split"]) == {"train", "test"}
    assert len(table) == 2 * len(EXPECTED_COMPARISON_MODELS)

    scalar_metrics = [key for key in modeling.METRIC_KEYS if key != "confusion_matrix"]
    for column in scalar_metrics + ["tn", "fp", "fn", "tp"]:
        assert column in table.columns

    row = table[
        (table["model"] == "hist_gradient_boosting") & (table["split"] == "test")
    ].iloc[0]
    expected = comparison_results["hist_gradient_boosting"].test_metrics
    assert row["pr_auc"] == expected["pr_auc"]
    assert row["tp"] == expected["confusion_matrix"]["tp"]


# ---------------------------------------------------------------------------
# Deterministic selection logic
# ---------------------------------------------------------------------------


def test_compare_and_select_are_deterministic():
    splits = make_splits()

    first = modeling.compare_models(splits)
    second = modeling.compare_models(splits)

    for name in first:
        assert first[name].train_metrics == second[name].train_metrics
        assert first[name].test_metrics == second[name].test_metrics

    first_selection = modeling.select_primary_model(first)
    second_selection = modeling.select_primary_model(second)
    assert first_selection.name == second_selection.name
    assert first_selection.ranking == second_selection.ranking
    assert first_selection.criteria == second_selection.criteria


def test_selection_prefers_highest_test_pr_auc_and_never_picks_dummy():
    results = {
        # Best metrics everywhere, but reference models are never selectable.
        "dummy": fake_result("dummy", test_pr_auc=0.99),
        "weaker": fake_result("weaker", test_pr_auc=0.40),
        "stronger": fake_result("stronger", test_pr_auc=0.45),
    }

    selection = modeling.select_primary_model(results)

    assert selection.name == "stronger"
    assert selection.ranking == ("stronger", "weaker")
    assert "dummy" not in selection.ranking
    assert "stronger" in selection.rationale


def test_selection_ignores_accuracy():
    results = {
        "high_accuracy": fake_result("high_accuracy", test_pr_auc=0.30, test_accuracy=0.99),
        "high_pr_auc": fake_result("high_pr_auc", test_pr_auc=0.45, test_accuracy=0.50),
    }

    assert modeling.select_primary_model(results).name == "high_pr_auc"


def test_selection_breaks_pr_auc_ties_with_f1_then_name():
    by_f1 = {
        "worse_f1": fake_result("worse_f1", test_pr_auc=0.40, test_f1=0.30),
        "better_f1": fake_result("better_f1", test_pr_auc=0.40, test_f1=0.50),
    }
    assert modeling.select_primary_model(by_f1).name == "better_f1"

    exact_tie = {
        "b_model": fake_result("b_model", test_pr_auc=0.40),
        "a_model": fake_result("a_model", test_pr_auc=0.40),
    }
    assert modeling.select_primary_model(exact_tie).name == "a_model"


def test_selection_deprioritizes_obvious_overfitting():
    gap = modeling.OVERFITTING_PR_AUC_GAP_LIMIT + 0.01
    results = {
        "overfit": fake_result("overfit", test_pr_auc=0.45, train_pr_auc=0.45 + gap),
        "stable": fake_result("stable", test_pr_auc=0.40),
    }

    selection = modeling.select_primary_model(results)

    assert selection.name == "stable"
    assert selection.ranking == ("stable", "overfit")
    assert selection.criteria["overfit"]["overfitting_risk"] is True
    assert selection.criteria["stable"]["overfitting_risk"] is False


def test_selection_requires_a_non_reference_candidate():
    with pytest.raises(ValueError, match="only reference models"):
        modeling.select_primary_model({"dummy": fake_result("dummy", test_pr_auc=0.2)})


# ---------------------------------------------------------------------------
# Scope guards: train rows only, no calibration usage, no reload/re-split,
# no artifact writes
# ---------------------------------------------------------------------------


def test_compare_models_trains_only_on_train_rows(monkeypatch):
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

    for attribute, name in (
        ("build_dummy_baseline", "dummy"),
        ("build_logistic_regression_baseline", "logistic_regression"),
        ("build_logistic_regression_balanced_variant", "logistic_regression_balanced"),
        ("build_hist_gradient_boosting_candidate", "hist_gradient_boosting"),
    ):
        monkeypatch.setattr(modeling, attribute, spying(name, getattr(modeling, attribute)))

    modeling.compare_models(splits)

    assert set(fitted_indices) == EXPECTED_COMPARISON_MODELS
    for indices in fitted_indices.values():
        assert indices == list(splits.train.index)


def test_compare_models_does_not_consume_calibration_split():
    # Poison the calibration split: any fit/predict/metric touching it would
    # propagate NaN or raise, so a clean run proves it was never consumed.
    splits = make_splits()
    poisoned = data.DataSplits(
        train=splits.train,
        calibration=splits.calibration * np.nan,
        test=splits.test,
    )

    results = modeling.compare_models(poisoned)

    assert set(results) == EXPECTED_COMPARISON_MODELS
    for result in results.values():
        assert not np.isnan(result.test_metrics["roc_auc"])


def test_p5_code_does_not_reload_resplit_or_serialize():
    # P5 consumes the P3 split contract and keeps everything in memory: no
    # raw-data reload, no new splits, and no model artifact writes without an
    # explicit decision (US-0803, D-017).
    source = inspect.getsource(modeling)
    for forbidden in (
        "train_test_split",
        "read_csv",
        "load_raw_data",
        "joblib",
        "pickle",
        "dump",
        "to_csv",
        "to_parquet",
        "open(",
    ):
        assert forbidden not in source


def test_comparison_and_selection_write_no_model_artifacts():
    models_dir = data.PROJECT_ROOT / "models"
    processed_dir = data.PROJECT_ROOT / "data" / "processed"

    def listing(directory):
        return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []

    before = (listing(models_dir), listing(processed_dir))

    results = modeling.compare_models(make_splits())
    modeling.comparison_table(results)
    modeling.select_primary_model(results)

    assert (listing(models_dir), listing(processed_dir)) == before
