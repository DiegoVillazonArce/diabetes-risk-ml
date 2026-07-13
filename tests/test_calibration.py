"""Tests for src/calibration.py: P8 calibration protocol (Epic E6).

All tests run on small synthetic frames that satisfy the P3 data contract
(reusing the fixtures from the earlier test modules), so the suite stays
fast and does not require the raw CSV. Splits always come from
`src.data.split_data`; P8 code must consume the P3 split contract without
re-splitting, reloading raw data, or writing artifacts, and the test split
must never reach the comparison, selection, or threshold code paths.
"""

import inspect

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from src import calibration, data
from tests.test_data import make_valid_df


def make_calibration_splits(
    n_rows: int = 1000, positive_rate: float = 0.2, seed: int = 0
) -> data.DataSplits:
    """Stratified splits sized so five-fold cross-fitting is well-posed.

    1000 rows give a 100-row calibration split with 20 positives: every
    fold's fit part keeps enough positives for the calibrators (and for the
    frozen-estimator interplay with scikit-learn's internal machinery)
    without slowing the suite down.
    """
    df = make_valid_df(n_rows=n_rows, positive_rate=positive_rate, seed=seed)
    return data.split_data(df)


@pytest.fixture(scope="module")
def frozen_setup():
    """One shared frozen base model and calibration data for read-only tests."""
    splits = make_calibration_splits()
    cal_data = calibration.to_calibration_data(splits)
    model = calibration.train_frozen_base_model(cal_data)
    return splits, cal_data, model


# ---------------------------------------------------------------------------
# CalibrationData conversion: train + calibration only, test unreachable
# ---------------------------------------------------------------------------


def test_to_calibration_data_takes_rows_exactly_from_p3_splits():
    splits = make_calibration_splits()

    cal_data = calibration.to_calibration_data(splits)

    assert list(cal_data.X_train.columns) == data.FEATURE_COLUMNS
    assert list(cal_data.X_calibration.columns) == data.FEATURE_COLUMNS
    assert data.TARGET not in cal_data.X_train.columns
    assert list(cal_data.X_train.index) == list(splits.train.index)
    assert list(cal_data.X_calibration.index) == list(splits.calibration.index)
    assert cal_data.y_train.equals(splits.train[data.TARGET])
    assert cal_data.y_calibration.equals(splits.calibration[data.TARGET])


def test_to_calibration_data_excludes_the_test_split():
    # Test participates in no P8 decision, so the P8 data structure must not
    # even carry it (mirroring how TrainTestData excludes calibration).
    splits = make_calibration_splits()

    cal_data = calibration.to_calibration_data(splits)

    used_indices = set(cal_data.X_train.index) | set(cal_data.X_calibration.index)
    assert used_indices.isdisjoint(splits.test.index)
    assert not hasattr(cal_data, "X_test")
    assert not hasattr(cal_data, "y_test")


def test_poisoned_test_split_does_not_affect_baseline_folds_or_data():
    # Poison the test split: any baseline, fold, or data helper touching it
    # would propagate NaN or raise, so identical results prove isolation.
    splits = make_calibration_splits()
    poisoned = data.DataSplits(
        train=splits.train,
        calibration=splits.calibration,
        test=splits.test * np.nan,
    )

    clean_data = calibration.to_calibration_data(splits)
    poisoned_data = calibration.to_calibration_data(poisoned)
    assert poisoned_data.X_train.equals(clean_data.X_train)
    assert poisoned_data.X_calibration.equals(clean_data.X_calibration)

    model = calibration.train_frozen_base_model(poisoned_data)
    evaluation = calibration.uncalibrated_baseline_evaluation(model, poisoned_data)
    assert all(
        np.isfinite(evaluation["metrics"][key])
        for key in calibration.PROBABILITY_METRIC_KEYS
    )

    clean_folds = calibration.stratified_calibration_folds(clean_data.y_calibration)
    poisoned_folds = calibration.stratified_calibration_folds(
        poisoned_data.y_calibration
    )
    for (clean_fit, clean_held), (poisoned_fit, poisoned_held) in zip(
        clean_folds, poisoned_folds
    ):
        assert np.array_equal(clean_fit, poisoned_fit)
        assert np.array_equal(clean_held, poisoned_held)


# ---------------------------------------------------------------------------
# Frozen base model: fits on exactly the train rows, calibration untouched
# ---------------------------------------------------------------------------


def test_base_model_fits_on_exactly_the_train_rows(monkeypatch):
    splits = make_calibration_splits()
    cal_data = calibration.to_calibration_data(splits)
    fitted_indices = {}
    original_builder = calibration.build_hist_gradient_boosting_candidate

    def spying_builder(random_state=data.RANDOM_SEED):
        model = original_builder(random_state=random_state)
        original_fit = model.fit

        def recording_fit(X, y, **kwargs):
            fitted_indices["rows"] = list(X.index)
            return original_fit(X, y, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(
        calibration, "build_hist_gradient_boosting_candidate", spying_builder
    )

    calibration.train_frozen_base_model(cal_data)

    assert fitted_indices["rows"] == list(splits.train.index)


def test_base_model_fit_does_not_consume_calibration_rows():
    # Poison the calibration split: the base model must fit on train only,
    # so training and train-row scoring still work.
    splits = make_calibration_splits()
    poisoned = data.DataSplits(
        train=splits.train,
        calibration=splits.calibration * np.nan,
        test=splits.test,
    )
    cal_data = calibration.to_calibration_data(poisoned)

    model = calibration.train_frozen_base_model(cal_data)

    from src.modeling import predict_positive_proba

    probabilities = predict_positive_proba(model, cal_data.X_train)
    assert np.all(np.isfinite(probabilities))


# ---------------------------------------------------------------------------
# Metric helpers: sklearn-consistent values, invalid probabilities rejected
# ---------------------------------------------------------------------------


def test_per_row_losses_match_sklearn_aggregates():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 25, dtype=float)
    probabilities = rng.uniform(0.0, 1.0, size=50)

    assert np.mean(
        calibration.brier_losses(y, probabilities)
    ) == pytest.approx(brier_score_loss(y, probabilities))
    assert np.mean(
        calibration.log_losses(y, probabilities)
    ) == pytest.approx(log_loss(y, probabilities))


def test_log_losses_stay_finite_for_exact_zero_and_one_probabilities():
    # Isotonic calibration can emit exact 0/1 probabilities; the sklearn
    # clipping convention keeps the per-row loss finite either way.
    y = np.array([1.0, 0.0, 1.0, 0.0])
    probabilities = np.array([0.0, 1.0, 1.0, 0.0])

    losses = calibration.log_losses(y, probabilities)

    assert np.all(np.isfinite(losses))
    assert np.mean(losses) == pytest.approx(log_loss(y, probabilities))


def test_probability_metrics_reports_the_p8_metric_set():
    y = np.array([0, 0, 0, 1, 1, 0], dtype=float)
    probabilities = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.05])

    metrics = calibration.probability_metrics(y, probabilities)

    assert set(metrics) == set(calibration.PROBABILITY_METRIC_KEYS)
    assert metrics["brier"] == pytest.approx(
        brier_score_loss(y, probabilities)
    )
    assert metrics["log_loss"] == pytest.approx(log_loss(y, probabilities))
    assert metrics["roc_auc"] == pytest.approx(7 / 8)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        np.array([0.2, np.nan, 0.4]),
        np.array([0.2, np.inf, 0.4]),
        np.array([-0.1, 0.5, 0.4]),
        np.array([0.2, 0.5, 1.1]),
        np.array([[0.2, 0.5], [0.3, 0.4]]),
        np.array([]),
    ],
)
def test_metrics_reject_invalid_probabilities(bad_probabilities):
    with pytest.raises(ValueError):
        calibration.validate_probabilities(bad_probabilities)
    if bad_probabilities.ndim == 1 and bad_probabilities.size == 3:
        y = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            calibration.probability_metrics(y, bad_probabilities)
        with pytest.raises(ValueError):
            calibration.brier_losses(y, bad_probabilities)
        with pytest.raises(ValueError):
            calibration.log_losses(y, bad_probabilities)


def test_metrics_reject_non_binary_or_single_class_targets():
    probabilities = np.array([0.2, 0.5, 0.4])
    with pytest.raises(ValueError, match="binary"):
        calibration.probability_metrics(np.array([0.0, 1.0, 2.0]), probabilities)
    with pytest.raises(ValueError, match="both classes"):
        calibration.probability_metrics(np.array([0.0, 0.0, 0.0]), probabilities)


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        calibration.brier_losses(np.array([0.0, 1.0]), np.array([0.2, 0.5, 0.4]))


# ---------------------------------------------------------------------------
# Reliability diagram and histogram evidence
# ---------------------------------------------------------------------------


def test_reliability_table_bins_predicted_and_observed_rates():
    y = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    probabilities = np.array([0.05, 0.15, 0.95, 0.85, 0.05, 0.95])

    table = calibration.reliability_table(y, probabilities, n_bins=10)

    assert len(table) == 10
    assert table["n_rows"].sum() == len(y)
    first = table.iloc[0]
    assert first["n_rows"] == 2
    assert first["mean_predicted_probability"] == pytest.approx(0.05)
    assert first["observed_positive_rate"] == 0.0
    last = table.iloc[9]
    assert last["n_rows"] == 2
    assert last["observed_positive_rate"] == 1.0
    empty = table.iloc[5]
    assert empty["n_rows"] == 0
    assert np.isnan(empty["mean_predicted_probability"])


def test_probability_histogram_counts_every_row_once():
    probabilities = np.array([0.0, 0.05, 0.5, 0.95, 1.0])

    table = calibration.probability_histogram_table(probabilities, n_bins=10)

    assert len(table) == 10
    assert table["n_rows"].sum() == len(probabilities)
    assert table["share"].sum() == pytest.approx(1.0)
    # Probability 1.0 belongs to the last bin, not an overflow bin.
    assert table.iloc[9]["n_rows"] == 2


def test_uncalibrated_baseline_evaluation_reports_the_recorded_evidence(
    frozen_setup,
):
    _, cal_data, model = frozen_setup

    evaluation = calibration.uncalibrated_baseline_evaluation(model, cal_data)

    assert set(evaluation) == {"metrics", "reliability", "histogram"}
    assert set(evaluation["metrics"]) == set(calibration.PROBABILITY_METRIC_KEYS)
    assert evaluation["reliability"]["n_rows"].sum() == len(cal_data.X_calibration)
    assert evaluation["histogram"]["n_rows"].sum() == len(cal_data.X_calibration)


def test_uncalibrated_baseline_evaluation_is_reproducible(frozen_setup):
    _, cal_data, model = frozen_setup

    first = calibration.uncalibrated_baseline_evaluation(model, cal_data)
    second = calibration.uncalibrated_baseline_evaluation(model, cal_data)

    assert first["metrics"] == second["metrics"]
    assert first["reliability"].equals(second["reliability"])


# ---------------------------------------------------------------------------
# Stratified folds: deterministic, exhaustive, both classes everywhere
# ---------------------------------------------------------------------------


def test_folds_are_deterministic_with_the_fixed_seed(frozen_setup):
    _, cal_data, _ = frozen_setup

    first = calibration.stratified_calibration_folds(cal_data.y_calibration)
    second = calibration.stratified_calibration_folds(cal_data.y_calibration)

    assert len(first) == calibration.N_CALIBRATION_FOLDS
    for (fit_a, held_a), (fit_b, held_b) in zip(first, second):
        assert np.array_equal(fit_a, fit_b)
        assert np.array_equal(held_a, held_b)


def test_folds_change_with_a_different_seed(frozen_setup):
    _, cal_data, _ = frozen_setup

    default = calibration.stratified_calibration_folds(cal_data.y_calibration)
    shifted = calibration.stratified_calibration_folds(
        cal_data.y_calibration, random_state=data.RANDOM_SEED + 1
    )

    assert any(
        not np.array_equal(held_a, held_b)
        for (_, held_a), (_, held_b) in zip(default, shifted)
    )


def test_every_calibration_row_is_held_out_exactly_once(frozen_setup):
    _, cal_data, _ = frozen_setup

    folds = calibration.stratified_calibration_folds(cal_data.y_calibration)

    held_out = np.concatenate([held for _, held in folds])
    assert len(held_out) == len(cal_data.y_calibration)
    assert np.array_equal(np.sort(held_out), np.arange(len(cal_data.y_calibration)))
    for fit_positions, held_positions in folds:
        assert set(fit_positions).isdisjoint(held_positions)
        assert len(fit_positions) + len(held_positions) == len(
            cal_data.y_calibration
        )


def test_folds_require_both_classes_in_every_part():
    # Two positives cannot stratify into five folds with a positive in every
    # held-out part; the protocol must fail loudly instead of degrading.
    y = np.array([1.0, 1.0] + [0.0] * 48)

    with pytest.warns(UserWarning, match="least populated"):
        with pytest.raises(ValueError, match="both classes"):
            calibration.stratified_calibration_folds(y)


# ---------------------------------------------------------------------------
# Out-of-fold assembly integrity
# ---------------------------------------------------------------------------


def test_assemble_out_of_fold_recombines_fold_predictions():
    fold_predictions = [
        (np.array([0, 2]), np.array([0.1, 0.3])),
        (np.array([1, 3]), np.array([0.2, 0.4])),
    ]

    out_of_fold = calibration.assemble_out_of_fold(4, fold_predictions)

    assert out_of_fold == pytest.approx(np.array([0.1, 0.2, 0.3, 0.4]))


def test_assemble_out_of_fold_rejects_duplicate_rows():
    fold_predictions = [
        (np.array([0, 1]), np.array([0.1, 0.2])),
        (np.array([1, 2]), np.array([0.3, 0.4])),
    ]

    with pytest.raises(ValueError, match="more than one"):
        calibration.assemble_out_of_fold(3, fold_predictions)


def test_assemble_out_of_fold_rejects_missing_rows():
    fold_predictions = [(np.array([0, 1]), np.array([0.1, 0.2]))]

    with pytest.raises(ValueError, match="no\\s+out-of-fold"):
        calibration.assemble_out_of_fold(3, fold_predictions)


def test_assemble_out_of_fold_rejects_out_of_range_positions():
    fold_predictions = [(np.array([0, 3]), np.array([0.1, 0.2]))]

    with pytest.raises(ValueError, match="outside"):
        calibration.assemble_out_of_fold(3, fold_predictions)


def test_assemble_out_of_fold_rejects_invalid_probabilities():
    fold_predictions = [
        (np.array([0, 1]), np.array([0.1, np.nan])),
        (np.array([2]), np.array([0.3])),
    ]

    with pytest.raises(ValueError):
        calibration.assemble_out_of_fold(3, fold_predictions)


def test_assemble_out_of_fold_rejects_length_mismatch():
    fold_predictions = [(np.array([0, 1, 2]), np.array([0.1, 0.2]))]

    with pytest.raises(ValueError, match="differ in length"):
        calibration.assemble_out_of_fold(3, fold_predictions)


# ---------------------------------------------------------------------------
# Cross-fitting: calibrators fit on exactly their four folds, base frozen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", calibration.CALIBRATION_METHODS)
def test_per_fold_calibrators_fit_on_exactly_their_four_folds(
    frozen_setup, monkeypatch, method
):
    _, cal_data, model = frozen_setup
    folds = calibration.stratified_calibration_folds(cal_data.y_calibration)
    fitted_rows: list[list] = []
    original_builder = calibration.build_calibrator

    def spying_builder(base_model, method):
        calibrator = original_builder(base_model, method)
        original_fit = calibrator.fit

        def recording_fit(X, y, **kwargs):
            fitted_rows.append(list(X.index))
            return original_fit(X, y, **kwargs)

        calibrator.fit = recording_fit
        return calibrator

    monkeypatch.setattr(calibration, "build_calibrator", spying_builder)

    calibration.cross_fit_out_of_fold(model, cal_data, method)

    assert len(fitted_rows) == calibration.N_CALIBRATION_FOLDS
    calibration_index = list(cal_data.X_calibration.index)
    train_and_test_free = set(cal_data.X_train.index)
    for (fit_positions, held_out_positions), rows in zip(folds, fitted_rows):
        expected = [calibration_index[position] for position in fit_positions]
        assert rows == expected  # exactly the four assigned folds
        held_out = {calibration_index[position] for position in held_out_positions}
        assert held_out.isdisjoint(rows)  # never its held-out fold
        assert train_and_test_free.isdisjoint(rows)  # never train rows


def test_final_calibrator_fits_on_exactly_the_full_calibration_split(
    frozen_setup, monkeypatch
):
    _, cal_data, model = frozen_setup
    fitted_rows: list[list] = []
    original_builder = calibration.build_calibrator

    def spying_builder(base_model, method):
        calibrator = original_builder(base_model, method)
        original_fit = calibrator.fit

        def recording_fit(X, y, **kwargs):
            fitted_rows.append(list(X.index))
            return original_fit(X, y, **kwargs)

        calibrator.fit = recording_fit
        return calibrator

    monkeypatch.setattr(calibration, "build_calibrator", spying_builder)

    calibration.fit_final_calibrator(model, cal_data, "sigmoid")

    assert fitted_rows == [list(cal_data.X_calibration.index)]


@pytest.mark.parametrize("method", calibration.CALIBRATION_METHODS)
def test_cross_fitting_never_refits_the_frozen_base_model(frozen_setup, method):
    _, cal_data, model = frozen_setup

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("the frozen base model must never be refitted in P8")

    original_fit = model.fit
    model.fit = forbidden_fit
    try:
        out_of_fold = calibration.cross_fit_out_of_fold(model, cal_data, method)
        calibration.fit_final_calibrator(model, cal_data, method)
    finally:
        model.fit = original_fit

    assert len(out_of_fold) == len(cal_data.X_calibration)


def test_build_calibrator_rejects_unknown_methods(frozen_setup):
    _, _, model = frozen_setup

    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibration.build_calibrator(model, "temperature")


def test_build_calibrator_wraps_the_frozen_base_model(frozen_setup):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    _, _, model = frozen_setup

    calibrator = calibration.build_calibrator(model, "sigmoid")

    assert isinstance(calibrator, CalibratedClassifierCV)
    assert isinstance(calibrator.estimator, FrozenEstimator)
    assert calibrator.estimator.estimator is model
    assert calibrator.ensemble is False


@pytest.mark.parametrize("method", calibration.CALIBRATION_METHODS)
def test_out_of_fold_probabilities_are_valid_and_reproducible(
    frozen_setup, method
):
    _, cal_data, model = frozen_setup

    first = calibration.cross_fit_out_of_fold(model, cal_data, method)
    second = calibration.cross_fit_out_of_fold(model, cal_data, method)

    assert len(first) == len(cal_data.X_calibration)
    assert np.all(np.isfinite(first))
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert np.array_equal(first, second)


def test_out_of_fold_probabilities_covers_every_contract(frozen_setup):
    _, cal_data, model = frozen_setup

    probabilities = calibration.out_of_fold_probabilities(model, cal_data)

    assert set(probabilities) == {
        calibration.NO_CALIBRATION,
        *calibration.CALIBRATION_METHODS,
    }
    baseline = calibration.uncalibrated_calibration_probabilities(model, cal_data)
    assert np.array_equal(probabilities[calibration.NO_CALIBRATION], baseline)
    for method in calibration.CALIBRATION_METHODS:
        assert len(probabilities[method]) == len(cal_data.X_calibration)
        # Calibration must actually transform the scores.
        assert not np.array_equal(probabilities[method], baseline)


# ---------------------------------------------------------------------------
# Paired bootstrap: deterministic, correct convention, strict validation
# ---------------------------------------------------------------------------


def test_bootstrap_interval_on_constant_delta_is_exact():
    candidate = np.full(50, 0.05)
    reference = np.full(50, 0.30)

    interval = calibration.paired_bootstrap_interval(candidate, reference)

    assert interval.mean_delta == pytest.approx(-0.25)
    assert interval.ci_lower == pytest.approx(-0.25)
    assert interval.ci_upper == pytest.approx(-0.25)
    assert interval.improves_reference
    assert interval.n_resamples == calibration.BOOTSTRAP_RESAMPLES
    assert interval.confidence == calibration.BOOTSTRAP_CONFIDENCE


def test_bootstrap_interval_is_deterministic_and_seed_sensitive():
    rng = np.random.default_rng(1)
    candidate = rng.uniform(0.0, 0.5, size=200)
    reference = rng.uniform(0.0, 0.5, size=200)

    first = calibration.paired_bootstrap_interval(candidate, reference)
    second = calibration.paired_bootstrap_interval(candidate, reference)
    shifted = calibration.paired_bootstrap_interval(
        candidate, reference, random_state=data.RANDOM_SEED + 1
    )

    assert first == second
    assert (first.ci_lower, first.ci_upper) != (shifted.ci_lower, shifted.ci_upper)


def test_bootstrap_interval_batching_does_not_change_the_stream():
    # The batch size only bounds memory; the fixed seed must produce the
    # identical resample stream for any batch partitioning.
    rng = np.random.default_rng(2)
    candidate = rng.uniform(0.0, 0.5, size=100)
    reference = rng.uniform(0.0, 0.5, size=100)

    default = calibration.paired_bootstrap_interval(candidate, reference)
    single_batch = calibration.paired_bootstrap_interval(
        candidate, reference, batch_size=calibration.BOOTSTRAP_RESAMPLES
    )

    assert default == single_batch


def test_bootstrap_interval_straddles_zero_for_symmetric_noise():
    rng = np.random.default_rng(3)
    reference = rng.uniform(0.2, 0.4, size=500)
    noise = rng.normal(0.0, 0.01, size=500)

    interval = calibration.paired_bootstrap_interval(reference + noise, reference)

    assert interval.ci_lower < 0.0 < interval.ci_upper
    assert not interval.improves_reference
    assert not interval.excludes_zero


def test_bootstrap_interval_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="identical shapes"):
        calibration.paired_bootstrap_interval(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="non-finite"):
        calibration.paired_bootstrap_interval(
            np.array([0.1, np.nan]), np.array([0.1, 0.2])
        )
    with pytest.raises(ValueError, match="non-empty"):
        calibration.paired_bootstrap_interval(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# D-018 selection rules on hand-crafted evidence
# ---------------------------------------------------------------------------

# Hand-crafted contracts over 200 alternating-class rows. The baseline ranks
# perfectly (ROC-AUC 1.0) but is deliberately under-confident (Brier 0.16),
# so candidates can beat it on Brier while ranking behavior stays exact.
HAND_Y = np.tile([1.0, 0.0], 100)


def hand_probabilities(positive: float, negative: float) -> np.ndarray:
    return np.where(HAND_Y == 1.0, positive, negative)


HAND_BASELINE = hand_probabilities(0.6, 0.4)  # Brier 0.16, ROC-AUC 1.0


def hand_degraded_ranking() -> np.ndarray:
    # Near-perfect probabilities (Brier ~0.026) with two positive/negative
    # pairs flipped: ROC-AUC drops to 0.98, violating the 0.005 guard.
    values = hand_probabilities(0.9, 0.1)
    values[[0, 2]] = 0.1  # two positives pushed to the bottom
    values[[1, 3]] = 0.9  # two negatives pushed to the top
    return values


def test_selection_picks_the_clearly_better_method():
    selection = calibration.select_calibration_method(
        HAND_Y,
        {
            "none": HAND_BASELINE,
            "sigmoid": hand_probabilities(0.9, 0.1),  # Brier 0.01
            "isotonic": hand_probabilities(0.7, 0.3),  # Brier 0.09
        },
    )

    assert selection.selected_method == "sigmoid"
    assert selection.adoptable == {"sigmoid": True, "isotonic": True}
    assert selection.pairwise_brier is not None
    assert selection.pairwise_brier.ci_upper < 0.0
    assert selection.pairwise_log_loss is None
    assert selection.ranking_guard["sigmoid"]["passes"]


def test_selection_picks_isotonic_when_it_clearly_wins():
    selection = calibration.select_calibration_method(
        HAND_Y,
        {
            "none": HAND_BASELINE,
            "sigmoid": hand_probabilities(0.7, 0.3),
            "isotonic": hand_probabilities(0.9, 0.1),
        },
    )

    assert selection.selected_method == "isotonic"
    assert selection.pairwise_brier.ci_lower > 0.0


def test_selection_prefers_sigmoid_on_demonstrated_equivalence():
    equivalent = hand_probabilities(0.8, 0.2)

    selection = calibration.select_calibration_method(
        HAND_Y,
        {"none": HAND_BASELINE, "sigmoid": equivalent, "isotonic": equivalent},
    )

    assert selection.selected_method == "sigmoid"
    assert selection.pairwise_brier is not None
    assert not selection.pairwise_brier.excludes_zero
    assert selection.pairwise_log_loss is not None
    assert not selection.pairwise_log_loss.excludes_zero
    assert "equivalent" in selection.rationale


def test_selection_returns_none_when_no_method_is_adoptable():
    selection = calibration.select_calibration_method(
        HAND_Y,
        {
            "none": HAND_BASELINE,
            "sigmoid": HAND_BASELINE.copy(),
            "isotonic": hand_probabilities(0.5, 0.5 - 1e-9),
        },
    )

    assert selection.selected_method == calibration.NO_CALIBRATION
    assert selection.adoptable == {"sigmoid": False, "isotonic": False}
    assert selection.pairwise_brier is None


def test_selection_takes_the_single_adoptable_method():
    selection = calibration.select_calibration_method(
        HAND_Y,
        {
            "none": HAND_BASELINE,
            "sigmoid": HAND_BASELINE.copy(),  # never adoptable (identical)
            "isotonic": hand_probabilities(0.8, 0.2),
        },
    )

    assert selection.selected_method == "isotonic"
    assert selection.adoptable == {"sigmoid": False, "isotonic": True}
    assert selection.pairwise_brier is None


def test_ranking_guard_disqualifies_and_falls_back_to_the_other_method():
    selection = calibration.select_calibration_method(
        HAND_Y,
        {
            "none": HAND_BASELINE,
            # Best Brier but degraded ranking: preferred, then disqualified.
            "sigmoid": hand_degraded_ranking(),
            "isotonic": hand_probabilities(0.7, 0.3),
        },
    )

    assert selection.selected_method == "isotonic"
    assert not selection.ranking_guard["sigmoid"]["passes"]
    assert selection.ranking_guard["sigmoid"]["roc_auc_drop"] > (
        calibration.RANKING_REGRESSION_LIMIT
    )
    assert selection.ranking_guard["isotonic"]["passes"]
    assert "ranking-preservation guard" in selection.rationale


def test_ranking_guard_returns_none_when_no_alternative_qualifies():
    selection = calibration.select_calibration_method(
        HAND_Y,
        {
            "none": HAND_BASELINE,
            "sigmoid": hand_degraded_ranking(),
            "isotonic": HAND_BASELINE.copy(),  # not adoptable
        },
    )

    assert selection.selected_method == calibration.NO_CALIBRATION


def test_selection_rejects_incomplete_or_mismatched_contracts():
    with pytest.raises(ValueError, match="exactly"):
        calibration.select_calibration_method(
            HAND_Y, {"none": HAND_BASELINE, "sigmoid": HAND_BASELINE}
        )
    with pytest.raises(ValueError, match="calibration rows"):
        calibration.select_calibration_method(
            HAND_Y,
            {
                "none": HAND_BASELINE,
                "sigmoid": HAND_BASELINE,
                "isotonic": HAND_BASELINE[:-1],
            },
        )


def test_selection_is_deterministic_end_to_end(frozen_setup):
    _, cal_data, model = frozen_setup

    first_probabilities = calibration.out_of_fold_probabilities(model, cal_data)
    second_probabilities = calibration.out_of_fold_probabilities(model, cal_data)
    first = calibration.select_calibration_method(
        cal_data.y_calibration, first_probabilities
    )
    second = calibration.select_calibration_method(
        cal_data.y_calibration, second_probabilities
    )

    assert first.selected_method == second.selected_method
    assert first.metrics == second.metrics
    assert first.brier_vs_none == second.brier_vs_none
    assert first.rationale == second.rationale


def test_poisoned_test_split_does_not_change_oof_selection_or_thresholds():
    # Test rows must never reach the comparison, selection, or threshold
    # code paths: poisoning the test split has to leave all of them
    # entirely unchanged.
    splits = make_calibration_splits()
    poisoned = data.DataSplits(
        train=splits.train,
        calibration=splits.calibration,
        test=splits.test * np.nan,
    )

    clean_data = calibration.to_calibration_data(splits)
    poisoned_data = calibration.to_calibration_data(poisoned)
    clean_model = calibration.train_frozen_base_model(clean_data)
    poisoned_model = calibration.train_frozen_base_model(poisoned_data)

    clean_probabilities = calibration.out_of_fold_probabilities(
        clean_model, clean_data
    )
    poisoned_probabilities = calibration.out_of_fold_probabilities(
        poisoned_model, poisoned_data
    )
    for contract in clean_probabilities:
        assert np.array_equal(
            clean_probabilities[contract], poisoned_probabilities[contract]
        )

    clean_selection = calibration.select_calibration_method(
        clean_data.y_calibration, clean_probabilities
    )
    poisoned_selection = calibration.select_calibration_method(
        poisoned_data.y_calibration, poisoned_probabilities
    )
    assert clean_selection == poisoned_selection

    selected = clean_probabilities[clean_selection.selected_method]
    clean_table = calibration.threshold_table(
        clean_data.y_calibration, selected
    )
    poisoned_table = calibration.threshold_table(
        poisoned_data.y_calibration,
        poisoned_probabilities[poisoned_selection.selected_method],
    )
    assert clean_table.equals(poisoned_table)
    assert calibration.select_threshold_scenarios(
        clean_data.y_calibration, selected
    ) == calibration.select_threshold_scenarios(
        poisoned_data.y_calibration, selected
    )


# ---------------------------------------------------------------------------
# Threshold analysis (US-0606): tables, scenario rules, strict validation
# ---------------------------------------------------------------------------


def test_threshold_metrics_match_hand_computed_values():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.2, 0.6, 0.4, 0.9])

    metrics = calibration.threshold_metrics(y, probabilities, 0.5)

    assert metrics == {
        "threshold": 0.5,
        "recall": 0.5,
        "precision": 0.5,
        "f1": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }


def test_threshold_metrics_handle_degenerate_cells():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    probabilities = np.array([0.1, 0.2, 0.1, 0.2])

    metrics = calibration.threshold_metrics(y, probabilities, 0.9)

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["tp"] == 0 and metrics["fp"] == 0
    assert metrics["fn"] == 2 and metrics["tn"] == 2


def test_threshold_metrics_reject_invalid_thresholds():
    y = np.array([0.0, 1.0])
    probabilities = np.array([0.1, 0.9])

    for invalid in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="Threshold"):
            calibration.threshold_metrics(y, probabilities, invalid)


def test_threshold_table_covers_the_grid_with_monotone_recall():
    y = HAND_Y
    probabilities = hand_probabilities(0.9, 0.1)

    table = calibration.threshold_table(y, probabilities)

    assert len(table) == len(calibration.THRESHOLD_GRID)
    assert list(table["threshold"]) == list(calibration.THRESHOLD_GRID)
    assert (table["recall"].diff().dropna() <= 0).all()
    assert ((table[["tp", "fp", "fn", "tn"]].sum(axis=1)) == len(y)).all()


def test_precision_recall_points_are_valid():
    y = HAND_Y
    probabilities = hand_probabilities(0.9, 0.1)

    points = calibration.precision_recall_points(y, probabilities)

    assert {"precision", "recall", "threshold"} == set(points.columns)
    assert points["precision"].between(0, 1).all()
    assert points["recall"].between(0, 1).all()


def test_scenario_rules_pick_the_documented_thresholds():
    # 0.9/0.1 probabilities: every grid threshold in (0.1, 0.9] classifies
    # perfectly, so max_f1 resolves to the lowest such threshold (0.11) and
    # both recall floors resolve to the highest (0.90).
    y = HAND_Y
    probabilities = hand_probabilities(0.9, 0.1)

    scenarios = calibration.select_threshold_scenarios(y, probabilities)

    assert set(scenarios) == set(calibration.THRESHOLD_SCENARIO_RULES)
    assert scenarios["default_half"]["threshold"] == pytest.approx(0.5)
    assert scenarios["max_f1"]["threshold"] == pytest.approx(0.11)
    assert scenarios["max_f1"]["f1"] == pytest.approx(1.0)
    assert scenarios["recall_floor_050"]["threshold"] == pytest.approx(0.9)
    assert scenarios["recall_floor_075"]["threshold"] == pytest.approx(0.9)
    for metrics in scenarios.values():
        assert metrics["scenario"] in calibration.THRESHOLD_SCENARIO_RULES
        assert isinstance(metrics["tp"], int)


def test_scenario_rules_fail_loudly_when_a_recall_floor_is_unreachable():
    y = HAND_Y
    probabilities = np.full(len(y), 0.005)  # recall 0 at every grid threshold

    with pytest.raises(ValueError, match="recall"):
        calibration.select_threshold_scenarios(y, probabilities)


def test_scenario_selection_is_deterministic(frozen_setup):
    _, cal_data, model = frozen_setup
    probabilities = calibration.uncalibrated_calibration_probabilities(
        model, cal_data
    )

    first = calibration.select_threshold_scenarios(
        cal_data.y_calibration, probabilities
    )
    second = calibration.select_threshold_scenarios(
        cal_data.y_calibration, probabilities
    )

    assert first == second


# ---------------------------------------------------------------------------
# Official P8 test evaluation: gated on frozen scenarios, single test consumer
# ---------------------------------------------------------------------------


def test_official_evaluation_requires_frozen_scenarios(frozen_setup):
    splits, _, model = frozen_setup

    with pytest.raises(ValueError, match="frozen"):
        calibration.official_test_evaluation(splits, model, None, {})


def test_official_evaluation_reports_the_frozen_contract_on_test(frozen_setup):
    splits, _, model = frozen_setup
    scenarios = {"default_half": 0.5, "max_f1": 0.2}

    evaluation = calibration.official_test_evaluation(
        splits, model, None, scenarios
    )

    assert set(evaluation) == {
        "contract_metrics",
        "uncalibrated_reference_metrics",
        "reliability",
        "scenario_metrics",
    }
    # With D-018 = none the served contract IS the uncalibrated model.
    assert evaluation["contract_metrics"] == (
        evaluation["uncalibrated_reference_metrics"]
    )
    from src.modeling import predict_positive_proba

    expected = calibration.probability_metrics(
        splits.test[data.TARGET],
        predict_positive_proba(model, splits.test[data.FEATURE_COLUMNS]),
    )
    assert evaluation["contract_metrics"] == expected
    assert set(evaluation["scenario_metrics"]) == set(scenarios)
    for name, threshold in scenarios.items():
        assert evaluation["scenario_metrics"][name]["threshold"] == (
            pytest.approx(threshold)
        )


def test_selection_cannot_be_recomputed_from_test_rows(frozen_setup):
    # The selection API is structurally calibration-only: it validates row
    # counts against the targets it was given, and the P8 data structure
    # carries no test rows at all. Feeding test-sized probability vectors
    # into a calibration-sized selection must fail loudly.
    splits, cal_data, model = frozen_setup
    from src.modeling import predict_positive_proba

    test_probabilities = predict_positive_proba(
        model, splits.test[data.FEATURE_COLUMNS]
    )

    with pytest.raises(ValueError, match="calibration rows"):
        calibration.select_calibration_method(
            cal_data.y_calibration,
            {
                "none": test_probabilities,
                "sigmoid": test_probabilities,
                "isotonic": test_probabilities,
            },
        )


# ---------------------------------------------------------------------------
# Module scope guards
# ---------------------------------------------------------------------------


def test_calibration_module_does_not_reload_resplit_or_serialize():
    # P8 consumes the P3 split contract and keeps results in memory: no
    # raw-data reload, no new splits of the dataset, and no artifact writes.
    source = inspect.getsource(calibration)
    for forbidden in (
        "train_test_split",
        "read_csv",
        "load_raw_data",
        "joblib",
        "pickle",
        "to_parquet",
    ):
        assert forbidden not in source
