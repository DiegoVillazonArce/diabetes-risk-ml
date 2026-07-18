"""P12 fairness-audit contract tests.

Synthetic fixtures exercise cohort semantics, aggregate formulas, unavailable
states, common bins, directional gaps, and deterministic bootstrap behavior.
Real-data integrations validate only frozen support/provenance and official
serving isolation; they never select P12 configuration from test results.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src import artifacts, data, fairness
from src.calibration import FROZEN_THRESHOLD_SCENARIOS
from tests.reference_profiles import REFERENCE_PROFILES
from tests.test_data import requires_raw_data


def complete_feature_fixture() -> pd.DataFrame:
    """Cover every Sex/Age/Income code while preserving contract order."""
    base = artifacts.example_input()
    rows = []
    for sex in (0, 1):
        for age in range(1, 14):
            for income in range(1, 9):
                row = dict(base)
                row.update({"Sex": sex, "Age": age, "Income": income})
                rows.append(row)
    return pd.DataFrame(rows, columns=data.FEATURE_COLUMNS).astype("uint8")


def balanced_labels(features: pd.DataFrame) -> np.ndarray:
    # Every Sex, age band, income code, and Sex x Age cell has both classes.
    return (
        (features["Age"].to_numpy() + features["Income"].to_numpy()) % 2
    ).astype(float)


def balanced_probabilities(labels: np.ndarray) -> np.ndarray:
    positions = np.arange(len(labels))
    return np.where(labels == 1.0, 0.65 + (positions % 5) * 0.02, 0.05 + (positions % 5) * 0.02)


@pytest.fixture(scope="module")
def complete_fixture():
    features = complete_feature_fixture()
    labels = balanced_labels(features)
    probabilities = balanced_probabilities(labels)
    return features, labels, probabilities


@pytest.fixture(scope="module")
def prepared_splits():
    splits, _ = data.prepare_data()
    return splits


def four_row_features() -> pd.DataFrame:
    base = artifacts.example_input()
    rows = []
    for sex, age, income in ((0, 1, 1), (0, 1, 1), (1, 7, 8), (1, 7, 8)):
        row = dict(base)
        row.update({"Sex": sex, "Age": age, "Income": income})
        rows.append(row)
    return pd.DataFrame(rows, columns=data.FEATURE_COLUMNS).astype("uint8")


ZERO_FLOOR = fairness.SupportRule(min_rows=0, min_positives=0, min_negatives=0)


# ---------------------------------------------------------------------------
# Cohort contract and support
# ---------------------------------------------------------------------------


def test_cohort_keys_labels_and_order_are_exact(complete_fixture):
    features, _, _ = complete_fixture

    cohorts = fairness.cohort_slices(features, include_whole=True)

    assert [cohort.cohort_axis for cohort in cohorts[:3]] == ["whole", "sex", "sex"]
    assert [(cohort.group_key, cohort.group_label) for cohort in cohorts[1:3]] == [
        ("sex_0", "Female"),
        ("sex_1", "Male"),
    ]
    assert [(cohort.group_key, cohort.group_label) for cohort in cohorts[3:7]] == [
        ("age_18_49", "18-49"),
        ("age_50_64", "50-64"),
        ("age_65_74", "65-74"),
        ("age_75_plus", "75+"),
    ]
    assert [cohort.group_key for cohort in cohorts[7:15]] == [
        f"income_{code}" for code in range(1, 9)
    ]
    assert [cohort.group_key for cohort in cohorts[15:]] == [
        f"sex_{sex}__age_{age}"
        for sex in (0, 1)
        for age in ("18_49", "50_64", "65_74", "75_plus")
    ]


def test_cohort_assignments_are_exhaustive_and_mutually_exclusive(complete_fixture):
    features, _, _ = complete_fixture
    cohorts = fairness.cohort_slices(features, include_whole=False)

    for axis in ("sex", "age", "income", "sex_x_age"):
        masks = np.vstack([c.mask for c in cohorts if c.cohort_axis == axis])
        assert np.all(masks.sum(axis=0) == 1)


def test_age_mapping_uses_exact_predeclared_codes(complete_fixture):
    features, _, _ = complete_fixture
    cohorts = {c.group_key: c for c in fairness.cohort_slices(features)}

    assert set(features.loc[cohorts["age_18_49"].mask, "Age"]) == set(range(1, 7))
    assert set(features.loc[cohorts["age_50_64"].mask, "Age"]) == {7, 8, 9}
    assert set(features.loc[cohorts["age_65_74"].mask, "Age"]) == {10, 11}
    assert set(features.loc[cohorts["age_75_plus"].mask, "Age"]) == {12, 13}


@pytest.mark.parametrize(
    ("rows", "positives", "negatives", "expected"),
    [
        (500, 100, 400, True),
        (499, 100, 399, False),
        (500, 99, 401, False),
        (500, 400, 99, False),
    ],
)
def test_support_floor_is_exact(rows, positives, negatives, expected):
    assert fairness.SupportRule().is_supported(rows, positives, negatives) is expected


def test_calibration_support_schema_excludes_performance_metrics(complete_fixture):
    features, labels, _ = complete_fixture
    frame = fairness.support_table(features, labels, include_whole=False)

    assert list(frame.columns) == [
        "cohort_axis",
        "group_key",
        "group_label",
        "row_count",
        "positive_count",
        "negative_count",
        "prevalence",
        "meets_support_floor",
    ]
    assert not ({"brier_score", "roc_auc", "pr_auc", "mean_probability"} & set(frame))


# ---------------------------------------------------------------------------
# Point metrics, unavailable states, gaps, thresholds, reliability
# ---------------------------------------------------------------------------


def test_manual_probability_formulas_and_calibration_gap_sign():
    features = four_row_features()
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])

    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    whole = audit.probability_metrics[
        audit.probability_metrics["group_key"].eq("whole")
    ].set_index("metric")

    assert whole.loc["prevalence", "value"] == pytest.approx(0.5)
    assert whole.loc["mean_probability", "value"] == pytest.approx(0.5)
    assert whole.loc["brier_score", "value"] == pytest.approx(0.05)
    expected_log = -np.mean(
        labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)
    )
    assert whole.loc["log_loss", "value"] == pytest.approx(expected_log)
    assert whole.loc["roc_auc", "value"] == pytest.approx(1.0)
    assert whole.loc["pr_auc", "value"] == pytest.approx(1.0)
    assert whole.loc["calibration_gap", "value"] == pytest.approx(0.0)

    positive_gap = fairness._point_probability_values(
        np.array([0.0, 1.0]), np.array([0.4, 0.8]), True
    )
    negative_gap = fairness._point_probability_values(
        np.array([0.0, 1.0]), np.array([0.1, 0.5]), True
    )
    assert positive_gap["calibration_gap"][1] > 0
    assert negative_gap["calibration_gap"][1] < 0


def test_group_gap_direction_is_group_minus_whole():
    features = four_row_features()
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    gap = audit.metric_gaps[
        audit.metric_gaps["group_key"].eq("sex_0")
        & audit.metric_gaps["metric"].eq("mean_probability")
    ].iloc[0]

    assert gap.gap_direction == "group_minus_whole_cohort"
    assert gap.gap == pytest.approx(0.2 - 0.5)


def test_unsupported_group_keeps_support_and_prevalence_only():
    features = four_row_features()
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])

    audit = fairness.audit_point_estimates(features, labels, probabilities)
    group = audit.probability_metrics[
        audit.probability_metrics["group_key"].eq("sex_0")
    ].set_index("metric")

    assert group.loc["prevalence", "status"] == "available"
    for metric in fairness.PROBABILITY_METRICS[1:]:
        assert group.loc[metric, "status"] == "unavailable"
        assert group.loc[metric, "unavailable_reason"] == "support_floor_not_met"
        assert pd.isna(group.loc[metric, "value"])


def test_one_class_metrics_requiring_both_classes_are_unavailable():
    values = fairness._point_probability_values(
        np.ones(4), np.array([0.2, 0.4, 0.6, 0.8]), True
    )

    assert values["mean_probability"][0] == "available"
    assert values["brier_score"][0] == "available"
    assert values["roc_auc"] == (
        "unavailable",
        None,
        "both_classes_required",
    )
    assert values["pr_auc"] == (
        "unavailable",
        None,
        "both_classes_required",
    )


def test_threshold_confusion_counts_and_normalized_metrics():
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.2, 0.6, 0.4, 0.9])

    values = fairness._point_threshold_values(labels, probabilities, 0.5, True)

    assert values["recall"][1] == pytest.approx(0.5)
    assert values["precision"][1] == pytest.approx(0.5)
    assert values["false_positive_rate"][1] == pytest.approx(0.5)
    assert values["false_positive_count"][1] == 1
    assert values["false_negative_count"][1] == 1


def test_threshold_output_uses_all_four_frozen_scenarios_in_order(complete_fixture):
    features, labels, probabilities = complete_fixture
    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    whole = audit.threshold_metrics[audit.threshold_metrics["group_key"].eq("whole")]

    assert list(dict.fromkeys(whole["scenario"])) == list(FROZEN_THRESHOLD_SCENARIOS)
    assert list(dict.fromkeys(whole["threshold"])) == list(
        FROZEN_THRESHOLD_SCENARIOS.values()
    )


def test_reliability_uses_common_bins_keeps_empty_bins_and_includes_one():
    features = four_row_features()
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.0, 0.05, 0.5, 1.0])
    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    whole = audit.reliability[audit.reliability["group_key"].eq("whole")]

    assert len(whole) == 10
    assert list(whole["bin_label"]) == list(fairness.RELIABILITY_BIN_LABELS)
    assert whole.iloc[-1]["row_count"] == 1
    empty = whole[whole["row_count"].eq(0)]
    assert not empty.empty
    assert set(empty["status"]) == {"unavailable"}
    assert set(empty["unavailable_reason"]) == {"empty_bin"}


def test_official_output_contract_contains_no_excluded_metric_name(complete_fixture):
    features, labels, probabilities = complete_fixture
    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    payload = b"\n".join(
        fairness.dataframe_csv_bytes(frame)
        for frame in (
            audit.support,
            audit.probability_metrics,
            audit.metric_gaps,
            audit.reliability,
            audit.threshold_metrics,
        )
    ).lower()

    assert b"ece" not in payload


# ---------------------------------------------------------------------------
# Bootstrap correctness and reproducibility
# ---------------------------------------------------------------------------


def test_weighted_bootstrap_ranking_matches_sklearn(complete_fixture):
    features, labels, probabilities = complete_fixture
    n_resamples = 12
    samples = fairness.bootstrap_metric_samples(
        features,
        labels,
        probabilities,
        n_resamples=n_resamples,
        random_state=data.RANDOM_SEED,
        batch_size=5,
        support_rule=fairness.SupportRule(1, 1, 1),
    )
    rng = np.random.default_rng(data.RANDOM_SEED)
    expected_roc = []
    expected_pr = []
    for _ in range(n_resamples):
        positions = rng.integers(0, len(labels), size=len(labels), dtype=np.int32)
        expected_roc.append(roc_auc_score(labels[positions], probabilities[positions]))
        expected_pr.append(
            average_precision_score(labels[positions], probabilities[positions])
        )

    np.testing.assert_allclose(
        samples[("whole", "probability", "", "roc_auc")], expected_roc
    )
    np.testing.assert_allclose(
        samples[("whole", "probability", "", "pr_auc")], expected_pr
    )


def test_bootstrap_is_reproducible_and_batch_partition_independent(complete_fixture):
    features, labels, probabilities = complete_fixture
    kwargs = dict(
        n_resamples=40,
        random_state=data.RANDOM_SEED,
        support_rule=fairness.SupportRule(1, 1, 1),
    )

    first = fairness.bootstrap_intervals(
        features, labels, probabilities, batch_size=7, **kwargs
    )
    second = fairness.bootstrap_intervals(
        features, labels, probabilities, batch_size=40, **kwargs
    )

    assert fairness.dataframe_csv_bytes(first) == fairness.dataframe_csv_bytes(second)


def test_changing_seed_changes_intervals_not_point_estimates(complete_fixture):
    features, labels, probabilities = complete_fixture
    kwargs = dict(
        n_resamples=60,
        batch_size=10,
        support_rule=fairness.SupportRule(1, 1, 1),
    )

    first = fairness.bootstrap_intervals(
        features, labels, probabilities, random_state=42, **kwargs
    )
    shifted = fairness.bootstrap_intervals(
        features, labels, probabilities, random_state=43, **kwargs
    )

    np.testing.assert_allclose(first["point_estimate"], shifted["point_estimate"])
    assert not np.array_equal(first["ci_lower"], shifted["ci_lower"])


def test_exactly_5000_resamples_are_executed(complete_fixture):
    features, labels, probabilities = complete_fixture

    intervals = fairness.bootstrap_intervals(
        features,
        labels,
        probabilities,
        n_resamples=5_000,
        batch_size=5_000,
        support_rule=fairness.SupportRule(1, 1, 1),
    )

    assert set(intervals["n_resamples"]) == {5_000}
    available = intervals[intervals["status"].eq("available")]
    assert available["valid_resamples"].between(1, 5_000).all()
    assert 5_000 in set(available["valid_resamples"])


def test_count_metrics_receive_no_intervals_or_gaps(complete_fixture):
    features, labels, probabilities = complete_fixture
    intervals = fairness.bootstrap_intervals(
        features,
        labels,
        probabilities,
        n_resamples=10,
        batch_size=5,
        support_rule=fairness.SupportRule(1, 1, 1),
    )

    assert not (set(fairness.THRESHOLD_COUNT_METRICS) & set(intervals["metric"]))


# ---------------------------------------------------------------------------
# Evidence, frozen serving, isolation, privacy, and source guards
# ---------------------------------------------------------------------------


@requires_raw_data
def test_calibration_support_is_byte_identical_and_matches_prepare_data(
    tmp_path, prepared_splits
):
    splits = prepared_splits
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    first = fairness.write_calibration_support(splits, first_path)
    fairness.write_calibration_support(splits, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes() == fairness.CALIBRATION_SUPPORT_PATH.read_bytes()
    expected = fairness.calibration_support_table(splits)
    pd.testing.assert_frame_equal(first, expected)
    assert len(splits.calibration) == 25_368
    assert len(splits.test) == 50_736 == 2 * len(splits.calibration)
    assert first["meets_candidate_floor"].all()


@requires_raw_data
def test_calibration_support_gate_accepts_exact_file_read_only(prepared_splits):
    before = fairness.CALIBRATION_SUPPORT_PATH.read_bytes()

    observed = fairness._load_and_validate_calibration_support(prepared_splits)

    assert fairness.CALIBRATION_SUPPORT_PATH.read_bytes() == before
    assert list(observed.columns) == list(fairness.CALIBRATION_SUPPORT_COLUMNS)
    assert len(observed) == fairness.CALIBRATION_SUPPORT_COHORTS == 22
    assert observed["meets_candidate_floor"].all()


@requires_raw_data
def test_calibration_support_gate_rejects_missing_file(
    tmp_path, monkeypatch, prepared_splits
):
    monkeypatch.setattr(
        fairness, "CALIBRATION_SUPPORT_PATH", tmp_path / "missing-support.csv"
    )

    with pytest.raises(RuntimeError, match="D-029 calibration support evidence is missing"):
        fairness._load_and_validate_calibration_support(prepared_splits)


@requires_raw_data
def test_calibration_support_gate_rejects_changed_byte(
    tmp_path, monkeypatch, prepared_splits
):
    expected = fairness.dataframe_csv_bytes(
        fairness.calibration_support_table(prepared_splits)
    )
    changed = expected.replace(b"1011", b"1012", 1)
    assert changed != expected
    path = tmp_path / "changed-support.csv"
    path.write_bytes(changed)
    monkeypatch.setattr(fairness, "CALIBRATION_SUPPORT_PATH", path)

    with pytest.raises(RuntimeError, match="was altered"):
        fairness._load_and_validate_calibration_support(prepared_splits)


@requires_raw_data
def test_calibration_support_gate_rejects_column_or_order_change(
    tmp_path, monkeypatch, prepared_splits
):
    frame = fairness.calibration_support_table(prepared_splits)
    reordered = frame[[frame.columns[1], frame.columns[0], *frame.columns[2:]]]
    path = tmp_path / "reordered-support.csv"
    path.write_bytes(fairness.dataframe_csv_bytes(reordered))
    monkeypatch.setattr(fairness, "CALIBRATION_SUPPORT_PATH", path)

    with pytest.raises(RuntimeError, match="was altered"):
        fairness._load_and_validate_calibration_support(prepared_splits)


@requires_raw_data
def test_missing_calibration_support_blocks_before_official_test_scoring(
    tmp_path, monkeypatch, prepared_splits
):
    score_called = False

    def fail_if_scored(*_args, **_kwargs):
        nonlocal score_called
        score_called = True
        raise AssertionError("Official test scoring must not run after a D-029 failure.")

    monkeypatch.setattr(fairness, "CALIBRATION_SUPPORT_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(fairness, "_assert_decisions_accepted", lambda: None)
    monkeypatch.setattr(
        fairness,
        "verify_frozen_artifacts",
        lambda: {
            "model": fairness.MODEL_ARTIFACT_SHA256,
            "shap_background": fairness.SHAP_BACKGROUND_SHA256,
        },
    )
    monkeypatch.setattr(
        fairness,
        "prepare_data",
        lambda random_state: (prepared_splits, {"n_rows": 253_680}),
    )
    monkeypatch.setattr(fairness, "score_official_test", fail_if_scored)

    with pytest.raises(RuntimeError, match="D-029 calibration support evidence is missing"):
        fairness.official_audit(evidence_dir=tmp_path / "evidence")
    assert score_called is False


def _benchmark_payload() -> dict:
    return json.loads(fairness.BOOTSTRAP_BENCHMARK_PATH.read_text(encoding="utf-8"))


def _install_benchmark_payload(tmp_path, monkeypatch, payload: dict) -> Path:
    path = tmp_path / "bootstrap_benchmark.json"
    path.write_bytes(fairness._json_bytes(payload))
    monkeypatch.setattr(fairness, "BOOTSTRAP_BENCHMARK_PATH", path)
    return path


def test_benchmark_gate_accepts_complete_evidence_read_only():
    before = fairness.BOOTSTRAP_BENCHMARK_PATH.read_bytes()

    observed = fairness._load_and_validate_benchmark()

    assert fairness.BOOTSTRAP_BENCHMARK_PATH.read_bytes() == before
    assert observed["artifacts"] == {
        "model_sha256": fairness.MODEL_ARTIFACT_SHA256,
        "shap_background_sha256": fairness.SHAP_BACKGROUND_SHA256,
    }


def test_benchmark_gate_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fairness, "BOOTSTRAP_BENCHMARK_PATH", tmp_path / "missing-benchmark.json"
    )

    with pytest.raises(RuntimeError, match="D-030 benchmark evidence is missing"):
        fairness._load_and_validate_benchmark()


def test_benchmark_gate_rejects_invalid_json(tmp_path, monkeypatch):
    path = tmp_path / "invalid-benchmark.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(fairness, "BOOTSTRAP_BENCHMARK_PATH", path)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        fairness._load_and_validate_benchmark()


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        (("schema_version",), 2),
        (("schema_version",), True),
        (("source_split",), "test"),
        (("source_rows",), 25_367),
        (("source_rows",), 25_368.0),
        (("bootstrap", "method"), "cluster bootstrap"),
        (("bootstrap", "n_resamples"), 4_999),
        (("bootstrap", "confidence"), 0.90),
        (("bootstrap", "interval"), "basic"),
        (("bootstrap", "random_seed"), 43),
        (("bootstrap", "batch_size"), 64),
        (("bootstrap", "rng_order"), "group-major"),
        (("project_operational_guardrails", "warm_runtime_seconds_max"), 601.0),
        (
            ("project_operational_guardrails", "incremental_python_memory_mib_max"),
            513.0,
        ),
        (("measurement", "interval_rows"), 854),
        (("measurement", "result_sha256"), "not-a-sha256"),
        (("measurement", "all_requested_resamples_recorded"), False),
        (("determinism", "fixed_seed"), 43),
    ],
)
def test_benchmark_gate_rejects_contract_mutations(
    tmp_path, monkeypatch, field_path, bad_value
):
    payload = copy.deepcopy(_benchmark_payload())
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = bad_value
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="D-030 benchmark evidence"):
        fairness._load_and_validate_benchmark()


def test_benchmark_gate_recomputes_runtime_pass(tmp_path, monkeypatch):
    payload = copy.deepcopy(_benchmark_payload())
    payload["measurement"]["warm_runtime_seconds"] = 601.0
    payload["measurement"]["passes"] = True
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="recomputed operational guardrails"):
        fairness._load_and_validate_benchmark()


def test_benchmark_gate_recomputes_memory_pass(tmp_path, monkeypatch):
    payload = copy.deepcopy(_benchmark_payload())
    payload["measurement"]["incremental_python_peak_bytes"] = 513 * 1024 * 1024
    payload["measurement"]["incremental_python_peak_mib"] = 513.0
    payload["measurement"]["passes"] = True
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="recomputed operational guardrails"):
        fairness._load_and_validate_benchmark()


def test_benchmark_gate_rejects_false_stored_pass_when_recomputed_true(
    tmp_path, monkeypatch
):
    payload = copy.deepcopy(_benchmark_payload())
    payload["measurement"]["passes"] = False
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="recomputed operational guardrails"):
        fairness._load_and_validate_benchmark()


@pytest.mark.parametrize("bad_value", [None, "fast", float("nan"), float("inf"), -1.0])
def test_benchmark_gate_rejects_invalid_runtime_values(
    tmp_path, monkeypatch, bad_value
):
    payload = copy.deepcopy(_benchmark_payload())
    payload["measurement"]["warm_runtime_seconds"] = bad_value
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="non-negative finite value"):
        fairness._load_and_validate_benchmark()


def test_benchmark_gate_rejects_incomplete_json(tmp_path, monkeypatch):
    payload = copy.deepcopy(_benchmark_payload())
    del payload["measurement"]["warm_runtime_seconds"]
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="D-030 benchmark evidence"):
        fairness._load_and_validate_benchmark()


def test_benchmark_gate_rejects_wrong_artifact_hash(tmp_path, monkeypatch):
    payload = copy.deepcopy(_benchmark_payload())
    payload["artifacts"]["model_sha256"] = "0" * 64
    _install_benchmark_payload(tmp_path, monkeypatch, payload)

    with pytest.raises(RuntimeError, match="frozen artifact hashes"):
        fairness._load_and_validate_benchmark()


@requires_raw_data
def test_invalid_benchmark_blocks_before_official_test_scoring(
    tmp_path, monkeypatch, prepared_splits
):
    payload = copy.deepcopy(_benchmark_payload())
    payload["measurement"]["warm_runtime_seconds"] = 601.0
    payload["measurement"]["passes"] = True
    _install_benchmark_payload(tmp_path, monkeypatch, payload)
    score_called = False

    def fail_if_scored(*_args, **_kwargs):
        nonlocal score_called
        score_called = True
        raise AssertionError("Official test scoring must not run after a D-030 failure.")

    monkeypatch.setattr(fairness, "_assert_decisions_accepted", lambda: None)
    monkeypatch.setattr(
        fairness,
        "verify_frozen_artifacts",
        lambda: {
            "model": fairness.MODEL_ARTIFACT_SHA256,
            "shap_background": fairness.SHAP_BACKGROUND_SHA256,
        },
    )
    monkeypatch.setattr(
        fairness,
        "prepare_data",
        lambda random_state: (prepared_splits, {"n_rows": 253_680}),
    )
    monkeypatch.setattr(
        fairness, "_load_and_validate_calibration_support", lambda _splits: None
    )
    monkeypatch.setattr(fairness, "score_official_test", fail_if_scored)

    with pytest.raises(RuntimeError, match="recomputed operational guardrails"):
        fairness.official_audit(evidence_dir=tmp_path / "evidence")
    assert score_called is False


@requires_raw_data
def test_train_and_calibration_poisoning_cannot_change_official_test_scoring(
    prepared_splits,
):
    splits = prepared_splits
    bundle = artifacts.load_artifact()
    poisoned = data.DataSplits(
        train=splits.train * np.nan,
        calibration=splits.calibration * np.nan,
        test=splits.test,
    )

    clean = fairness.score_official_test(bundle, splits)
    changed = fairness.score_official_test(bundle, poisoned)

    assert clean[0].equals(changed[0])
    assert np.array_equal(clean[1], changed[1])
    assert np.array_equal(clean[2], changed[2])


def test_test_values_cannot_select_cohorts_or_protocol(complete_fixture):
    features, labels, probabilities = complete_fixture
    original = fairness.cohort_slices(features)
    poisoned_probabilities = probabilities[::-1]
    poisoned_labels = 1.0 - labels

    changed = fairness.cohort_slices(features)
    fairness.validate_audit_inputs(features, poisoned_labels, poisoned_probabilities)

    assert [(c.cohort_axis, c.group_key, c.group_label) for c in original] == [
        (c.cohort_axis, c.group_key, c.group_label) for c in changed
    ]
    np.testing.assert_allclose(
        fairness.RELIABILITY_BIN_EDGES, [index / 10 for index in range(11)]
    )
    assert fairness.BOOTSTRAP_RESAMPLES == 5_000
    assert fairness.RANDOM_SEED == 42


def test_fairness_module_has_no_training_calibration_selection_or_streamlit_import():
    source = inspect.getsource(fairness)

    assert "import streamlit" not in source
    assert ".fit(" not in source
    for forbidden in (
        "train_frozen_base_model(",
        "fit_final_calibrator(",
        "select_calibration_method(",
        "select_threshold_scenarios(",
        "create_default_artifact(",
        "save_artifact(",
        "joblib.dump(",
    ):
        assert forbidden not in source


def test_report_distinguishes_fixed_bin_reliability_from_group_calibration_plot():
    source = inspect.getsource(fairness._build_report)

    assert "fixed-bin reliability data are published" in source
    assert "calibration-in-the-large" in source
    assert "observed prevalence with its mean served probability" in source


def test_published_table_columns_cannot_expose_rows_or_individual_outputs(complete_fixture):
    features, labels, probabilities = complete_fixture
    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    forbidden = set(data.FEATURE_COLUMNS) | {
        data.TARGET,
        "probability",
        "target",
        "row_index",
        "split_index",
        "shap_value",
    }

    for frame in (
        audit.support,
        audit.probability_metrics,
        audit.metric_gaps,
        audit.reliability,
        audit.threshold_metrics,
    ):
        assert not (forbidden & set(frame.columns))
        assert len(frame) < len(features) * 10


def test_frozen_artifact_hashes_and_reference_profiles_are_intact():
    observed = fairness.verify_frozen_artifacts()
    bundle = artifacts.load_artifact()
    references = fairness._reference_profile_contract(bundle)

    assert observed == {
        "model": fairness.MODEL_ARTIFACT_SHA256,
        "shap_background": fairness.SHAP_BACKGROUND_SHA256,
    }
    assert [row["probability"] for row in references] == [
        profile.expected_probability for profile in REFERENCE_PROFILES
    ]
    assert [row["display"] for row in references] == [
        "0.3%",
        "60.0%",
        "70.0%",
        "79.9%",
    ]


def test_aggregate_csv_serialization_is_utf8_lf_without_index(complete_fixture):
    features, labels, probabilities = complete_fixture
    audit = fairness.audit_point_estimates(
        features, labels, probabilities, support_rule=ZERO_FLOOR
    )
    payload = fairness.dataframe_csv_bytes(audit.support)

    payload.decode("utf-8")
    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")
    assert not payload.startswith(b",")
