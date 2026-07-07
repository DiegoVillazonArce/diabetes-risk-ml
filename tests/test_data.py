"""Tests for src/data.py: loading, validation, downcasting, and splits (P3).

Most tests run on small synthetic frames that satisfy the documented data
contract, so the suite does not require the raw CSV. Tests marked with
`requires_raw_data` validate the real dataset and are skipped when the
git-ignored raw file is not present locally.
"""

import numpy as np
import pandas as pd
import pytest

from src import data

requires_raw_data = pytest.mark.skipif(
    not data.RAW_DATA_PATH.is_file(),
    reason="raw BRFSS CSV not present locally (see data/README.md)",
)


def make_valid_df(n_rows: int = 1000, positive_rate: float = 0.14, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic frame that satisfies the documented raw data contract.

    Mirrors the raw CSV: float64 dtypes, expected column order, whole-number
    values within the documented ranges, and an exact positive count so that
    stratified-split prevalence checks are meaningful.
    """
    rng = np.random.default_rng(seed)
    target = np.zeros(n_rows)
    target[: round(n_rows * positive_rate)] = 1
    rng.shuffle(target)

    columns = {data.TARGET: target}
    for feature in data.BINARY_FEATURES:
        columns[feature] = rng.integers(0, 2, n_rows)
    for feature in data.ORDINAL_FEATURES + data.NUMERIC_FEATURES:
        lower, upper = data.VALUE_RANGES[feature]
        columns[feature] = rng.integers(lower, upper + 1, n_rows)
    return pd.DataFrame(columns)[data.EXPECTED_COLUMNS].astype("float64")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_raw_data_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        data.load_raw_data(missing_path)


def test_load_raw_data_reads_csv(tmp_path):
    df = make_valid_df(n_rows=50)
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loaded = data.load_raw_data(csv_path)

    assert list(loaded.columns) == data.EXPECTED_COLUMNS
    assert len(loaded) == 50


# ---------------------------------------------------------------------------
# Schema and value validation
# ---------------------------------------------------------------------------


def test_validate_accepts_contract_compliant_data():
    data.validate_raw_data(make_valid_df())


def test_validate_rejects_missing_column():
    df = make_valid_df().drop(columns=["BMI"])
    with pytest.raises(ValueError, match="Missing: \\['BMI'\\]"):
        data.validate_raw_data(df)


def test_validate_rejects_unexpected_column():
    df = make_valid_df()
    df["ExtraColumn"] = 0.0
    with pytest.raises(ValueError, match="Unexpected: \\['ExtraColumn'\\]"):
        data.validate_raw_data(df)


def test_validate_rejects_wrong_column_order():
    df = make_valid_df()
    reordered = df[list(reversed(data.EXPECTED_COLUMNS))]
    with pytest.raises(ValueError, match="exact names and order"):
        data.validate_raw_data(reordered)


def test_validate_rejects_non_numeric_column():
    df = make_valid_df()
    df["Sex"] = df["Sex"].map({0.0: "female", 1.0: "male"})
    with pytest.raises(ValueError, match="Non-numeric columns.*Sex"):
        data.validate_raw_data(df)


def test_validate_rejects_missing_values():
    df = make_valid_df()
    df.loc[0, "BMI"] = np.nan
    with pytest.raises(ValueError, match="Missing values.*BMI"):
        data.validate_raw_data(df)


def test_validate_rejects_non_integer_values():
    df = make_valid_df()
    df.loc[0, "BMI"] = 25.5
    with pytest.raises(ValueError, match="Non-integer values.*BMI"):
        data.validate_raw_data(df)


def test_validate_rejects_invalid_target_values():
    df = make_valid_df()
    df.loc[0, data.TARGET] = 2.0
    with pytest.raises(ValueError, match=f"'{data.TARGET}'.*\\[0, 1\\]"):
        data.validate_raw_data(df)


def test_validate_rejects_out_of_range_binary_feature():
    df = make_valid_df()
    df.loc[0, "HighBP"] = 3.0
    with pytest.raises(ValueError, match="'HighBP'.*\\[0, 1\\]"):
        data.validate_raw_data(df)


def test_validate_rejects_out_of_range_ordinal_feature():
    df = make_valid_df()
    df.loc[0, "GenHlth"] = 6.0
    with pytest.raises(ValueError, match="'GenHlth'.*\\[1, 5\\]"):
        data.validate_raw_data(df)


def test_validate_rejects_out_of_range_numeric_feature():
    df = make_valid_df()
    df.loc[0, "BMI"] = 11.0
    with pytest.raises(ValueError, match="'BMI'.*\\[12, 98\\]"):
        data.validate_raw_data(df)


def test_validate_allows_duplicate_rows():
    # Exact duplicate rows are kept per D-014, so they must not fail validation.
    df = make_valid_df(n_rows=100)
    with_duplicates = pd.concat([df, df.head(20)], ignore_index=True)

    data.validate_raw_data(with_duplicates)


# ---------------------------------------------------------------------------
# Dataset summary (D-014 selected population)
# ---------------------------------------------------------------------------


def test_summarize_dataset_reports_rows_duplicates_and_prevalence():
    df = make_valid_df(n_rows=100, positive_rate=0.14)
    with_duplicates = pd.concat([df, df.head(20)], ignore_index=True)

    summary = data.summarize_dataset(with_duplicates)

    assert summary["n_rows"] == 120
    assert summary["n_duplicate_rows"] == 20
    assert summary["positive_prevalence"] == pytest.approx(
        with_duplicates[data.TARGET].mean()
    )


# ---------------------------------------------------------------------------
# Downcasting
# ---------------------------------------------------------------------------


def test_downcast_to_uint8_is_lossless_and_smaller():
    df = make_valid_df()

    downcast = data.downcast_to_uint8(df)

    assert (downcast.dtypes == "uint8").all()
    assert downcast.astype("float64").equals(df)
    assert downcast.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()


def test_downcast_rejects_fractional_values():
    df = make_valid_df()
    df.loc[0, "BMI"] = 25.5
    with pytest.raises(ValueError):
        data.downcast_to_uint8(df)


def test_downcast_rejects_values_outside_uint8_range():
    df = make_valid_df()
    df.loc[0, "BMI"] = 300.0
    with pytest.raises(ValueError):
        data.downcast_to_uint8(df)


# ---------------------------------------------------------------------------
# Stratified 70/10/20 split
# ---------------------------------------------------------------------------


def test_split_sizes_follow_70_10_20():
    df = make_valid_df(n_rows=1000)

    splits = data.split_data(df)

    assert len(splits.train) == 700
    assert len(splits.calibration) == 100
    assert len(splits.test) == 200


def test_split_is_a_partition_of_the_input():
    df = make_valid_df(n_rows=1000)

    splits = data.split_data(df)

    all_indices = (
        list(splits.train.index)
        + list(splits.calibration.index)
        + list(splits.test.index)
    )
    assert len(all_indices) == len(df)
    assert set(all_indices) == set(df.index)


def test_split_is_reproducible_with_fixed_seed():
    df = make_valid_df(n_rows=1000)

    first = data.split_data(df)
    second = data.split_data(df)

    assert first.train.equals(second.train)
    assert first.calibration.equals(second.calibration)
    assert first.test.equals(second.test)


def test_split_changes_with_different_seed():
    df = make_valid_df(n_rows=1000)

    first = data.split_data(df, random_state=data.RANDOM_SEED)
    second = data.split_data(df, random_state=data.RANDOM_SEED + 1)

    assert list(first.train.index) != list(second.train.index)


def test_split_preserves_positive_prevalence():
    df = make_valid_df(n_rows=1000, positive_rate=0.14)
    overall_prevalence = df[data.TARGET].mean()

    splits = data.split_data(df)

    for split in (splits.train, splits.calibration, splits.test):
        assert split[data.TARGET].mean() == pytest.approx(
            overall_prevalence, abs=0.01
        )


def test_split_keeps_duplicate_rows():
    # D-014: duplicates are part of the selected population and must survive
    # splitting; no rows are dropped anywhere in the pipeline.
    df = make_valid_df(n_rows=100)
    with_duplicates = pd.concat([df, df.head(20)], ignore_index=True)

    splits = data.split_data(with_duplicates)

    total = len(splits.train) + len(splits.calibration) + len(splits.test)
    assert total == 120


def test_split_requires_target_column():
    df = make_valid_df().drop(columns=[data.TARGET])
    with pytest.raises(ValueError, match=data.TARGET):
        data.split_data(df)


# ---------------------------------------------------------------------------
# Integration tests against the real raw dataset (skipped if absent)
# ---------------------------------------------------------------------------


@requires_raw_data
def test_real_dataset_matches_documented_population():
    df = data.load_raw_data()
    data.validate_raw_data(df)

    summary = data.summarize_dataset(df)

    assert summary["n_rows"] == data.EXPECTED_RAW_ROW_COUNT
    assert summary["n_duplicate_rows"] == 24_206
    assert summary["positive_prevalence"] == pytest.approx(
        data.EXPECTED_POSITIVE_PREVALENCE
    )


@requires_raw_data
def test_prepare_data_end_to_end_on_real_dataset():
    splits, summary = data.prepare_data()

    assert summary["n_rows"] == data.EXPECTED_RAW_ROW_COUNT
    # Exact 70/10/20 sizes for 253,680 rows; duplicates kept (D-014).
    assert len(splits.train) == 177_576
    assert len(splits.calibration) == 25_368
    assert len(splits.test) == 50_736
    for split in (splits.train, splits.calibration, splits.test):
        assert (split.dtypes == "uint8").all()
        assert split[data.TARGET].mean() == pytest.approx(
            data.EXPECTED_POSITIVE_PREVALENCE, abs=0.002
        )
