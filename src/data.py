"""Data contract, loading, validation, downcasting, and split logic for the
BRFSS 2015 diabetes dataset (P3: Data Preparation and Splits).

The data contract (column order, feature groups, and value ranges) mirrors the
P2 EDA findings in `notebooks/01_data_understanding_eda.ipynb`. Per decision
D-014, exact duplicate rows are kept: the selected analysis population is the
full 253,680-row dataset with ~13.9% positive prevalence, and the stratified
70/10/20 train/calibration/test split must preserve that prevalence. Per
decision D-015, splits are returned in memory; no files are written to
`data/processed/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"

TARGET = "Diabetes_binary"

BINARY_FEATURES = [
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
    "Sex",
]
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

# Exact column order of the raw Kaggle CSV; validation requires this order so
# that every consumer of the dataset sees an identical frame.
EXPECTED_COLUMNS = [
    TARGET,
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
FEATURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column != TARGET]

# Inclusive valid value ranges per column, from the P2 EDA range checks.
VALUE_RANGES: dict[str, tuple[int, int]] = {
    TARGET: (0, 1),
    **{feature: (0, 1) for feature in BINARY_FEATURES},
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
    "BMI": (12, 98),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
}

# Documented properties of the selected raw file (D-014). These describe the
# expected full dataset and are used by integration tests; `validate_raw_data`
# does not enforce them so that it also works on synthetic or partial frames.
EXPECTED_RAW_ROW_COUNT = 253_680
EXPECTED_RAW_POSITIVE_COUNT = 35_346
EXPECTED_POSITIVE_PREVALENCE = EXPECTED_RAW_POSITIVE_COUNT / EXPECTED_RAW_ROW_COUNT

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.10
TEST_FRACTION = 0.20


@dataclass(frozen=True)
class DataSplits:
    """In-memory stratified train/calibration/test splits (D-015)."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw BRFSS CSV, failing clearly if the file is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. Download "
            "'diabetes_binary_health_indicators_BRFSS2015.csv' from Kaggle and "
            "place it at 'data/raw/'; see data/README.md for instructions."
        )
    return pd.read_csv(path)


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate a dataframe against the documented raw data contract.

    Checks exact columns and column order, numeric dtypes, missing values,
    integer-like values, and per-column value ranges (target and binary
    features restricted to 0/1). Raises ValueError on the first violation.
    Exact duplicate rows are allowed and intentionally not treated as an
    error (D-014).
    """
    if list(df.columns) != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
        unexpected = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            "Columns do not match the expected schema (exact names and order "
            f"required). Missing: {missing}. Unexpected: {unexpected}. "
            f"Observed order: {list(df.columns)}."
        )

    non_numeric = [
        column for column in EXPECTED_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}.")

    missing_counts = df.isna().sum()
    if missing_counts.any():
        with_missing = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing values found: {with_missing}.")

    non_integer = [
        column for column in EXPECTED_COLUMNS if (df[column] % 1 != 0).any()
    ]
    if non_integer:
        raise ValueError(
            f"Non-integer values found in columns: {non_integer}. All values "
            "are expected to be whole numbers."
        )

    for column, (lower, upper) in VALUE_RANGES.items():
        out_of_range = df[(df[column] < lower) | (df[column] > upper)]
        if not out_of_range.empty:
            raise ValueError(
                f"Column '{column}' has {len(out_of_range)} value(s) outside "
                f"the valid range [{lower}, {upper}]."
            )


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Report the selected analysis population explicitly (D-014).

    Returns row count, exact duplicate row count (kept, not dropped), and
    positive target prevalence.
    """
    return {
        "n_rows": int(len(df)),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "positive_prevalence": float(df[TARGET].mean()),
    }


def downcast_to_uint8(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast all columns to uint8, verifying the cast is lossless.

    Intended to run after `validate_raw_data` (all values whole numbers in
    [0, 98]), but independently verifies losslessness via a float64 round
    trip and raises ValueError if any value would change.
    """
    try:
        downcast = df.astype("uint8")
    except (ValueError, TypeError) as error:
        raise ValueError(f"Cannot downcast to uint8: {error}") from error
    if not downcast.astype("float64").equals(df.astype("float64")):
        raise ValueError(
            "Downcasting to uint8 would change values (non-integer or outside "
            "[0, 255]); run validate_raw_data first."
        )
    return downcast


def split_data(df: pd.DataFrame, random_state: int = RANDOM_SEED) -> DataSplits:
    """Create a reproducible stratified 70/10/20 train/calibration/test split.

    Stratification on the target preserves the selected positive prevalence
    (~13.9% on the full dataset, per D-014) in every split. All rows are kept,
    including exact duplicates, and splits are returned in memory (D-015).
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataframe.")

    train_calibration, test = train_test_split(
        df,
        test_size=TEST_FRACTION,
        stratify=df[TARGET],
        random_state=random_state,
    )
    # Calibration is 10% of the full dataset, taken from the remaining 80%.
    calibration_fraction_of_rest = CALIBRATION_FRACTION / (1.0 - TEST_FRACTION)
    train, calibration = train_test_split(
        train_calibration,
        test_size=calibration_fraction_of_rest,
        stratify=train_calibration[TARGET],
        random_state=random_state,
    )
    return DataSplits(train=train, calibration=calibration, test=test)


def prepare_data(
    path: Path | str = RAW_DATA_PATH, random_state: int = RANDOM_SEED
) -> tuple[DataSplits, dict]:
    """Run the full P3 preparation pipeline: load, validate, downcast, split.

    Returns the in-memory splits together with the dataset summary metadata
    that makes the selected row count and prevalence explicit (D-014).
    """
    df = load_raw_data(path)
    validate_raw_data(df)
    summary = summarize_dataset(df)
    df = downcast_to_uint8(df)
    return split_data(df, random_state=random_state), summary
