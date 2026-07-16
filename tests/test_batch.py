"""Focused contract tests for the pure P11 batch workflow."""

from __future__ import annotations

import csv
import inspect
import io
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from src import artifacts, batch, calibration, data
from src.feature_labels import (
    BINARY_VALUE_LABELS,
    ORDINAL_VALUE_LABELS,
    feature_label,
    format_feature_value,
)
from tests.reference_profiles import REFERENCE_PROFILES, format_display
from tests.test_modeling import make_splits


@pytest.fixture(scope="module")
def bundle():
    return artifacts.build_artifact_bundle(make_splits())


@pytest.fixture(scope="module")
def official_bundle():
    return artifacts.load_artifact()


def valid_row() -> dict[str, object]:
    return dict(artifacts.example_input())


def csv_bytes(
    rows: list[dict[str, object] | list[object] | tuple[object, ...]],
    *,
    header: list[str] | tuple[str, ...] = tuple(data.FEATURE_COLUMNS),
    delimiter: str = ",",
    bom: bool = False,
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        if isinstance(row, dict):
            writer.writerow([row.get(feature, "") for feature in header])
        else:
            writer.writerow(row)
    payload = stream.getvalue().encode("utf-8")
    return (b"\xef\xbb\xbf" + payload) if bom else payload


def exact_size_csv(size: int) -> bytes:
    """Build 1,000 valid rows whose UTF-8 payload is exactly ``size`` bytes."""
    header_line = ",".join(data.FEATURE_COLUMNS) + "\n"
    values = [str(valid_row()[feature]) for feature in data.FEATURE_COLUMNS]
    base_line = ",".join(values) + "\n"
    base_size = (
        len(header_line.encode())
        + batch.MAX_DATA_ROWS * len(base_line.encode())
    )
    padding = size - base_size
    assert padding >= 0
    per_row, remainder = divmod(padding, batch.MAX_DATA_ROWS)
    lines = [header_line]
    for index in range(batch.MAX_DATA_ROWS):
        padded = list(values)
        padded[0] += " " * (per_row + (1 if index < remainder else 0))
        lines.append(",".join(padded) + "\n")
    payload = "".join(lines).encode("utf-8")
    assert len(payload) == size
    return payload


# ---------------------------------------------------------------------------
# Template and field guide
# ---------------------------------------------------------------------------


def test_template_has_exact_contract_order_and_only_a_synthetic_example():
    frame = batch.template_dataframe()

    assert list(frame.columns) == data.FEATURE_COLUMNS
    assert len(frame) == 1
    assert frame.iloc[0].to_dict() == artifacts.example_input()
    assert (frame.dtypes == "uint8").all()
    assert data.TARGET not in frame


def test_template_bytes_are_utf8_lf_deterministic_and_round_trip(bundle):
    first = batch.template_csv_bytes()
    second = batch.template_csv_bytes()

    assert first == second
    assert not first.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in first
    assert first.decode("utf-8")
    result = batch.process_batch(first, bundle)
    assert (result.total_rows, result.valid_rows, result.invalid_rows) == (1, 1, 0)


def test_field_guide_cannot_drift_from_schema_ranges_or_labels():
    guide = batch.field_guide_dataframe()

    assert tuple(guide.columns) == batch.FIELD_GUIDE_COLUMNS
    assert guide["feature"].tolist() == data.FEATURE_COLUMNS
    for row in guide.itertuples(index=False):
        lower, upper = data.VALUE_RANGES[row.feature]
        assert row.label == feature_label(row.feature)
        assert (row.minimum, row.maximum) == (lower, upper)
        mapping = BINARY_VALUE_LABELS.get(row.feature) or ORDINAL_VALUE_LABELS.get(
            row.feature
        )
        if mapping:
            assert row.representation == "integer code"
            for code in range(lower, upper + 1):
                assert f"{code} = {format_feature_value(row.feature, code)}" in (
                    row.accepted_values
                )
        else:
            assert row.representation == "whole number"
            assert format_feature_value(row.feature, lower) in row.accepted_values
            assert format_feature_value(row.feature, upper) in row.accepted_values


def test_field_guide_bytes_are_utf8_lf_and_deterministic():
    first = batch.field_guide_csv_bytes()
    assert first == batch.field_guide_csv_bytes()
    assert first.decode("utf-8")
    assert not first.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in first
    rows = list(csv.reader(io.StringIO(first.decode("utf-8"))))
    assert rows[0] == list(batch.FIELD_GUIDE_COLUMNS)
    assert [row[0] for row in rows[1:]] == data.FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"", "empty_file"),
        (csv_bytes([]), "no_data_rows"),
        (
            (",".join(data.FEATURE_COLUMNS) + '\n"unterminated').encode(),
            "malformed_csv",
        ),
        (b"\xff\xfe\xfa", "unsupported_encoding"),
        (csv_bytes([valid_row()], delimiter=";"), "unsupported_delimiter"),
        (csv_bytes([valid_row()]).replace(b"HighBP", b"High\x00BP", 1), "null_byte"),
    ],
)
def test_file_level_failures_are_deterministic(payload, code):
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(payload)
    assert caught.value.code == code


def test_utf8_with_a_leading_bom_is_accepted():
    parsed = batch.parse_batch_csv(csv_bytes([valid_row()], bom=True))
    assert len(parsed.rows) == 1


def test_utf16_is_rejected_via_the_nul_guard():
    payload = csv_bytes([valid_row()]).decode().encode("utf-16le")
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(payload)
    assert caught.value.code == "null_byte"


def test_headerless_numeric_first_record_is_rejected():
    row = [valid_row()[feature] for feature in data.FEATURE_COLUMNS]
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerow(row)
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(stream.getvalue().encode())
    assert caught.value.code == "missing_header"


def test_duplicate_headers_are_rejected_before_schema_canonicalization():
    header = list(data.FEATURE_COLUMNS)
    header[1] = header[0]
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(csv_bytes([valid_row()], header=header))
    assert caught.value.code == "duplicate_headers"
    assert data.FEATURE_COLUMNS[0] in str(caught.value)


def test_excessive_header_count_is_rejected_with_a_small_constant_message():
    header = [f"column_{index}" for index in range(10_000)]
    payload = (",".join(header) + "\n").encode("utf-8")
    assert len(payload) < batch.MAX_UPLOAD_BYTES

    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(payload)

    message = str(caught.value)
    assert caught.value.code == "too_many_columns"
    assert "10000 columns" in message
    assert "column_9999" not in message
    assert len(message) <= batch.MAX_FILE_ERROR_MESSAGE_CHARS


@pytest.mark.parametrize(
    ("header", "code"),
    [
        (
            [*data.FEATURE_COLUMNS[:-1], "X" * 100_000],
            "column_schema_mismatch",
        ),
        (
            ["Y" * 100_000, "Y" * 100_000, *data.FEATURE_COLUMNS[2:]],
            "duplicate_headers",
        ),
    ],
    ids=("long-unexpected", "long-duplicate"),
)
def test_user_controlled_header_names_are_bounded_in_structural_errors(
    header, code
):
    payload = csv_bytes([valid_row()], header=header)
    assert len(payload) < batch.MAX_UPLOAD_BYTES

    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(payload)

    message = str(caught.value)
    assert caught.value.code == code
    assert len(message) <= batch.MAX_FILE_ERROR_MESSAGE_CHARS
    assert "X" * 1_000 not in message
    assert "Y" * 1_000 not in message


def test_batch_file_error_has_an_absolute_message_length_backstop():
    error = batch.BatchFileError("test", "Z" * 10_000)
    message = str(error)

    assert len(message) == batch.MAX_FILE_ERROR_MESSAGE_CHARS
    assert message.endswith("... [truncated]")


@pytest.mark.parametrize(
    ("header", "code"),
    [
        (data.FEATURE_COLUMNS[:-1], "column_schema_mismatch"),
        ([*data.FEATURE_COLUMNS, "Extra"], "column_schema_mismatch"),
        ([*data.FEATURE_COLUMNS, data.TARGET], "target_column_not_allowed"),
        (["Unnamed: 0", *data.FEATURE_COLUMNS], "exported_index_not_allowed"),
        ([*data.FEATURE_COLUMNS, "patient_id"], "identifier_not_allowed"),
    ],
)
def test_exact_column_policy_rejects_missing_extra_target_index_and_id(header, code):
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(csv_bytes([valid_row()], header=header))
    assert caught.value.code == code


def test_columns_may_arrive_in_any_order_but_rows_are_canonicalized():
    reversed_header = list(reversed(data.FEATURE_COLUMNS))
    parsed = batch.parse_batch_csv(
        csv_bytes([valid_row()], header=reversed_header)
    )
    expected = tuple(str(valid_row()[feature]) for feature in data.FEATURE_COLUMNS)
    assert parsed.rows == (expected,)


def test_wrong_record_width_is_a_whole_file_failure():
    payload = csv_bytes([[1, 2, 3]])
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(payload)
    assert caught.value.code == "malformed_record"


def test_blank_logical_record_is_preserved_for_row_validation(bundle):
    header = ",".join(data.FEATURE_COLUMNS)
    payload = (header + "\n\n").encode()
    result = batch.process_batch(payload, bundle)

    assert result.total_rows == 1
    assert result.invalid_rows == 1
    assert len(result.rows[0].validation_errors) == len(data.FEATURE_COLUMNS)
    assert result.rows[0].model_probability is None


def test_exact_byte_limit_is_accepted_and_one_extra_byte_is_rejected():
    exact = exact_size_csv(batch.MAX_UPLOAD_BYTES)
    parsed = batch.parse_batch_csv(exact)
    assert len(parsed.rows) == batch.MAX_DATA_ROWS

    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(exact + b"x")
    assert caught.value.code == "file_too_large"


def test_exact_row_limit_is_accepted_and_one_extra_row_is_rejected():
    row = valid_row()
    exact = csv_bytes([row] * batch.MAX_DATA_ROWS)
    assert len(batch.parse_batch_csv(exact).rows) == batch.MAX_DATA_ROWS

    excess = csv_bytes([row] * (batch.MAX_DATA_ROWS + 1))
    with pytest.raises(batch.BatchFileError) as caught:
        batch.parse_batch_csv(excess)
    assert caught.value.code == "too_many_rows"


# ---------------------------------------------------------------------------
# Cell validation, partial success, and scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    [
        ("BMI", "", "value is missing"),
        ("BMI", "large", "value is not numeric"),
        ("HighBP", "true", "boolean literals are not allowed"),
        ("BMI", "NaN", "value must be finite"),
        ("BMI", "Inf", "value must be finite"),
        ("BMI", "25.5", "value must be a whole number"),
        ("BMI", "99", "value is outside [12, 98]"),
    ],
)
def test_invalid_cell_classes_are_reported_without_repair(
    bundle, feature, value, message
):
    row = valid_row()
    row[feature] = value
    result = batch.process_batch(csv_bytes([row]), bundle)

    assert result.invalid_rows == 1
    assert result.rows[0].model_probability is None
    assert any(message in error for error in result.rows[0].validation_errors)


def test_all_applicable_errors_follow_feature_and_rule_order(bundle):
    row = valid_row()
    row["HighBP"] = "1.5"  # fraction and range
    row["BMI"] = ""
    row["Age"] = "NaN"
    result = batch.process_batch(csv_bytes([row]), bundle)

    assert result.rows[0].validation_errors == (
        "HighBP: value must be a whole number",
        "HighBP: value is outside [0, 1]",
        "BMI: value is missing",
        "Age: value must be finite",
    )


def test_integer_like_decimal_and_scientific_notation_are_valid(bundle):
    row = valid_row()
    row["HighBP"] = "1.0"
    row["HighChol"] = "1e0"
    result = batch.process_batch(csv_bytes([row]), bundle)
    assert result.valid_rows == 1


def test_partial_success_scores_only_valid_rows_once_and_preserves_duplicates(
    bundle, monkeypatch
):
    duplicate = valid_row()
    invalid = valid_row()
    invalid["BMI"] = "not-a-number"
    payload = csv_bytes([duplicate, invalid, duplicate])
    original = artifacts.predict_probability_frame
    calls: list[pd.DataFrame] = []

    def spy(scorer, frame):
        calls.append(frame.copy())
        return original(scorer, frame)

    monkeypatch.setattr(artifacts, "predict_probability_frame", spy)
    result = batch.process_batch(payload, bundle)

    assert (result.total_rows, result.valid_rows, result.invalid_rows) == (3, 2, 1)
    assert [row.row_number for row in result.rows] == [1, 2, 3]
    assert result.rows[0].values == result.rows[2].values
    assert result.rows[0].model_probability == result.rows[2].model_probability
    assert result.rows[1].model_probability is None
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_artifact_is_validated_once_per_structurally_valid_batch(
    bundle, monkeypatch
):
    original = artifacts.validate_artifact_bundle
    calls = 0

    def spy(candidate):
        nonlocal calls
        calls += 1
        return original(candidate)

    monkeypatch.setattr(artifacts, "validate_artifact_bundle", spy)
    batch.process_batch(csv_bytes([valid_row(), valid_row()]), bundle)
    assert calls == 1


def test_no_probability_call_occurs_when_every_row_is_invalid(bundle, monkeypatch):
    row = valid_row()
    row["BMI"] = ""

    def forbidden(*args, **kwargs):
        raise AssertionError("no valid row may reach the probability scorer")

    monkeypatch.setattr(artifacts, "predict_probability_frame", forbidden)
    result = batch.process_batch(csv_bytes([row, row]), bundle)
    assert result.valid_rows == 0
    assert result.invalid_rows == 2


@pytest.mark.parametrize("invalid_probability", [float("nan"), -0.1, 1.1])
def test_nonfinite_or_out_of_range_probability_is_rejected(
    bundle, monkeypatch, invalid_probability
):
    def bad_positive_probability(scorer, frame):
        return np.full(len(frame), invalid_probability)

    monkeypatch.setattr(artifacts, "predict_positive_proba", bad_positive_probability)
    with pytest.raises(ValueError, match="invalid probability"):
        batch.process_batch(csv_bytes([valid_row()]), bundle)


def test_batch_probabilities_equal_individual_serving_within_1e12(official_bundle):
    payload = csv_bytes([dict(profile.features) for profile in REFERENCE_PROFILES])
    result = batch.process_batch(payload, official_bundle)

    assert result.valid_rows == len(REFERENCE_PROFILES)
    for row_result, profile in zip(result.rows, REFERENCE_PROFILES, strict=True):
        individual = artifacts.predict_risk_probability(
            official_bundle, dict(profile.features)
        )
        assert row_result.model_probability == pytest.approx(individual, abs=1e-12)
        assert format_display(row_result.model_probability) == profile.expected_display


def test_selected_calibrator_is_shared_by_batch_and_individual_serving():
    splits = make_splits()
    candidate = artifacts.build_artifact_bundle(splits)
    candidate["metadata"]["calibration_method"] = "sigmoid"
    candidate["calibrator"] = calibration.fit_final_calibrator(
        candidate["model"], calibration.to_calibration_data(splits), "sigmoid"
    )
    artifacts.validate_artifact_bundle(candidate)

    result = batch.process_batch(csv_bytes([valid_row()]), candidate)
    individual = artifacts.predict_risk_probability(candidate, valid_row())
    assert result.rows[0].model_probability == pytest.approx(individual, abs=1e-12)


def test_batch_result_is_frozen_and_dataframe_views_are_independent(bundle):
    result = batch.process_batch(csv_bytes([valid_row()]), bundle)
    with pytest.raises(FrozenInstanceError):
        result.upload_sha256 = "changed"
    first = result.to_dataframe()
    first.loc[0, "BMI"] = "changed"
    second = result.to_dataframe()
    assert second.loc[0, "BMI"] != "changed"


# ---------------------------------------------------------------------------
# Deterministic safe export and scope guards
# ---------------------------------------------------------------------------


def test_export_has_exact_columns_utf8_lf_fixed_float_and_blank_invalid(bundle):
    invalid = valid_row()
    invalid["BMI"] = ""
    result = batch.process_batch(csv_bytes([valid_row(), invalid]), bundle)
    payload = batch.result_csv_bytes(result)

    text = payload.decode("utf-8")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert "\r" not in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0]) == batch.RESULT_COLUMNS
    assert rows[0]["validation_status"] == batch.VALID_STATUS
    assert rows[0]["model_probability"] == (
        f"{result.rows[0].model_probability:.15f}"
    )
    assert rows[1]["validation_status"] == batch.INVALID_STATUS
    assert rows[1]["model_probability"] == ""
    assert rows[1]["validation_errors"] == "BMI: value is missing"


def test_repeated_processing_and_export_produce_identical_bytes(bundle):
    payload = csv_bytes([valid_row(), valid_row()])
    first = batch.result_csv_bytes(batch.process_batch(payload, bundle))
    second = batch.result_csv_bytes(batch.process_batch(payload, bundle))
    assert first == second


def test_export_neutralizes_spreadsheet_formula_injection_without_hiding_errors(
    bundle,
):
    row = valid_row()
    attacks = {
        "HighBP": "=2+3",
        "HighChol": "+cmd|calc",
        "CholCheck": "-cmd|calc",
        "BMI": "  @SUM(1,1)",
    }
    row.update(attacks)
    result = batch.process_batch(csv_bytes([row]), bundle)
    raw = result.to_dataframe().iloc[0]
    exported = next(
        csv.DictReader(io.StringIO(batch.result_csv_bytes(result).decode("utf-8")))
    )

    for feature, attack in attacks.items():
        assert raw[feature] == attack
        assert exported[feature] == "'" + attack
    assert exported["validation_status"] == batch.INVALID_STATUS
    assert exported["validation_errors"].startswith("HighBP:")


def test_batch_module_has_no_project_data_fit_write_remote_or_logging_path():
    source = inspect.getsource(batch).lower()
    for forbidden in (
        "raw_data_path",
        "prepare_data",
        "load_raw_data",
        "train_test_split",
        ".fit(",
        "build_artifact_bundle",
        "create_default_artifact",
        "joblib",
        "requests",
        "urllib",
        "http://",
        "https://",
        "analytics",
        "logging.",
        "open(",
        "write_text",
        "write_bytes",
        "to_csv",
    ):
        assert forbidden not in source
    assert "import streamlit" not in source
    assert "from streamlit" not in source
