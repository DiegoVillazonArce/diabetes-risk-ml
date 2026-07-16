"""Pure in-memory P11 CSV parsing, validation, scoring, and export.

The module implements D-026 and D-027 without importing Streamlit or touching
the filesystem. Uploaded bytes are bounded before decoding, structurally
validated before column canonicalization, and then validated cell by cell.
Only complete valid rows reach one vectorized call to the scorer selected by
the schema-version-2 P8 artifact.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import pandas as pd

from src import artifacts
from src.data import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    ORDINAL_FEATURES,
    TARGET,
    VALUE_RANGES,
)
from src.feature_labels import (
    BINARY_VALUE_LABELS,
    ORDINAL_VALUE_LABELS,
    feature_label,
    format_feature_value,
)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_DATA_ROWS = 1_000
MAX_PARSED_HEADER_COLUMNS = 64
MAX_FILE_ERROR_MESSAGE_CHARS = 1_000
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"
CSV_LINE_TERMINATOR = "\n"
PROBABILITY_DECIMAL_PLACES = 15
VALIDATION_ERROR_SEPARATOR = " | "

VALID_STATUS = "valid"
INVALID_STATUS = "invalid"

RESULT_COLUMNS = (
    "row_number",
    *FEATURE_COLUMNS,
    "validation_status",
    "validation_errors",
    "model_probability",
)
FIELD_GUIDE_COLUMNS = (
    "feature",
    "label",
    "representation",
    "minimum",
    "maximum",
    "accepted_values",
)

_ALTERNATIVE_DELIMITERS = (";", "\t", "|")
_BOOLEAN_LITERALS = {"true", "false"}
_FORMULA_PREFIXES = ("=", "+", "-", "@")
_MAX_REPORTED_HEADER_NAMES = 5
_MAX_REPORTED_HEADER_NAME_CHARS = 80
_TRUNCATED_MESSAGE_SUFFIX = "... [truncated]"


class BatchFileError(ValueError):
    """Safe deterministic rejection of one structurally invalid upload."""

    def __init__(self, code: str, message: str) -> None:
        if len(message) > MAX_FILE_ERROR_MESSAGE_CHARS:
            retained = MAX_FILE_ERROR_MESSAGE_CHARS - len(_TRUNCATED_MESSAGE_SUFFIX)
            message = message[:retained] + _TRUNCATED_MESSAGE_SUFFIX
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedBatch:
    """Canonical raw cells from one structurally valid upload."""

    rows: tuple[tuple[str, ...], ...]
    upload_sha256: str


@dataclass(frozen=True)
class BatchRowResult:
    """One logical input record and its complete validation/scoring result."""

    row_number: int
    values: tuple[str, ...]
    validation_errors: tuple[str, ...]
    model_probability: float | None

    @property
    def validation_status(self) -> str:
        return INVALID_STATUS if self.validation_errors else VALID_STATUS


@dataclass(frozen=True)
class BatchResult:
    """Immutable ordered result; dataframe views are created as fresh copies."""

    rows: tuple[BatchRowResult, ...]
    upload_sha256: str

    @property
    def total_rows(self) -> int:
        return len(self.rows)

    @property
    def valid_rows(self) -> int:
        return sum(row.validation_status == VALID_STATUS for row in self.rows)

    @property
    def invalid_rows(self) -> int:
        return self.total_rows - self.valid_rows

    def to_dataframe(self) -> pd.DataFrame:
        """Return an independent combined result table in D-027 order."""
        return batch_result_dataframe(self)


def upload_sha256(uploaded_bytes: bytes) -> str:
    """Return the stable active-session identity of one uploaded byte string."""
    if not isinstance(uploaded_bytes, bytes):
        raise BatchFileError("invalid_upload_type", "Uploaded content must be bytes.")
    return hashlib.sha256(uploaded_bytes).hexdigest()


def _duplicates_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _bounded_header_name(name: str) -> str:
    rendered = repr(name)
    if len(rendered) <= _MAX_REPORTED_HEADER_NAME_CHARS:
        return rendered
    retained = _MAX_REPORTED_HEADER_NAME_CHARS - 3
    return rendered[:retained] + "..."


def _summarize_header_names(names: list[str]) -> str:
    shown = names[:_MAX_REPORTED_HEADER_NAMES]
    previews = ", ".join(_bounded_header_name(name) for name in shown)
    remainder = len(names) - len(shown)
    suffix = f"; {remainder} more not shown" if remainder else ""
    return f"{len(names)} total; first {len(shown)}: [{previews}]{suffix}"


def _looks_like_unsupported_delimiter(header: list[str]) -> str | None:
    if len(header) != 1:
        return None
    cell = header[0]
    for delimiter in _ALTERNATIVE_DELIMITERS:
        pieces = cell.split(delimiter)
        if len(pieces) > 1:
            return delimiter
    return None


def _looks_like_data_record(record: list[str]) -> bool:
    if len(record) != len(FEATURE_COLUMNS):
        return False
    for cell in record:
        token = cell.strip()
        if not token or token.casefold() in _BOOLEAN_LITERALS:
            continue
        try:
            Decimal(token)
        except InvalidOperation:
            return False
    return True


def _is_exported_index_header(header: str) -> bool:
    normalized = header.strip().casefold()
    return (
        normalized in {"", "index", "level_0", "row_number"}
        or normalized.startswith("unnamed:")
    )


def _is_identifier_header(header: str) -> bool:
    normalized = header.strip().casefold().replace("-", "_").replace(" ", "_")
    return (
        normalized in {
            "id",
            "identifier",
            "patientid",
            "patient_id",
            "recordid",
            "record_id",
            "respondentid",
            "respondent_id",
        }
        or normalized.endswith("_id")
    )


def _schema_error(header: list[str]) -> BatchFileError:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in header]
    unexpected = [name for name in header if name not in FEATURE_COLUMNS]
    unexpected_distinct = list(dict.fromkeys(unexpected))
    details = (
        f"Missing columns ({len(missing)}): {missing}. "
        f"Unexpected columns ({len(unexpected)} entries): "
        f"{_summarize_header_names(unexpected_distinct)}."
    )
    if any(name.casefold() == TARGET.casefold() for name in unexpected):
        return BatchFileError(
            "target_column_not_allowed",
            f"Target column '{TARGET}' is not allowed in batch input. {details}",
        )
    if any(_is_exported_index_header(name) for name in unexpected):
        return BatchFileError(
            "exported_index_not_allowed",
            f"Exported index columns are not allowed in batch input. {details}",
        )
    if any(_is_identifier_header(name) for name in unexpected):
        return BatchFileError(
            "identifier_not_allowed",
            f"Identifier columns are not allowed in batch input. {details}",
        )
    return BatchFileError(
        "column_schema_mismatch",
        "Batch input must contain exactly the 21 FEATURE_COLUMNS headers. "
        f"{details}",
    )


def parse_batch_csv(uploaded_bytes: bytes) -> ParsedBatch:
    """Parse one D-026 upload and reject every file-level structural error.

    Column order is canonicalized only after the header has passed exact-set
    validation. A blank logical record is preserved as 21 empty cells so row
    validation can report it honestly instead of dropping it.
    """
    if not isinstance(uploaded_bytes, bytes):
        raise BatchFileError("invalid_upload_type", "Uploaded content must be bytes.")
    byte_count = len(uploaded_bytes)
    if byte_count == 0:
        raise BatchFileError("empty_file", "The uploaded CSV is empty.")
    if byte_count > MAX_UPLOAD_BYTES:
        raise BatchFileError(
            "file_too_large",
            f"The uploaded CSV exceeds the {MAX_UPLOAD_BYTES}-byte limit.",
        )

    try:
        text = uploaded_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise BatchFileError(
            "unsupported_encoding",
            "The uploaded CSV must use UTF-8 encoding, with an optional leading BOM.",
        ) from error
    if "\x00" in text:
        raise BatchFileError(
            "null_byte",
            "The uploaded CSV contains NUL bytes and is not accepted as UTF-8 CSV.",
        )

    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=CSV_DELIMITER,
        strict=True,
    )
    try:
        header = next(reader)
    except StopIteration as error:
        raise BatchFileError("empty_file", "The uploaded CSV is empty.") from error
    except csv.Error as error:
        raise BatchFileError(
            "malformed_csv", f"The uploaded CSV is malformed: {error}."
        ) from error

    if not header or all(cell == "" for cell in header):
        raise BatchFileError("missing_header", "The uploaded CSV has no header row.")
    alternative_delimiter = _looks_like_unsupported_delimiter(header)
    if alternative_delimiter is not None:
        label = "tab" if alternative_delimiter == "\t" else alternative_delimiter
        raise BatchFileError(
            "unsupported_delimiter",
            f"The uploaded CSV uses an unsupported '{label}' delimiter; use comma.",
        )
    if len(header) > MAX_PARSED_HEADER_COLUMNS:
        raise BatchFileError(
            "too_many_columns",
            f"The uploaded CSV header contains {len(header)} columns; "
            f"at most {MAX_PARSED_HEADER_COLUMNS} are inspected and exactly "
            f"{len(FEATURE_COLUMNS)} feature columns are required.",
        )
    if _looks_like_data_record(header):
        raise BatchFileError(
            "missing_header",
            "The first CSV record looks like data; an exact FEATURE_COLUMNS "
            "header is required.",
        )

    duplicates = _duplicates_in_order(header)
    if duplicates:
        raise BatchFileError(
            "duplicate_headers",
            "The uploaded CSV contains duplicate headers: "
            f"{_summarize_header_names(duplicates)}.",
        )
    if set(header) != set(FEATURE_COLUMNS) or len(header) != len(FEATURE_COLUMNS):
        raise _schema_error(header)

    positions = {name: index for index, name in enumerate(header)}
    canonical_rows: list[tuple[str, ...]] = []
    try:
        for row_number, record in enumerate(reader, start=1):
            if row_number > MAX_DATA_ROWS:
                raise BatchFileError(
                    "too_many_rows",
                    f"The uploaded CSV exceeds the {MAX_DATA_ROWS}-data-row limit.",
                )
            if not record:
                record = [""] * len(header)
            if len(record) != len(header):
                raise BatchFileError(
                    "malformed_record",
                    f"Data row {row_number} has {len(record)} cells; "
                    f"expected {len(header)}.",
                )
            canonical_rows.append(
                tuple(record[positions[feature]] for feature in FEATURE_COLUMNS)
            )
    except csv.Error as error:
        raise BatchFileError(
            "malformed_csv", f"The uploaded CSV is malformed: {error}."
        ) from error

    if not canonical_rows:
        raise BatchFileError(
            "no_data_rows", "The uploaded CSV contains a header but no data rows."
        )
    return ParsedBatch(
        rows=tuple(canonical_rows),
        upload_sha256=upload_sha256(uploaded_bytes),
    )


def _validate_cell(raw_value: str, feature: str) -> tuple[int | None, tuple[str, ...]]:
    token = raw_value.strip()
    if not token:
        return None, (f"{feature}: value is missing",)
    if token.casefold() in _BOOLEAN_LITERALS:
        return None, (f"{feature}: boolean literals are not allowed",)
    if "_" in token:
        return None, (f"{feature}: value is not numeric",)
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None, (f"{feature}: value is not numeric",)
    if not value.is_finite():
        return None, (f"{feature}: value must be finite",)

    errors: list[str] = []
    if value != value.to_integral_value():
        errors.append(f"{feature}: value must be a whole number")
    lower, upper = VALUE_RANGES[feature]
    if not Decimal(lower) <= value <= Decimal(upper):
        errors.append(f"{feature}: value is outside [{lower}, {upper}]")
    if errors:
        return None, tuple(errors)
    return int(value), ()


def process_batch(uploaded_bytes: bytes, bundle: dict) -> BatchResult:
    """Validate and score one upload through the unchanged P8 contract."""
    parsed = parse_batch_csv(uploaded_bytes)
    scorer = artifacts.select_probability_scorer(bundle)

    row_errors: list[tuple[str, ...]] = []
    valid_positions: list[int] = []
    valid_values: list[list[int]] = []
    for position, raw_row in enumerate(parsed.rows):
        errors: list[str] = []
        canonical_values: list[int] = []
        row_is_valid = True
        for feature, raw_value in zip(FEATURE_COLUMNS, raw_row, strict=True):
            numeric_value, cell_errors = _validate_cell(raw_value, feature)
            errors.extend(cell_errors)
            if numeric_value is None:
                row_is_valid = False
            else:
                canonical_values.append(numeric_value)
        row_errors.append(tuple(errors))
        if row_is_valid:
            valid_positions.append(position)
            valid_values.append(canonical_values)

    probabilities_by_position: dict[int, float] = {}
    if valid_positions:
        frame = pd.DataFrame(valid_values, columns=FEATURE_COLUMNS).astype("uint8")
        probabilities = artifacts.predict_probability_frame(scorer, frame)
        probabilities_by_position = {
            position: float(probability)
            for position, probability in zip(
                valid_positions, probabilities, strict=True
            )
        }

    rows = tuple(
        BatchRowResult(
            row_number=position + 1,
            values=raw_row,
            validation_errors=row_errors[position],
            model_probability=probabilities_by_position.get(position),
        )
        for position, raw_row in enumerate(parsed.rows)
    )
    return BatchResult(rows=rows, upload_sha256=parsed.upload_sha256)


def template_dataframe() -> pd.DataFrame:
    """Return the code-generated one-row synthetic D-026 template."""
    example = artifacts.example_input()
    return pd.DataFrame(
        [[example[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    ).astype("uint8")


def _mapped_values(feature: str) -> dict[int, str] | None:
    if feature in BINARY_VALUE_LABELS:
        return BINARY_VALUE_LABELS[feature]
    if feature in ORDINAL_VALUE_LABELS:
        return ORDINAL_VALUE_LABELS[feature]
    return None


def field_guide_dataframe() -> pd.DataFrame:
    """Generate the exact field guide from schema, range, and label sources."""
    records: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        lower, upper = VALUE_RANGES[feature]
        mapped = _mapped_values(feature)
        if mapped is not None:
            accepted = "; ".join(
                f"{code} = {format_feature_value(feature, code)}"
                for code in range(lower, upper + 1)
            )
            representation = "integer code"
        else:
            accepted = (
                f"{format_feature_value(feature, lower)} to "
                f"{format_feature_value(feature, upper)} (inclusive whole numbers)"
            )
            representation = "whole number"
        records.append(
            {
                "feature": feature,
                "label": feature_label(feature),
                "representation": representation,
                "minimum": lower,
                "maximum": upper,
                "accepted_values": accepted,
            }
        )
    return pd.DataFrame(records, columns=FIELD_GUIDE_COLUMNS)


def _csv_bytes(rows: list[list[object]] | tuple[tuple[object, ...], ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(
        stream,
        delimiter=CSV_DELIMITER,
        lineterminator=CSV_LINE_TERMINATOR,
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writerows(rows)
    return stream.getvalue().encode(CSV_ENCODING)


def template_csv_bytes() -> bytes:
    """Return deterministic UTF-8/LF bytes for the synthetic template."""
    frame = template_dataframe()
    rows: list[list[object]] = [list(FEATURE_COLUMNS)]
    rows.extend(frame.astype(object).values.tolist())
    return _csv_bytes(rows)


def field_guide_csv_bytes() -> bytes:
    """Return deterministic UTF-8/LF bytes for the generated field guide."""
    frame = field_guide_dataframe()
    rows: list[list[object]] = [list(FIELD_GUIDE_COLUMNS)]
    rows.extend(frame.astype(object).values.tolist())
    return _csv_bytes(rows)


def batch_result_dataframe(result: BatchResult) -> pd.DataFrame:
    """Build a fresh combined table in the exact D-027 output schema."""
    records: list[dict[str, object]] = []
    for row in result.rows:
        record: dict[str, object] = {"row_number": row.row_number}
        record.update(
            {
                feature: value
                for feature, value in zip(FEATURE_COLUMNS, row.values, strict=True)
            }
        )
        record["validation_status"] = row.validation_status
        record["validation_errors"] = VALIDATION_ERROR_SEPARATOR.join(
            row.validation_errors
        )
        record["model_probability"] = row.model_probability
        records.append(record)
    return pd.DataFrame(records, columns=RESULT_COLUMNS)


def _spreadsheet_safe_cell(value: str) -> str:
    """Neutralize formula-capable user text in the downloadable CSV only."""
    first_visible = value.lstrip()
    if first_visible.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def result_csv_bytes(result: BatchResult) -> bytes:
    """Serialize one result deterministically with D-027 CSV-injection safety."""
    rows: list[list[object]] = [list(RESULT_COLUMNS)]
    for row in result.rows:
        probability = (
            ""
            if row.model_probability is None
            else f"{row.model_probability:.{PROBABILITY_DECIMAL_PLACES}f}"
        )
        rows.append(
            [
                row.row_number,
                *(_spreadsheet_safe_cell(value) for value in row.values),
                row.validation_status,
                VALIDATION_ERROR_SEPARATOR.join(row.validation_errors),
                probability,
            ]
        )
    return _csv_bytes(rows)


if set(BINARY_FEATURES) & set(ORDINAL_FEATURES):
    raise RuntimeError("Batch field groups must not overlap.")
if set(VALUE_RANGES) - {TARGET} != set(FEATURE_COLUMNS):
    raise RuntimeError("Batch ranges must cover exactly FEATURE_COLUMNS.")
