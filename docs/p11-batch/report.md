# P11 Batch Prediction Workflow Technical Report

**Local implementation date:** 2026-07-16

**Planning baseline:** `ba82106 docs: refine P11 batch prediction planning`

**Local status:** implementation and local verification complete; review and external closure pending
**Phase status:** P11 remains **Ready**, not Done

## 1. Scope and outcome

P11 now has a local, pure in-memory CSV workflow that generates its own template and field guide, rejects ambiguous files, reports all row-value problems, scores only valid rows through the unchanged P8 probability contract, previews a bounded result, and returns deterministic safe CSV bytes. P9 SHAP and P10 scenarios remain available only for the individual workflow.

This report does not claim a deployment or public smoke test. The reviewed local changes are unstaged and the current public application still contains P8-P10 only. US-0603 and US-0612 are Done; US-0613 remains In Progress until review, commit, push, deployment, and public valid-plus-mixed verification.

## 2. Ordered decision evidence

### D-026 -- upload and template contract -- Accepted 2026-07-16

The byte/parser spike used Python's standard `csv.reader` with `strict=True` over `io.StringIO(..., newline="")` and tested the candidates before the definitive contract was implemented:

- `bytes.decode("utf-8-sig", errors="strict")` accepted plain UTF-8 and one leading UTF-8 BOM while producing the same first header.
- UTF-16LE decoded as NUL-bearing text under UTF-8 and was therefore distinguishable and rejected.
- Strict parsing preserved duplicate header names and rejected an unterminated quoted record with `unexpected end of data`.
- Semicolon input appeared as one header cell, making the unsupported delimiter deterministic.
- A blank logical CSV record was visible as a zero-width record; the final parser preserves it as 21 empty cells so row validation can report it rather than silently discard it.

Each candidate was evaluated as follows:

| Candidate | Decision | Evidence/rationale |
|---|---|---|
| UTF-8 and UTF-8 with BOM | Accepted as strict UTF-8 with optional leading BOM | One decoder handles both without header drift; invalid byte sequences fail before parsing. |
| Comma delimiter | Accepted | It is unambiguous with the generated files; semicolon, tab, and pipe are detected and rejected. |
| 2 MiB | Accepted | The limit is enforced on bytes before decode/parse. The simultaneous exact 2,097,152-byte/1,000-row valid boundary is tested and benchmarked. `.streamlit/config.toml` also fixes transport to 2 MiB and a headless anti-drift assertion binds it to `MAX_UPLOAD_BYTES`. |
| 1,000 data rows | Accepted | It bounds validation/UI work while supporting a useful batch; exact-limit and one-over tests pass. |
| Exactly `FEATURE_COLUMNS` | Accepted | The artifact already consumes this schema. Case-sensitive headers may arrive in any order but are reordered only after exact-set validation. |
| Target/index/identifier/free text/passthrough | Rejected | They are not model inputs, create privacy or CSV-injection surface, and would invent a second output-retention contract. |
| Integer-like `VALUE_RANGES` values | Accepted | This is the executable artifact boundary. Decimal/scientific forms are allowed only when their mathematical value is an in-range integer. |
| `Age` code versus exact age | BRFSS code `1`-`13` accepted | Exact age is only an individual-UI convenience. Converting exact age in batch would add a second transformation schema; the field guide explains the existing model codes. |
| Template example row | One synthetic row accepted | `artifacts.example_input()` makes the template immediately round-trippable and is not a real dataset row. |

The accepted generated outputs use UTF-8 without BOM, comma delimiter, and LF line endings.

### D-027 -- validation, scoring, and export -- Accepted 2026-07-16

The row-validation spike used `Decimal` so missing, boolean literal, non-numeric, non-finite, fractional, and out-of-range states remain distinct without pandas coercion. It confirmed that one cell can have multiple applicable errors: `1.5` for a binary feature is both fractional and outside `[0, 1]`.

Plain CSV quoting does not neutralize spreadsheet formulas. The export prototype therefore prefixes one apostrophe when the first non-whitespace user character is `=`, `+`, `-`, or `@`. This happens only in the download; the retained raw value, validation, and model input are unchanged.

D-027 accepts:

- whole-file rejection for structural failures;
- partial success for row-value failures in a structurally valid file;
- every applicable error in stable `FEATURE_COLUMNS` and per-cell rule order;
- no scorer call for invalid rows and no probability when a row is invalid;
- original logical row position, order, blank records, and duplicates;
- one artifact validation/scorer selection per structurally valid batch;
- one vectorized positive-class scoring call for all valid rows;
- absolute batch-versus-individual equivalence tolerance `1e-12`;
- exactly 15 probability digits after the decimal point in export;
- deterministic UTF-8/LF output bytes and export-only formula neutralization;
- structural error messages capped at 1,000 characters, with at most five user-controlled header-name previews of at most 80 rendered characters each.

### D-028 -- Streamlit, privacy, failure, state, and performance -- Accepted 2026-07-16

The pre-integration review selected one explicit workflow radio, defaulting to individual, because rendering both dense workflows at once would blur the P9/P10 boundary. The batch branch uses explicit processing/reset, a 25-row preview, fixed filenames, controlled structural/internal errors, and visible privacy/probability-only/medical text.

Failure policy:

- upload removal or replacement clears a prior result;
- processing starts by clearing a prior result;
- a structural, artifact, scoring, or export error leaves no stale or partial table/download;
- an artifact SHA-256 change clears results before rendering;
- upload names are not interpolated and user content is not rendered as unsafe HTML.

The initial pre-integration performance fixture covered the accepted 1,000-row ceiling but used compact 43,197-byte input rather than the simultaneous byte ceiling. Review identified that gap before stage. The corrective benchmark in Section 9 uses exactly 2 MiB and 1,000 valid rows for 30 warm runs; it confirms the proposed bounds without changing or relaxing either one.

## 3. Exact input and template schema

Input contains exactly these case-sensitive headers, admitted in any order:

1. `HighBP` -- integer `0`-`1`
2. `HighChol` -- integer `0`-`1`
3. `CholCheck` -- integer `0`-`1`
4. `BMI` -- integer `12`-`98`
5. `Smoker` -- integer `0`-`1`
6. `Stroke` -- integer `0`-`1`
7. `HeartDiseaseorAttack` -- integer `0`-`1`
8. `PhysActivity` -- integer `0`-`1`
9. `Fruits` -- integer `0`-`1`
10. `Veggies` -- integer `0`-`1`
11. `HvyAlcoholConsump` -- integer `0`-`1`
12. `AnyHealthcare` -- integer `0`-`1`
13. `NoDocbcCost` -- integer `0`-`1`
14. `GenHlth` -- integer code `1`-`5`
15. `MentHlth` -- integer `0`-`30`
16. `PhysHlth` -- integer `0`-`30`
17. `DiffWalk` -- integer `0`-`1`
18. `Sex` -- integer code `0`-`1`
19. `Age` -- BRFSS integer group code `1`-`13`
20. `Education` -- integer code `1`-`6`
21. `Income` -- integer code `1`-`8`

After structural validation, data are reordered to exactly that canonical sequence. Header names, ranges, and guide text are generated from `FEATURE_COLUMNS`, `VALUE_RANGES`, `feature_label`, `format_feature_value`, `BINARY_VALUE_LABELS`, and `ORDINAL_VALUE_LABELS`; there is no handwritten second schema in the batch module or UI.

The template has the same 21-column canonical header and this synthetic `artifacts.example_input()` row:

```text
0,0,0,28,0,0,0,0,0,0,0,0,0,3,0,2,0,0,9,4,6
```

The field-guide CSV schema is exactly:

```text
feature,label,representation,minimum,maximum,accepted_values
```

It contains one row per feature in canonical order. The template, guide, and their bytes are regenerated from executable sources on demand and are deterministic.

## 4. File-level structural contract

Accepted files have all of the following properties:

- Python `bytes` input;
- byte length `1` through `2,097,152` inclusive;
- strict UTF-8, optionally with one leading UTF-8 BOM;
- comma delimiter and one exact header record;
- exactly 1 through 1,000 logical data records inclusive;
- exactly the 21 feature headers with no duplicates or additions;
- exactly 21 cells in every data record.

The accepted schema remains exactly 21 columns. As an abuse-safety inspection guard, a parsed header with more than 64 columns is rejected immediately after delimiter detection and before duplicate/schema detail generation. This does not broaden accepted input; it prevents a very wide invalid header from driving detailed diagnostics.

The complete safe structural error taxonomy is:

| Code | Boundary |
|---|---|
| `invalid_upload_type` | input is not `bytes` |
| `empty_file` | zero bytes or no CSV records after decoding |
| `file_too_large` | more than 2 MiB before decode/parse |
| `unsupported_encoding` | invalid strict UTF-8 |
| `null_byte` | NUL-bearing text, including UTF-16-like input |
| `malformed_csv` | strict CSV parser error |
| `missing_header` | blank or data-like first record |
| `unsupported_delimiter` | semicolon, tab, or pipe-delimited header |
| `too_many_columns` | header contains more than the 64-column inspection cap |
| `duplicate_headers` | repeated exact header name |
| `target_column_not_allowed` | `Diabetes_binary` is present |
| `exported_index_not_allowed` | blank/`index`/`level_0`/`row_number`/`Unnamed:*` header |
| `identifier_not_allowed` | identifier-like header such as `patient_id` |
| `column_schema_mismatch` | any other missing/additional header mismatch |
| `malformed_record` | data-record width differs from 21 |
| `too_many_rows` | logical data record 1,001 is reached |
| `no_data_rows` | valid header with no logical data record |

Any structural error rejects the whole file before scorer selection. No table or download is returned. Every `BatchFileError` message has an absolute 1,000-character backstop. Duplicate/unexpected-name diagnostics show at most the first five names, each represented with at most 80 characters, plus total/remainder counts. The 10,000-header/118,890-byte regression now produces a 117-character count-only `too_many_columns` error instead of reflecting all names.

## 5. Row validation and partial success

Every raw cell is retained as text for reporting. Validation strips surrounding whitespace only for interpretation and applies these ordered rules:

1. empty/whitespace-only: `<feature>: value is missing`;
2. case-insensitive `true`/`false`: `<feature>: boolean literals are not allowed`;
3. non-`Decimal` text, including underscore numeric forms: `<feature>: value is not numeric`;
4. `NaN`/`Inf`/infinity: `<feature>: value must be finite`;
5. non-integral finite number: `<feature>: value must be a whole number`;
6. outside the executable interval: `<feature>: value is outside [<min>, <max>]`.

Rules 5 and 6 can both apply. Errors are joined by ` | ` in canonical feature order. The engine never fills, rounds, clips, casts a fraction to integer, corrects, drops, deduplicates, or reorders a row.

For a structurally valid mixed file:

- all rows remain in the result;
- `row_number` is the one-based logical data-record position;
- only fully valid rows enter the scoring frame;
- valid duplicates enter and leave in their original positions;
- invalid rows have `validation_status = invalid`, all errors, and an empty probability;
- summary counts are exact totals over the retained rows.

## 6. Exact output schema and serialization

The combined table and download use exactly:

```text
row_number,
HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,
PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,
GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income,
validation_status,validation_errors,model_probability
```

The physical CSV has that sequence on one header line. It is comma-delimited UTF-8 without BOM, uses LF line endings, standard minimal quoting, and ends each record with LF. A valid probability is formatted with exactly 15 decimal places. An invalid probability is an empty field. Stable error text and ordering plus the fixed float format make repeated processing/export produce identical bytes.

CSV injection protection applies to every user-provided feature cell in the download. If the first non-whitespace character is `=`, `+`, `-`, or `@`, the exported cell gains one leading apostrophe. Error strings contain only fixed feature/rule text. The raw retained result is not altered, and such a value remains invalid and unscored.

## 7. Scoring fidelity

`src.artifacts.select_probability_scorer()` is now the single validated model/calibrator selection boundary for individual and batch serving. `predict_probability_frame()` performs the shared positive-class vector call and rejects a wrong result shape, non-finite probability, or value outside `[0, 1]`. `predict_risk_probability()` delegates to these helpers, so the P8 route is not duplicated.

For one batch:

1. structural validation completes;
2. the artifact bundle is validated once and the D-018 scorer is selected (`model` for the official `calibration_method = none`, otherwise its validated calibrator);
3. all row values are validated;
4. valid canonical rows are scored in one DataFrame call;
5. results are mapped back to original positions;
6. tests compare vectorized and individual values with absolute tolerance `1e-12`.

The four official profiles remained:

| Profile | Probability | Display |
|---|---:|---:|
| low-risk young/healthy | `0.0030013847190189` | `0.3%` |
| high-risk cardiac/smoker | `0.6000009431177805` | `60.0%` |
| high-risk poor health | `0.6999879500512149` | `70.0%` |
| high-risk severe-obesity/cardiac | `0.7990007166974580` | `79.9%` |

No threshold, category, diagnosis, recommendation, decision field, SHAP explanation, scenario, or population statistic is created.

## 8. Streamlit behavior, privacy, and state

The application exposes `Individual prediction` and `Batch CSV prediction` as separate branches. The individual branch preserves P8, P9, and P10 behavior. The batch branch contains:

- deterministic template and field-guide downloads;
- an optional field-guide table;
- a CSV-only uploader whose transport and engine limits are both 2 MiB;
- an explicit `Process batch` action;
- an explicit `Reset batch` action;
- total/valid/invalid metrics;
- a preview capped at 25 original-order rows;
- one deterministic full result download;
- controlled structural and internal failure messages;
- D-018 probability-contract caption, D-019 probability-only wording, privacy text, and the medical disclaimer.

Uploaded bytes, parsed values, errors, result objects, and download bytes live only in the active Streamlit session. The application does not write them to the repository or another filesystem location, database, object store, analytics service, external logger, or shared cache. The trusted artifact/explainer cache contains no user content.

A retained result stores the artifact SHA-256 and upload SHA-256. Upload removal/replacement, reset, parse/scoring/export failure, or artifact change invalidates the result before rendering, so an older table/download cannot masquerade as current. Fixed download names avoid unsafe filename rendering.

Source/headless guards inspect both `app/streamlit_app.py` and `src/batch.py` and prohibit raw-project CSV access, `prepare_data`, fitting/calibration, artifact generation, disk writes, remote fetches, external analytics/logging, global/batch SHAP, batch scenarios, and cross-session user-data caching.

## 9. Performance and memory

Benchmark target: the simultaneous accepted extreme against the official frozen artifact -- exactly 2,097,152 uploaded bytes and 1,000 valid synthetic rows -- including byte parsing, complete row validation, vectorized scoring, and deterministic result export. The fixture starts from `artifacts.example_input()` and pads accepted surrounding whitespace in the first numeric cell; it contains no dataset row and remains valid without changing its interpreted values.

- upload: 2,097,152 bytes;
- result: 2,126,110 bytes;
- separate cold first run: 2.1176360 seconds;
- 30 warm runs minimum: 0.1051285 seconds;
- 30 warm runs median: 0.1179472 seconds;
- 30 warm runs p95: 0.1294039 seconds;
- 30 warm runs maximum: 0.1627905 seconds;
- `tracemalloc` incremental current: 4,643,355 bytes;
- `tracemalloc` incremental peak: 12,869,837 bytes (12.2736 MiB);
- repeated result bytes: identical.

Accepted bounds remain warm processing at or below 2 seconds and incremental Python peak memory at or below 50 MiB. The observed maximum used 8.1% of the latency allowance and the peak used 24.5% of the memory allowance. The cold observation is reported separately and was not used to redefine the predeclared warm criterion.

The benchmark can be reproduced from the project virtual environment by generating the exact-size fixture with the same padding algorithm as `tests/test_batch.py::exact_size_csv`, loading `artifacts.load_artifact()`, timing `batch.process_batch()` plus `batch.result_csv_bytes()` with `time.perf_counter()`, recording the cold call separately, and collecting 30 warm calls. `tracemalloc.start()` after bundle/payload construction and `get_traced_memory()` around one complete process/export call record incremental Python allocations. The exact byte and row limits are additionally exercised by focused tests.

## 10. Verification executed

All pytest temporary directories were placed outside the repository with `--basetemp`.

| Command/scope | Result |
|---|---|
| `python -m pytest tests/test_batch.py -v -p no:cacheprovider --basetemp <temp>` | 50 passed, 1 warning, 6.26 s |
| `python -m pytest tests/test_batch.py tests/test_app.py -v -p no:cacheprovider --basetemp <temp>` | 91 passed, 15 warnings, 22.94 s |
| `python -m pytest tests -v -p no:cacheprovider --basetemp <temp>` on the hardened code state | 382 passed, 15 warnings, 59.40 s |
| `python -m pip check` | `No broken requirements found.` |
| `python -m compileall src app tests` | passed |
| `git diff --check` | passed; only Git's existing LF-to-CRLF checkout notices were printed |

The warnings were 14 Matplotlib/PyParsing deprecations from installed dependencies and one joblib physical-core-detection fallback. No project test failed.

### Local visual review

The application was served on `http://127.0.0.1:8765`, its health endpoint returned `ok`, and Edge headless was controlled through local Chrome DevTools Protocol for a real rendered-DOM review. The reviewer switched from the default individual branch to batch and confirmed:

- only the batch interface was visible;
- template, guide, uploader, process, and reset controls rendered cleanly;
- the uploader showed `Limit 2MB per file`, aligned with the 2 MiB engine bound;
- privacy text, D-018 caption, partial-success explanation, and medical disclaimer were visible;
- no individual form, P9 explanation, or P10 scenario appeared in the batch branch;
- layout at 1,440 by 1,400 pixels had no visible overlap or truncation in the reviewed initial batch state.

Headless tests supplied valid and mixed uploads and covered summaries, bounded preview, download, reset/replacement, stale hashes, oversized/malformed files, engine/export failures, and absence of batch SHAP/scenarios. The Streamlit and Edge review processes were stopped afterward. No public URL was opened or tested.

## 11. Artifact integrity

Both reviewed artifacts were hashed explicitly after implementation and were not regenerated:

```text
models/diabetes_risk_model.joblib
957c14ff5a490bbc60822121a889f92ee2a6a20f797eef741a710d887ecc9216

models/shap_background_v1.json
73d1ff21e3c98ee79fa7d72758517047f13e5f454d7ff95edb1ee93812cca120
```

These values exactly match the expected P8/P9 frozen hashes.

## 12. Limitations

- CSV is the only accepted batch format; URLs, archives, spreadsheets, and remote inputs are excluded.
- The input carries only model features. Identifiers, target values, arbitrary metadata, free text, and passthrough columns are intentionally unsupported.
- `Age` is a BRFSS model group code, not an exact age.
- The engine limit is 1,000 rows and 2 MiB; the UI preview is 25 rows.
- Formula neutralization protects the generated CSV's user feature cells but cannot control how third-party spreadsheet software treats files after a user modifies them.
- Batch output is row-level validation plus probability only. It does not explain rows, compare scenarios, assign risk labels, diagnose, recommend, or support population-level conclusions.
- The model remains an uncalibrated portfolio estimator based on self-reported BRFSS 2015 data and inherits the limitations documented for P8-P10.

## 13. Reproduction steps

From the repository root with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_batch.py -v -p no:cacheprovider --basetemp "$env:TEMP\diabetes-risk-p11-batch"
.\.venv\Scripts\python.exe -m pytest tests\test_app.py tests\test_reference_profiles.py tests\test_artifacts.py tests\test_explainability.py tests\test_scenarios.py -v -p no:cacheprovider --basetemp "$env:TEMP\diabetes-risk-p11-affected"
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider --basetemp "$env:TEMP\diabetes-risk-p11-full"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall src app tests
git diff --check
Get-FileHash models\diabetes_risk_model.joblib -Algorithm SHA256
Get-FileHash models\shap_background_v1.json -Algorithm SHA256
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

For manual local workflow review, select `Batch CSV prediction`, download the template, process it unchanged, then duplicate a row and make one `BMI` cell invalid to observe partial success and the blank invalid probability. Do not use project data for this check.

## 14. External closure boundary

The following are intentionally not complete in this local implementation task:

1. manual code review and authorization to stage/commit;
2. commit and push;
3. Streamlit Community Cloud deployment/reboot;
4. mandatory public smoke with one small valid template-derived file;
5. mandatory public smoke with one mixed-validity file, including safe download inspection.

Until all five occur, US-0613 remains In Progress, P11 remains Ready, and P12-P13 remain Future. This work performed no `git add`, stage, commit, push, deployment, public smoke test, artifact generation, or history modification.
