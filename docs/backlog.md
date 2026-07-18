# Backlog

## Workflow

This backlog is a living document. The roadmap defines broad direction; this file tracks user stories, tasks, priorities, and status.

Epics are ordered by MVP delivery order, not by numeric ID: Epic E7 (MVP Documentation and Deployment) appears before Epic E6 (Post-MVP Enhancements) because E7 is part of the MVP and E6 is not. Likewise, Epic E8 (Model Comparison and Selection, P5) appears before Epic E5 (Streamlit MVP, P6): E8 was split out of the former Epic E4 during a P4 backlog refinement and given a fresh, non-colliding ID (`US-05xx` was already in use under Epic E5) rather than renumbering existing epics.

## Status Values

- **To Do:** identified but not yet ready for implementation.
- **Ready:** sufficiently clear to start.
- **In Progress:** actively being worked on.
- **Review:** implemented and awaiting validation or documentation update.
- **Done:** completed and validated.
- **Deferred:** intentionally postponed.

## Priority Values

- **P0:** required for MVP correctness.
- **P1:** important for MVP quality.
- **P2:** useful post-MVP improvement.
- **P3:** optional enhancement.

## Epic E0: Planning and Analysis

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0001 | As a portfolio reviewer, I want the project purpose and scope to be explicit so that I can understand what the project demonstrates. | P0 | Done | Project charter exists and defines purpose, scope, non-scope, constraints, and disclaimer. |
| US-0002 | As the developer, I want a roadmap and backlog structure so that future work can be planned progressively. | P0 | Done | Roadmap and backlog exist with phases, status values, priorities, and MVP boundary. |
| US-0003 | As the developer, I want initial ML assumptions documented so that dataset and target choices are traceable. | P0 | Done | ML analysis plan records the current local dataset state and accepted binary target decision. |
| US-0004 | As the developer, I want major decisions logged so that project rationale is not lost. | P1 | Done | Decision log exists with initial accepted and pending decisions. |

## Epic E1: Project Setup and Data Governance

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0101 | As the developer, I want a clean project folder structure so that source code, notebooks, app files, tests, data, and artifacts are separated. | P0 | Done | Initial folders exist with placeholders where needed and the README points to the structure. |
| US-0102 | As the developer, I want a clear raw data policy so that raw data handling is intentional and reproducible. | P0 | Done | Raw CSV files are ignored, the CC0 dataset source is documented, and data acquisition instructions are planned. |
| US-0103 | As the developer, I want the target formulation resolved before modeling so that all downstream work is consistent. | P0 | Done | Decision made to use the official binary CSV with `Diabetes_binary` as target. |
| US-0104 | As a reviewer, I want dependency management to be explicit so that the project can be reproduced. | P0 | Done | Python version is selected and a pinned dependency file strategy is documented. |
| US-0105 | As a reviewer, I want clear data acquisition instructions so that I can reproduce the project without the raw CSV being committed. | P0 | Done | README or data documentation identifies the Kaggle dataset URL, required file, CC0 license, expected local path, and credential caveats. |
| US-0106 | As the developer, I want a testing strategy before implementation so that the MVP includes meaningful checks instead of vague "tests". | P1 | Done | Backlog identifies initial pytest targets for data validation, splits, pipeline behavior, and artifact loading. |

### Candidate Tasks for E1

All E1 setup tasks were completed in Iteration 1: the folder structure exists with lightweight placeholders, the raw dataset was moved to `data/raw/` and remains git-ignored, `data/README.md` documents acquisition and the data handling policy, and Python 3.12 with a pinned `requirements.txt` resolves the dependency strategy (D-012).

Intentionally deferred from E1:

- Test file placeholders will be created together with the first code modules in Epic E3, as planned; the pytest targets themselves are documented in the ML analysis plan.
- Validating the pinned environment (`pip install -r requirements.txt` in a fresh virtual environment) is the first task of the next iteration, before EDA work starts.

## Epic E2: Data Understanding and EDA

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0201 | As a reviewer, I want to see the dataset schema validated so that I can trust the inputs used for modeling. | P0 | Done | Feature names, target, data types, ranges, and unexpected values are reported. |
| US-0202 | As a reviewer, I want class balance and duplicate behavior documented so that evaluation choices are justified. | P0 | Done | Target distribution, missing values, duplicate count, and duplicate policy are documented. |
| US-0203 | As a non-technical viewer, I want the EDA to explain why imbalance matters so that model metrics are understandable. | P1 | Done | EDA narrative explains class imbalance and why accuracy is insufficient. |

### Candidate Tasks for E2

Implementation should focus on a reproducible EDA artifact, not on modeling or data preparation code yet.

- [x] Validate the environment in a fresh virtual environment with `pip install -r requirements.txt`; if a pin fails, adjust that pin explicitly and document the reason.
- [x] Create the first EDA notebook under `notebooks/`, using a clear name such as `01_data_understanding_eda.ipynb`.
- [x] Load the dataset from `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` and fail clearly if the file is missing.
- [x] Verify dataset shape, column order, target column, and row count against the documented expectations.
- [x] Classify features into initial modeling groups: binary indicators, ordinal indicators, and numeric indicators.
- [x] Report data types and feature ranges for each feature group.
- [x] Check missing values and unexpected values.
- [x] Count duplicate rows and record an initial duplicate interpretation without dropping rows in E2 unless there is a documented reason.
- [x] Analyze target distribution, positive class prevalence, and base-rate implications for baseline evaluation.
- [x] Add basic descriptive tables or plots for high-signal variables such as `BMI`, `GenHlth`, `Age`, `Income`, and `HighBP`.
- [x] Add a lightweight Spearman correlation review to identify obvious feature relationships or redundancy, without removing features in E2.
- [x] Analyze memory usage and record downcasting recommendations for E3, without permanently changing dtypes in E2.
- [x] Explain why class imbalance makes accuracy insufficient as the main evaluation metric.
- [x] Capture EDA findings and convert any cleaning, validation, or split implications into candidate E3 tasks.

### Definition of Done for E2

- [x] The EDA notebook can run top-to-bottom from the documented local raw data path.
- [x] The notebook reports schema, missing values, duplicates, feature ranges, and target imbalance.
- [x] The EDA records initial feature groups, positive class prevalence, correlation observations, and memory optimization recommendations.
- [x] The EDA includes a short written interpretation suitable for a portfolio reviewer.
- [x] No model training, balancing, train/test splitting, or feature engineering is introduced in E2.
- [x] Follow-up tasks for E3 are updated based on the findings.

All E2 acceptance criteria are satisfied; see `notebooks/01_data_understanding_eda.ipynb` and Iteration 2 in `docs/iteration-log.md` for findings.

## Epic E3: Reproducible Data Preparation

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0301 | As the developer, I want reusable data loading and validation code so that notebooks and training scripts share the same logic. | P0 | Done | Data module loads the selected dataset and validates schema/ranges. |
| US-0302 | As a reviewer, I want stratified train/calibration/test splits so that model evaluation is trustworthy. | P0 | Done | Split code is reproducible, stratified, and preserves the selected positive prevalence in each split. |
| US-0303 | As the developer, I want tests for data preparation so that schema, ranges, downcasting, and splits do not silently regress. | P1 | Done | pytest tests cover required columns, expected target values, feature ranges, downcasting, and split class proportions. |

### Candidate Tasks for E3

Sourced from the P2 EDA findings in `notebooks/01_data_understanding_eda.ipynb` (see Iteration 2 in `docs/iteration-log.md`):

- [x] Resolve D-014 before creating splits: exact duplicate rows will be kept for MVP data preparation. P2 found 24,206 exact duplicate rows (~9.5% of the dataset), heavily skewed negative; dropping them would shift positive prevalence from ~13.9% to ~15.3%. P3 must preserve the observed ~13.9% positive prevalence.
- [x] Define the P3 data contract in a reusable data module, likely `src/data.py`: raw data path, expected column order, target column (`Diabetes_binary`), feature groups, and valid value ranges for binary, ordinal, and numeric indicators.
- [x] Implement raw dataset loading with clear failure behavior when `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` is missing.
- [x] Implement validation for required/exact columns, missing values, integer-like values, target values, feature ranges, and duplicate-row policy compliance.
- [x] Apply the accepted duplicate policy before splitting: retain exact duplicate rows, keep the full 253,680-row dataset, and make the selected row count/prevalence explicit in returned metadata, logs, or reports.
- [x] Apply safe lossless downcasting to `uint8` only after validation confirms all values are whole numbers within the documented ranges.
- [x] Implement a reproducible stratified 70/10/20 train/calibration/test split with a fixed random seed, preserving the selected positive prevalence in train, calibration, and test sets.
- [x] Record D-015: P3 returns splits in memory instead of writing processed split files to `data/processed/`; `data/processed/` stays empty until a consumer needs persisted files.
- [x] Add pytest coverage for loading, validation failures, duplicate-policy behavior, lossless `uint8` downcasting, split sizes, split reproducibility, and split class-proportion preservation.
- [x] Keep P3 limited to data preparation and splitting: do not add balancing, SMOTE, model training, feature engineering, Streamlit/app work, calibration, SHAP, or model-selection logic.
- [x] Carry forward the EDA's correlation observations (`GenHlth`/`PhysHlth`/`DiffWalk`, `Education`/`Income`) into P4 model design discussions without dropping features solely on correlation grounds in P3.

All E3 acceptance criteria are satisfied; see `src/data.py`, `tests/test_data.py`, D-014, D-015, and Iteration 3 in `docs/iteration-log.md`.

## Epic E4: Baseline Modeling

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0401 | As a reviewer, I want a DummyClassifier baseline so that all real models are compared against a trivial reference. | P0 | Done | Dummy metrics are computed and included in comparison outputs. |
| US-0402 | As a reviewer, I want a first interpretable model so that a non-trivial reference performance exists before adding model complexity. | P0 | Done | Logistic Regression is trained through reusable P4 modeling code, likely `src/modeling.py`, using the `src.data.prepare_data()` splits and evaluated with ROC-AUC, PR-AUC, recall, precision, F1, confusion matrix, and accuracy (secondary) on train and test only. |

### Candidate Tasks for E4

- [x] Create reusable P4 modeling code, likely `src/modeling.py`, for baseline model construction, fitting, prediction, and metric calculation; keep notebooks optional for narrative/reporting rather than as the only implementation.
- [x] Use `src.data.prepare_data()` as the exclusive data entry point for P4; do not reload or re-split the raw data ad hoc.
- [x] Separate features (`X`) and target (`y`) for the train and test splits; keep the calibration split unused in P4 so it remains reserved for later probability calibration work.
- [x] Train a `DummyClassifier` baseline (most-frequent or stratified strategy).
- [x] Train `LogisticRegression` as the first interpretable model.
- [x] Define minimal preprocessing needed for Logistic Regression (e.g. scaling), without heavy feature engineering.
- [x] Evaluate both models on train and test with ROC-AUC, PR-AUC, recall, precision, F1, and confusion matrix, reporting accuracy only as a secondary metric.
- [x] Keep results in memory or in a lightweight report; do not serialize any model artifact in P4.
- [x] Add pytest coverage: the pipeline fits on a small sample, `predict_proba` returns probabilities in `[0, 1]`, metrics compute without errors, training never touches calibration/test rows, and P4 evaluation does not consume calibration rows.
- [x] Review the EDA correlation observations (`GenHlth`/`PhysHlth`/`DiffWalk`, `Education`/`Income`) during Logistic Regression preprocessing, without dropping features solely on correlation grounds.
- [x] Keep P4 limited to the Dummy and Logistic Regression baseline: do not add tree-based candidates, formal model comparison/selection, or serialization policy, which belong to Epic E8 (P5). Do not add calibration-set evaluation, SHAP, deep calibration, Streamlit/app work, fairness analysis, or advanced threshold tuning; those belong to P8 or later phases.

All E4 acceptance criteria are satisfied; see `src/modeling.py`, `tests/test_modeling.py`, and Iteration 4 in `docs/iteration-log.md`.

## Epic E8: Model Comparison and Selection

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0801 | As a reviewer, I want a tree-based candidate compared formally against the P4 baseline so that trade-offs are visible. | P0 | Done | Dummy, Logistic Regression, and at least one restrained tree-based candidate (preferably `HistGradientBoostingClassifier` or `RandomForestClassifier`) are trained through the P3/P4 contracts and evaluated with the same train/test metric protocol. |
| US-0802 | As the developer, I want a primary candidate model selected and justified so that later phases serve a deliberate choice. | P0 | Done | Selection criteria are defined before implementation, the selected model is documented with metric-based rationale, and PR-AUC plus positive-class recall/precision/F1 are prioritized over accuracy. |
| US-0803 | As the developer, I want a model serialization policy so that Streamlit only loads known project artifacts. | P1 | Done | Serialization format and timing are documented consistently with D-010 (`joblib`) while D-013 remains pending for deployment artifact distribution; any artifact write requires an explicit decision and a local load/predict check. |

### Candidate Tasks for E8

- [x] Use the existing P3/P4 contracts: consume `src.data.prepare_data()` or supplied `DataSplits`, convert splits through the existing train/test helpers, and avoid ad hoc raw-data loading, re-splitting, or processed split files.
- [x] Reuse or extend `src/modeling.py` for P5 model builders, evaluation, and comparison unless a separate orchestration module has a clear reason. P5 extends `src/modeling.py`; no separate module was needed.
- [x] Train Dummy, Logistic Regression, and at least one restrained tree-based candidate on the train split only. Prefer a practical MVP candidate such as `HistGradientBoostingClassifier` or `RandomForestClassifier` with deterministic settings. `HistGradientBoostingClassifier` with library defaults and a fixed seed was chosen.
- [x] Decide during P5 implementation whether a simple imbalance-aware variant such as `class_weight="balanced"` belongs in the comparison. Do not add SMOTE, advanced resampling, threshold tuning, or calibration unless P5 scope is explicitly expanded first. One variant was included: `class_weight="balanced"` Logistic Regression, isolating the reweighting effect against the P4 baseline.
- [x] Evaluate every candidate on train and test only with the same metric protocol used in P4: ROC-AUC, PR-AUC, positive-class recall, precision, F1, confusion matrix, and accuracy as secondary context.
- [x] Keep the calibration split untouched so it remains reserved for P8 probability calibration.
- [x] Produce an in-memory comparison table or structured result with metrics by model and split; do not require notebooks as the only comparison path. `compare_models()` returns structured results and `comparison_table()` a long-format DataFrame.
- [x] Define selection criteria before comparing models. Prioritize PR-AUC and positive-class recall/precision/F1 because the selected population has ~13.9% positive prevalence; treat accuracy as secondary because an always-negative model already scores about 86%.
- [x] Select and document the primary model with metric-based rationale. Do not create a final model-selection decision before the P5 metrics exist. D-016 resolved: `HistGradientBoostingClassifier` selected primarily on test PR-AUC, with strongest ROC-AUC/precision and a small train/test PR-AUC gap; the balanced Logistic Regression variant led default-threshold recall/F1 but not ranking metrics.
- [x] Confirm the selected model's serialization policy before writing artifacts: D-010 accepts `joblib` for MVP serialization, while D-013 still governs how artifacts reach deployment. Do not write model artifacts in P5 unless the timing is explicitly decided and tested. D-017 defers the artifact write to the start of P6, paired with a load/predict check; no artifact was written in P5.
- [x] Add pytest coverage expectations for tree-based fit/`predict_proba`, comparison result structure, deterministic selection behavior, no calibration-split usage, no ad hoc re-splitting/reloading, and no artifact writes unless explicitly authorized by a decision.
- [x] Keep P5 limited to model comparison and selection: no Streamlit/app work, SHAP, fairness analysis, deep calibration, advanced threshold tuning, batch prediction, or scenario exploration.

All E8 acceptance criteria are satisfied; see `src/modeling.py`, `tests/test_model_comparison.py`, D-016, D-017, and Iteration 5 in `docs/iteration-log.md`.

## Epic E5: Streamlit MVP

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0504 | As the developer, I want the D-016 model serialized into a reusable local artifact so that the Streamlit app can load a trained model without retraining. | P0 | Done | A small artifact helper module, likely `src/artifacts.py`, trains/serializes the D-016 `HistGradientBoostingClassifier` with `joblib` (D-010) at the start of P6 (D-017 timing), saves metadata (feature order, target name, model type, key P5 comparison metrics, and package versions or other minimal reproducibility metadata where feasible) beside the model, and a local load/predict check confirms the reloaded artifact returns probabilities in `[0, 1]` before the app depends on it. |
| US-0501 | As a user, I want to enter health indicators and receive an estimated risk percentage so that the model output is understandable. | P0 | Done | A minimal local Streamlit app, likely `app/streamlit_app.py`, loads the artifact through the artifact helper module, presents a single-case input form covering all 21 `src.data.FEATURE_COLUMNS`, grouped by binary/ordinal/numeric type where practical, validates inputs against the P3 feature ranges, assembles a one-row DataFrame in the exact training feature order, and calls `predict_proba` to display an educational risk percentage. |
| US-0502 | As a reviewer, I want to see model limitations in the app so that the project does not overclaim medical value. | P0 | Done | A clear medical disclaimer and limitations text (educational estimate only, self-reported survey data, not a diagnosis) are visible alongside every prediction in the MVP app. |
| US-0503 | As the developer, I want a smoke test for the serving path so that the app-facing prediction helper can produce probabilities in `[0, 1]` without launching Streamlit. | P1 | Done | Tests exercise input validation, the input-to-DataFrame helper, and the app-facing prediction helper directly (no Streamlit runtime), confirming preserved feature order and a `predict_proba` output in `[0, 1]`. |

### Candidate Tasks for E5

- [x] Keep P6 scoped to a local functional Streamlit MVP: no public deployment, no artifact distribution decision, and no resolution of D-013 in this phase. A durable test keeps artifact loading local-only (no remote sources or deployment secrets, per the ML analysis plan) without blocking legitimate P7 deployment files; D-013 remains Pending.
- [x] At the start of P6, train the D-016 `HistGradientBoostingClassifier` through the existing P3/P4/P5 contracts (`src.data.prepare_data()` and the `src/modeling.py` builders); do not reload or re-split raw data ad hoc. `src/artifacts.py` consumes `prepare_data()` and the P5 builder only; a source-guard test forbids ad hoc reloading/re-splitting.
- [x] Serialize the fitted model with `joblib`, per D-010, following the D-017 timing: no artifact exists before P6 starts. `save_artifact()` writes atomically, requires the `.joblib` extension, and refuses repository locations outside `models/` (only `models/*.joblib` is git-ignored).
- [x] Create a small artifact helper module, likely `src/artifacts.py`, that saves and loads the model artifact and its metadata as a single reusable contract shared by the app and tests. Implemented as `src/artifacts.py` with a single-bundle layout: the fitted model and its metadata travel in one joblib file, so they cannot drift apart.
- [x] Define the expected artifact contents: the fitted model/pipeline, the `src.data.FEATURE_COLUMNS` feature order, the `src.data.TARGET` name, the model type/name, key P5 comparison metrics for the selected model, and package versions or other minimal reproducibility metadata where feasible. `REQUIRED_METADATA_KEYS` covers schema version, model identity (D-016), exact feature order, target, random seed, P5-protocol train/test metrics, package versions, and creation timestamp.
- [x] Add a local load/predict verification (reload the saved artifact and confirm `predict_proba` returns probabilities in `[0, 1]`) before the app is built to depend on it. `load_predict_smoke_check()` runs as part of `python -m src.artifacts`; the real-data run returned example-case probability 0.0254.
- [x] Keep `models/*.joblib` git-ignored; do not commit generated artifacts, and do not treat this as an artifact distribution decision for public deployment (D-013 stays pending). Confirmed with `git check-ignore`; the generated artifact never appears in `git status`.
- [x] Create a minimal Streamlit app, likely `app/streamlit_app.py`, that loads the local artifact through the artifact helper module. The app loads once per process via `st.cache_resource` and never trains.
- [x] Build a single-case input form covering all 21 features in `src.data.FEATURE_COLUMNS`, grouping inputs by binary, ordinal, and numeric feature groups where practical for readability. Yes/no checkboxes (plus a female/male selector for `Sex`), human-readable ordinal selectboxes without exposing internal codes, an exact-age input mapped automatically to the matching BRFSS age interval, and bounded numeric inputs.
- [x] Validate user inputs against the P3 feature ranges/schema contract (`src.data.VALUE_RANGES`) before building a prediction input. `validate_input_values()` enforces exact features, numeric integer-like values, and the documented ranges.
- [x] Build a one-row DataFrame from validated inputs in the exact training feature order (`src.data.FEATURE_COLUMNS`) before calling the model. `input_to_dataframe()` returns one `uint8` row in exact training order.
- [x] Call `predict_proba` on the one-row input and display the positive-class probability as an educational risk percentage. `predict_risk_probability()` returns the raw positive-class probability; no custom threshold is applied (P8 scope).
- [x] Display a clear medical disclaimer and limitations text (educational estimate only, self-reported survey data, not a diagnosis) alongside every prediction.
- [x] Keep P6 limited to the local MVP: no calibration, threshold tuning, SHAP, fairness analysis, batch prediction, scenario exploration, public deployment, or artifact distribution decisions. Public deployment and artifact distribution belong to P7; calibration and threshold analysis belong to P8; SHAP belongs to P9; fairness analysis, batch prediction, and scenario exploration remain later-phase work.
- [x] Leave D-013 pending unless artifact distribution for public deployment is actually decided during P6; do not resolve it or add a new decision without a real architectural choice behind it. No distribution choice was made; D-013 remains Pending for P7.
- [x] Add pytest coverage expectations: artifact save/load round-trip; local load/predict returns a probability in `[0, 1]`; metadata includes feature order and the selected model identity; the input-to-DataFrame helper preserves feature order; input validation rejects missing/out-of-range values; an app-facing prediction helper works without launching Streamlit; and no public deployment logic is introduced in P6. Covered by `tests/test_artifacts.py` (50 tests) and `tests/test_app.py` (10 tests), including exact-age interval conversion, a headless render/predict check with Streamlit's `AppTest`, and hardened-validation cases added after a P6 review (impostor models, corrupt files, unignored save locations).

All E5 acceptance criteria are satisfied; see `src/artifacts.py`, `app/streamlit_app.py`, `tests/test_artifacts.py`, `tests/test_app.py`, and Iteration 6 in `docs/iteration-log.md`.

## Epic E7: MVP Documentation and Deployment

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0701 | As a reviewer, I want complete run instructions so that I can reproduce the environment, training, and app locally. | P0 | Done | The README, with links to supporting data documentation where useful, covers Python 3.12 virtual-environment creation and activation, installation from `requirements.txt`, the exact CSV download and local path, artifact generation with `python -m src.artifacts`, tests with `python -m pytest tests -v -p no:cacheprovider`, and app launch with `python -m streamlit run app/streamlit_app.py`, all through the same environment interpreter. It includes clear PowerShell equivalents where useful and explains how to detect and replace an artifact created with incompatible Python, scikit-learn, or joblib versions. Validation from a clean clone or environment must show that a reviewer with no prior project knowledge can reproduce training, tests, and the application. |
| US-0702 | As the developer, I want an explicit artifact distribution policy for deployment so that the deployed app can load a trained model even though model files are git-ignored. | P0 | Done | Before the first public deployment, D-013 records a formal comparison and final choice among a controlled Git exception, a GitHub Release asset, and generation during build, covering artifact size, reproducibility, security, maintenance, Git history, network dependency, cold start, and dataset availability. Only the selected alternative is implemented. The official artifact is generated with Python 3.12 and the pinned versions; the app loads only the known project artifact and never trains, downloads arbitrary artifacts, or requires the CSV at runtime. The chosen path is verified from a clean clone and in the deployment environment. |
| US-0703 | As a user, I want the MVP deployed on Streamlit Community Cloud so that the app is publicly accessible. | P0 | Done | A GitHub-backed Streamlit Community Cloud deployment runs on Linux with Python 3.12, installs an evidenced deployment-suitable `requirements.txt` without an unnecessary alternative requirements file, and loads the official artifact. P7 defines four reference profiles with exact 21-feature inputs and locally verified expected displays; all four must be submitted publicly and match 0.3%, 60.0%, 70.0%, and 79.9% before closure. Missing/corrupt-artifact behavior may be verified through deployment-equivalent headless tests that exercise the same app loading and error-rendering path, rather than deliberately breaking the healthy public deployment. The deployment publishes no raw CSV, secrets, or user data, and the README records the public URL and final run instructions. |

### Candidate Tasks for E7

P6 provides the planning evidence for this refinement: the validated single-bundle artifact is approximately 263 KB, the app loads it once and never trains, the form covers all 21 features, and exact age is converted to the corresponding BRFSS group. These facts inform D-013 but do not select a distribution alternative. (During P7 implementation, the formal comparison was performed and D-013 was accepted: controlled Git exception; see `docs/decisions.md`.)

#### US-0701: Reproducible Instructions

- [x] Document creation and activation of a Python 3.12 virtual environment from a clean clone, including clear PowerShell commands such as `.venv\Scripts\Activate.ps1` and equivalent commands for other supported shells where useful. README "Run It Locally" step 1 covers PowerShell (`py -3.12 -m venv .venv`, `Activate.ps1`, execution-policy fallback) and macOS/Linux equivalents.
- [x] Document dependency installation through the environment interpreter with `python -m pip install -r requirements.txt`, and make it clear that subsequent project commands must use that same interpreter. README step 2 plus an explicit same-interpreter rationale (version-mismatched artifacts, wrong `streamlit`/`pytest` executables).
- [x] Document the selected Kaggle CSV, its exact filename, and its required location at `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`; link to the existing data acquisition instructions instead of duplicating or changing the raw-data policy. README step 3 links to `data/README.md`; the raw-data policy is unchanged.
- [x] Document the exact artifact-generation command, `python -m src.artifacts`, and explain that it performs offline training through the existing project contract before the app is launched. README step 4; also notes the step is optional because the official artifact ships with the repository (D-013).
- [x] Document the complete test command, `python -m pytest tests -v -p no:cacheprovider`. README step 5, including the raw-data skip behavior.
- [x] Document the app command, `python -m streamlit run app/streamlit_app.py`, so Streamlit uses the same environment interpreter as training and tests. README step 6.
- [x] Add troubleshooting for artifact/runtime incompatibility: show how to confirm Python, scikit-learn, and joblib versions against the pinned environment and artifact metadata; explain recognizable load/validation failures; and instruct the reviewer to rebuild the environment and regenerate the trusted artifact rather than bypass validation or load an unknown file. README "Troubleshooting: artifact and environment incompatibility" maps each failure mode to the rebuild-and-regenerate fix and shows how to inspect the artifact's recorded `package_versions`.
- [x] Follow the written instructions from a fresh clone or otherwise clean Python 3.12 environment, recording enough evidence that a reviewer with no prior project knowledge can reproduce artifact training, the full test suite, and the local app. Validated 2026-07-11 on an isolated copy of the repository with a fresh python.org CPython 3.12.1 venv (a different installation than the development environment's 3.12.7 base): pinned install resolved with `pip check` clean; without the CSV the committed artifact served 143 passing tests with exactly the 3 documented raw-data skips; after placing the CSV per instructions, `python -m src.artifacts` reproduced the D-016 metrics and the 0.0254 smoke probability, the full suite passed 146/146, and `python -m streamlit run app/streamlit_app.py` answered HTTP 200 on `/_stcore/health`. The committed repository content was subsequently exercised by the successful Community Cloud deployment.

#### US-0702: Artifact Distribution Policy

- [x] Measure and record the official artifact's size (currently approximately 263 KB) and treat a controlled Git exception as a reasonable candidate at that size, without marking it as selected or accepted before D-013 is resolved. Measured 268,815 bytes and recorded in D-013; the candidate was not selected until the comparison ran.
- [x] Compare the three D-013 alternatives formally: a controlled exception that versions the official artifact in Git, a trusted GitHub Release asset, and deterministic artifact generation during the deployment build. Recorded in the "D-013 Artifact Distribution Comparison" section of `docs/decisions.md`, grounded in current official Streamlit Community Cloud, scikit-learn persistence, and GitHub documentation (consulted 2026-07-11).
- [x] Evaluate each alternative against artifact size, reproducibility, deserialization security and provenance, maintenance burden, Git-history impact, runtime/build network dependency, cold-start behavior, and availability of the git-ignored dataset in the build environment. All criteria appear per alternative in the comparison table, plus Community Cloud viability and the training/serving separation.
- [x] Record the selected alternative, rationale, constraints, and verification approach in D-013 before the first public deployment; keep D-013 Pending until this comparison produces an actual choice. D-013 moved Pending -> Accepted on 2026-07-11: controlled Git exception, selected from the comparison evidence (the platform copies repository files and offers no pre-runtime hook; a runtime download would violate the local-only loading policy; build-time generation is not possible and would need the git-ignored CSV).
- [x] Implement only the alternative selected in D-013; do not add speculative support for the rejected alternatives. `.gitignore` now excepts exactly `models/diabetes_risk_model.joblib` (temporary `*.tmp.joblib` and alternative artifacts stay ignored); no release-download or build-generation code exists.
- [x] Generate the official artifact with Python 3.12 and the versions pinned in `requirements.txt`, and verify its metadata and local load/predict contract before distribution. Regenerated 2026-07-11 with Python 3.12.7 and the exact pins; verified metadata identity, package versions, feature order (metadata and fitted model), classes `[0, 1]`, D-016 hyperparameters, reproduced selection metrics (test PR-AUC 0.423065, ROC-AUC 0.826955), and smoke probability 0.025419.
- [x] Confirm that the app resolves and loads only the artifact and provenance defined by D-013, and rejects absent, corrupt, incompatible, or untrusted bundles with clear guidance. The app loads only `models/diabetes_risk_model.joblib` through the hardened validator; missing/corrupt/incompatible bundles fail with regeneration guidance, provenance and runtime versions are enforced against the P7 pins, and the official artifact's absence fails rather than skips the deployment-reference tests. The public deployment confirms the valid-artifact path in the target environment.
- [x] Confirm that Streamlit never trains or retrains a model, never downloads an arbitrary artifact, and never needs the raw CSV at runtime; build-time generation, if selected by D-013, must remain separate from the Streamlit serving process. Source-guard tests forbid training/data access in the app and remote sources in the serving path; the clean-environment run served predictions with no CSV present.
- [x] Verify the implemented policy end to end from both a clean environment and the actual deployment environment. The clean-environment committed-artifact and full-training journeys passed on 2026-07-11; the public Community Cloud deployment then confirmed that the repository-delivered artifact is found, validated, and loaded without the raw CSV or a model download.

#### US-0703: Public Deployment

- [x] Verify the current app and artifact path contract on a Linux, Python 3.12, Streamlit Community Cloud-compatible environment, including case-sensitive paths and repository-root execution assumptions. Static checks passed on 2026-07-11, and the successful public deployment confirmed the contract on the target platform.
- [x] Review whether the existing `requirements.txt` is suitable for deployment; create no alternate requirements file unless an observed platform constraint provides a documented reason. Reviewed 2026-07-11: it is the repository's only dependency file (so Community Cloud will select it), it pins every package the app imports, and the clean-environment install resolved with `pip check` clean. The EDA/notebook pins (jupyterlab, matplotlib, seaborn, imbalanced-learn, pytest) make the cloud install heavier than serving strictly needs -- a documented trade-off accepted to keep one reproducible environment; no second requirements file was created.
- [x] Configure the Streamlit Community Cloud deployment from repository `DiegoVillazonArce/diabetes-risk-ml`, branch `main`, entry point `app/streamlit_app.py`, and Python 3.12.
- [x] Review installation and startup behavior and record the final configuration; the public app starts successfully with the repository dependency and artifact contract.
- [x] Confirm that the deployed app finds, validates, and loads the official artifact delivered through the D-013 policy.
- [x] Run all four reference profiles against [brfss-diabetes-risk-estimator.streamlit.app](https://brfss-diabetes-risk-estimator.streamlit.app/) and confirm exact displays of 0.3%, 60.0%, 70.0%, and 79.9%. All four public submissions matched their recorded expectations, and the public form renders all 21 model features.
- [x] Verify boundary and representative exact ages map to the correct BRFSS age-group codes before inference. Automated and public checks confirm ages 24, 65, 70, and 80 map to codes 1, 10, 11, and 13 and display groups 18-24, 65-69, 70-74, and 80 or older respectively.
- [x] Define and verify a deployment-smoke reference set: `tests/reference_profiles.py` records four exact 21-feature profiles, boundary ages and BRFSS codes, official-artifact probabilities, expected displays (0.3%, 60.0%, 70.0%, 79.9%), a +/-0.0002 probability tolerance, and a 0.00025 display-rounding margin. The 31 tests in `tests/test_reference_profiles.py` cover the contract, artifact recomputation, provenance, and end-to-end headless form runs; `python -m tests.reference_profiles` prints the public checklist.
- [x] Verify clear user-facing errors for an absent or invalid artifact without leaving the public app broken. Per the revised US-0703 criterion, deployment-equivalent headless runs exercise the same app loading and error-rendering path and show regeneration guidance for missing/corrupt artifacts; the healthy public deployment separately confirms the valid-artifact path. A deliberate corrupt production deployment is not required.
- [x] Confirm the educational/medical disclaimer remains visible alongside prediction output on the public deployment.
- [x] Confirm the raw CSV, secrets, and submitted user inputs are not published, logged by project code, or persisted. No CSV or secrets are tracked, the serving source contains no input logging/persistence, and the local-only loading guard passes.
- [x] Close P7 after all three stories satisfy their acceptance criteria. The public URL and final local/deployment instructions are recorded, and all four public reference profiles match their expected outputs and age groups.

#### P7 Scope Guardrails

- [x] Keep calibration and threshold selection or modification out of P7; P8 remains responsible for probability calibration and threshold analysis, and the calibration split remained untouched throughout P7.
- [x] Do not perform experimental retraining or compare new models in P7. Only deterministic generation of the already selected D-016 model was performed for the D-013 distribution policy.
- [x] Keep SHAP out of P7; P9 remains responsible for global and local SHAP explainability.
- [x] Keep fairness analysis, batch prediction, scenario exploration, authentication, persistence of user inputs, and user analytics out of P7 and in their later phases or future backlog refinement.

## Epic E6: Post-MVP Enhancements

The P8 stories -- US-0601 (calibration evaluation and selection), US-0606 (threshold analysis and trade-offs), and US-0607 (schema-version-2 artifact and app integration) -- were refined to Ready on 2026-07-11, implemented on 2026-07-12/13, and closed on 2026-07-13 after implementation commit `5798a0e` was pushed and the Streamlit deployment passed its public smoke verification. D-018 and D-019 are Accepted, and all three P8 stories are Done. The leakage-safe protocol lives in `docs/ml-analysis-plan.md`, evidence in `docs/p8-calibration/report.md`, and execution history in `docs/iteration-log.md`. P9 was implemented and closed on 2026-07-14: D-020, D-021, and D-022 are Accepted; US-0602, US-0608, and US-0609 are Done. P10 was implemented and closed on 2026-07-15: D-023, D-024, and D-025 are Accepted; US-0605, US-0610, and US-0611 are Done. P11 was implemented and closed on 2026-07-16: D-026, D-027, and D-028 are Accepted; US-0603, US-0612, and US-0613 are Done; implementation commit `246d5ff` was pushed; and the deployed Streamlit batch workflow passed mandatory valid-plus-mixed public verification including the safe result download. P12 implementation commit `1f600e8` was pushed with the complete audit package; the phase was then reviewed and closed on 2026-07-17 after D-029, D-030, and D-031 were Accepted in the required order. US-0604, US-0614, and US-0615 are Done, and D-031 made Streamlit deployment verification not applicable.

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0601 | As a user, I want calibrated probabilities so that risk percentages are more honest. | P1 | Done | Before any calibrator is fitted, an uncalibrated baseline for the frozen train-only D-016 model is recorded on the calibration split: reliability diagram, Brier score, and log loss, with ROC-AUC and PR-AUC as ranking context. Sigmoid and isotonic calibration are then compared against that baseline exclusively on stratified five-fold out-of-fold predictions within the calibration split, fitting each calibrator on the frozen model's scores; the base model is never refit and calibrators never consume train or test rows. The comparison reports reliability diagrams (visual diagnostics), Brier score, and log loss, plus ROC-AUC and PR-AUC as ranking-preservation checks (ECE may appear as a descriptive secondary metric only). The selection follows the operationalized D-018 criteria pre-declared in `docs/ml-analysis-plan.md` -- a paired-bootstrap Brier adoption rule against the uncalibrated baseline, log loss under the same rule as tie-break, a fixed ROC-AUC/PR-AUC regression bound, and `calibration_method = none` when no method qualifies -- declared before out-of-fold results are observed. The method choice uses no test data and D-018 is resolved before any threshold freezing, final refit, or test use; a selected calibrator is refit on the full calibration split; and the frozen final contract receives one official P8 test evaluation, which later runs may only repeat as a deterministic regression check without modifying any decision. |
| US-0606 | As a reviewer, I want a documented threshold analysis with explicit trade-offs so that decision cut-offs are understandable without turning the risk estimate into a diagnosis. | P1 | Done | Precision-recall curves and threshold tables (positive-class recall, precision, F1, false-positive and false-negative counts, and confusion matrices per candidate threshold) are computed exclusively from the out-of-fold probabilities of the D-018-selected probability contract on the calibration split; test rows are never used to explore or select thresholds. Candidate threshold scenarios are frozen, and D-019 resolved, after the D-018 selection and before the official P8 test evaluation. The analysis keeps probability estimation explicitly separate from any decision layer, makes no clinical claims, and never presents a threshold as a validated screening or diagnostic rule. The app keeps displaying a probability without high/low-risk labels unless D-019 later records an explicit, justified product decision. |
| US-0607 | As a user, I want the deployed app to serve the P8-selected probability contract so that the public estimates reflect the P8 calibration work. | P1 | Done | The artifact moves to schema version 2 and serves the D-018-selected contract: a single bundle holding the frozen D-016 base model, the selected `calibration_method` (`sigmoid`, `isotonic`, or `none`), and -- only when the method is sigmoid or isotonic -- the fitted final calibrator, with metadata recording the method, the out-of-fold protocol, calibration metrics (calibration-split out-of-fold and the official P8 test evaluation), and package versions. `validate_artifact_bundle()` enforces the version-2 contract with the same strictness as version 1, including that a calibrator is present if and only if the method requires one, with direct checks of the calibrator object. App wording matches the served contract: it stops describing the probability as uncalibrated only when a calibrator ships, and keeps the uncalibrated notice if `none` wins. The four deployment reference profiles are regenerated with newly recorded expected probabilities and displays when the served probabilities change, and re-verified unchanged against the version-2 artifact when they do not. Local, headless, and public smoke tests pass; the official artifact ships through the D-013 controlled Git exception; and Streamlit still never trains or calibrates at runtime. |
| US-0602 | As a reviewer, I want a defined explanation contract and global SHAP analysis so that the final P8 probability behavior can be audited without changing it. | P1 | Done | Validate SHAP compatibility with the pinned Python 3.12, NumPy 2.2.6, scikit-learn 1.7.1 stack and the frozen D-016 `HistGradientBoostingClassifier`; determine and document the explained output, ideally the positive-class probability served by `predict_risk_probability`; produce global importance from mean absolute SHAP values with at least a bar plot and beeswarm; record the model, output, background, analysis sample, seed, package versions, limitations, mathematical-fidelity checks, and reproduction procedure; and leave the P8 model, artifact, and served probabilities unchanged. |
| US-0608 | As a user and reviewer, I want reproducible local explanations for the public reference profiles so that each estimate can be understood and audited. | P1 | Done | Explain all four public reference profiles with the correct positive class and exact `FEATURE_COLUMNS` order; show the base value, final served estimate, and per-feature contributions; produce waterfall plots or an evidenced equivalent local representation; translate feature names and encoded values into understandable labels; describe contributions only with wording such as "increased/decreased the model's estimate"; prohibit causal, diagnostic, or medical-recommendation claims; and verify additivity, finiteness, feature order, and fixed-seed reproducibility. |
| US-0609 | As a non-technical user and technical reviewer, I want explanation communication at two levels so that the app stays understandable while GitHub evidence remains auditable. | P1 | Done | Provide a simple, progressive, visual Streamlit section similar to "How the model interprets this estimate" for a non-technical audience, while preserving the medical disclaimer; provide a reproducible technical report and evidence under `docs/p9-explainability/`; resolve D-022 before implementation chooses dynamic, precomputed, or hybrid local explanations; verify performance, privacy, regressions, and public deployment behavior; and never expose real dataset rows in the app or published artifacts. |
| US-0605 | As a user, I want to compare my original model estimate with a controlled hypothetical scenario so that I can explore model sensitivity without treating it as medical advice. | P1 | Done | D-023 was resolved before engine/UI implementation from the complete 21-field BRFSS semantic audit. The accepted whitelist is exactly `PhysActivity`, `Fruits`, and `Veggies`; every other field is excluded with a documented reason. The result is described only as a hypothetical change in model output, never as an expected medical benefit, causal effect, recommendation, or way to reduce real risk. |
| US-0610 | As a maintainer, I want a deterministic and independently tested scenario engine so that hypothetical comparisons cannot silently change the serving contract or original input. | P1 | Done | `src/scenarios.py` reuses `validate_input_values`, exact `FEATURE_COLUMNS` order, and the P8 `predict_risk_probability` contract; never mutates its baseline; rejects unknown, unapproved, missing, non-finite, incorrectly typed, or out-of-range changes; and implements the accepted one-field D-024 schema. Focused tests prove direct-scoring equality and the signed formula within absolute tolerance `1e-12`; no training, calibration, explanation, optimization, ranking, or persistence path exists. |
| US-0611 | As a non-technical user, I want the scenario comparison presented clearly and safely in Streamlit so that I can distinguish a model experiment from a health prediction about changing my behavior. | P1 | Done | D-025 was resolved before integration. The app provides the progressive original-versus-scenario view, explicit reset, neutral positive/negative/zero treatment, effective changes, non-causal/non-medical statement, and controlled scenario-only fallback while preserving P8, P9, the disclaimer, privacy boundary, and reference displays. Implementation commit `fb50ed9` was pushed and mandatory public frontend verification passed on 2026-07-15. |
| US-0603 | As a user, I want a documented CSV template and downloadable batch results so that I can score multiple cases without guessing the model schema. | P1 | Done | D-026 is resolved before publishing the template. Streamlit offers a code-generated template and concise field guide derived from `FEATURE_COLUMNS`, `VALUE_RANGES`, and shared labels; accepts only the frozen encoding/delimiter/size/row contract; reports file-level structural failures clearly; summarizes valid and invalid rows without exposing thresholds or risk categories; and downloads deterministic results under the D-027 schema. Uploaded data and output remain in memory and are never persisted or externally logged. |
| US-0612 | As a maintainer, I want a pure validated batch-scoring engine so that CSV processing cannot bypass the P8 artifact and input contracts. | P1 | Done | A Streamlit-independent module parses bounded uploaded bytes, rejects malformed or structurally ambiguous files, validates every row against the exact 21-feature contract without silent coercion, preserves input order and duplicates, and implements the D-027 file-level/row-level policy. Only valid rows are scored through the validated D-018-selected positive-class probability contract; each output equals individual `predict_risk_probability` scoring within absolute tolerance `1e-12`; invalid rows have no probability; and export is deterministic and in memory. No raw project data, training, calibration, threshold, label, SHAP, scenario, artifact generation, persistence, or external logging path exists. |
| US-0613 | As a non-technical user, I want a safe batch workflow in Streamlit so that I can understand validation problems and download probabilities without confusing them with diagnoses. | P1 | Done | D-028 was resolved from UX/privacy/failure/performance evidence before integration. The app separates single-case and batch workflows, explains the template and privacy boundary, provides bounded preview/summary/download behavior, preserves D-019 probability-only wording and the medical disclaimer, and handles empty, malformed, oversized, mixed-validity, scoring, and stale artifact/upload states without showing prior results as current. P9 explanations and P10 scenarios remain single-case only. Implementation commit `246d5ff` was pushed and the deployed valid-plus-mixed workflow, validation summary, and safe result download passed mandatory public verification on 2026-07-16. |
| US-0604 | As a reviewer, I want a predeclared subgroup audit of the frozen probability contract so that demographic and socioeconomic differences are measured rather than ignored. | P1 | Done | Before the official P12 audit, D-029 freezes semantic labels, cohort mappings, intersectional scope, and minimum-support behavior using only feature semantics and calibration-split support counts. The primary audit covers the dataset's binary `Sex` codes, four aggregated BRFSS `Age` bands, all eight `Income` codes, and the predeclared `Sex x Age` intersection. Every cohort reports support and observed prevalence; full probability, ranking, calibration, and D-019 frozen-scenario metrics appear only under the accepted support rule. No group, metric, or narrative is selected from official P12 test results, no real row is published, and the audit never treats the available fields as complete coverage of protected identities. |
| US-0614 | As a maintainer, I want a deterministic subgroup-metrics engine with uncertainty estimates so that the audit is reproducible and cannot silently alter the serving contract. | P1 | Done | A Streamlit-independent module consumes explicit labels and the positive-class probabilities returned by the unchanged P8 scorer; assigns every row to exactly one D-029 cohort per declared axis; computes the D-030 support, prevalence, mean estimate, Brier, log loss, ROC-AUC, PR-AUC, signed calibration gap, common-bin reliability, and all four frozen D-019 scenario metrics; and produces fixed-seed percentile bootstrap intervals and group-minus-overall gaps under the accepted resampling contract. It handles unsupported or one-class groups explicitly, is deterministic, fits no model or calibrator, selects no threshold, and leaves both official artifacts and the four reference probabilities unchanged. |
| US-0615 | As a non-technical reader and technical reviewer, I want the audit communicated with uncertainty and dataset limitations so that subgroup differences are not mistaken for causal, clinical, or universal fairness conclusions. | P1 | Done | D-031 was resolved before the official P12 audit and commits to publishing all predeclared eligible groups regardless of result. GitHub receives a reproducible technical report, aggregate CSVs, and accessible plots; README receives a concise limitations summary. Communication distinguishes model behavior from health mechanisms, explains base-rate and uncertainty effects, and states that BRFSS 2015 uses self-reported outcomes, a binary `Sex` field, ordinal age/income groups, and no race/ethnicity variable in this dataset. P12 performs audit and communication only: mitigation, retraining, reweighting, group-specific thresholds, individual fairness claims, and Streamlit subgroup judgments remain out of scope. Human review of the complete publication package and its responsible interpretation was completed on 2026-07-17. |

### P8 Implementation Tasks (US-0601, US-0606, US-0607)

P5-P7 provided the planning evidence for this refinement: the calibration split remained untouched until P8 and holds exactly 25,368 rows at the preserved ~13.9% prevalence (3,535 positives), enough for isotonic calibration to be a serious candidate alongside sigmoid; the selected D-016 model's low default-threshold recall (0.157 on test, against 0.563 precision) motivated the threshold analysis; and the pre-P8 artifact used schema version 1 with no calibrator or decision threshold. The deployed P8 artifact now uses schema version 2, still with no calibrator because D-018 selected `none`, and D-019 intentionally keeps no served threshold.

#### US-0601: Calibration Evaluation and Selection

- [x] Record the uncalibrated baseline first: score the calibration split with the frozen train-only D-016 model and report the reliability diagram, Brier score, and log loss, with ROC-AUC and PR-AUC as ranking context, before any calibrator exists. Recorded 2026-07-12 through `src/calibration.py`: Brier 0.096940, log loss 0.313828, ROC-AUC 0.827060, PR-AUC 0.432421, with the reliability table and probability histogram in `docs/p8-calibration/report.md`.
- [x] Implement stratified five-fold cross-fitting within the calibration split, operating only on the frozen base model's predicted scores with a fixed random seed: for each method (sigmoid, isotonic), fit the calibrator on four folds, predict the held-out fold, and assemble complete out-of-fold calibrated probabilities. No calibration row ever refits the `HistGradientBoostingClassifier`, and no train or test row ever fits a calibrator. Implemented in `src/calibration.py` with `CalibratedClassifierCV(FrozenEstimator(...), ensemble=False)` per the pinned scikit-learn 1.7.1 semantics (calibrators consume the frozen model's `decision_function` scores identically during cross-fitting and serving); fit-spy and out-of-fold integrity tests enforce the protocol in `tests/test_calibration.py`.
- [x] Compare sigmoid, isotonic, and the uncalibrated baseline exclusively on the out-of-fold predictions using the operationalized D-018 criteria declared before results are observed: the paired-bootstrap Brier adoption rule (10,000 fixed-seed resamples; 95% confidence intervals of the mean paired per-row difference `delta = candidate loss - reference loss`, improvement requiring the interval's upper limit below zero), log loss under the same convention as tie-break, the project-defined 0.005 absolute ROC-AUC/PR-AUC regression bound, and sigmoid preferred on demonstrated equivalence; `calibration_method = none` is selected when no method qualifies. Report reliability diagrams as visual diagnostics and ECE as a bin-dependent descriptive metric only, never as deciding criteria. Ran 2026-07-12: neither method passed the Brier adoption rule (sigmoid CI [-0.56e-05, +5.46e-05], isotonic CI [-11.15e-05, +23.18e-05]); isotonic also failed the PR-AUC guard (drop 0.00736). Evidence in `docs/decisions.md` and `docs/p8-calibration/report.md`.
- [x] Resolve D-018 with the selected outcome (sigmoid, isotonic, or none) and its recorded evidence, before any threshold freezing, final refit, or test use; do not pre-register any outcome before the comparison runs. D-018 Accepted 2026-07-12: `calibration_method = none`; the uncalibrated frozen D-016 output is retained and `src.calibration.SELECTED_CALIBRATION_METHOD` pins the outcome.
- [x] Refit the selected calibrator on the full calibration split (skipped when `none` is selected; per-fold calibrators are discarded either way), and record the official P8 test evaluation of the frozen contract, only after the calibration method and the US-0606 threshold scenarios are frozen. D-018 selected `none`, so no final calibrator was fitted. After D-019 was frozen, the official test evaluation recorded Brier 0.097381, log loss 0.314394, ROC-AUC 0.826955, and PR-AUC 0.423065; deterministic repeats cannot modify either decision.

#### US-0606: Threshold Analysis and Trade-Offs

- [x] Build precision-recall curves and threshold tables exclusively from the selected contract's calibration-split probabilities. Exact tables and the curve are recorded under `docs/p8-calibration/`.
- [x] Document the D-016 threshold trade-offs in product-neutral language, including false negatives and false positives, without clinical claims. See `docs/p8-calibration/report.md`.
- [x] Freeze the documented scenarios after D-018 and before test, then resolve D-019. D-019 Accepted 2026-07-12 with 0.50, 0.25, 0.29, and 0.15 as documentation scenarios only.
- [x] Keep estimation and decisions separate end to end: D-019 keeps the app probability-only with no high/low-risk labels and no served threshold.

#### US-0607: Schema-Version-2 Artifact and App Integration (final P8 increment)

- [x] Bump the artifact contract to schema version 2 with the frozen model, conditional calibrator, method, fixed protocol, OOF and official P8 metrics, frozen D-019 scenarios, and package versions.
- [x] Enforce the version-2 contract strictly: conditional calibrator presence, direct object checks, complete P8 metadata, exact layout, and rejection of schema version 1 or inconsistent combinations.
- [x] Update the app texts conditionally. The accepted `none` outcome remains explicitly uncalibrated; calibrated contract tests drop that wording. The medical disclaimer remains intact.
- [x] Re-verify the four profiles against the official version-2 artifact. Because `none` leaves serving unchanged, the same inputs and displays 0.3%, 60.0%, 70.0%, and 79.9% remain valid.
- [x] Regenerate the official version-2 artifact per D-013 and verify it locally/headlessly, including the reference suite. The reviewed artifact shipped in implementation commit `5798a0e`.
- [x] Push implementation commit `5798a0e`, redeploy the Streamlit app, and rerun the public startup, complete-form, contract-wording, disclaimer, probability-only, age-group, and four-profile checks. Public verification passed on 2026-07-13 with exact displays of 0.3%, 60.0%, 70.0%, and 79.9%.
- [x] Confirm the serving path still never trains, calibrates, downloads models, or reads the raw CSV at runtime; source guards enforce this boundary.

#### Expected Tests for P8

- [x] Fit-spy checks: the base model consumes exactly the train rows; each per-fold calibrator is fitted on exactly its four assigned training folds and never on its held-out fold; the final calibrator (when one is selected) is fitted on exactly the full calibration split; no calibrator is ever fitted on train or test rows.
- [x] Test isolation: test rows never reach the comparison, selection, or threshold-analysis code paths; poisoning the test split leaves the out-of-fold comparison, D-018 selection, and threshold tables unchanged. Test enters only the official P8 evaluation and deterministic regression repeats.
- [x] Reproducibility: the fixed seed reproduces fold assignment, out-of-fold probabilities, bootstrap intervals, method comparison results, and the selection outcome.
- [x] Out-of-fold integrity: every calibration row receives exactly one out-of-fold prediction per method, produced by a calibrator that was not fitted on that row's fold.
- [x] Version-2 artifact contract: save/load round-trip for each `calibration_method` outcome, conditional calibrator presence, rejection of version 1, tampered or unfitted calibrators, inconsistent method/calibrator combinations, missing calibration metadata, and provenance/runtime pins.
- [x] Controlled reference-profile update: all four profiles recompute against the version-2 artifact within tolerance and render the unchanged recorded displays through the headless app.
- [x] Source guards: no calibrator fitting or training code in the app serving path; calibration fitting happens only in offline P8 code.

#### P8 Scope Guardrails

- [x] Do not reopen the D-016 model selection and do not compare new models; P8 calibrates the already selected frozen model only.
- [x] Never use test rows to select the calibration method or the threshold scenarios; test participates in no P8 decision. The official P8 test evaluation was recorded after method and scenarios were frozen; later runs only repeat it as a deterministic regression check.
- [x] Never train the base model on calibration rows; never fit a calibrator on train or test rows.
- [x] No calibration or training inside Streamlit; the app only loads and serves the artifact.
- [x] Keep SHAP out of P8; P9 owns global and local explainability so explanations can target the final P8 serving contract.
- [x] Keep fairness analysis, batch prediction, and scenario exploration in their later phases (P10-P12).
- [x] No medical recommendations, diagnostic or high/low-risk labels, or causal interpretation of calibration curves or threshold trade-offs; D-019 explicitly retains a probability-only product.

### Candidate Tasks for P9

P9 explains the final probability contract selected in P8: the frozen D-016 `HistGradientBoostingClassifier`, artifact schema version 2, and `calibration_method = none`, serving the model's positive-class probability through `predict_risk_probability`. The phase does not change that contract. The 2026-07-14 spike resolved D-020 as a direct positive-probability explanation with SHAP 0.52.0 `TreeExplainer`, resolved D-021 with one privacy-safe 256-centroid train-derived background plus a proportionally stratified 5,000-row calibration sample, and resolved D-022 as hybrid delivery. Exact evidence is under `docs/p9-explainability/`.

#### Increment 1 -- Compatibility and Technical Contract (approximately 1 day)

- [x] Run a focused compatibility spike for SHAP with Python 3.12, NumPy 2.2.6, scikit-learn 1.7.1, and the frozen D-016 `HistGradientBoostingClassifier`.
- [x] Evaluate `TreeExplainer`; the selected probability path passed, and the raw-margin and model-agnostic alternatives plus their rejection reasons are recorded rather than substituted silently.
- [x] Validate returned dimensions, the explained positive class, exact `FEATURE_COLUMNS` order, finite values, and additivity against the selected contract.
- [x] Compare the base value plus the sum of SHAP contributions with `predict_risk_probability`; direct probability passed the unchanged `1e-4` absolute tolerance with a global maximum error of `1.3185956326822179e-08`.
- [x] Measure representative explainer-creation, global-sample and one-row local time, plus approximate peak memory, against limits fixed before the final run.
- [x] Verify SHAP's background retention: the implicit masker retained only 100 rows, while the accepted explicit `Independent` masker retains all 256 aggregate rows. No real row is serialized or deployed.
- [x] Resolve D-020 and D-021 from spike evidence before the full global analysis.
- [x] Pin only the compatible top-level dependency, `shap==0.52.0`, after the selected route passed.

#### Increment 2 -- Offline Global Analysis (approximately 1-2 days)

- [x] Build the D-021 256-centroid background deterministically from train rows only; calibration and test never supply background rows.
- [x] Use a deterministic, proportionally stratified 5,000-row calibration sample: 697 positive and 4,303 negative rows preserve source prevalence within deterministic rounding. Test selects nothing.
- [x] Evaluate the proposed sizes of 256 and 5,000. The deterministic centroid builder uses no RNG; project seed 42 governs the proportionally stratified global sample.
- [x] Keep both proposed sizes because they passed the predeclared performance and memory limits; no explanation result was used to resize them.
- [x] Produce only aggregate mean-absolute importance CSV/bar output and a rendered beeswarm; publish no source row or individual global SHAP matrix.
- [x] Create `docs/p9-explainability/report.md` with the complete reproducible technical contract and evidence.

#### Increment 3 -- Local Explanations (approximately 1-2 days)

- [x] Explain the four synthetic public reference profiles already defined by the deployment contract.
- [x] Generate one waterfall plot per profile and one contribution table showing the base value, all per-feature contributions, and final served estimate under D-020.
- [x] Translate encoded features and values through the shared pure `src/feature_labels.py` source; `Age` remains a BRFSS group, not an exact age.
- [x] Validate positive-class selection, exact feature order, additivity, finiteness, and fixed-seed reproducibility for every profile.
- [x] Provide simple and technical content using increased/decreased model-estimate language, with no causal, diagnostic, clinical, or prescriptive interpretation.

#### Increment 4 -- Integration, Regression, and Deployment (approximately 1-2 days)

- [x] Resolve D-022 as hybrid delivery from performance, memory, privacy, fidelity, maintenance, and UX evidence before Streamlit integration.
- [x] Implement the selected hybrid route: dynamic cached local explanations plus precomputed aggregate global/reference evidence.
- [x] Validate the deployed 256-centroid aggregate background under D-020/D-021 and prove it contains no exact real train row or target/index field.
- [x] Run the complete test suite and Streamlit headless tests locally.
- [x] Confirm that the exact probabilities for all four reference profiles remain unchanged from the P8 contract.
- [x] Verify simple/technical wording, disclaimer visibility, performance/timeout bounds, error handling, and privacy locally.
- [x] Deploy implementation commit `25c4ed4` through the existing process and pass the mandatory public smoke test. After one transient process crash during the initial reboot, a clean reboot remained stable; all four reference profiles preserved 0.3%, 60.0%, 70.0%, and 79.9%, rendered their dynamic explanations, and retained the disclaimer without triggering the explanation fallback.

#### Expected Tests for P9

- [x] The SHAP matrix is finite and has exact shape `n x 21`.
- [x] The explanation targets positive class `1` and the D-020 direct-probability contract.
- [x] Explanation columns preserve the exact order of `src.data.FEATURE_COLUMNS`.
- [x] Base plus contributions satisfies the unchanged `1e-4` tolerance and explicitly reproduces `predict_risk_probability`.
- [x] Fixed-seed runs reproduce background/sample selection, global importance, and all four local explanations within predeclared tolerances.
- [x] The aggregate background is derived exclusively from train.
- [x] The deterministic stratified calibration sample preserves prevalence within its predeclared allocation/rounding tolerance.
- [x] Test rows cannot affect explainer configuration, background, sample selection, narrative, plots, or any P9 decision; poisoning test leaves P9 evidence unchanged.
- [x] The four reference-profile probabilities and recorded displays remain exactly conserved.
- [x] User-facing content avoids causal, diagnostic, clinical, and prescriptive claims and preserves the disclaimer.
- [x] Dynamic local explanations meet predeclared creation and warm-local time bounds.
- [x] The deployed asset exposes no real train, calibration, or test row and contains no target, identifier, or split index.
- [x] Importing Streamlit performs no model fitting, data download, raw-CSV read, or global SHAP computation.

#### Implemented Local P9 Deliverables

- `src/explainability.py`.
- `tests/test_explainability.py`.
- `docs/p9-explainability/report.md`.
- Aggregate CSV of global importance and CSV of local contributions for the four public synthetic profiles.
- Global bar and beeswarm plots, plus local waterfall plots or the D-020/D-022-evidenced equivalent.
- Controlled changes to `app/streamlit_app.py` that deliver the required simple explanation according to the accepted D-022 outcome.
- A pinned SHAP dependency only after the compatibility spike succeeds.

#### P9 Scope Guardrails

- [x] Do not retrain, recalibrate, retune, or replace the D-016 model; the official artifact was not regenerated or modified.
- [x] Do not change thresholds, labels, or any probability returned by the P8 serving contract.
- [x] Do not use test data to select explanation configuration, samples, text, plots, or decisions.
- [x] Do not publish or deploy real dataset rows; global artifacts are aggregate, local artifacts use only the four synthetic profiles, and the runtime asset contains only non-matching aggregate centroids.
- [x] Keep P10 scenario exploration and P12 fairness analysis outside P9.
- [x] Make no causal, diagnostic, clinical, or prescriptive claim from SHAP values.
- [x] Keep CI as an optional quality-track candidate or independent increment, never a prerequisite on the SHAP critical path.

#### Definition of Done for P9

- [x] US-0602 and US-0608 satisfy their acceptance criteria with reproducible evidence.
- [x] US-0609 satisfies its acceptance criteria locally and in the mandatory public verification.
- [x] D-020, D-021, and D-022 are resolved from recorded implementation evidence.
- [x] Global and local explanations are mathematically faithful to D-020, and their relationship to the P8 served probability is explicit and verified.
- [x] The D-016 model, official artifact, four reference probabilities, and probability-only P8 product behavior remain unchanged.
- [x] The local Streamlit implementation provides the approved simple explanation, while GitHub provides sufficient technical evidence for audit and reproduction.
- [x] Published language is non-causal, non-diagnostic, non-clinical, and non-prescriptive; the medical disclaimer remains visible.
- [x] The complete local suite and headless checks pass.
- [x] The updated Streamlit application passes mandatory public verification after deployment.
- [x] P9 moved to Done after implementation commit `25c4ed4`, redeployment, reboot, and successful public verification on 2026-07-14.

### Iteration 10 -- P10 Model Scenario Explorer

P10 is complete. It adds a constrained sensitivity explorer around the unchanged P8 probability contract, comparing the submitted profile with one hypothetical variant. It does not estimate what would happen to a person's health after an intervention. The frozen D-016 model, schema-version-2 artifact, `calibration_method = none`, D-019 probability-only product, P9 SHAP contract, and four public reference estimates remain unchanged. Implementation commit `fb50ed9` was pushed and the updated Streamlit application passed mandatory public frontend verification on 2026-07-15.

#### Increment 1 -- Feature Semantics and Safety Contract (approximately 1 day)

- [x] Audit every `FEATURE_COLUMNS` field against its BRFSS meaning, encoding, valid values, reversibility, and risk of causal or prescriptive misinterpretation.
- [x] Evaluate `BMI`, `PhysActivity`, `Fruits`, `Veggies`, and `HvyAlcoholConsump` as the initial candidate set; candidate status does not authorize a field for the UI before D-023 is Accepted.
- [x] Exclude age, sex, education, and income from improvement-lever framing. Treat diagnoses, historical events, access-to-care variables, and subjective health summaries as non-editable unless D-023 records exceptional evidence and safe wording.
- [x] Explicitly reject `Smoker` as an improvement lever because the BRFSS field means having smoked at least 100 cigarettes over a lifetime and cannot be reversed by a present scenario choice.
- [x] Resolve D-023 with the approved whitelist, field labels, allowed values/ranges, excluded-field rationale, and wording constraints before implementing the scenario engine or UI.

#### Increment 2 -- Deterministic Scenario Engine (approximately 1 day)

- [x] Implement a pure scenario module, expected at `src/scenarios.py`, that copies a validated baseline profile, applies only approved changes, and scores both profiles through `predict_risk_probability`.
- [x] Return a structured comparison containing the unchanged baseline, exact scenario profile, original probability, scenario probability, and signed percentage-point delta.
- [x] Reject unknown, unapproved, missing, non-finite, incorrectly typed, or out-of-range changes; preserve exact feature order and every non-edited value.
- [x] Resolve D-024 with the number of simultaneously editable fields, comparison/reset behavior, numerical contract, and explicit prohibition of optimization, ranked scenarios, recommended presets, or threshold labels before Streamlit integration.
- [x] Record the engine contract and evidence in `docs/p10-scenarios/report.md` without using test rows or regenerating an artifact.

#### Increment 3 -- Streamlit Communication and Integration (approximately 1-2 days)

- [x] Resolve D-025 from a focused UX/privacy review before changing Streamlit: placement, progressive disclosure, transient state, neutral delta presentation, fallback behavior, and public wording.
- [x] Present the original and hypothetical probabilities side by side, identify exactly which approved inputs changed, show the signed difference in model-estimate percentage points, and provide an unambiguous reset to the submitted profile.
- [x] Use wording such as "the model estimate changed" and "hypothetical scenario" for positive, negative, and zero differences. Never use "improves your health", "reduces your risk", "recommended", "best", or equivalent prescriptive language.
- [x] Keep the existing medical disclaimer and make clear that the scenario is not a causal estimate, diagnosis, treatment effect, or forecast of what will happen if a user changes behavior.
- [x] Keep P9's explanation of the submitted estimate intact and separate. Do not add scenario-specific SHAP explanations in P10.

#### Increment 4 -- Regression, Deployment, and Closure (approximately 1 day)

- [x] Run the complete pytest suite and Streamlit headless checks, including accepted, rejected, reset, zero-delta, positive-delta, negative-delta, and scenario-failure paths.
- [x] Verify that the original estimates and exact displays for the four reference profiles remain 0.3%, 60.0%, 70.0%, and 79.9% and that their P9 explanations still render.
- [x] Verify that `models/diabetes_risk_model.joblib` retains SHA-256 `957c14ff5a490bbc60822121a889f92ee2a6a20f797eef741a710d887ecc9216` and `models/shap_background_v1.json` retains SHA-256 `73d1ff21e3c98ee79fa7d72758517047f13e5f454d7ff95edb1ee93812cca120`; neither is regenerated.
- [x] Confirm Streamlit performs no fitting, calibration, artifact generation, raw-data access, global SHAP analysis, scenario persistence, or external input logging.
- [x] Push reviewed implementation commit `fb50ed9`, reboot/redeploy the existing Streamlit application, and pass the mandatory healthy-path public smoke test before marking US-0611 or P10 Done. Completed 2026-07-15; invalid/error paths remain covered locally/headlessly rather than by deliberately breaking the public app.

#### Expected Tests for P10

- [x] The baseline input is never mutated, and applying no change reproduces the original probability and a zero delta.
- [x] Only D-023-approved fields can change; every other feature and exact `FEATURE_COLUMNS` order are preserved.
- [x] Unknown, excluded, missing, non-finite, incorrectly typed, and out-of-range scenario values are rejected deterministically.
- [x] The scenario probability equals `predict_risk_probability` for the exact modified profile, and `delta_percentage_points = 100 * (scenario_probability - original_probability)` within an absolute `1e-12` numerical tolerance.
- [x] Reset reconstructs the submitted baseline exactly, including all 21 feature values.
- [x] Positive, negative, and zero differences use neutral model-response language and no threshold or high/low-risk category.
- [x] Existing reference-profile probabilities, P9 local explanations, disclaimer, artifact-error behavior, and explanation-error fallback remain unchanged; failed original resubmission and artifact-hash changes invalidate prior session results before rendering.
- [x] Source and headless guards prohibit optimization, ranked recommendations, scenario-specific SHAP, persistence, external logging, fitting, raw-data reads, and artifact regeneration in the app.
- [x] No train, calibration, or test row is needed to run the scenario engine or Streamlit path; test data cannot influence D-023 through D-025 or the product wording.
- [x] The deployed healthy path passes a mandatory public smoke test after the implementation is pushed. The user confirmed the planned frontend cases, and an independent public check confirmed prediction, P9 preservation, progressive P10 rendering, the exact whitelist, neutral safety wording, the disclaimer, and no browser-console errors.

#### P10 Deliverables

- `src/scenarios.py` with the pure validated comparison contract.
- `tests/test_scenarios.py` plus controlled regression/headless additions for the app.
- `docs/p10-scenarios/report.md` with the approved whitelist, excluded-field rationale, decision evidence, reproducible checks, limitations, and public-verification record.
- Controlled `app/streamlit_app.py` changes only after D-023, D-024, and D-025 are resolved.
- No new or regenerated model, calibrator, SHAP-background, dataset, or row-level evidence artifact.

#### P10 Scope Guardrails

- [x] Do not retrain, recalibrate, retune, or replace the D-016 model, and do not regenerate either reviewed artifact.
- [x] Do not change the P8 probability, threshold, label, or reference-profile contract, and do not reinterpret P9 SHAP contributions as intervention effects.
- [x] Do not use train, calibration, or test data to operate the explorer or choose its fields, defaults, narrative, or UI after the semantic contract is fixed.
- [x] Do not present sensitive, contextual, historical, diagnostic, or access-to-care fields as actions a person should take.
- [x] Do not optimize, search, rank, prescribe, or recommend scenarios; do not create a "best" profile or claim a real-world risk reduction.
- [x] Do not persist or externally log submitted profiles or scenarios; widget state remains transient in the active session.
- [x] Keep batch prediction (P11), fairness analysis (P12), and broader P9 explanation/UX expansion outside P10.

#### Definition of Done for P10

- [x] US-0605, US-0610, and US-0611 satisfy their acceptance criteria.
- [x] D-023, D-024, and D-025 are Accepted from recorded pre-integration evidence, not assumed during planning.
- [x] The approved editable-feature whitelist and all exclusions are traceable to exact BRFSS semantics and safe communication constraints.
- [x] The pure engine is deterministic, validated, non-mutating, and exactly consistent with `predict_risk_probability`.
- [x] Streamlit clearly separates the submitted estimate from one hypothetical scenario, provides reset, and uses non-causal, non-clinical, non-prescriptive wording.
- [x] The probability-only P8 contract, P9 explanations, medical disclaimer, four reference displays, and both reviewed artifact hashes remain unchanged.
- [x] The complete local suite and Streamlit headless checks pass.
- [x] The updated public application passes mandatory healthy-path smoke verification after deployment.
- [x] P10 moved from Ready to Done after implementation commit `fb50ed9` was pushed, deployed, and publicly verified on 2026-07-15.

### Iteration 11 -- P11 Batch Prediction Workflow

P11 is a delivery extension over the unchanged schema-version-2 P8 probability contract. It does not create a second model, calibrator, decision layer, explanation contract, or scenario engine. The implemented app and strengthened guards prohibit access to the project training CSV and disk/external persistence while allowing only bounded user-uploaded CSV bytes in memory. The pure batch boundary reuses the executable schema/range/label sources and the shared P8 scorer selection without looping through UI code or duplicating feature rules.

The initial candidates were evaluated rather than assumed and are now the Accepted D-026 through D-028 contract: strict UTF-8 comma CSV with optional leading BOM, at most 2 MiB and 1,000 logical data rows, exactly the 21 technical feature names in any input order followed by canonical reordering, generated numeric-code guidance, BRFSS `Age` codes `1`-`13`, and no target/index/identifier/free-text/passthrough column. Structural failures reject the whole file; row-value failures use partial success with complete stable errors and no invalid-row probability. The parser/export spikes, engine evidence, UX/privacy review, and official-artifact performance measurement were completed in that required order before their dependent implementation steps.

#### Increment 1 -- Input, Validation, and Resource Contract (approximately 1 day)

- [x] Inventory the exact batch-facing sources of truth: `FEATURE_COLUMNS`, `VALUE_RANGES`, `feature_label`, `format_feature_value`, schema-version-2 artifact validation, and the D-018/D-019 serving policy. Document where batch behavior may adapt presentation without changing model inputs.
- [x] Spike CSV parsing from bytes with UTF-8 and optional UTF-8 BOM, comma delimiter, one header, duplicate-header detection, empty/malformed-file handling, and deterministic differentiation between structural errors and row errors. Do not accept spreadsheet, URL, ZIP, JSON, or remote inputs.
- [x] Evaluate and record the proposed 2 MiB/1,000-row limits, exact-column/any-order policy, `Age` representation, identifier exclusion, numeric-code requirement, and code-generated template/field-guide design. D-026 was Accepted before the definitive parser/template implementation.
- [x] Prototype complete per-row validation without silent fill, rounding, clipping, truthy conversion, type coercion, or dropping duplicates. D-027 was Accepted before Streamlit integration with file rejection, partial success, exact output, 15-decimal probabilities, stable errors, and export-only formula neutralization.

#### Increment 2 -- Pure Batch Validation and Scoring Engine (approximately 1-2 days)

- [x] Implement `src/batch.py` as a Streamlit-independent module that accepts uploaded bytes plus a bundle and returns immutable/independent structured validation and scoring results.
- [x] Reject empty, malformed, wrong-encoding/delimiter, headerless, duplicate-header, missing-column, unexpected-column, target-bearing, and over-limit files before scoring. Preserve every structurally valid row and its original order, including exact duplicate profiles.
- [x] Validate all 21 values per row against integer-like, finite `VALUE_RANGES` rules and return deterministic complete error information. Never silently repair an invalid value and never score an invalid row.
- [x] Score valid rows through one vectorized call to the same validated P8 scorer selected by the artifact, and prove each probability equals the existing single-case helper within absolute tolerance `1e-12`.
- [x] Generate the template, field guide, combined result table, and UTF-8 CSV download entirely in memory with stable canonical column/error ordering and D-027 formula-injection safety.

#### Increment 3 -- Streamlit Batch Delivery (approximately 1-2 days)

- [x] Perform the D-028 UX/privacy/failure/performance review before changing Streamlit: navigation relative to the single-case form, explicit processing action, bounded preview, summary counts, safe error visibility, reset/replacement behavior, download naming, transient state, and medical/privacy wording.
- [x] Measure the simultaneous accepted maximum on the official artifact: exactly 2,097,152 bytes and 1,000 valid rows, including parse, validation, vectorized score, and export. Thirty warm runs measured 0.1179472-second median, 0.1294039-second p95, 0.1627905-second maximum, and 12.2736 MiB incremental Python peak memory. The accepted 2-second/50-MiB bounds remain unchanged.
- [x] Integrate template/field-guide download, upload, validation summary, bounded preview, and result download without putting CSV parsing or serialization logic in `app/streamlit_app.py`. Retained results bind to both artifact and upload SHA-256 and invalidate after replacement, failure, reset, or artifact change.
- [x] Keep uploaded bytes and generated results only in active-session memory. Add no filesystem/object-store/database write, analytics, external logging, URL fetch, or cross-session cache of user content.
- [x] Keep P9 explanations and P10 scenarios exclusively in the single-case workflow. Batch output contains probabilities and validation status only, with D-019 wording and the medical disclaimer visible.

#### Increment 4 -- Regression, Deployment, and Closure (approximately 1 day)

- [x] Run focused batch tests, app/headless tests, the complete suite, dependency validation, and whitespace checks. Headless coverage exercises valid, invalid, mixed, duplicate, maximum-size, reset/replacement, and download cases; the real localhost browser review covers the initial batch layout, controls, wording, and visible 2 MiB transport limit.
- [x] Confirm the four reference profiles produce unchanged probabilities/displays when scored individually and as one batch, and verify both reviewed artifact SHA-256 values remain unchanged.
- [x] Update source guards so they continue to prohibit the project training CSV and disk/external persistence while allowing only the reviewed in-memory uploaded-CSV helper. Do not weaken guards merely by moving prohibited operations to another module.
- [x] Review, commit, and push implementation commit `246d5ff`; deploy/reboot the existing Streamlit application; and pass mandatory public verification with a small valid template-derived file and a mixed-validity file whose safe validation report/download are confirmed. Completed 2026-07-16; internal, oversized, corrupt-artifact, and forced-scoring failures remain local/headless.
- [x] Record implementation, contract decisions, performance/privacy evidence, exact tests, artifact hashes, visual review, limitations, and public closure evidence in `docs/p11-batch/report.md`; P11 moved to Done only after external closure evidence existed.

#### Expected Tests for P11

- [x] The generated template and field guide cover exactly `FEATURE_COLUMNS`/`VALUE_RANGES`, use the D-026 encoding, and cannot drift independently from the executable contract.
- [x] Empty, malformed, wrong-encoding/delimiter, headerless, duplicate-header, excessive-column, missing, unexpected, target, index, identifier, over-byte, over-row, and zero-data-row files fail at the correct boundary with deterministic safe errors. Structural messages are capped at 1,000 characters and user-controlled header previews are bounded.
- [x] Missing cells, text, booleans, non-finite values, fractions, and out-of-range numbers are reported by row/feature without coercion; all applicable row errors are returned in stable feature order.
- [x] Structural validity plus mixed row validity scores only valid rows, leaves invalid probabilities blank, preserves all row positions and duplicates, and reports exact valid/invalid totals.
- [x] Vectorized probabilities equal per-row `predict_risk_probability` results within absolute tolerance `1e-12`, including all four reference profiles; probabilities are finite and within `[0, 1]`.
- [x] Output columns/order, line endings, encoding, float serialization, blank invalid probabilities, error ordering, and repeated-run bytes are deterministic under D-027.
- [x] Uploaded bytes and outputs are never written, externally logged, sent to analytics, fetched remotely, or cached across sessions; artifact/upload hash changes and failed processing invalidate stale results.
- [x] Batch processing performs no fitting, calibration, thresholding, labeling, SHAP, scenario exploration, raw-project-data access, artifact generation, or model/background modification.
- [x] App/headless coverage verifies progressive batch disclosure, template and result downloads, valid/invalid summaries, bounded preview, limits, reset/replacement, controlled failures, D-019 wording, and disclaimer visibility.
- [x] The simultaneous accepted maximum batch of 2 MiB/1,000 valid rows passes the D-028 latency/memory limits, and the complete local suite passes.
- [x] Mandatory public valid-plus-mixed workflow verification succeeded after deployment on 2026-07-16, including summary, preview, validation details, blank invalid probability, and safe result download.

#### P11 Deliverables

- `src/batch.py` with pure in-memory parsing, validation, scoring, template, field-guide, and export contracts.
- `tests/test_batch.py` plus controlled app/headless and reference-regression additions.
- `docs/p11-batch/report.md` with accepted D-026 through D-028 evidence, exact schemas, limits, performance/privacy evidence, tests, artifact hashes, limitations, and public-verification closure evidence.
- Controlled `app/streamlit_app.py` changes only after D-028, with parsing/export kept outside the UI module.
- No new or regenerated model, calibrator, SHAP background, dataset, persisted upload, or row-level project-data artifact.

#### P11 Scope Guardrails

- [x] Do not retrain, recalibrate, retune, replace, or regenerate the D-016/schema-version-2 artifact or P9 background.
- [x] Do not change the P8 probability contract or D-019 probability-only policy; add no threshold, diagnosis, recommendation, high/low-risk category, or screening interpretation.
- [x] Do not compute per-row SHAP explanations or scenarios in batch. P9/P10 remain single-case features.
- [x] Do not use the project raw/training, calibration, or test CSV at runtime; parse only bounded user-uploaded bytes in memory.
- [x] Do not accept remote URLs, archives, arbitrary identifiers/free text, target labels, or passthrough columns under the D-026 accepted contract.
- [x] Do not persist, externally log, analyze, or share uploaded profiles or results. No cross-session cache may contain user data.
- [x] Do not mix P11 with P12 fairness analysis, P13 product polish, CI, `skops`, authentication, general data storage, or user accounts.

#### Definition of Done for P11

- [x] US-0603, US-0612, and US-0613 satisfy their acceptance criteria.
- [x] D-026, D-027, and D-028 are Accepted from ordered evidence before their dependent implementation steps.
- [x] Template, schema, validation, partial-success, scoring, export, resource, privacy, failure, and UI contracts are explicit and reproducible.
- [x] Every valid batch probability is faithful to the unchanged P8 serving helper, while invalid rows can never receive a probability.
- [x] No uploaded profile or output is written or externally logged, and the public wording remains probability-only, educational, and non-diagnostic.
- [x] P9 explanations, P10 scenarios, reference displays, medical disclaimer, and both reviewed artifact hashes remain unchanged.
- [x] Focused, full-suite, headless, performance, and local visual checks pass.
- [x] The deployed valid and mixed-validity workflows passed mandatory public verification, including downloads, on 2026-07-16.
- [x] P11 moved from Ready to Done after implementation commit `246d5ff` was pushed, deployed, and publicly verified on 2026-07-16; P12-P13 remain Future.

### P12 Implementation and Closure Status (US-0604, US-0614, US-0615)

P12 audits the already frozen P8 probability contract; it does not reopen model selection, calibration, or the D-019 product policy. The calibration-only support evidence was generated first from `prepare_data()` without subgroup performance metrics. All candidate cells passed, after which D-029 was operationally accepted in the working tree. D-030 was accepted after the calibration-only computational benchmark, and D-031 was accepted report-first before official test scoring. The official 50,736-row test audit, support evidence, implementation, interpretation, and applicable verification gates were reviewed together on 2026-07-17. All three stories are Done and P12 is closed without a Streamlit deployment gate under D-031.

#### Increment 1 -- Cohorts, Metrics, and Publication Contract (approximately 1 day)

- [x] Audit the source semantics and limitations of `Sex`, `Age`, and `Income`. Use the dataset's documented binary `Sex` labels without treating them as a complete representation of sex or gender; preserve all eight ordinal `Income` codes; and evaluate the candidate age mapping `18-49` = codes `1`-`6`, `50-64` = `7`-`9`, `65-74` = `10`-`11`, and `75+` = `12`-`13`.
- [x] Produce `docs/p12-fairness/calibration_support.csv` first from `prepare_data()` and calibration cohort grouping, without calibration subgroup-performance metrics. The evidence and D-029 acceptance were reviewed together as one closure package.
- [x] Resolve D-029 before the official audit with the full-metric rule of at least 500 rows, 100 positive labels, and 100 negative labels in the audited split; unsupported groups remain visible with support and prevalence while other metrics are explicitly unavailable.
- [x] Resolve D-030 before the official audit with the exact metric, reliability-bin, gap, unavailable-state, ordering, and 5,000-resample seed-42 percentile-bootstrap contract after the calibration benchmark passed.
- [x] Resolve D-031 before the official audit with mandatory all-results publication and a report-first boundary: no Streamlit change, deployment, or public smoke test is part of P12.

#### Increment 2 -- Pure Audit Engine and Tests (approximately 1-2 days)

- [x] Add `src/fairness.py` as a Streamlit-independent engine over explicit targets and served positive-class probabilities. Reuse `FEATURE_COLUMNS`, split contracts, the validated artifact loader/scorer, `FROZEN_THRESHOLD_SCENARIOS`, and project seed rather than duplicating them.
- [x] Make cohort assignment exhaustive and mutually exclusive within each declared axis, deterministic, and auditable. Preserve source row order internally, but publish no row, target, probability, index, or combination that could expose an individual record.
- [x] Implement support/prevalence, mean probability, Brier score, log loss, ROC-AUC, PR-AUC, signed calibration gap (`mean_probability - prevalence`), shared-bin reliability data, and recall/precision/FPR/FN/FP for every frozen D-019 scenario. D-019 remains documentation-only and no group-specific threshold may be selected.
- [x] Implement the accepted fixed-seed bootstrap and group-minus-whole-cohort gaps without significance tests or a fairness pass/fail rule. Degenerate or unsupported metrics return an explicit unavailable state rather than silently propagating `NaN` or raising an uncontrolled error.
- [x] Keep evidence generation offline and controlled under `docs/p12-fairness/` using the existing pinned NumPy, pandas, scikit-learn, and plotting stack.

#### Increment 3 -- Official P12 Audit and Evidence (approximately 1-2 days)

- [x] After D-029 through D-031 were Accepted and the engine passed synthetic fixtures, load the official schema-version-2 artifact, score the unchanged P3 test split through the D-018-selected probability contract, and record one official P12 descriptive audit without modifying any contract or configuration.
- [x] Export deterministic aggregate support, probability/ranking/calibration, bootstrap-gap, reliability, and frozen-threshold CSVs plus accessible metric-interval and calibration plots. Publish every predeclared eligible cohort regardless of direction.
- [x] Record exact split provenance, cohort definitions, support decisions, metric formulas, bootstrap configuration, package versions, artifact hashes, reproduction command, runtime, and limitations in `docs/p12-fairness/report.md`.
- [x] Interpret observed differences alongside uncertainty, prevalence, sample size, and label/access limitations without causal, biological, discriminatory, clinical, equalized-odds, demographic-parity, or universal conclusions.

#### Increment 4 -- Communication, Regression, and Closure (approximately 1 day)

- [x] Add a concise README summary that links to the technical report and communicates the audit's scope and limitations in everyday language. Keep detailed methods, all aggregates, and plots in GitHub documentation rather than presenting individual subgroup judgments in the prediction UI.
- [x] Verify deterministic evidence regeneration, exact cohort coverage, absence of real-row outputs, and isolation from train/calibration performance decisions. `calibration_support.csv` regenerated byte-identically, and two complete official-audit reproductions matched byte-for-byte across all ten regenerated JSON/CSV/PNG/report files.
- [x] Run focused fairness tests, the complete suite, `pip check`, compile checks, `git diff --check`, artifact-hash checks, and all four reference-profile regressions. Fairness passed 66 tests after the evidence-gate hardening, the targeted data/artifact/calibration/profile set passed 190, and the full suite passed 448. D-031 makes Streamlit execution, deployment, and public smoke testing not applicable.
- [x] Update roadmap, backlog, decisions, iteration log, ML plan, and README only after evidence exists. Human review completed the closure: P12 and all three stories are Done.

#### Expected P12 Tests

- [x] Cohort mappings are exhaustive, mutually exclusive, correctly labeled, and independent of model results; every audited row belongs to exactly one group per declared axis.
- [x] Minimum-support behavior is enforced identically for each metric family; unsupported and one-class fixtures expose explicit unavailable metrics while retaining support/prevalence.
- [x] Metric and D-019 scenario outputs match hand-calculated synthetic fixtures, including confusion counts, signed calibration gap, and group-minus-overall direction.
- [x] Bootstrap intervals and output bytes are deterministic under the fixed seed; changing the seed changes sampled evidence without changing point estimates.
- [x] Poisoning train or calibration values after the protocol is frozen cannot change official P12 metrics; test rows never select cohorts, metrics, bins, resampling, thresholds, narrative inclusion, or publication policy.
- [x] The audit loads and validates the official artifact without fitting, calibrating, threshold-selecting, writing artifacts, or importing Streamlit. Both official SHA-256 values and the four probabilities/displays remain unchanged.
- [x] Published files contain only declared aggregates and synthetic metadata: no real feature row, target vector, per-row probability, split index, SHAP vector, or user-uploaded content.
- [x] Documentation and any user-facing copy prohibit causal, clinical, prescriptive, discriminatory, or blanket fair/unfair claims and describe unavailable demographic coverage explicitly.

#### P12 Deliverables

- `src/fairness.py`.
- `tests/test_fairness.py`.
- `docs/p12-fairness/calibration_support.csv`: the pre-audit calibration-only support table from `prepare_data()` and cohort grouping, generated and validated before operational D-029 acceptance and reviewed with the acceptance update.
- `docs/p12-fairness/report.md` with the accepted D-029 through D-031 contracts, official aggregate results, uncertainty, interpretation, limitations, hashes, and reproduction procedure.
- Deterministic aggregate CSVs for group support, probability/ranking/calibration metrics, group-minus-overall gaps, common-bin reliability data, and all D-019 scenario metrics.
- Accessible offline plots for metric intervals and subgroup calibration; no real row or individual explanation output.
- Controlled README and planning-document updates; Streamlit changes only if D-031 explicitly accepts them.

#### P12 Guardrails

- [x] Do not retrain, reweight, resample training data, recalibrate, regenerate an artifact, compare a new model, optimize a threshold, or implement mitigation in P12.
- [x] Do not create group-specific thresholds, risk labels, recommendations, diagnoses, or different serving behavior. D-019 remains probability-only.
- [x] Do not use SHAP importance or local contributions as fairness evidence, and do not reinterpret P9/P10 outputs as causal or intervention effects.
- [x] Do not hide, merge, rename, or remove a predeclared group after observing official results. Apply the accepted support rule mechanically and publish every eligible result.
- [x] Do not claim that observed differences prove discrimination or causality, or that small/non-significant differences prove fairness. Confidence intervals are descriptive uncertainty, not certification.
- [x] Do not generalize beyond the 2015 self-reported BRFSS population or imply coverage of identities absent from the dataset, including race/ethnicity and non-binary sex/gender identities.
- [x] Do not publish real records, row-level predictions, targets, split indices, or small-cell drill-downs. P12 output is aggregate and offline.
- [x] Keep P13 product polish, CI, `skops`, authentication, persistence, analytics, and any fairness-mitigation project outside P12.

#### Definition of Done for P12

- [x] US-0604, US-0614, and US-0615 are Done; D-029 through D-031 are Accepted in the required pre-audit order; and `calibration_support.csv`, the decision updates, and the complete evidence package passed joint human review.
- [x] Every predeclared cohort has transparent support/prevalence output; every eligible cohort has the complete probability, ranking, calibration, uncertainty, gap, and frozen-threshold evidence.
- [x] The official audit is reproducible from the documented command and publishes only aggregates, with exact formulas, seed, versions, split provenance, and artifact hashes.
- [x] Conclusions communicate differences, uncertainty, base rates, historical/self-reported-label limitations, and missing demographic coverage without causal, clinical, or fairness-certification claims.
- [x] The model, artifact schema, calibration method, D-019 policy, P9/P10/P11 behavior, both official artifacts, and four reference displays remain unchanged.
- [x] Focused and full tests, dependency/compile/diff checks, deterministic regeneration, privacy checks, and every applicable gate pass; D-031 makes UI/deployment verification not applicable.
- [x] P12 moved from Ready to Done after implementation commit `1f600e8` versioned the complete audit package and human review completed evidence closure. P13 remains Future.

## Epic E9: Product Polish and Portfolio Packaging

P13 is a presentation and delivery-quality phase over the frozen, publicly verified product. It does not reopen data preparation, model selection, calibration, thresholds, SHAP mathematics, scenario semantics, batch contracts, or fairness analysis. Rolling-wave refinement completed on 2026-07-17: the four stories below are Ready, while D-032 through D-035 remain Pending until the evidence required by each decision exists. No P13 feature, CI workflow, demo asset package, or serialization change has been implemented yet.

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0901 | As a non-technical user, I want a polished and navigable product experience so that I can find the prediction workflows, supporting explanations, and project context without confusion. | P1 | Ready | Before navigation changes, D-032 is resolved from desktop/mobile and headless state evidence. The accepted UI keeps individual prediction as the obvious default, preserves the separate batch workflow and progressive P9/P10 sections, provides concise project/architecture context at an accessible level, and improves hierarchy, labels, wrapping, responsive layout, and non-color-only cues without weakening the medical disclaimer. Artifact failures, failed resubmission, navigation changes, upload replacement, and resets cannot display stale results. No analytics, persistence, account, diagnosis, recommendation, or individual fairness judgment is introduced. |
| US-0902 | As a technical reviewer, I want an accurate architecture explanation so that I can understand how data, offline analysis, validated artifacts, application modules, and deployment boundaries fit together. | P1 | Ready | A versioned architecture page and diagram trace manual data acquisition, deterministic splits, offline training/calibration/explainability/fairness evidence, the schema-version-2 model and safe SHAP-background artifacts, Streamlit serving, individual/scenario/batch paths, tests, and deployment. Every node and claim maps to real repository code or an Accepted decision; offline and runtime boundaries, trusted-artifact assumptions, privacy limits, and non-goals are explicit. Streamlit presents only a short plain-language overview and link, while GitHub carries the detailed technical explanation. |
| US-0903 | As a portfolio reviewer, I want concise, evidence-backed demo and project materials so that I can assess the work quickly without reading every implementation document first. | P1 | Ready | D-033 is resolved before screenshots or headline narrative are finalized. README provides a clear project pitch, live-app entry point, architecture/result/limitation summary, and links to deeper evidence. The portfolio package contains an approved desktop/mobile screenshot set or an evidenced equivalent, useful text alternatives, and short/medium/technical CV or interview narratives. Assets use only the four public synthetic profiles or generated safe batch examples, contain no real BRFSS/user row, and make no causal, clinical, production-scale, discrimination, or certified-fairness claim unsupported by versioned evidence. |
| US-0904 | As a maintainer and reviewer, I want automated clean-clone verification and a closed serialization assessment so that the final portfolio exposes its quality controls and remaining trust boundary honestly. | P1 | Ready | D-034 is resolved before a CI workflow is added, and D-035 is resolved before final packaging claims artifact hardening. The accepted least-privilege workflow installs the pinned Python 3.12 environment without private data or credentials, runs the accepted dependency/compile/test checks, and passes remotely before any status badge is shown. The complete raw-data suite still passes locally. D-035 records either a justified final retention of the current controlled `joblib` contract or a separately planned migration; P13 itself does not regenerate or replace either official artifact. Final local, headless, visual, link, hash, and public Streamlit checks pass. |

### Candidate Tasks for P13

Planning evidence is concrete: P12 is Done; the app currently exposes individual and batch workflows through one Streamlit entry point; the README has reproducible run/deployment/audit material but no dedicated architecture or recruiter-oriented overview; there is no `.github/` CI workflow; D-010 explicitly left a `skops` evaluation for final packaging; and the current regression baseline is 448 passing tests with exact model/background hashes and reference displays of 0.3%, 60.0%, 70.0%, and 79.9%. These facts justify refinement but do not pre-accept a navigation, asset, CI, or serialization outcome.

#### Increment 1 -- UX, Architecture, Publication, and Quality Contracts (approximately 1 day)

- [ ] Audit the current deployed/local app on representative desktop and narrow/mobile viewports. Record navigation depth, initial comprehension, label wrapping, content density, keyboard-visible controls where supported, disclaimer visibility, state transitions, and failure/fallback behavior without changing code.
- [ ] Inventory the real offline/runtime architecture from `src/`, `app/`, tests, artifacts, deployment configuration, and Accepted decisions. Identify exactly which details belong in a short Streamlit overview versus the technical GitHub architecture page.
- [ ] Resolve D-032 from the UX/state spike before changing Streamlit navigation or layout. Keep individual prediction as the default and predeclare the state/fallback behavior of every accepted view transition.
- [ ] Resolve D-033 from a written asset and claims inventory before capturing screenshots or drafting final headline/CV copy. Freeze approved synthetic inputs, asset types, accessibility treatment, and evidence links.
- [ ] Resolve D-034 from a clean-clone CI spike before creating `.github/workflows/`. Freeze triggers, permissions, Python/install/test commands, expected raw-data skips, cache behavior, timeout, and badge gate.
- [ ] Resolve D-035 from a `joblib` versus `skops` compatibility, threat-model, maintenance, deployment, and hash-cascade comparison. Do not create or migrate an artifact during the evaluation.

#### Increment 2 -- Product UX and Architecture Communication (approximately 1-2 days)

- [ ] Implement only the D-032-selected navigation and presentation route. Preserve the current individual/batch separation, prediction-first default, hash-bound state, resets, safe fallbacks, and all P8-P11 wording/contracts.
- [ ] Add concise non-technical project and architecture context to Streamlit with progressive disclosure and a clear route back to prediction. Do not load raw data, global evidence, or offline analysis at app runtime.
- [ ] Create a detailed versioned architecture page and diagram covering data provenance, offline pipeline, artifacts, serving paths, privacy boundaries, deployment, tests, and limitations. Link every important component to its real source or decision.
- [ ] Verify responsive layout, complete labels, readable wrapping, heading order, non-color-only meaning, disclaimers, downloads, explanation chart, scenario comparison, and error states on desktop and narrow viewports.

#### Increment 3 -- README, Demo Assets, and Portfolio Narrative (approximately 1-2 days)

- [ ] Restructure the top of README for fast review: concise problem/solution statement, educational boundary, live demo, key capabilities, architecture link, validated results, responsible limitations, and reproduction path. Preserve the detailed sections and source links already present.
- [ ] Generate the D-033-approved screenshot/demo set only from public synthetic reference profiles and generated safe batch examples. Store optimized assets under `docs/p13-portfolio/` with stable names and textual alternatives; do not commit real or user-submitted rows.
- [ ] Add a portfolio narrative with a one-sentence summary, evidence-backed CV bullets, a short recruiter explanation, and a technical interview version. Distinguish built functionality, measured results, limitations, and future work.
- [ ] Check every numeric, performance, fairness, privacy, deployment, and reproducibility claim against versioned reports or executable tests. Remove unsupported superlatives and never describe the model as diagnostic, clinically validated, unbiased, secure for untrusted artifacts, or production-scale.

#### Increment 4 -- CI, Regression, Deployment, and Final Closure (approximately 1-2 days)

- [ ] Implement only the D-034-accepted least-privilege CI workflow. It must work from a clean clone without the raw CSV or credentials, use the pinned Python 3.12 dependency contract, run the accepted checks, avoid training/evidence/artifact writes, and prove a real remote green run before adding a badge.
- [ ] Record D-035's accepted final serialization outcome. If `joblib` is retained, document the trusted-source/hash/validator/environment boundary; if migration is recommended, create a separate future hardening item without changing P13 artifacts.
- [ ] Run focused app/navigation/portfolio/CI tests, the clean-clone no-data command, the complete local raw-data suite, `pip check`, compile checks, link/asset validation, `git diff --check`, both official SHA-256 checks, and the four exact reference-profile regressions.
- [ ] Review the final UI in a real browser on desktop and narrow viewports, push and redeploy any Streamlit changes, then pass mandatory public smoke checks for the healthy individual prediction/explanation/scenario path, valid-plus-mixed batch path, navigation/about path, disclaimer, and downloads.
- [ ] Update README, roadmap, backlog, decisions, iteration log, and this plan only after evidence exists. Move P13 from Ready to Done only after human review, the implementation commit, a green remote CI run, and applicable public verification.

#### Expected P13 Tests

- [ ] Navigation preserves a clear prediction-first default and cannot retain an old individual or batch result after failed scoring, artifact-hash change, reset, upload replacement, or an incompatible view transition.
- [ ] The architecture/about route performs no training, calibration, evidence regeneration, raw-data access, remote data fetch, user-data write, analytics, or external logging.
- [ ] P8-P12 contracts remain exact: artifact schema/method/scenarios, P9 explanation fidelity, P10 whitelist/comparison semantics, P11 CSV contract, P12 report-first boundary, both official hashes, and the four reference probabilities/displays.
- [ ] Architecture links, README links, asset paths, captions/text alternatives, and live-app links resolve; diagrams and claims match real modules and Accepted decisions.
- [ ] A source/asset privacy guard permits only approved synthetic profiles and generated safe batch examples in P13 demo material and rejects raw/test/user rows, targets, row-level audit outputs, or identifying free text.
- [ ] The CI workflow uses pinned Python 3.12 commands, least-privilege permissions, no secrets/private data, no artifact generation, and the accepted no-data test behavior; a clean local clone-equivalent command matches the remote result.
- [ ] Desktop and narrow-viewport review confirms no clipped controls, illegible wrapping, hidden disclaimer, color-only meaning, broken chart/download, or navigation dead end.
- [ ] The deployed app passes the D-032/D-033 public healthy-path smoke contract after the final P13 commit.

#### P13 Deliverables

- Controlled `app/streamlit_app.py` and app-test changes selected by D-032.
- `docs/architecture.md` with an accurate technical diagram and offline/runtime/privacy boundaries.
- `docs/p13-portfolio/` containing only approved optimized demo assets and any asset manifest/evidence required by D-033.
- `docs/portfolio-summary.md` with short, CV, recruiter, and technical-interview narratives grounded in versioned evidence.
- A D-034-approved `.github/workflows/` CI workflow and badge only after its first successful remote run.
- D-035 comparison evidence and the final retained-or-deferred serialization rationale, with no P13 artifact migration.
- README and planning-document updates, focused tests, visual evidence, and public closure record.

#### P13 Guardrails

- [ ] Do not retrain, reweight, recalibrate, regenerate or replace artifacts, change the P8 probability contract, introduce thresholds/categories, or alter the four reference outputs.
- [ ] Do not expand SHAP mathematics/background, scenario inputs, batch schemas/limits, or P12 cohorts/interpretation. P13 presents existing verified behavior; it does not reopen it.
- [ ] Do not publish real BRFSS rows, user uploads, per-row probabilities/targets, split indices, or individual fairness claims in screenshots, diagrams, examples, logs, CI artifacts, or portfolio text.
- [ ] Do not add authentication, accounts, persistence, analytics, external user-input logging, remote model/data downloads, or general data storage.
- [ ] Do not claim diagnosis, medical advice, causal effects, clinical validation, universal fairness, discrimination findings, production-scale reliability, or safe loading of untrusted artifacts.
- [ ] Do not migrate to `skops`, add a second official artifact, or weaken current validation during P13. A recommended migration requires a separate planned phase.
- [ ] Do not make CI depend on Kaggle credentials, the ignored raw dataset, deployment secrets, or mutable external evidence; do not display a CI badge before a real green run.
- [ ] Keep future mitigation, additional demographic data, dependency/lockfile migration, authentication, persistence, analytics, and artifact-format migration outside the P13 critical path.

#### Definition of Done for P13

- [ ] US-0901 through US-0904 are Done and D-032 through D-035 are Accepted before their dependent implementation or publication steps.
- [ ] The final app is clearer on desktop and narrow viewports, preserves every validated serving/privacy contract, and provides accessible non-technical project context without overwhelming the prediction path.
- [ ] The technical architecture page is accurate, linked, reproducible, and explicit about offline/runtime, artifact-trust, privacy, deployment, and limitation boundaries.
- [ ] README, demo assets, and portfolio/CV narratives are complete, accessible, synthetic-only, and trace every material claim to versioned evidence.
- [ ] Clean-clone CI passes remotely without private data or credentials, the full local raw-data suite passes, and dependency/compile/link/asset/diff checks are clean.
- [ ] Both official artifact hashes and the four exact reference probabilities/displays remain unchanged; D-035 is closed without a P13 artifact migration.
- [ ] Human review, implementation commit, redeployment, and mandatory public individual/batch/navigation smoke verification pass before P13 moves from Ready to Done.
