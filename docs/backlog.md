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
| US-0701 | As a reviewer, I want complete run instructions so that I can reproduce the environment, training, and app locally. | P0 | Ready | The README, with links to supporting data documentation where useful, covers Python 3.12 virtual-environment creation and activation, installation from `requirements.txt`, the exact CSV download and local path, artifact generation with `python -m src.artifacts`, tests with `python -m pytest tests -v -p no:cacheprovider`, and app launch with `python -m streamlit run app/streamlit_app.py`, all through the same environment interpreter. It includes clear PowerShell equivalents where useful and explains how to detect and replace an artifact created with incompatible Python, scikit-learn, or joblib versions. Validation from a clean clone or environment must show that a reviewer with no prior project knowledge can reproduce training, tests, and the application. |
| US-0702 | As the developer, I want an explicit artifact distribution policy for deployment so that the deployed app can load a trained model even though model files are git-ignored. | P0 | Ready | Before the first public deployment, D-013 records a formal comparison and final choice among a controlled Git exception, a GitHub Release asset, and generation during build, covering artifact size, reproducibility, security, maintenance, Git history, network dependency, cold start, and dataset availability. Only the selected alternative is implemented. The official artifact is generated with Python 3.12 and the pinned versions; the app loads only the known project artifact and never trains, downloads arbitrary artifacts, or requires the CSV at runtime. The chosen path is verified from a clean clone and in the deployment environment. |
| US-0703 | As a user, I want the MVP deployed on Streamlit Community Cloud so that the app is publicly accessible. | P0 | Ready | A GitHub-backed Streamlit Community Cloud deployment runs on Linux with Python 3.12, installs an evidenced deployment-suitable `requirements.txt` without an unnecessary alternative requirements file, and loads the official artifact. Before public verification, P7 defines and commits a small set of reference profiles with their exact 21-feature inputs and locally verified expected display outputs. Public smoke tests reuse those profiles and verify all 21 features, exact-age conversion to the BRFSS group, documented probability tolerance/rounding, clear missing/invalid-artifact errors, and a visible medical disclaimer. The deployment publishes no raw CSV, secrets, or user data, and the README records the public URL and final run instructions. |

### Candidate Tasks for E7

P6 provides the planning evidence for this refinement: the validated single-bundle artifact is approximately 263 KB, the app loads it once and never trains, the form covers all 21 features, and exact age is converted to the corresponding BRFSS group. These facts inform D-013 but do not select a distribution alternative.

#### US-0701: Reproducible Instructions

- [ ] Document creation and activation of a Python 3.12 virtual environment from a clean clone, including clear PowerShell commands such as `.venv\Scripts\Activate.ps1` and equivalent commands for other supported shells where useful.
- [ ] Document dependency installation through the environment interpreter with `python -m pip install -r requirements.txt`, and make it clear that subsequent project commands must use that same interpreter.
- [ ] Document the selected Kaggle CSV, its exact filename, and its required location at `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`; link to the existing data acquisition instructions instead of duplicating or changing the raw-data policy.
- [ ] Document the exact artifact-generation command, `python -m src.artifacts`, and explain that it performs offline training through the existing project contract before the app is launched.
- [ ] Document the complete test command, `python -m pytest tests -v -p no:cacheprovider`.
- [ ] Document the app command, `python -m streamlit run app/streamlit_app.py`, so Streamlit uses the same environment interpreter as training and tests.
- [ ] Add troubleshooting for artifact/runtime incompatibility: show how to confirm Python, scikit-learn, and joblib versions against the pinned environment and artifact metadata; explain recognizable load/validation failures; and instruct the reviewer to rebuild the environment and regenerate the trusted artifact rather than bypass validation or load an unknown file.
- [ ] Follow the written instructions from a fresh clone or otherwise clean Python 3.12 environment, recording enough evidence that a reviewer with no prior project knowledge can reproduce artifact training, the full test suite, and the local app.

#### US-0702: Artifact Distribution Policy

- [ ] Measure and record the official artifact's size (currently approximately 263 KB) and treat a controlled Git exception as a reasonable candidate at that size, without marking it as selected or accepted before D-013 is resolved.
- [ ] Compare the three D-013 alternatives formally: a controlled exception that versions the official artifact in Git, a trusted GitHub Release asset, and deterministic artifact generation during the deployment build.
- [ ] Evaluate each alternative against artifact size, reproducibility, deserialization security and provenance, maintenance burden, Git-history impact, runtime/build network dependency, cold-start behavior, and availability of the git-ignored dataset in the build environment.
- [ ] Record the selected alternative, rationale, constraints, and verification approach in D-013 before the first public deployment; keep D-013 Pending until this comparison produces an actual choice.
- [ ] Implement only the alternative selected in D-013; do not add speculative support for the rejected alternatives.
- [ ] Generate the official artifact with Python 3.12 and the versions pinned in `requirements.txt`, and verify its metadata and local load/predict contract before distribution.
- [ ] Confirm that the deployed app resolves and loads only the artifact and provenance defined by D-013, and rejects absent, corrupt, incompatible, or untrusted bundles as required by the selected policy with clear guidance.
- [ ] Confirm that Streamlit never trains or retrains a model, never downloads an arbitrary artifact, and never needs the raw CSV at runtime; build-time generation, if selected by D-013, must remain separate from the Streamlit serving process.
- [ ] Verify the implemented policy end to end from both a clean clone and the actual deployment environment.

#### US-0703: Public Deployment

- [ ] Verify the current app and artifact path contract on a Linux, Python 3.12, Streamlit Community Cloud-compatible environment, including case-sensitive paths and repository-root execution assumptions.
- [ ] Review whether the existing `requirements.txt` is suitable for deployment; create no alternate requirements file unless an observed platform constraint provides a documented reason.
- [ ] Configure the Streamlit Community Cloud deployment from the correct GitHub repository, branch, and entry point, and select Python 3.12 in the platform settings.
- [ ] Review installation and startup logs, resolving only evidenced deployment issues and recording the final configuration.
- [ ] Confirm that the deployed app finds, validates, and loads the official artifact delivered through the D-013 policy.
- [ ] Run public-URL smoke tests that submit the form and confirm it covers all 21 model features in the expected contract.
- [ ] Verify boundary and representative exact ages map to the correct BRFSS age-group codes before inference.
- [ ] Define and commit a small deployment-smoke reference set before public verification: record every profile's exact 21-feature inputs, derive its expected displayed probability locally from the official artifact, document an appropriate comparison tolerance or display-rounding rule, and then exercise those same recorded profiles on the public app without changing the model or threshold to force a match. The manually observed P6 outputs near 0.3%, 60%, 70%, and 79.9% may guide profile selection, but are not established fixtures until their inputs and expectations are recorded and verified in P7.
- [ ] Verify clear user-facing errors for an absent or invalid artifact using a controlled deployment check without leaving the public app broken.
- [ ] Confirm the educational/medical disclaimer remains visible alongside prediction output on the public deployment.
- [ ] Confirm the raw CSV, secrets, and submitted user inputs are not published, logged by project code, or persisted.
- [ ] Add the verified public URL and final local/deployment instructions to the README, then close P7 only after the clean-clone and public smoke-test evidence satisfies all three stories.

#### P7 Scope Guardrails

- [ ] Keep calibration and threshold selection or modification out of P7; P8 remains responsible for probability calibration and threshold analysis, and the calibration split must remain untouched throughout P7.
- [ ] Do not perform experimental retraining or compare new models in P7. Deterministic generation of the already selected D-016 model is permitted only as required by the D-013 distribution policy.
- [ ] Keep SHAP out of P7; P9 remains responsible for global and local SHAP explainability.
- [ ] Keep fairness analysis, batch prediction, scenario exploration, authentication, persistence of user inputs, and user analytics out of P7 and in their later phases or future backlog refinement.

## Epic E6: Post-MVP Enhancements

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0601 | As a user, I want calibrated probabilities so that risk percentages are more honest. | P1 | Deferred | Calibration method and reliability diagrams are implemented and documented. |
| US-0602 | As a user, I want explanations for model predictions so that I can understand which features influenced the output. | P1 | Deferred | SHAP global and local explanations are available with non-causal wording. |
| US-0603 | As a user, I want batch CSV prediction so that multiple cases can be scored at once. | P2 | Deferred | CSV upload, validation report, and downloadable predictions are implemented. |
| US-0604 | As a reviewer, I want a fairness audit so that subgroup performance is not ignored. | P2 | Deferred | Metrics are reported by sex, age group, and income group where applicable. |
| US-0605 | As a user, I want to explore how selected modifiable inputs affect the model output so that I can understand model sensitivity without treating it as medical advice. | P2 | Deferred | Scenario explorer only changes approved modifiable features and clearly states that outputs are model responses, not causal health effects. |
