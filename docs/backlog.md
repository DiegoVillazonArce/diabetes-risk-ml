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
| US-0801 | As a reviewer, I want a tree-based candidate compared formally against the P4 baseline so that trade-offs are visible. | P0 | Ready | Dummy, Logistic Regression, and at least one restrained tree-based candidate (preferably `HistGradientBoostingClassifier` or `RandomForestClassifier`) are trained through the P3/P4 contracts and evaluated with the same train/test metric protocol. |
| US-0802 | As the developer, I want a primary candidate model selected and justified so that later phases serve a deliberate choice. | P0 | Ready | Selection criteria are defined before implementation, the selected model is documented with metric-based rationale, and PR-AUC plus positive-class recall/precision/F1 are prioritized over accuracy. |
| US-0803 | As the developer, I want a model serialization policy so that Streamlit only loads known project artifacts. | P1 | Ready | Serialization format and timing are documented consistently with D-010 (`joblib`) while D-013 remains pending for deployment artifact distribution; any artifact write requires an explicit decision and a local load/predict check. |

### Candidate Tasks for E8

- [ ] Use the existing P3/P4 contracts: consume `src.data.prepare_data()` or supplied `DataSplits`, convert splits through the existing train/test helpers, and avoid ad hoc raw-data loading, re-splitting, or processed split files.
- [ ] Reuse or extend `src/modeling.py` for P5 model builders, evaluation, and comparison unless a separate orchestration module has a clear reason.
- [ ] Train Dummy, Logistic Regression, and at least one restrained tree-based candidate on the train split only. Prefer a practical MVP candidate such as `HistGradientBoostingClassifier` or `RandomForestClassifier` with deterministic settings.
- [ ] Decide during P5 implementation whether a simple imbalance-aware variant such as `class_weight="balanced"` belongs in the comparison. Do not add SMOTE, advanced resampling, threshold tuning, or calibration unless P5 scope is explicitly expanded first.
- [ ] Evaluate every candidate on train and test only with the same metric protocol used in P4: ROC-AUC, PR-AUC, positive-class recall, precision, F1, confusion matrix, and accuracy as secondary context.
- [ ] Keep the calibration split untouched so it remains reserved for P8 probability calibration.
- [ ] Produce an in-memory comparison table or structured result with metrics by model and split; do not require notebooks as the only comparison path.
- [ ] Define selection criteria before comparing models. Prioritize PR-AUC and positive-class recall/precision/F1 because the selected population has ~13.9% positive prevalence; treat accuracy as secondary because an always-negative model already scores about 86%.
- [ ] Select and document the primary model with metric-based rationale. Do not create a final model-selection decision before the P5 metrics exist.
- [ ] Confirm the selected model's serialization policy before writing artifacts: D-010 accepts `joblib` for MVP serialization, while D-013 still governs how artifacts reach deployment. Do not write model artifacts in P5 unless the timing is explicitly decided and tested.
- [ ] Add pytest coverage expectations for tree-based fit/`predict_proba`, comparison result structure, deterministic selection behavior, no calibration-split usage, no ad hoc re-splitting/reloading, and no artifact writes unless explicitly authorized by a decision.
- [ ] Keep P5 limited to model comparison and selection: no Streamlit/app work, SHAP, fairness analysis, deep calibration, advanced threshold tuning, batch prediction, or scenario exploration.

## Epic E5: Streamlit MVP

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0501 | As a user, I want to enter health indicators and receive an estimated risk percentage so that the model output is understandable. | P0 | To Do | Streamlit page loads model artifact and returns a probability with a visible disclaimer. |
| US-0502 | As a reviewer, I want to see model limitations in the app so that the project does not overclaim medical value. | P0 | To Do | Limitations and medical disclaimer are visible in the MVP app. |
| US-0503 | As the developer, I want a smoke test for the serving path so that the app-facing artifact can produce probabilities in `[0, 1]`. | P1 | To Do | Test or script verifies artifact loading and `predict_proba` output shape/range. |

## Epic E7: MVP Documentation and Deployment

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0701 | As a reviewer, I want complete run instructions so that I can reproduce the environment, training, and app locally. | P0 | To Do | README documents environment setup, data download, training command, and app launch command. |
| US-0702 | As the developer, I want an explicit artifact distribution policy for deployment so that the deployed app can load a trained model even though model files are git-ignored. | P0 | To Do | A decision records how the artifact reaches deployment (committed exception, release asset, or build step) before the first public deploy. |
| US-0703 | As a user, I want the MVP deployed on Streamlit Community Cloud so that the app is publicly accessible. | P0 | To Do | A public URL serves the app, loads the trained artifact, and returns a probability with the medical disclaimer visible. |

## Epic E6: Post-MVP Enhancements

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0601 | As a user, I want calibrated probabilities so that risk percentages are more honest. | P1 | Deferred | Calibration method and reliability diagrams are implemented and documented. |
| US-0602 | As a user, I want explanations for model predictions so that I can understand which features influenced the output. | P1 | Deferred | SHAP global and local explanations are available with non-causal wording. |
| US-0603 | As a user, I want batch CSV prediction so that multiple cases can be scored at once. | P2 | Deferred | CSV upload, validation report, and downloadable predictions are implemented. |
| US-0604 | As a reviewer, I want a fairness audit so that subgroup performance is not ignored. | P2 | Deferred | Metrics are reported by sex, age group, and income group where applicable. |
| US-0605 | As a user, I want to explore how selected modifiable inputs affect the model output so that I can understand model sensitivity without treating it as medical advice. | P2 | Deferred | Scenario explorer only changes approved modifiable features and clearly states that outputs are model responses, not causal health effects. |
