# Backlog

## Workflow

This backlog is a living document. The roadmap defines broad direction; this file tracks user stories, tasks, priorities, and status.

Epics are ordered by MVP delivery order, not by numeric ID: Epic E7 (MVP Documentation and Deployment) appears before Epic E6 (Post-MVP Enhancements) because E7 is part of the MVP and E6 is not.

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
| US-0301 | As the developer, I want reusable data loading and validation code so that notebooks and training scripts share the same logic. | P0 | Ready | Data module loads the selected dataset and validates schema/ranges. |
| US-0302 | As a reviewer, I want stratified train/calibration/test splits so that model evaluation is trustworthy. | P0 | Ready | Split code is reproducible, stratified, and preserves original test distribution. |
| US-0303 | As the developer, I want tests for data preparation so that schema, ranges, and splits do not silently regress. | P1 | Ready | pytest tests cover required columns, expected target values, feature ranges, and split class proportions. |

### Candidate Tasks for E3

Sourced from the P2 EDA findings in `notebooks/01_data_understanding_eda.ipynb` (see Iteration 2 in `docs/iteration-log.md`):

- [ ] Decide and document an explicit duplicate-row policy (keep vs. drop). Dropping duplicates shifts positive prevalence from ~13.9% to ~15.3% (24,206 rows, ~9.5% of the dataset, skew heavily negative) -- this must be a recorded decision, not an implicit default.
- [ ] Implement `uint8` downcasting for all 22 columns in the data-loading module, validated against the documented ranges from the EDA (all observed values are whole numbers in `[0, 98]`).
- [ ] Implement the stratified 70/10/20 train/calibration/test split, preserving whichever positive prevalence results from the duplicate-policy decision above.
- [ ] Add pytest coverage for schema/range/target validation and split class-proportion checks, per the testing plan in `docs/ml-analysis-plan.md`.
- [ ] Carry forward the EDA's correlation observations (`GenHlth`/`PhysHlth`/`DiffWalk`, `Education`/`Income`) into preprocessing/model design discussions without dropping features solely on correlation grounds.

## Epic E4: Baseline and Candidate Modeling

| ID | User Story | Priority | Status | Acceptance Criteria |
|---|---|---:|---|---|
| US-0401 | As a reviewer, I want a DummyClassifier baseline so that all real models are compared against a trivial reference. | P0 | To Do | Dummy metrics are computed and included in comparison outputs. |
| US-0402 | As a reviewer, I want interpretable and tree-based candidates so that trade-offs are visible. | P0 | To Do | Logistic Regression and at least one tree-based model are trained and evaluated. |
| US-0403 | As the developer, I want a model serialization policy so that Streamlit only loads known project artifacts. | P1 | To Do | Artifact format is selected, documented, and tested with a local load/predict check. |

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
