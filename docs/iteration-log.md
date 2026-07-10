# Iteration Log

This file tracks short planning and development iterations. It is intentionally lightweight: each entry should capture what was planned, what changed, what was completed, and what should happen next.

## Iteration 0: Planning and Analysis

**Date:** 2026-07-04 to 2026-07-05

**Status:** Completed

**Goal:** Establish the project planning structure and identify early data/modeling assumptions before implementation.

### Planned

- Create repository-facing planning documentation.
- Define the roadmap/backlog workflow.
- Record key project decisions.
- Inspect the local dataset at a high level.
- Identify any immediate planning risks.

### Completed

- Created project charter.
- Created high-level roadmap.
- Created living backlog.
- Created ML analysis plan.
- Created decision log.
- Created iteration log.
- Updated README with links to planning documentation.
- Added initial `.gitignore` rules for local raw data and generated model artifacts.
- Initially detected a multiclass `dataset.csv`; later replaced and confirmed the local file now uses `Diabetes_binary` as target.
- Confirmed binary target distribution: 218,334 negative cases and 35,346 positive cases.
- Recorded the binary target formulation as an accepted decision.
- Refined planning documentation based on review feedback around dependencies, data acquisition, artifact serialization, and testing.
- Clarified that what-if or scenario simulations are post-MVP in-scope features, while causal claims and medical recommendations remain out of scope.

### Decisions Added

- Keep the Spanish blueprint outside the repository.
- Use English for repository-facing documentation.
- Use lightweight Scrum-inspired rolling-wave planning.
- Treat the model as educational, not medical.
- Keep raw CSV data out of git for now.
- Train offline and serve serialized artifacts in Streamlit.
- Use the original imbalanced binary BRFSS dataset with `Diabetes_binary` as target.
- Start with pinned `requirements.txt` for MVP dependency management.
- Document manual Kaggle data acquisition before adding automation.
- Define focused pytest targets for the MVP.
- Use `joblib` for MVP model artifact serialization, with `skops` left as a later safety evaluation.

### Follow-Up

- Create the initial repository structure.
- Define data acquisition instructions.
- Select Python version and create the first pinned dependency file.
- Implement and test `joblib` artifact serialization before Streamlit deployment.
- Start Data Understanding and EDA with the confirmed binary dataset.

## Iteration 1: Project Setup and Data Governance

**Date:** 2026-07-05

**Status:** Completed

**Goal:** Create the repository structure, enforce the raw data policy, and resolve the MVP dependency strategy (D-012) so that EDA can start on a reproducible foundation.

### Planned

- Create the initial folder structure for source code, notebooks, app, tests, data, and models.
- Move the local raw dataset to its standard git-ignored path.
- Write data acquisition and data handling documentation.
- Select the supported Python version and create the first pinned `requirements.txt`.
- Update planning documents to reflect the completed setup.

### Completed

- Created `src/`, `notebooks/`, `app/`, `tests/`, `data/raw/`, `data/processed/`, and `models/` with lightweight placeholders only where needed (`__init__.py` for the Python packages, `.gitkeep` for empty directories).
- Moved `dataset.csv` from the repository root to `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` and verified with `git check-ignore` that the file remains ignored.
- Added `data/README.md` documenting the Kaggle source, CC0 license, expected local path, manual download steps, and who actually needs the raw file (EDA/training reproduction only; the Streamlit app serves trained artifacts).
- Selected Python 3.12 (local default, mature wheel support, deployable on Streamlit Community Cloud) and recorded it in `.python-version`.
- Created the first hand-curated pinned `requirements.txt` covering data analysis, modeling, visualization, notebooks, the app, and testing, instead of freezing the noisy global environment.
- Updated the README with the project structure, environment setup instructions, and dataset policy.
- Marked all Epic E1 stories as Done and set P1 to Done in the roadmap.
- A `scripts/` directory was intentionally not created yet; it will be added when the first training or download script exists, consistent with rolling-wave planning.

### Decisions Added

- Resolved D-012: Python 3.12 with a hand-curated pinned `requirements.txt` (Accepted).
- D-013 (artifact distribution for deployment) intentionally remains Pending until P7.

### Follow-Up

- Create the first virtual environment and run `pip install -r requirements.txt` to validate the pins; adjust any pin that fails to resolve.
- Start P2: Data Understanding and EDA with the first notebook under `notebooks/`.
- Keep D-013 in view before the first public deployment.

## Iteration 2: Data Understanding and EDA

**Date:** 2026-07-05

**Status:** Completed

**Goal:** Validate the dataset structure and produce the first reproducible EDA artifact so that data preparation decisions in P3 are based on observed evidence.

### Planned

- Validate the pinned environment in a fresh virtual environment before starting EDA.
- Create the first notebook under `notebooks/` for data understanding and exploratory analysis.
- Load the raw CSV from the documented `data/raw/` path.
- Verify shape, columns, target, row count, data types, and feature ranges.
- Classify columns into initial binary, ordinal, and numeric feature groups for later pipeline design.
- Report missing values, unexpected values, duplicate rows, and target distribution.
- Document the positive class prevalence and its implications for baseline evaluation.
- Add basic descriptive summaries or plots for important features.
- Add a lightweight Spearman correlation review to identify obvious feature relationships without selecting or dropping features.
- Analyze memory usage and record downcasting recommendations for P3 without changing the raw data contract.
- Explain the class imbalance and why accuracy alone is not an adequate evaluation metric.
- Record P3 follow-up tasks for validation, cleaning, duplicate handling, and split logic.

### Scope Guardrails

- Do not train models in this iteration.
- Do not create train/test splits yet.
- Do not apply balancing, SMOTE, calibration, SHAP, or Streamlit work.
- Do not select or remove features based on correlation in this iteration.
- Do not drop duplicate rows in E2 unless the reason is explicitly documented and carried into P3.

### Completed

- Created a local virtual environment (`.venv`, Python 3.12.7) and validated `pip install -r requirements.txt`: all pinned versions resolved with no conflicts (`pip check` reported no broken requirements). No pin adjustments were needed.
- Created `notebooks/01_data_understanding_eda.ipynb`, executed top-to-bottom with no errors, loading the raw CSV from `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` with a clear `FileNotFoundError` if the file is missing.
- Confirmed shape (253,680 rows x 22 columns), all 22 documented columns present with no unexpected columns, and the target column (`Diabetes_binary`, strictly `{0.0, 1.0}`, no missing values).
- Classified features into binary (14), ordinal (4: `GenHlth`, `Age`, `Education`, `Income`), and numeric (3: `BMI`, `MentHlth`, `PhysHlth`) groups; confirmed the raw file interleaves these rather than grouping by type.
- Confirmed all columns are `float64` but every value is a whole number in `[0, 98]`; no out-of-range or unexpected values found in any binary/ordinal/numeric feature; zero missing values across the dataset.
- Found 24,206 exact duplicate rows (~9.5% of rows, by `drop_duplicates()` semantics; 35,575 rows / 14.0% are involved in some duplicate group). Duplicated rows are heavily skewed toward the negative class (~1% positive vs. ~13.9% overall) -- dropping them would raise prevalence to ~15.3%. No rows were dropped.
- Confirmed target prevalence: 218,334 negative (86.07%) vs. 35,346 positive (13.93%), consistent with the P0/P1 inspection.
- Added descriptive plots/tables for `BMI`, `GenHlth`, `Age`, `Income`, and `HighBP`, showing the positive rate rising with worse `GenHlth`, older `Age` groups, and lower `Income`, and markedly higher with `HighBP`.
- Ran a Spearman correlation review: strongest individual associations with the target are `GenHlth` (0.29), `HighBP` (0.26), `BMI` (0.23), `DiffWalk` (0.22), `HighChol` (0.20), and `Age` (0.18); no features were removed. Flagged moderate feature-pair redundancy (`Education`/`Income` 0.45, `GenHlth`/`PhysHlth` 0.45, `GenHlth`/`DiffWalk` 0.42, `PhysHlth`/`DiffWalk` 0.42) for P3 awareness only.
- Measured current memory usage (42.58 MB, all `float64`) and confirmed a projected 87.5% reduction (to 5.32 MB) if all columns were downcast to `uint8`, with a round-trip equality check confirming no value loss. Downcasting was not applied to the raw dataframe.
- Wrote a portfolio-facing explanation of why accuracy alone is insufficient for this ~13.9%-prevalence imbalanced classification problem, reinforcing the metrics already selected in `docs/ml-analysis-plan.md`.

### Decisions Added

- Added D-014 as a pending decision: duplicate-row handling policy (keep vs. drop) must be resolved before P3 splits because it changes the prevalence each split must preserve. No duplicate rows were dropped and no policy was chosen during EDA.

### Follow-Up

- Decide and document an explicit duplicate-row policy (keep vs. drop) before building splits, accounting for the ~13.9% -> ~15.3% prevalence shift if duplicates are dropped.
- Implement `uint8` downcasting for all 22 columns inside the P3 data-loading module (e.g. `src/data.py`), validated against the documented ranges.
- Build the stratified 70/10/20 train/calibration/test split described in `docs/ml-analysis-plan.md`, preserving whichever prevalence results from the duplicate-policy decision.
- Add pytest coverage for schema/range/target validation and split class-proportion checks.
- Carry forward the correlation observations into preprocessing/model design discussions without pre-emptively dropping features.

### Expected Deliverables

- A committed-ready EDA notebook under `notebooks/`. -- Done: `notebooks/01_data_understanding_eda.ipynb`.
- Updated backlog status for E2 stories. -- Done: US-0201, US-0202, US-0203 marked Done in `docs/backlog.md`.
- Updated follow-up tasks for E3 based on EDA findings. -- Done: added a "Candidate Tasks for E3" section in `docs/backlog.md`.
- Any dependency pin adjustment documented if environment validation requires it. -- Not needed: all pins resolved cleanly.

## Iteration 3: Data Preparation and Splitting

**Date:** 2026-07-06 to 2026-07-07

**Status:** Completed

**Goal:** Convert the P2 findings into reusable data loading, validation, downcasting, and stratified split logic without introducing modeling work.

### Completed

- D-014 is resolved: exact duplicate rows will be kept for MVP data preparation.
- The accepted duplicate policy keeps the full 253,680-row dataset and the observed ~13.9% positive prevalence, which the stratified 70/10/20 train/calibration/test split must preserve.
- D-015 is resolved: P3 split outputs are returned in memory for now, and `data/processed/` remains empty until a downstream consumer needs persisted files.
- `src/data.py` implements the P3 data contract, raw loading, validation, safe `uint8` downcasting, dataset summary metadata, and reproducible in-memory stratified splits.
- `tests/test_data.py` covers loading, schema/value validation, duplicate handling, downcasting, split sizes, split reproducibility, and prevalence preservation.
- P3 remains scoped to data preparation and splitting: no balancing, SMOTE, model training, feature engineering, app work, calibration, or explainability was added.
- Verified `python -m pytest tests/test_data.py -v -p no:cacheprovider --basetemp .pytest_tmp_precommit`: 27 passed, including integration checks against the local raw CSV.
- Verified `python -m compileall src tests`: OK.
- Marked US-0301, US-0302, and US-0303 Done; set roadmap P3 to Done; updated the README current status to point to P4.

### Follow-Up

- Refine P4 baseline modeling tasks before implementation.
- Use the in-memory P3 split functions from `src/data.py` for baseline training.
- Keep D-013 in view before deployment work begins.

## Iteration 4: Baseline Modeling

**Date:** 2026-07-07 to 2026-07-08

**Status:** Completed

**Goal:** Establish a first verifiable training/evaluation baseline (`DummyClassifier` and `LogisticRegression`) on top of the P3 data preparation module, without expanding into tree-based candidates or formal model comparison yet.

### Planning Notes

- A backlog micro-refinement split the former Epic E4 (Baseline and Candidate Modeling) into Epic E4 (Baseline Modeling, P4) and a new Epic E8 (Model Comparison and Selection, P5): E4 now keeps only `DummyClassifier` and `LogisticRegression`; E8 owns the tree-based candidate, formal comparison, model selection, and the serialization policy story (moved from the former US-0403 to US-0803, since `US-05xx` was already used by Epic E5).
- P4 should implement reusable modeling code, likely `src/modeling.py`, so baseline training and evaluation are testable outside notebooks.
- P4 must use `src.data.prepare_data()` as the exclusive data entry point, train only on the training split, and report baseline metrics on train and test only.
- The calibration split should remain unused in P4 so it stays reserved for later probability calibration work.
- P4 metrics: ROC-AUC, PR-AUC, recall, precision, F1, confusion matrix, and accuracy as a secondary metric only, per `docs/ml-analysis-plan.md`.
- P4 should not serialize a model artifact, train a tree-based candidate, or select a primary model; those steps belong to P5 (Epic E8).
- P4 should not add SHAP, deep calibration, Streamlit/app work, fairness analysis, or advanced threshold tuning.
- Add lightweight pytest coverage: the pipeline fits on a small sample, `predict_proba` returns probabilities in `[0, 1]`, metrics compute without errors, training never touches calibration/test rows, and P4 evaluation does not consume calibration rows.

### Completed

- `src/modeling.py` implements the P4 baseline contract: conversion of P3 `DataSplits` into X/y train/test frames (calibration deliberately absent), a `DummyClassifier` (most-frequent) trivial reference, a `StandardScaler` + `LogisticRegression` pipeline as the first interpretable model, positive-class probability prediction, and the documented metric set (ROC-AUC, PR-AUC, recall, precision, F1, confusion matrix, and accuracy as secondary context).
- `train_and_evaluate_baselines()` fits both baselines on the train split only, evaluates on train and test only, and returns lightweight in-memory `BaselineResult` structures; no model artifact or processed split file is written, and `data/processed/` and `models/` remain empty.
- Scaling is the only preprocessing: features share no common scale (`BMI` in [12, 98] vs. 0/1 indicators) and scaling keeps lbfgs convergence well-conditioned. The P2 correlation observations (`GenHlth`/`PhysHlth`/`DiffWalk`, `Education`/`Income`, Spearman ~0.42-0.45) were reviewed: all 21 features are kept, since moderate collinearity mainly affects coefficient interpretability and the default L2 regularization stabilizes the fit; no features were dropped on correlation grounds.
- `tests/test_modeling.py` covers small-sample fits for both baselines, `predict_proba` validity in `[0, 1]`, metric keys and hand-computed metric values, degenerate all-negative predictions, train/test evaluation via the orchestrator, determinism, a fit-spy check that training uses exactly the train rows, a poisoned-calibration check that P4 never consumes the calibration split, and a guard that the module never reloads or re-splits raw data.
- Verified `python -m pytest tests -v`: 40 passed (27 P3 + 13 new P4 tests). Verified `python -m compileall src tests`: OK.
- Smoke-ran the full pipeline on the real dataset via `prepare_data()`: the dummy baseline scores ROC-AUC 0.5, recall/precision/F1 0.0, and ~86.1% accuracy (the prevalence floor that motivates the secondary role of accuracy); Logistic Regression reaches test ROC-AUC ~0.819, PR-AUC ~0.394, recall ~0.16, precision ~0.52, F1 ~0.24 at the default 0.5 threshold, with train and test metrics closely aligned (no overfitting signal). The low default-threshold recall is expected without balancing/threshold tuning, which remain out of P4 scope.
- Marked US-0401 and US-0402 Done in the backlog; set roadmap P4 to Done; updated the README current status to point to P5.

### Follow-Up

- Refine P5 (Epic E8) tasks before implementation: tree-based candidate on the same P3 splits, formal comparison of Dummy/Logistic Regression/tree-based with the documented imbalanced-classification metrics, primary model selection, and the serialization policy (US-0803, D-010).
- Consider `class_weight="balanced"` during P5 if it fits the model-comparison scope; leave threshold analysis for P8 unless P5 scope is explicitly expanded.
- Keep the calibration split untouched until probability calibration work (P8).
- Keep D-013 (artifact distribution for deployment) in view before P7.

## Iteration 5: Model Comparison and Selection

**Date:** 2026-07-09

**Status:** Completed

**Goal:** Compare the P4 baselines against at least one restrained tree-based candidate and select the primary MVP model using the documented imbalanced-classification metric protocol.

### Planned Scope

- Use the existing P3/P4 contracts: `prepare_data()` / `DataSplits`, train/test conversion, and reusable metrics in `src/modeling.py`.
- Train candidates on the train split only and evaluate on train and test only.
- Keep the calibration split untouched for P8 probability calibration.
- Compare Dummy, Logistic Regression, and a tree-based candidate such as `HistGradientBoostingClassifier` or `RandomForestClassifier` in an in-memory comparison result.
- Define selection criteria before implementation, prioritizing PR-AUC and positive-class recall/precision/F1 over accuracy because the selected population is only ~13.9% positive.
- Decide whether a simple imbalance-aware variant such as `class_weight="balanced"` belongs in P5; keep SMOTE, advanced threshold tuning, calibration, SHAP, fairness analysis, Streamlit/app work, batch prediction, and scenario exploration out of scope.
- Confirm serialization timing before any artifact write. D-010 already accepts `joblib` as the MVP format, but D-013 remains pending for deployment distribution.

### Prepared Updates

- Refined Epic E8 stories and candidate tasks in the backlog.
- Moved roadmap P5 from Planned to Ready.
- Added D-016 as a pending decision for the primary MVP model selection.

### Completed

- Extended `src/modeling.py` (no separate module was needed) with the P5 comparison contract: a restrained tree-based candidate (`HistGradientBoostingClassifier`, library defaults, fixed seed), a single imbalance-aware variant (`class_weight="balanced"` Logistic Regression, isolating the reweighting effect against the P4 baseline), `compare_models()` reusing the shared fit/evaluate loop, `comparison_table()` producing a long-format in-memory DataFrame with metrics by model and split, and `select_primary_model()`.
- Selection criteria were fixed in code before the comparison ran: the dummy baseline is never selectable; candidates with a train/test PR-AUC gap above 0.10 are deprioritized as obvious overfitting risk; remaining candidates rank by test PR-AUC, then positive-class F1, recall, precision, and ROC-AUC, with the model name as a final deterministic tie-break. Accuracy is deliberately excluded from ranking because an always-negative model already scores ~86% at ~13.9% positive prevalence.
- All four candidates (dummy, Logistic Regression, balanced Logistic Regression, `HistGradientBoostingClassifier`) were trained on the train split only and evaluated on train and test only via the P3/P4 contracts; the calibration split was never read, no raw data was reloaded or re-split, and everything stayed in memory.
- Ran the comparison on the real dataset (`prepare_data()`, 253,680 rows). Test-split results: `HistGradientBoostingClassifier` ROC-AUC 0.827, PR-AUC 0.423, recall 0.157, precision 0.563, F1 0.246, accuracy 0.866, train/test PR-AUC gap 0.032; Logistic Regression ROC-AUC 0.819, PR-AUC 0.394 (matching the P4 smoke run); balanced Logistic Regression ROC-AUC 0.820, PR-AUC 0.393, recall 0.760, precision 0.311, F1 0.441, accuracy 0.732. No candidate triggered the overfitting flag.
- Resolved D-016: `HistGradientBoostingClassifier` is the primary MVP model because it leads the primary ranking metric (test PR-AUC), also has the strongest ROC-AUC and positive-class precision, and shows a small train/test PR-AUC gap. The balanced variant improved default-threshold recall/F1 by construction but did not improve PR-AUC or ROC-AUC, so simple reweighting was not selected; the low default-threshold recall of the selected model is documented as a P8 threshold-analysis concern.
- Confirmed serialization timing as D-017: no artifact is written in P5; the selected model will be serialized at the start of P6 with the D-010 `joblib` format plus a local load/predict check (US-0503). `models/` and `data/processed/` remain empty; D-013 stays pending for deployment distribution.
- Added `tests/test_model_comparison.py` (15 tests, all on small synthetic splits reusing the P3/P4 fixtures): tree-based fit/evaluation, valid `predict_proba` in `[0, 1]` for both new candidates, comparison completeness and metric keys, comparison-table structure by model and split, compare/select determinism, selection unit tests on hand-crafted results (PR-AUC priority, accuracy ignored, F1 and name tie-breaks, overfitting deprioritization, dummy never selectable, reference-only error), a fit-spy check that all four models train on exactly the train rows, a poisoned-calibration check, a source guard against reloading/re-splitting/serializing, and a runtime check that no files appear in `models/` or `data/processed/`.
- Verified `python -m pytest tests -v -p no:cacheprovider`: 55 passed (27 P3 + 13 P4 + 15 P5). Verified `python -m compileall src tests`: OK.
- Marked US-0801, US-0802, and US-0803 Done in the backlog; set roadmap P5 to Done; updated the README current status to point to P6.

### Decisions Added

- Resolved D-016: `HistGradientBoostingClassifier` (library defaults, fixed seed) selected as the primary MVP model (Accepted).
- Added D-017: model artifact serialization is deferred to the start of P6, paired with a local load/predict verification (Accepted).

### Follow-Up

- Refine P6 (Epic E5, Streamlit MVP) tasks before implementation: offline training/serialization of the D-016 model per D-017 and D-010, artifact load/predict smoke test (US-0503), individual prediction page with visible disclaimers (US-0501, US-0502).
- Resolve D-013 (artifact distribution for deployment) before the first public deploy in P7.
- Keep the calibration split untouched until P8; revisit the selected model's low default-threshold recall in the P8 threshold analysis.

## Next Iteration Planning: P6 Streamlit MVP

**Date:** 2026-07-09

**Status:** Ready for implementation

**Goal:** Serialize the D-016 primary model as a local artifact per the D-017 timing and D-010 format, then build a minimal local Streamlit app that loads it for single-case educational risk prediction.

### Planned Scope

- Train the D-016 `HistGradientBoostingClassifier` through the existing P3/P4/P5 contracts (`src.data.prepare_data()`, `src/modeling.py` builders) at the start of P6; do not reload or re-split raw data ad hoc.
- Serialize the fitted model with `joblib` (D-010) following the D-017 timing, through a small artifact helper module, likely `src/artifacts.py`, that also saves metadata: feature order, target name, model type, key P5 comparison metrics, and package versions or other minimal reproducibility metadata where feasible.
- Add a local load/predict verification (reloaded artifact returns probabilities in `[0, 1]`) before the app depends on it; keep `models/*.joblib` git-ignored.
- Build a minimal Streamlit app, likely `app/streamlit_app.py`, with a single-case input form for all 21 `src.data.FEATURE_COLUMNS`, grouped by binary/ordinal/numeric type, validated against the P3 feature ranges, assembled into a one-row DataFrame in training feature order, and scored with `predict_proba` into an educational risk percentage alongside a visible medical disclaimer and limitations text.
- Keep P6 a local functional MVP only: no calibration, threshold tuning, SHAP, fairness analysis, batch prediction, scenario exploration, public deployment, or artifact distribution decisions. D-013 stays pending for P7 unless a real deployment-distribution decision is actually made.
- Expected tests: artifact save/load round-trip; local load/predict probability in `[0, 1]`; metadata includes feature order and selected model identity; the input-to-DataFrame helper preserves feature order; input validation rejects missing/out-of-range values; an app-facing prediction helper is testable without launching Streamlit; and no public deployment logic is introduced.

### Prepared Updates

- Refined Epic E5 stories and candidate tasks in the backlog, including a new US-0504 for the artifact helper module.
- Moved roadmap P6 from Planned to Ready.
- Clarified the local artifact/app boundary and the P6/P7 split in the ML analysis plan.
