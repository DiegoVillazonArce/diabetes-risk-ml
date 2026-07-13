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

## Iteration 6: Streamlit MVP

**Date:** 2026-07-09 to 2026-07-10

**Status:** Completed

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

### Completed

- `src/artifacts.py` implements the P6 artifact contract (US-0504): the D-016 `HistGradientBoostingClassifier` is trained on the P3 train split only, through `src.data.prepare_data()` and the existing P5 builder/evaluation helpers (the calibration split is never read), and serialized with `joblib` (D-010) as a single bundle holding the fitted model plus serving metadata: schema version, model identity with the D-016 reference, exact `FEATURE_COLUMNS` order, target name, random seed, P5-protocol train/test metrics, dataset summary, package versions, and creation timestamp.
- Artifact safety helpers: `validate_artifact_bundle()` rejects incomplete or incompatible bundles (wrong layout, missing metadata keys, schema-version mismatch, wrong declared model class or selection decision, feature-order drift against the current contract, or wrong target) and verifies the model object directly instead of trusting self-declared metadata: it must be a fitted D-016 `HistGradientBoostingClassifier` with the contract's fitted feature order, binary `[0, 1]` classes, and the D-016 hyperparameters (library defaults with the recorded seed). `load_artifact()` fails with regeneration instructions when the artifact is absent, and wraps deserialization failures (truncated or corrupt files) in the same clear guidance; `save_artifact()` writes atomically, requires the `.joblib` extension, and refuses repository locations outside `models/` because only `models/*.joblib` is git-ignored.
- The app-facing serving path (US-0503) also lives in `src/artifacts.py` and never imports Streamlit: `validate_input_values()` requires exactly the 21 features with integer-like values inside the P3 `VALUE_RANGES`, `input_to_dataframe()` builds one `uint8` row in exact training feature order, and `predict_risk_probability()` returns the raw positive-class `predict_proba` output with no custom threshold or calibration (both remain P8 scope).
- `python -m src.artifacts` generates the local artifact and runs the required load/predict smoke check (D-017). Real-data run: artifact saved to `models/diabetes_risk_model.joblib` (~263 KB, git-ignored), test metrics reproduce the D-016 selection evidence exactly (PR-AUC 0.423, ROC-AUC 0.827, recall 0.157, precision 0.563, F1 0.246), and the example-case probability was 0.0254.
- `app/streamlit_app.py` implements the local MVP (US-0501, US-0502) as a thin UI over the serving contract: a cached one-time artifact load (`st.cache_resource`; the app never trains or retrains), a single-case form covering all 21 features grouped as yes/no checkboxes, human-readable ordinal selectboxes that hide internal codes, an exact-age input (18-120) mapped automatically to the matching BRFSS age interval, and range-bounded numeric inputs, an educational risk percentage from `predict_proba`, a medical disclaimer visible alongside every prediction, and clear errors plus regeneration instructions when the artifact is missing or invalid.
- Added `tests/test_artifacts.py` (50 tests) and `tests/test_app.py` (10 tests): bundle contents and required metadata, save/load round-trip, exact feature-order preservation, rejection of twelve incompatible-bundle mutations plus direct model-object checks (impostor estimators with self-consistent metadata, unfitted models, drifted fitted feature order, non-binary classes, tuned hyperparameters), corrupt-file load errors that keep the regeneration command, refusal of unignored repository save locations, probability validity in `[0, 1]`, input validation (missing, unexpected, non-integer, non-numeric, and out-of-range values), exact-age-to-BRFSS-interval conversion and validation, a fit-spy check that training uses exactly the train rows, a poisoned-calibration check that the calibration split is never consumed, source guards against reloading/re-splitting, git-ignore verification for the default artifact path, app import safety (importing the app loads no artifact and trains nothing), a durable local-only artifact-loading guard (no remote sources or deployment secrets; deliberately phrased so legitimate P7 deployment files will not break it), and headless render/predict workflow checks with Streamlit's built-in `AppTest` harness (form renders without exposing internal dropdown codes, submission shows a valid percentage with the disclaimer, and missing or corrupt artifacts show a clear error). All artifact writes go to pytest temporary directories; a real-data integration test regenerates and verifies the artifact end to end and is skipped when the raw CSV is absent.
- A post-implementation review hardened the artifact contract: the validator originally trusted the self-declared `model_class`/`selection_decision` metadata, deserialization errors from corrupt files escaped the app's error handling, a phase-temporal "no deployment files" test would have blocked legitimate P7 work, and the `.joblib` extension check overclaimed git-ignore coverage for paths outside `models/`. All four findings were fixed and covered by the tests above.
- Verified `python -m pytest tests -v -p no:cacheprovider`: 115 passed (27 P3 + 13 P4 + 15 P5 + 60 P6). Verified `python -m compileall src app tests`: OK. The app imports in ~2 s without retraining, and `streamlit run app/streamlit_app.py` serves locally (HTTP 200).
- Scope held: no calibration usage, threshold tuning, SHAP, fairness analysis, batch prediction, scenario exploration, public deployment, or artifact distribution work. `data/processed/` stays empty and the generated artifact remains git-ignored and unstaged.
- Marked US-0501, US-0502, US-0503, and US-0504 Done in the backlog; set roadmap P6 to Done; updated the README with the current status and the artifact/app commands.

### Decisions Added

- None. Serialization followed the already accepted D-010 (joblib), D-016 (selected model), and D-017 (timing plus load/predict check); the single-bundle artifact layout is an implementation detail within D-010. D-013 (artifact distribution for deployment) intentionally remains Pending for P7.

### Follow-Up

- Refine P7 (Epic E7, MVP Documentation and Deployment) before implementation: complete run instructions (US-0701), resolve D-013 before the first public deploy (US-0702), and deploy the MVP to Streamlit Community Cloud (US-0703).
- Keep the calibration split untouched until P8; the selected model's low default-threshold recall and the uncalibrated probabilities remain documented P8 concerns.
- Evaluate `skops` as a safer serialization option before final portfolio packaging, per D-010.

## Iteration 7: MVP Documentation and Deployment

**Date:** 2026-07-10 to 2026-07-11

**Status:** Completed

**Goal:** Turn the completed P6 local MVP into a reviewer-reproducible and publicly accessible deployment by validating the run documentation, resolving D-013 from evidence, implementing only the selected artifact-distribution policy, and verifying the public app without expanding the modeling scope.

### Planned Scope

Execute P7 as small, verifiable increments, targeting 1-2 days per iteration where practical, in this order:

1. Complete and validate reproducible documentation for Python 3.12 environment setup, exact CSV acquisition, artifact generation, the full test suite, local app launch, and incompatible-artifact recovery.
2. Evaluate the controlled Git exception, GitHub Release asset, and build-time generation alternatives against the D-013 criteria, incorporating the P6 evidence that the current single-bundle artifact is approximately 263 KB.
3. Resolve D-013 with the selected distribution policy and rationale before any public deployment work depends on that choice.
4. Implement and verify only the policy selected by D-013, preserving the known-artifact contract and the separation between offline/build-time artifact generation and Streamlit runtime serving.
5. Prepare Streamlit Community Cloud from GitHub by verifying Linux and Python 3.12 compatibility, reviewing the existing `requirements.txt`, selecting the correct interpreter, and configuring the repository, branch, and entry point.
6. Deploy the MVP and review dependency-installation, startup, artifact-resolution, validation, and load logs.
7. Define and commit exact-input reference profiles with locally verified expected display outputs and a documented comparison tolerance, then reuse them in public smoke tests covering all 21 features, exact-age-to-BRFSS conversion, missing/invalid-artifact errors, disclaimer visibility, and absence of published CSV, secrets, or persisted user data. The manually observed P6 outputs near 0.3%, 60%, 70%, and 79.9% are candidate coverage targets, not established fixtures until P7 records and verifies them.
8. Add the verified public URL and final instructions to the README, record the validation evidence, and close P7 only when all acceptance criteria are satisfied.

Use four suggested 1-2 day slices while preserving that sequence: reproducible documentation and clean-clone validation (step 1); alternative comparison and D-013 (steps 2-3); selected-policy implementation and deployment preparation/launch (steps 4-6); and public verification plus documentation/closure (steps 7-8).

Throughout P7, keep the calibration split untouched. Do not add calibration, threshold selection or modification, experimental retraining, new-model comparison, SHAP, fairness analysis, batch prediction, scenario exploration, authentication, user-input persistence, or user analytics. P8 remains responsible for calibration and threshold analysis; P9 remains responsible for SHAP.

### Planning Completed

- Refined US-0701, US-0702, and US-0703 into Ready stories with verifiable acceptance criteria and detailed candidate tasks based on P6 evidence.
- Defined the clean-clone reproducibility check, formal D-013 comparison gate, selected-policy-only implementation rule, Streamlit Community Cloud preparation, and public smoke-test contract.
- Moved roadmap P7 from Planned to Ready while leaving P8, P9, and later phases at high level.
- Added explicit P7 scope guardrails and preserved the calibration split for P8.
- Corrected only planning ambiguities about the timing of D-013 and the ownership of calibration and threshold analysis; no deployment implementation was performed.

### Completed (2026-07-11)

- US-0701: rewrote the README into a validated "Run It Locally" sequence -- Python 3.12 venv creation/activation with PowerShell and macOS/Linux commands, `python -m pip install -r requirements.txt`, the exact CSV path with a link to `data/README.md`, `python -m src.artifacts`, `python -m pytest tests -v -p no:cacheprovider`, and `python -m streamlit run app/streamlit_app.py`, all through the same interpreter, with an explicit explanation of why mixing the global Python with `.venv` breaks artifact loading, plus a troubleshooting section that maps recognizable load/validation failures to the rebuild-and-regenerate fix instead of validation bypasses.
- Clean-environment validation: on an isolated copy of the repository with a fresh python.org CPython 3.12.1 venv (different installation from the development 3.12.7 base), the pinned install resolved with `pip check` clean; the committed-artifact journey worked without the CSV (artifact loads, smoke probability identical, 143 tests passed with exactly the 3 documented raw-data skips); after placing the CSV per the instructions, `python -m src.artifacts` reproduced the D-016 metrics and 0.0254 smoke probability, the full suite passed 146/146, and the app served HTTP 200 headlessly. The later Community Cloud deployment exercised the pushed repository content in the target environment.
- US-0702: performed the formal D-013 comparison against current official documentation (Streamlit Community Cloud dependencies/file-organization/deploy pages, scikit-learn model persistence, GitHub file-size limits; consulted 2026-07-11) and accepted the controlled Git exception; the full criteria table lives in `docs/decisions.md`. Key evidence: Community Cloud copies repository files and offers no pre-runtime build or download hook; a Release asset would force an in-app runtime download that the local-only loading policy forbids; build-time generation is impossible on the platform and would need the git-ignored CSV.
- D-013 implementation: `.gitignore` now excepts exactly `models/diabetes_risk_model.joblib` (temporary and alternative artifacts stay ignored); the official artifact was regenerated with Python 3.12 and the pinned versions and fully verified (metadata identity and package versions, feature order in metadata and the fitted model, classes `[0, 1]`, D-016 hyperparameters, reproduced selection metrics PR-AUC 0.423065 / ROC-AUC 0.826955, smoke probability 0.025419); docstrings, comments, and tests that assumed the artifact is always git-ignored were updated, including a new test pinning the exception to exactly the official filename. The artifact was committed in `facfb24` and delivered successfully to the public deployment.
- Reference profiles (US-0703 prerequisite): `tests/reference_profiles.py` records four synthetic profiles with exact 21-feature inputs, exact UI ages at BRFSS group boundaries (24 -> 1, 80 -> 13, 70 -> 11, 65 -> 10), official-artifact probabilities, and expected displays 0.3%, 60.0%, 70.0%, and 79.9% (the P6 manual observations guided the targets; their inputs were reconstructed, recorded, and verified rather than assumed). A documented +/-0.0002 probability tolerance stays below the 0.00025 display-rounding margin so tolerance-accepted values always render the recorded display. `tests/test_reference_profiles.py` adds 31 tests: contract checks, exact-age conversion, recomputation against the official artifact, an anti-tampering check that the artifact remains the untampered D-016 configuration reproducing its selection evidence, and end-to-end headless AppTest runs that drive all 21 form widgets per profile and assert the exact displayed percentage. `python -m tests.reference_profiles` prints the manual checklist for the public smoke tests.
- Post-review hardening: the official D-013 artifact is now a mandatory test dependency rather than a conditional skip, so omitting the tracked `.joblib` makes the deployment-reference suite fail with explicit restore/regeneration guidance. `validate_artifact_bundle()` treats package provenance as a real contract: it requires the exact metadata layout, Python 3.12 at major/minor level, exact NumPy/pandas/scikit-learn/joblib pins, and a matching runtime; tests also pin those constants back to `requirements.txt` and reject malformed metadata, false package/Python versions, and incompatible runtimes.
- Streamlit Community Cloud readiness: entry point `app/streamlit_app.py` tracked with matching case; all paths `pathlib`-based (forward slashes; absolute paths derived from module location, compatible with the platform's repository-root `streamlit run`); `requirements.txt` is the only dependency file so the platform will select it; Python 3.12 is the platform default; no secrets and no `.streamlit/` directory exist; no `packages.txt` needed. The heavier EDA/notebook pins slow the one-time cloud install but keep a single reproducible environment (documented trade-off; no second requirements file).
- Verification battery: `python -m pytest tests -v -p no:cacheprovider` passed 153/153 (the previous 146 plus 7 package-provenance/runtime hardening cases) with a `--basetemp` outside the repository; the targeted negative check also proved that an absent official artifact now fails instead of skipping. `python -m compileall src app tests` OK; `git diff --check` clean; local Streamlit serves HTTP 200 from the main repository; no CSV or secrets were added.
- Scope guardrails held: no calibration or threshold work (the calibration split remains untouched -- enforced by existing tests), no experimental retraining or new model comparisons (the D-016 regeneration is required by the D-013 policy and reproduces the selection evidence exactly), no SHAP, fairness analysis, batch prediction, scenario exploration, authentication, input persistence, or analytics.

### Public Deployment Closure (2026-07-11)

- Deployed the GitHub-backed `main` branch to Streamlit Community Cloud with entry point `app/streamlit_app.py` and Python 3.12: [brfss-diabetes-risk-estimator.streamlit.app](https://brfss-diabetes-risk-estimator.streamlit.app/).
- Confirmed successful public startup and that the D-013 repository artifact is found, validated, and loaded without training, model download, or raw-CSV access.
- Verified the public 21-feature form, visible educational/medical disclaimer, and all four reference profiles. Ages 24, 65, 70, and 80 map to groups 18-24, 65-69, 70-74, and 80 or older; their displayed probabilities match the recorded expectations exactly at 0.3%, 79.9%, 70.0%, and 60.0% respectively.
- Accepted deployment-equivalent headless tests for missing/corrupt-artifact behavior because they exercise the same app loading and error-rendering path without deliberately breaking the healthy public deployment. The public valid-artifact path is verified separately.
- Confirmed that the repository publishes no raw CSV or secrets and that project code does not log or persist submitted inputs.
- Added the public URL and final deployment description to the README. US-0701, US-0702, and US-0703 are Done; roadmap P7 is Done, and P8 is Planned pending detailed refinement.

### Decisions Added

- During the refinement (2026-07-10): none; D-013 stayed Pending until the planned comparison could run.
- During implementation (2026-07-11): D-013 resolved and Accepted -- the official model artifact is distributed as a controlled Git exception (`models/diabetes_risk_model.joblib` version-controlled; all other artifacts under `models/` remain ignored), selected from the formal three-alternative comparison recorded in `docs/decisions.md`.

### Follow-Up

- Refine P8 (probability calibration and threshold analysis) into a concrete iteration goal, user stories, acceptance criteria, and leakage-safe evaluation protocol before implementation.
- Preserve the existing train/calibration/test contract: fit the selected model on train, reserve calibration work for the calibration split, and keep test for final evaluation.
- Leave SHAP for P9 so explanations target the final serving probability contract selected in P8.

## Iteration 8: P8 Probability Calibration and Threshold Analysis

**Start date:** 2026-07-11

**Completion date:** 2026-07-13

**Status:** Completed -- implementation commit `5798a0e` is on `main`, and the public schema-version-2 deployment passed the P8 smoke verification

**Goal:** Make the public app's risk percentages more honest by evaluating calibration of the frozen D-016 model's probabilities through a leakage-safe protocol on the untouched calibration split, documenting decision-threshold trade-offs on the selected contract's out-of-fold evidence, and only then integrating the D-018-selected probability contract (sigmoid, isotonic, or a justified `none`) into a schema-version-2 artifact and the deployed app -- without reopening model selection or introducing a decision layer by default.

### Completed Scope

P8 was executed as four ordered 1-2 day increments:

1. **Increment 1 -- refinement, uncalibrated baseline, and metric/test infrastructure.** Confirm this refinement; score the calibration split with the frozen train-only D-016 model and record the uncalibrated baseline (reliability diagram, Brier score, log loss, with ROC-AUC/PR-AUC as ranking context); build the reusable calibration-metric helpers and the stratified fold/out-of-fold assembly with a fixed seed; land the fit-spy and poisoned-split guards from the start so leakage safety is enforced before any calibrator exists.
2. **Increment 2 -- cross-fitting comparison and D-018 resolution.** Stratified five-fold cross-fitting within the calibration split on the frozen model's scores; sigmoid and isotonic calibrators fitted per fold and combined into complete out-of-fold predictions; out-of-fold reliability diagrams and the paired-bootstrap metric comparison against the uncalibrated baseline; select the method (sigmoid, isotonic, or none) strictly per the D-018 pre-declared operationalized criteria, using no test data, and resolve D-018 with the recorded evidence.
3. **Increment 3 -- threshold analysis, D-019, and the official P8 test evaluation.** In this order: precision-recall curves and threshold tables from the selected contract's out-of-fold probabilities; freeze the threshold scenarios and resolve D-019; refit the selected calibrator (when one was selected) on the full calibration split; run the official P8 test evaluation of the frozen contract; record the results without revisiting D-018 or D-019. After D-018 and D-019 are frozen there is exactly one official P8 evaluation on test; later runs (deterministic artifact regeneration, regression tests, full-suite repeats, deployment verification) may only repeat it as a deterministic regression check and never modify decisions, methods, or thresholds.
4. **Increment 4 -- schema-version-2 artifact, app integration, and public redeploy.** Bundle the frozen base model with the D-018-selected contract (`calibration_method` plus the fitted final calibrator only when the method requires one) and calibration metadata (method, protocol, metrics, versions) under artifact schema version 2 with validation as strict as version 1; update the app wording to match the served contract (dropping the uncalibrated notice only when a calibrator ships); regenerate the four deployment reference profiles when the served probabilities change or re-verify them unchanged; run the full local and headless battery; regenerate and ship the official version-2 artifact per D-013; redeploy and rerun the public smoke tests.

The protocol details live in `docs/ml-analysis-plan.md` ("Calibration and Threshold Analysis Plan"); the refined stories and implementation tasks live under Epic E6 in `docs/backlog.md` (US-0601, US-0606, US-0607).

### Guardrails

- Do not reopen the D-016 model selection and do not compare new models; only the already selected frozen model is calibrated.
- Never use test rows to select the calibration method or the threshold scenarios; test participates in no P8 decision. One official P8 test evaluation is recorded after method and scenarios are frozen (test's last prior use was the P5 selection protocol); later runs may only repeat it as a deterministic regression check, and nothing changes after its results are observed.
- Never train the base model on calibration rows; never fit a calibrator on train or test rows; the cross-fitting operates on the frozen model's output scores only.
- No calibration or training inside Streamlit; the app only loads and serves the validated artifact.
- Keep SHAP out of P8 (P9 owns explainability); keep fairness analysis, batch prediction, and scenario exploration in their later phases.
- No medical recommendations, no diagnostic or high/low-risk labels without an explicit D-019 resolution, and no causal interpretation of calibration or threshold results.
- The implementation keeps the P8 changes limited to offline calibration analysis, the artifact/serving contract, tests, and documentation; no later-phase feature is pulled forward.
- CI remains only a quality-track candidate and is not implemented in P8.

### Implementation and Closure (2026-07-12 to 2026-07-13)

- **Increment 1 completed.** `src/calibration.py` implements the P8 infrastructure on top of the P3/P5 contracts: `CalibrationData` (train and calibration rows only; the test split is structurally absent, mirroring how `TrainTestData` excluded calibration), the frozen train-only D-016 base model, per-row Brier/log-loss helpers that reproduce the pinned scikit-learn aggregates (including the machine-epsilon clipping that keeps isotonic's exact 0/1 outputs finite), reliability-diagram and probability-histogram tables, deterministic stratified five-fold assignment with both-classes validation, and out-of-fold assembly that rejects duplicate, missing, out-of-range, or invalid predictions. The uncalibrated baseline was recorded on the calibration split before any calibrator existed: Brier 0.096940, log loss 0.313828, ROC-AUC 0.827060, PR-AUC 0.432421.
- **Increment 2 completed.** Calibrators use the public scikit-learn 1.7.1 API selected after consulting the pinned version's source and documentation: `CalibratedClassifierCV(FrozenEstimator(base_model), method=..., ensemble=False)`, the documented replacement for the deprecated `cv="prefit"`. With a frozen estimator, `fit(X, y)` fits exactly one sigmoid/isotonic calibrator on the frozen model's `decision_function` scores for all provided rows and never refits the base model, so cross-fitting and final serving share one score representation by construction. Stratified five-fold cross-fitting produced complete out-of-fold probabilities per method; the paired bootstrap (10,000 fixed-seed resamples, batched to bound memory without changing the resample stream) and the pre-declared selection rules ran exactly as operationalized. Result: neither method is adoptable (Brier deltas against the baseline: sigmoid +2.44e-05, CI [-0.56e-05, +5.46e-05]; isotonic +6.01e-05, CI [-11.15e-05, +23.18e-05]; both log-loss deltas strictly positive; isotonic also fails the 0.005 PR-AUC guard with a 0.00736 drop). D-018 resolved as Accepted: `calibration_method = none`; the frozen D-016 model's own probabilities are retained. Evidence: `docs/decisions.md`, `docs/p8-calibration/report.md`, and 67 focused tests in `tests/test_calibration.py` covering leakage, fit scope, OOF integrity, reproducibility, bootstrap rules, method selection, thresholds, and test isolation.
- **Increment 3 completed.** The selected `none` contract's calibration probabilities produced the full precision-recall curve and 0.01--0.99 threshold table. D-019 froze four documentation scenarios before test -- 0.50, 0.25 (maximum F1), 0.29 (recall floor 0.50), and 0.15 (recall floor 0.75) -- while retaining a probability-only app with no decision labels. The official P8 test evaluation then recorded Brier 0.097381, log loss 0.314394, ROC-AUC 0.826955, and PR-AUC 0.423065 without revisiting D-018 or D-019. Exact evidence and figures live in `docs/p8-calibration/report.md`.
- **Increment 4 completed.** `src/artifacts.py` now creates and strictly validates schema-version-2 bundles with a conditional calibrator, fixed P8 protocol/decision metadata, OOF metrics, official test metrics, frozen scenarios, and package provenance. The accepted artifact has `calibration_method = none` and `calibrator = None`; Streamlit explains that outcome while preserving the medical disclaimer and probability-only behavior. The official D-013 artifact was regenerated, and all four reference profiles re-verified unchanged at 0.3%, 60.0%, 70.0%, and 79.9%. Contract tests cover `none`, `sigmoid`, and `isotonic`, schema-v1 rejection, missing/inconsistent/unfitted calibrators, metadata integrity, serving, and headless rendering.
- **Local validation passed 2026-07-13.** `python -m src.calibration` reproduced D-018, D-019, all evidence tables/figures, and the official P8 test metrics. The pinned `.venv` then passed the complete test battery: 229 passed, including real-data integrations, artifact schema outcomes, reference profiles, and Streamlit headless tests. `python -m compileall -q src app tests` and `git diff --check` also passed. The only pytest warning is joblib's sandbox-specific inability to detect physical CPU cores; it falls back to logical cores and does not affect results.
- **Public deployment closure passed 2026-07-13.** Implementation commit `5798a0e` was pushed to `main` and the Streamlit app was rebooted/redeployed. Public verification confirmed startup and schema-version-2 artifact loading, the complete form, contract-consistent uncalibrated-probability wording, the medical disclaimer, and probability-only behavior with no high/low-risk labels or decision threshold. The four reference profiles matched exactly: 0.3% for age 24 (18-24), 60.0% for age 80 (80 or older), 70.0% for age 70 (70-74), and 79.9% for age 65 (65-69). US-0607 and roadmap P8 are Done.

### Delivered

- Reusable, offline P8 calibration code (fold assembly, calibrator fitting, out-of-fold prediction, reliability/Brier/log-loss reporting) built on the existing P3 split contract.
- Recorded uncalibrated baseline, paired-bootstrap out-of-fold method comparison, threshold trade-off tables, and the official P8 test evaluation.
- D-018 and D-019 resolved from evidence, in that order and both before the official test evaluation. Both are now Accepted with their evidence recorded.
- A schema-version-2 official artifact serving the D-018-selected probability contract, reference profiles re-verified to match it, and app wording consistent with the served contract. The artifact is deployed and the public smoke verification passed.

### Validation Coverage

- Fit-spy checks: the base model consumes exactly the train rows; each per-fold calibrator fits on exactly its four assigned training folds and never on its held-out fold; the final calibrator (when one is selected) fits on exactly the full calibration split; no calibrator ever fits on train or test rows.
- Test isolation: test rows never reach the comparison, selection, or threshold-analysis code paths (poisoning the test split leaves them unchanged); test enters only the official P8 evaluation and its deterministic regression repeats.
- Fixed-seed reproducibility of fold assignment, out-of-fold probabilities, bootstrap intervals, and the selection outcome.
- Out-of-fold integrity: every calibration row gets exactly one out-of-fold prediction per method from a calibrator not fitted on its fold.
- Schema-version-2 contract tests: round-trip per `calibration_method` outcome, a calibrator present if and only if the method requires one, rejection of version-1 bundles, tampered/unfitted calibrators, inconsistent method/calibrator combinations, missing calibration metadata, provenance pins carried over.
- Reference profiles regenerated (when outputs changed) or re-verified (when not) against the version-2 artifact and through the headless app.
- Source guards forbidding calibrator fitting in the serving path.

### Definition of Done

- All US-0601, US-0606, and US-0607 acceptance criteria are satisfied with recorded evidence.
- D-018 and D-019 are resolved (Accepted) from the out-of-fold evidence, in that order and before the official P8 test evaluation, whose results are recorded without changing either decision; any later test run only repeats that evaluation as a deterministic regression check.
- The full pytest battery passes locally, including the reference-profile suite; the public app serves the D-018-selected probability contract with wording consistent with it and the medical disclaimer intact, and the four reference profiles (regenerated when the served probabilities changed, re-verified otherwise) match their recorded displays publicly.
- The calibration split protocol is documented well enough that a reviewer can verify no leakage occurred, and the roadmap/backlog/iteration log reflect P8 closure.

### Decisions Added

- D-018 (Accepted 2026-07-12): select `none`; neither post-hoc method passed the pre-declared OOF Brier adoption rule.
- D-019 (Accepted 2026-07-12): freeze four documentation scenarios and retain a probability-only app with no served threshold or high/low-risk labels.
- Artifact schema version 2 is documented as an implementation detail within the accepted D-010/D-013 artifact policies, following the P6 precedent that the single-bundle layout was an implementation detail within D-010; no separate schema decision is created.

### Follow-Up

- Refine P9 (SHAP explainability) before implementation so global and local explanations target the final serving probability contract selected in P8.
- Evaluate the CI quality-track candidate (automated pytest on pushes) during the next implementation increment.
- Keep evaluating `skops` as a safer serialization option before final portfolio packaging, per D-010.
