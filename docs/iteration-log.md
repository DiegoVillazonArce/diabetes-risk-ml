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

## Next Iteration Planning: P3 Data Preparation and Splitting

**Date:** 2026-07-06

**Status:** Planned

**Goal:** Convert the P2 findings into reusable data loading, validation, downcasting, and stratified split logic without introducing modeling work.

### Planning Notes

- D-014 is resolved: exact duplicate rows will be kept for MVP data preparation.
- The accepted duplicate policy keeps the full 253,680-row dataset and the observed ~13.9% positive prevalence, which the stratified 70/10/20 train/calibration/test split must preserve.
- P3 should define the data contract in reusable code, validate the raw schema and values, apply safe lossless `uint8` downcasting, and add focused pytest coverage.
- P3 should decide whether split outputs belong in `data/processed/` now or whether the split function should return in-memory data for the next modeling phase.
- P3 should not add balancing, SMOTE, model training, feature engineering, app work, calibration, or explainability.
