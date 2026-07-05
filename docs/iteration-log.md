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
