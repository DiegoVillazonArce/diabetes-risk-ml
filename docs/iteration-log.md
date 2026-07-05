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
