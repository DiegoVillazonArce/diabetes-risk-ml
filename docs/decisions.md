# Decision Log

This file records project decisions that affect scope, data handling, modeling, evaluation, deployment, or public communication.

## Decisions

| ID | Date | Status | Decision | Rationale | Consequences |
|---|---|---|---|---|---|
| D-001 | 2026-07-04 | Accepted | Keep the original Spanish blueprint outside the repository. | The blueprint is useful as an internal guide but is not structured as public project documentation. | Public repo documentation will be concise, English-first, and portfolio-oriented. |
| D-002 | 2026-07-04 | Accepted | Use English for repository-facing documentation. | The project is intended for recruiters, portfolio reviewers, and technical interviewers. | Docs should be written in clear English, while private planning conversations can remain Spanish. |
| D-003 | 2026-07-04 | Accepted | Use a lightweight Scrum-inspired workflow with rolling-wave planning. | ML work often changes after data inspection and model results, so future work should be refined progressively. | Roadmap stays high-level; backlog is updated frequently with user stories, tasks, and acceptance criteria. |
| D-004 | 2026-07-04 | Accepted | Treat the project as educational, not medical. | The dataset is self-reported survey data and the model is not clinically validated. | Disclaimers must appear in documentation and app output. |
| D-005 | 2026-07-04 | Accepted | Use the original imbalanced binary BRFSS dataset with `Diabetes_binary` as target. | The current local CSV matches the intended binary dataset: 253,680 rows, 218,334 negative cases, and 35,346 positive cases. | Modeling can proceed with the binary target. The original class imbalance must be preserved for evaluation. |
| D-006 | 2026-07-05 | Accepted | Keep the raw CSV out of git for the MVP, while documenting the exact Kaggle source and CC0 license. | The selected dataset is CC0, so licensing is not the blocker; keeping it out of Git keeps the repository focused on code, docs, and lightweight deployable artifacts. | `dataset.csv` and raw/processed CSV outputs are ignored. Reproducers should download `diabetes_binary_health_indicators_BRFSS2015.csv` from Kaggle and place it at `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`. |
| D-007 | 2026-07-04 | Accepted | Train offline and serve serialized artifacts in Streamlit. | Streamlit should stay responsive and reproducible. | Training scripts generate artifacts; the app only loads and uses them. |
| D-008 | 2026-07-04 | Accepted | Start with a pinned `requirements.txt` for MVP dependency management. | This keeps the project easy for reviewers to run while still making package versions explicit. | A lockfile workflow such as `uv` can be evaluated later if reproducibility needs increase. |
| D-009 | 2026-07-04 | Accepted | Document manual Kaggle data acquisition before adding automation. | Kaggle API automation requires user credentials and can create friction for reviewers. | README or data docs must identify the dataset URL, required file, license, and expected local path; an optional download script can be added later. |
| D-010 | 2026-07-05 | Accepted | Use `joblib` for MVP model artifact serialization and evaluate `skops` before final packaging. | `joblib` is practical for scikit-learn pipelines and keeps the MVP simple, while `skops` may provide a safer artifact-loading story for a later hardening pass. | MVP artifacts will use `joblib` and should include local load/predict tests. This decision can be superseded if the project later adopts `skops`. |
| D-011 | 2026-07-04 | Accepted | Define focused pytest targets before implementation. | "Add tests" is too vague to guide development. | Initial tests should cover schema validation, feature ranges, stratified splits, pipeline predictions, and artifact loading. |
| D-012 | 2026-07-05 | Accepted | Use Python 3.12 with a hand-curated pinned `requirements.txt` as the first reproducible environment. | Python 3.12 is the locally installed default, has mature wheel support across the scientific Python stack, and is supported by Streamlit Community Cloud for the later deployment phase. Hand-curating top-level pins avoids a noisy `pip freeze` from the global environment. | `requirements.txt` pins numpy, pandas, scikit-learn, imbalanced-learn, joblib, matplotlib, seaborn, jupyterlab, streamlit, and pytest; `.python-version` records the interpreter version. The pins are validated when the first virtual environment is installed; any pin that fails to resolve should be adjusted individually, not unpinned. Unblocks US-0104. |
| D-013 | 2026-07-05 | Pending | Define how the trained model artifact reaches the deployed Streamlit app. | `models/*.joblib` is git-ignored, so deployment needs an explicit artifact distribution path (committed exception, release asset, or build step). | Required before the first public deployment in P7. |

## Status Values

- **Proposed:** under discussion.
- **Accepted:** active decision.
- **Rejected:** considered but not used.
- **Superseded:** replaced by a later decision.
- **Pending:** decision required before dependent work continues.
