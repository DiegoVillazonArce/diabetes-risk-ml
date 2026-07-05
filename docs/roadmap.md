# Roadmap

## Methodology

The project follows a lightweight Scrum-inspired workflow with rolling-wave planning. The roadmap defines the high-level direction, while the backlog is refined progressively as data, modeling results, implementation constraints, and deployment constraints become clearer.

Near-term work is planned in detail. Later phases remain intentionally broader until the project reaches them.

## Planning Rules

- Work is organized into short iterations of 1-2 days when possible.
- Each iteration should produce a verifiable increment: code, analysis, documentation, model artifact, test, or app feature.
- The backlog is the source of truth for user stories and tasks.
- The roadmap is updated at phase boundaries, not after every small task.
- Technical decisions that affect reproducibility, evaluation, or product behavior are recorded in the decision log.

## MVP Boundary

The MVP should prove the complete ML product lifecycle without trying to include every advanced feature from the long-term project vision.

MVP includes:

- Data understanding and quality analysis.
- Reproducible preprocessing and splits.
- Documented data acquisition instructions.
- Explicit dependency strategy for local reproduction.
- Baseline and candidate model training.
- Correct imbalanced classification evaluation.
- Initial model artifact serialization.
- Streamlit individual prediction page.
- Minimal tests for data validation and model artifact behavior.
- Basic model comparison and limitations documentation.
- Professional README.

Advanced post-MVP features include calibration depth, SHAP polish, model scenario exploration, batch prediction, fairness audit, additional app pages, and final portfolio packaging.

## Phases

| Phase | Name | Goal | MVP | Status |
|---|---|---|---|---|
| P0 | Planning and Analysis | Define scope, workflow, documentation, and initial ML assumptions. | Yes | Done |
| P1 | Project Setup and Data Governance | Create repo structure, dependency strategy, data acquisition instructions, and data handling policy. | Yes | Planned |
| P2 | Data Understanding and EDA | Analyze schema, target distribution, missing values, duplicates, and feature ranges. | Yes | Planned |
| P3 | Data Preparation and Splits | Build reproducible cleaning, validation, and train/calibration/test split logic. | Yes | Planned |
| P4 | Baseline Modeling | Train Dummy and simple interpretable models to establish reference performance. | Yes | Planned |
| P5 | Model Comparison and Selection | Compare candidate models using appropriate imbalanced classification metrics. | Yes | Planned |
| P6 | Streamlit MVP | Serve a trained model for individual prediction with clear disclaimers. | Yes | Planned |
| P7 | MVP Documentation and Deployment | Prepare README, run instructions, tests, artifact notes, and first public deployment. | Yes | Planned |
| P8 | Probability Calibration and Threshold Analysis | Improve probability honesty and explain decision threshold trade-offs. | No | Future |
| P9 | Explainability with SHAP | Add global and local model explanations with non-causal framing. | No | Future |
| P10 | Model Scenario Explorer | Add carefully framed what-if simulations for selected modifiable inputs. | No | Future |
| P11 | Batch Prediction Workflow | Add CSV upload, schema validation, templates, and downloadable predictions. | No | Future |
| P12 | Fairness Audit | Evaluate model behavior across demographic and socioeconomic subgroups. | No | Future |
| P13 | Product Polish and Portfolio Packaging | Improve UX, architecture page, demo assets, and final CV narrative. | No | Future |

## Quality Tracks

These tracks should be refined as the project moves through the roadmap:

- **Dependencies:** start with a pinned `requirements.txt` for the MVP; evaluate `uv` or another lockfile-based workflow if reproducibility needs increase.
- **Data acquisition:** document manual Kaggle download instructions first; consider an optional Kaggle API script later because automated download requires user credentials.
- **Model artifacts:** use a clear serialization policy before deployment; `joblib` is the practical MVP default, while safer formats such as `skops` may be evaluated before final packaging.
- **Testing:** begin with pytest coverage for data validation, split reproducibility, pipeline prediction behavior, and artifact loading; add app smoke tests after the Streamlit MVP exists.

## Definition of Done

A phase or story is done when:

- The intended artifact exists and is committed-ready.
- Acceptance criteria are satisfied.
- Relevant decisions or assumptions are documented.
- The work can be reproduced or verified locally.
- The next iteration has clear candidate tasks.

## Roadmap Review Cadence

The roadmap should be reviewed:

- At the end of each major phase.
- When data findings change the modeling strategy.
- When deployment constraints change product scope.
- Before moving advanced features into MVP scope.
