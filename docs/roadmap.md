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

## Roadmap Status Values

- **Done:** completed and validated.
- **Ready:** next phase has a clear iteration goal, tasks, and acceptance criteria.
- **Planned:** phase is part of the roadmap but not yet refined for implementation.
- **Future:** post-MVP or later improvement.

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
| P1 | Project Setup and Data Governance | Create repo structure, dependency strategy, data acquisition instructions, and data handling policy. | Yes | Done |
| P2 | Data Understanding and EDA | Analyze schema, target distribution, missing values, duplicates, and feature ranges. | Yes | Done |
| P3 | Data Preparation and Splits | Build reproducible cleaning, validation, and train/calibration/test split logic. | Yes | Done |
| P4 | Baseline Modeling | Train Dummy and simple interpretable models to establish reference performance. | Yes | Done |
| P5 | Model Comparison and Selection | Compare candidate models using appropriate imbalanced classification metrics. | Yes | Done |
| P6 | Streamlit MVP | Serve a trained model for individual prediction with clear disclaimers. | Yes | Done |
| P7 | MVP Documentation and Deployment | Prepare README, run instructions, tests, artifact notes, and first public deployment. | Yes | Done |
| P8 | Probability Calibration and Threshold Analysis | Improve probability honesty and explain decision threshold trade-offs. | No | Done |
| P9 | Explainability with SHAP | Explain the final P8 probability contract globally and locally with SHAP, preserving its predictions and providing both an accessible Streamlit explanation and reproducible technical evidence, with non-causal, non-medical wording. | No | Done |
| P10 | Model Scenario Explorer | Compare the original model estimate with one controlled hypothetical scenario over an explicitly approved feature whitelist, using non-causal, non-medical language and preserving the P8/P9 contracts. | No | Done |
| P11 | Batch Prediction Workflow | Add a privacy-safe in-memory CSV template, deterministic file/row validation, probability-only batch scoring, and downloadable results without changing the P8-P10 contracts. | No | Done |
| P12 | Fairness Audit | Audit the frozen P8 probability contract across predeclared sex, age, income, and supported intersectional cohorts with uncertainty-aware probability, ranking, calibration, and frozen-threshold metrics; communicate limitations without changing the model or claiming fairness. | No | Done |
| P13 | Product Polish and Portfolio Packaging | Polish the final product experience, explain the offline-to-serving architecture at accessible and technical levels, package privacy-safe demo/CV evidence, and add a clean-clone CI quality signal without changing the validated ML contracts. | No | Done |

P11 completed on 2026-07-16 and P12 on 2026-07-17. P13 completed on 2026-07-20 after D-032 through D-035 were Accepted from audit/spike evidence, implementation commit `7f642af` and cross-platform CI fix `08170c3` were pushed, the least-privilege clean-clone GitHub Actions workflow passed remotely, and the redeployed Streamlit app passed the mandatory public individual/explanation/scenario, valid-plus-mixed batch/download, navigation/About, link, disclaimer, and responsive-layout verification. US-0901 through US-0904 are Done. The completed package includes the three-section product navigation, accessible About view, technical architecture, synthetic-only portfolio assets, tiered narratives, README overview, CI badge, and documented `joblib` trust boundary. Both official artifact hashes and all four reference displays remain unchanged, and neither artifact was regenerated. P0-P13 now form the completed portfolio baseline; additional work requires explicit future planning rather than reopening P13.

## Quality Tracks

These tracks should be refined as the project moves through the roadmap:

- **Dependencies:** start with a pinned `requirements.txt` for the MVP; evaluate `uv` or another lockfile-based workflow if reproducibility needs increase.
- **Data acquisition:** document manual Kaggle download instructions first; consider an optional Kaggle API script later because automated download requires user credentials.
- **Model artifacts:** D-010's controlled `joblib` policy remains binding. P13 completed D-035 by retaining the reviewed repository artifact while documenting that SHA-256 tracking/runtime binding and post-load semantic validation do not make untrusted pickle deserialization safe. Any `skops` migration is a separately authorized hardening project rather than an implicit P13 regeneration.
- **Testing:** the completed P13 baseline passes 469 local tests; its clean-clone validation passes 441 tests with 28 explicit CSV-gated skips. Focused, clean-clone, complete local, visual, hash, remote-CI, and public smoke gates passed without adding a model experiment.
- **Continuous integration:** the D-034 bounded Python 3.12 clean-clone workflow passed remotely without private data or credentials; its status badge is published in README after that green run.

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
