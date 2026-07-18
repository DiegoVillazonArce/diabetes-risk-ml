# diabetes-risk-ml

Machine Learning app to estimate self-reported diabetes/prediabetes risk using BRFSS 2015 data, with model comparison, probability calibration, SHAP explainability, responsible offline model auditing, post-MVP extensions, and Streamlit deployment.

> This is an academic portfolio project. It is not a medical device and must not be used as diagnosis or medical advice.

## Current Status

Phases P0-P11 are complete, including public deployment, probability-quality work, SHAP explainability, the constrained model scenario explorer, and the privacy-safe batch prediction workflow. The frozen D-016 `HistGradientBoostingClassifier` remains the selected model; D-018 accepted `calibration_method = none`, so the schema-version-2 artifact serves the model's positive-class probability without a post-hoc calibrator; and D-019 retains a probability-only product with no decision threshold or high/low-risk label. P9 explains that exact probability without changing it, P10 compares one approved hypothetical input using model-sensitivity wording only, and P11 scores bounded CSV batches through the same probability contract. P12 implementation and aggregate evidence are complete locally under D-029 through D-031, but still await human review and a user-created commit; P12 therefore remains Ready rather than Done, US-0604 and US-0614 are Done, and US-0615 is in Review. The audit is descriptive and does not certify fairness. D-031 keeps it report-first, so the public Streamlit application remains functionally unchanged and P12 requires no deployment or public smoke test. P13 remains Future.

## Run It Locally

The supported Python version is **3.12** (see `.python-version`); dependencies are pinned in `requirements.txt` (decision D-012). All commands run from the project root.

### 1. Create and activate a Python 3.12 virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version    # must print Python 3.12.x
```

If the `py` launcher is not installed, create the environment with any Python 3.12 interpreter (`python -m venv .venv` after confirming `python --version` prints 3.12.x). If activation is blocked by the PowerShell execution policy, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, or use `.venv\Scripts\activate.bat` from `cmd`.

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version    # must print Python 3.12.x
```

**Use the `.venv` interpreter for every project command.** After activation, `python` resolves to the environment's interpreter, and the `python -m ...` forms below guarantee that training, tests, and the app all run on that same interpreter with the same pinned packages. Mixing the global Python with `.venv` is the main source of hard-to-diagnose failures here: scikit-learn does not support loading a model serialized under a different scikit-learn version, and a `streamlit` or `pytest` executable from another installation silently brings that installation's package versions instead of the pinned ones.

### 2. Install the pinned dependencies

```
python -m pip install -r requirements.txt
```

### 3. Download the dataset (only to reproduce training or the full test suite)

Download `diabetes_binary_health_indicators_BRFSS2015.csv` (CC0) from the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv) on Kaggle (a free account is required) and place it at exactly:

```
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
```

See [data/README.md](data/README.md) for the full acquisition instructions and data handling policy. The raw CSV is never committed. The Streamlit app does not need it (the app serves the committed model artifact), and the test suite skips its raw-data integration tests when the file is absent.

### 4. Generate the model artifact (optional: it ships with the repository)

The official trained artifact is version-controlled at `models/diabetes_risk_model.joblib` (decision D-013), so the app runs without any training step. To reproduce training — or to regenerate the artifact after an environment change — run:

```
python -m src.artifacts
```

This trains the selected D-016 model offline through the project's data contract, rebuilds the accepted P8 probability contract, records its calibration/threshold provenance under schema version 2, and overwrites `models/diabetes_risk_model.joblib` after a load/predict smoke check. With D-018's current `none` result, no calibrator is fitted or stored. Training and any conditional calibration only happen through offline project code (decision D-007): Streamlit loads the validated artifact and never trains, calibrates, downloads models, or reads the CSV.

### 5. Run the test suite

```
python -m pytest tests -v -p no:cacheprovider
```

With the raw CSV in place the whole suite runs, including the real-data integration tests; without it, those tests are skipped and the rest still run.

### 6. Launch the app

```
python -m streamlit run app/streamlit_app.py
```

The app provides an explicit `Individual prediction` / `Batch CSV prediction` selector. The individual workflow retains the 21-feature form, the P8 probability, P9 local explanation, and P10 one-field hypothetical explorer unchanged. The P11 batch workflow provides a code-generated CSV template and field guide, accepts a UTF-8 comma CSV of at most 2 MiB and 1,000 rows, scores only valid rows in one vectorized call, previews at most 25 rows, and downloads one deterministic result CSV. Uploaded bytes and batch results remain in active-session memory, are bound to both artifact and upload SHA-256, and are cleared after replacement, failure, reset, or artifact change; project code does not write or externally log them or put user content in a shared cache. Batch produces no SHAP, scenarios, thresholds, categories, diagnosis, recommendations, or population conclusions. The current D-018 outcome remains `none`, the D-019 probability-only wording and medical disclaimer remain visible, and Streamlit still never reads the project training CSV, trains, calibrates, or generates artifacts.

To reproduce the complete P9 technical evidence offline (the local CSV is required), run:

```
python -m src.explainability
```

The command deterministically rebuilds only aggregate global outputs and explanations for the four already-public synthetic reference profiles under `docs/p9-explainability/`. It also refreshes the separately versioned `models/shap_background_v1.json`, which contains 256 train-derived aggregate centroids and no target, split indices, identifiers, or exact real row. It never rewrites the official P8 model artifact.

## Troubleshooting: artifact and environment incompatibility

The artifact bundle records the exact package versions that produced it, and `src/artifacts.py` validates every load. Failures are designed to be recognizable:

- **`FileNotFoundError ... Generate it from the project root with 'python -m src.artifacts'`** — the artifact file is missing (for example, deleted locally). Restore it from git, or regenerate it (requires the CSV).
- **`ValueError: Could not deserialize the model artifact ...`** — the file is truncated or corrupt, or it was written under an incompatible Python/scikit-learn/joblib combination and cannot even be unpickled.
- **`ValueError` naming a schema version, model class, feature order, classes, hyperparameters, package version, or runtime version** — the file deserialized but either the artifact provenance or the active environment does not match the current project contract.
- **`InconsistentVersionWarning` from scikit-learn** — the artifact was created by a different scikit-learn version; scikit-learn does not support cross-version loading, so treat the artifact as untrusted output.

In every case the fix is the same, deliberately: do not bypass validation and do not load artifacts from unknown sources. Recreate the pinned environment and regenerate the trusted artifact with the same interpreter:

```powershell
python --version                              # 1. confirm Python 3.12.x from .venv
python -m pip install -r requirements.txt     # 2. restore the pinned versions
python -m pip show scikit-learn joblib        # 3. verify 1.7.1 / 1.5.1, as pinned
python -m src.artifacts                       # 4. retrain and overwrite the artifact
```

The versions recorded inside a loadable artifact can be inspected with:

```
python -c "from src.artifacts import load_artifact; print(load_artifact()['metadata']['package_versions'])"
```

## Deployment

The app is deployed on Streamlit Community Cloud at [https://brfss-diabetes-risk-estimator.streamlit.app/](https://brfss-diabetes-risk-estimator.streamlit.app/). It uses branch `main`, entry point `app/streamlit_app.py`, Python 3.12, the pinned `requirements.txt`, and the D-013 version-controlled artifact. P9 was introduced by commit `25c4ed4`, P10 by `fb50ed9`, and P11 batch prediction by `246d5ff`. Public P11 verification on 2026-07-16 confirmed the deployed workflow with a small valid template-derived CSV and a mixed-validity CSV, including summary, preview, blank invalid probability, validation details, and safe result download. Streamlit does not train or calibrate at runtime, download a model, read the raw project CSV, derive the SHAP background, run global explanation analysis, optimize scenarios, persist uploaded profiles or results, or externally log user content.

## P12 Fairness Audit

The local P12 evidence audits the unchanged P8 probability contract on all 50,736 rows of the frozen P3 test split. It covers the dataset's two documented `Sex` codes, four predeclared `Age` bands, all eight ordinal `Income` codes, and the eight `Sex x Age` intersections. All 22 subgroup cells passed the predeclared floor of 500 rows, 100 positives, and 100 negatives; every eligible result is included with 5,000 seed-42 percentile-bootstrap resamples and directional `group - whole cohort` gaps.

The whole test cohort has 13.93% observed prevalence, 13.95% mean served probability, Brier score 0.0974, log loss 0.3144, ROC-AUC 0.8270, and PR-AUC 0.4231. Subgroup estimates differ, sometimes materially, but those differences do not establish causes, discrimination, clinical validity, or a universal fairness conclusion. BRFSS 2015 is historical and self-reported; the target can reflect diagnosis access and reporting; prevalence affects precision, PR-AUC, and error rates; the processed data has only binary `Sex`, ordinal `Age`/`Income`, and no race/ethnicity field. Group averages also cannot determine whether an individual prediction is fair.

The complete formulas, support evidence, uncertainty intervals, aggregate tables, plots, limitations, hashes, and reproduction command are in the [P12 technical report](docs/p12-fairness/report.md). These local files and their Accepted decision updates are not yet versioned: they are intended to be reviewed and committed together by the user. P12 made no mitigation, retraining, recalibration, threshold, artifact, or Streamlit change.

## Project Structure

```
src/               # Reusable data, modeling, artifact, explanation, scenario,
                   # batch, and offline fairness-audit modules
.streamlit/        # Server transport configuration (2 MiB upload ceiling)
notebooks/         # EDA and analysis notebooks
app/               # Streamlit application code
tests/             # Pytest suite, including deployment reference profiles
data/
  raw/             # Local raw dataset (git-ignored; manual Kaggle download)
  processed/       # Generated intermediate datasets (git-ignored; currently empty)
models/            # Official P8 model artifact plus the separate privacy-safe P9
                   # aggregate SHAP background asset
docs/              # Planning and reproducible aggregate analysis documentation
```

## Dataset

This project uses the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv) from Kaggle.

Selected file:

`diabetes_binary_health_indicators_BRFSS2015.csv`

License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

The raw CSV is not committed to the repository, and only the offline EDA/training workflows need it. See [data/README.md](data/README.md) for the acquisition instructions and data handling policy.

## Planning Documentation

- [Project Charter](docs/project-charter.md)
- [Roadmap](docs/roadmap.md)
- [Backlog](docs/backlog.md)
- [ML Analysis Plan](docs/ml-analysis-plan.md)
- [Decision Log](docs/decisions.md)
- [Iteration Log](docs/iteration-log.md)

## License

The project code is licensed under the [MIT License](LICENSE). The dataset is distributed by its authors under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
