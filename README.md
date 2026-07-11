# diabetes-risk-ml

Machine Learning app to estimate self-reported diabetes/prediabetes risk using BRFSS 2015 data, with model comparison, probability calibration, SHAP explainability, fairness analysis, and Streamlit deployment.

> This is an academic portfolio project. It is not a medical device and must not be used as diagnosis or medical advice.

## Current Status

The MVP lifecycle through public deployment (P0-P7) is complete. The selected `HistGradientBoostingClassifier` (decision D-016) is trained offline through `src/artifacts.py`; the official artifact ships with the repository under the controlled D-013 Git exception; and `app/streamlit_app.py` serves single-case educational risk predictions without training, downloading a model, or reading the raw CSV at runtime. The app is live at [brfss-diabetes-risk-estimator.streamlit.app](https://brfss-diabetes-risk-estimator.streamlit.app/). Public startup, form rendering, artifact loading, disclaimer visibility, exact-age conversion, and all four deployment reference profiles are verified. P8, probability calibration and threshold analysis, is refined and Ready (stories US-0601/US-0606/US-0607; decisions D-018/D-019 pending); the next step is implementing its first increment. The public app continues to serve uncalibrated probabilities until P8 resolves D-018 and deploys the selected probability contract through a schema-version-2 artifact.

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

This trains the selected D-016 model offline through the project's data contract on the local CSV, then overwrites `models/diabetes_risk_model.joblib` after a load/predict smoke check. Training only ever happens through this offline command (decision D-007): the Streamlit app loads the artifact and never trains, downloads models, or reads the CSV.

### 5. Run the test suite

```
python -m pytest tests -v -p no:cacheprovider
```

With the raw CSV in place the whole suite runs, including the real-data integration tests; without it, those tests are skipped and the rest still run.

### 6. Launch the app

```
python -m streamlit run app/streamlit_app.py
```

The app loads the local artifact once, renders the 21-feature input form, and shows the model's uncalibrated positive-class probability as an educational risk percentage with a visible medical disclaimer.

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

The MVP is deployed on Streamlit Community Cloud at [https://brfss-diabetes-risk-estimator.streamlit.app/](https://brfss-diabetes-risk-estimator.streamlit.app/). The deployment uses branch `main`, entry point `app/streamlit_app.py`, Python 3.12, the pinned `requirements.txt`, and the version-controlled official artifact selected by D-013. The deployed app performs no training, model download, or CSV access. Verification confirmed successful startup, artifact loading, the complete 21-feature form, the visible medical disclaimer, and all four reference profiles with exact displayed outputs of 0.3%, 60.0%, 70.0%, and 79.9% and their expected age-group messages.

## Project Structure

```
src/               # Reusable Python modules (data contract, modeling, artifacts)
notebooks/         # EDA and analysis notebooks
app/               # Streamlit application code
tests/             # Pytest suite, including deployment reference profiles
data/
  raw/             # Local raw dataset (git-ignored; manual Kaggle download)
  processed/       # Generated intermediate datasets (git-ignored; currently empty)
models/            # Model artifacts: diabetes_risk_model.joblib is the official
                   # version-controlled artifact (D-013); all others are git-ignored
docs/              # Planning and analysis documentation
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
