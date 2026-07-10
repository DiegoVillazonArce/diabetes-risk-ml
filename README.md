# diabetes-risk-ml

Machine Learning app to estimate self-reported diabetes/prediabetes risk using BRFSS 2015 data, with model comparison, probability calibration, SHAP explainability, fairness analysis, and Streamlit deployment.

> This is an academic portfolio project. It is not a medical device and must not be used as diagnosis or medical advice.

## Current Status

Reproducible data preparation/splitting (P3), baseline modeling (P4), model comparison/selection (P5), and the local Streamlit MVP (P6) are complete: the selected `HistGradientBoostingClassifier` (decision D-016) is trained and serialized locally through `src/artifacts.py`, and `app/streamlit_app.py` serves single-case educational risk predictions from that artifact. The next phase is MVP documentation and the first public deployment (P7), which will also resolve the pending artifact-distribution decision (D-013).

### Generate the model artifact and run the local app

With the project environment installed and activated and the raw CSV in place (see Dataset below), run from the project root:

```
python -m src.artifacts             # train the selected model and save models/diabetes_risk_model.joblib
streamlit run app/streamlit_app.py  # launch the local prediction app
```

The generated artifact stays local (`models/` contents are git-ignored); the app only loads it and never retrains. Predictions are educational risk estimates with a visible medical disclaimer, not diagnoses.

## Project Structure

```
src/               # Reusable Python modules (data loading, validation, training)
notebooks/         # EDA and analysis notebooks
app/               # Streamlit application code
tests/             # Pytest suite
data/
  raw/             # Local raw dataset (git-ignored)
  processed/       # Generated intermediate datasets (git-ignored)
models/            # Serialized model artifacts (git-ignored)
docs/              # Planning and analysis documentation
```

## Environment Setup

The supported Python version is **3.12**. Dependencies are pinned in `requirements.txt`.

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Dataset

This project uses the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv) from Kaggle.

Selected file:

`diabetes_binary_health_indicators_BRFSS2015.csv`

License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

The raw CSV is not committed to the repository. It is only needed to reproduce the EDA and the offline training pipeline; Streamlit app users do not need it because the app serves serialized model artifacts trained offline. To reproduce EDA or training, download the selected file from Kaggle and place it at:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

See [data/README.md](data/README.md) for the full data acquisition instructions and data handling policy.

## Planning Documentation

- [Project Charter](docs/project-charter.md)
- [Roadmap](docs/roadmap.md)
- [Backlog](docs/backlog.md)
- [ML Analysis Plan](docs/ml-analysis-plan.md)
- [Decision Log](docs/decisions.md)
- [Iteration Log](docs/iteration-log.md)

## License

The project code is licensed under the [MIT License](LICENSE). The dataset is distributed by its authors under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
