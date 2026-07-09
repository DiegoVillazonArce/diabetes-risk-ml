# diabetes-risk-ml

Machine Learning app to estimate self-reported diabetes/prediabetes risk using BRFSS 2015 data, with model comparison, probability calibration, SHAP explainability, fairness analysis, and Streamlit deployment.

> This is an academic portfolio project. It is not a medical device and must not be used as diagnosis or medical advice.

## Current Status

Reproducible data preparation/splitting (P3) and baseline modeling (P4, Dummy and Logistic Regression baselines in `src/modeling.py`) are complete. The next phase is model comparison and selection (P5).

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
