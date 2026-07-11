# Data Directory

This directory holds local dataset files. Raw and processed data files are intentionally **not committed** to the repository (see `.gitignore` and decision D-006).

## Dataset Source

- Kaggle dataset: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv)
- Selected file: `diabetes_binary_health_indicators_BRFSS2015.csv`
- License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- Target column: `Diabetes_binary`

## Expected Local Layout

```
data/
  raw/
    diabetes_binary_health_indicators_BRFSS2015.csv   (manual download, git-ignored)
  processed/
    ...                                               (generated outputs, git-ignored)
```

## Who Needs the Raw Dataset

- **Streamlit app users: nobody needs it.** The app only loads the serialized model artifact produced by the offline training pipeline (decision D-007), which ships with the repository at `models/diabetes_risk_model.joblib` (decision D-013). It never reads the raw CSV.
- **Reproducing EDA or training: required.** The raw CSV is needed only to rerun the exploratory data analysis notebooks and the offline training pipeline that regenerates the model artifact.

## How to Obtain the Data

1. Sign in to Kaggle (a free account is required to download; this is why the download is manual rather than automated).
2. Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the dataset page linked above.
3. Place it at:

   `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

An optional Kaggle API download script may be added later as a convenience, but manual download remains the primary supported path because API access requires user-specific credentials (decision D-009).

## Why the Raw CSV Is Not Committed

The selected file is CC0, so licensing is not the blocker. The raw CSV is kept out of git to keep the repository focused on code, documentation, and lightweight deployable artifacts. `.gitignore` excludes `data/raw/*.csv`, `data/processed/*.csv`, and `data/processed/*.parquet`.

## Expected File Characteristics

Quick integrity reference for the selected file:

- Rows: 253,680 (plus a header row).
- Columns: 22.
- Target counts: `0.0` -> 218,334, `1.0` -> 35,346 (positive rate approx. 13.9%).
