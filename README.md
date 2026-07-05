# diabetes-risk-ml

Machine Learning app to estimate self-reported diabetes/prediabetes risk using BRFSS 2015 data, with model comparison, probability calibration, SHAP explainability, fairness analysis, and Streamlit deployment.

> This is an academic portfolio project. It is not a medical device and must not be used as diagnosis or medical advice.

## Current Status

Planning and analysis is complete. The next phase is project setup and data governance.

## Dataset

This project uses the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv) from Kaggle.

Selected file:

`diabetes_binary_health_indicators_BRFSS2015.csv`

License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

The raw CSV is not committed to the repository. To reproduce the training pipeline, download the selected file from Kaggle and place it at:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

## Planning Documentation

- [Project Charter](docs/project-charter.md)
- [Roadmap](docs/roadmap.md)
- [Backlog](docs/backlog.md)
- [ML Analysis Plan](docs/ml-analysis-plan.md)
- [Decision Log](docs/decisions.md)
- [Iteration Log](docs/iteration-log.md)

## License

The project code is licensed under the [MIT License](LICENSE). The dataset is distributed by its authors under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
