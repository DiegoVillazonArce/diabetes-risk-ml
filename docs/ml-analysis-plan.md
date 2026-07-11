# ML Analysis Plan

## Objective

Build a supervised classification pipeline that estimates self-reported diabetes or prediabetes risk from BRFSS 2015 health indicators.

The model output should be communicated as an educational risk estimate, not as diagnosis or medical advice.

## Current Local Dataset State

The raw dataset was moved to its standard local path during project setup (P1) and is git-ignored:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

Dataset source:

- Kaggle dataset: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv)
- Selected file: `diabetes_binary_health_indicators_BRFSS2015.csv`
- License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

Current inspection:

- Rows: 253,680.
- Columns: 22.
- Target column found: `Diabetes_binary`.
- Observed target counts:
  - `0.0`: 218,334.
  - `1.0`: 35,346.
- Positive class rate: approximately 13.9%.

This confirms that the current local file matches the original imbalanced binary BRFSS dataset selected for the MVP.

The P2 EDA (`notebooks/01_data_understanding_eda.ipynb`) additionally found 24,206 exact duplicate rows (~9.5%), skewed heavily toward the negative class (~1% positive among duplicates vs. ~13.9% overall). No rows were dropped during EDA. Decision D-014 accepts keeping exact duplicate rows for MVP data preparation, so P3 splits must preserve the observed ~13.9% positive prevalence.

## Target Definition

The project uses the original binary target:

- `Diabetes_binary = 0`: no self-reported diabetes or prediabetes.
- `Diabetes_binary = 1`: self-reported prediabetes or diabetes.

The target is not derived from the multiclass `Diabetes_012` file. The project remains an imbalanced binary classification problem, and the selected analysis population's class distribution must be preserved in evaluation splits.

Because D-014 keeps exact duplicate rows for MVP data preparation, the selected analysis population remains the full 253,680-row dataset and the selected prevalence remains approximately 13.9% positive.

## Data Handling Policy

- The selected raw Kaggle CSV is CC0, so there is no known license blocker to redistribution.
- The raw CSV is still kept out of Git for the MVP to keep the repository focused on code, documentation, and lightweight deployable artifacts.
- Large generated files such as raw splits, serialized models, and binary artifacts should be ignored unless intentionally selected for deployment.
- Public documentation should explain how to obtain the data rather than assuming the raw CSV is versioned.
- The MVP should use manual Kaggle download instructions first, because automated Kaggle API download requires user-specific credentials.
- A Kaggle API helper script can be considered later as an optional convenience, not as the only supported data path.

## Reproducibility and Dependency Plan

The initial MVP dependency strategy is:

- Select and document the supported Python version during project setup.
- Use a `requirements.txt` file with explicit package versions for the first reproducible environment.
- Avoid generating the main requirements file from a noisy global environment.
- Revisit a lockfile-based workflow, such as `uv`, if dependency reproducibility becomes a larger concern before deployment.

This strategy is recorded in the decision log as D-012: Python 3.12 with a hand-curated pinned `requirements.txt`, created during project setup (P1).

## Initial Feature Set

Expected BRFSS features:

- Binary health indicators: `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.
- Numeric indicators: `BMI`, `MentHlth`, `PhysHlth`.
- Ordinal indicators: `GenHlth`, `Age`, `Education`, `Income`.

Important notes:

- `Age` is an ordinal age group in the dataset and model, not exact age in years. The P6 UI may collect an exact adult age as a convenience, but it must deterministically map that value to the documented BRFSS age-group code before inference.
- `Education` and `Income` are ordinal socioeconomic indicators.
- The data is self-reported survey data, not clinical measurement data.

## Data Quality Analysis

The EDA phase should report:

- Shape and schema.
- Target distribution.
- Positive class prevalence and base-rate implications.
- Missing values.
- Duplicate rows.
- Feature ranges and unexpected values.
- Confirmation of the documented feature groups (see Initial Feature Set) against observed dtypes and values.
- Lightweight Spearman correlation observations to identify obvious feature relationships or redundancy.
- Memory usage and downcasting recommendations for later data preparation.
- Expected class balance impact of any proposed cleaning decision, without applying cleaning during EDA.
- Potential limitations from self-reported survey data.

EDA should generate evidence for later preparation and modeling decisions, but it should not perform permanent dtype conversion, feature removal, train/test splitting, balancing, or model training.

## Preparation and Splitting Expectations

P3 should turn the EDA findings into reusable loading, validation, downcasting, and split logic. D-014 has resolved that exact duplicate rows are kept for MVP data preparation.

P3 should validate the raw schema, missing values, integer-like values, target values, and feature ranges; safely downcast validated values to `uint8`; and create reproducible stratified 70/10/20 train/calibration/test splits that preserve the observed ~13.9% positive prevalence in every split.

Per D-015, P3 returns split outputs in memory and does not write processed split files under `data/processed/` for now. It should not introduce balancing, SMOTE, model training, feature engineering, calibration, explainability, or app work.

## Split Strategy

Use stratified splits:

- Train set: 70%.
- Calibration set: 10%.
- Test set: 20%.

Train, calibration, and test sets must each preserve the selected target distribution. Any class balancing or sampling technique must be applied only inside the training process.

Note: D-014 keeps duplicate rows for MVP data preparation, so the selected target distribution is the observed ~13.9% positive prevalence from the full 253,680-row dataset. Dropping duplicates would have shifted positive prevalence to ~15.3%, but that is not the default P3 path.

P4 baseline modeling should fit models on the training split and report baseline metrics on train and test only. The calibration split should remain unused in P4 so it is still clean for later probability calibration work.

## Baseline and Candidate Models

MVP candidates:

- `DummyClassifier`.
- `LogisticRegression`.
- One tree-based model, such as `RandomForestClassifier` or `HistGradientBoostingClassifier`.

P4 establishes the Dummy and Logistic Regression baseline only; P5 extends to the tree-based candidate and formal model comparison/selection (see Epic E8 in `docs/backlog.md`).

P4 modeling code should live in a reusable source module, likely `src/modeling.py`; notebooks may summarize results, but they should not be the only implementation path.

P5 should build on the P3/P4 contracts rather than reloading or re-splitting data. Candidate models should train on the train split and be evaluated on train and test only, leaving the calibration split untouched for P8 probability calibration. The comparison should return an in-memory table or structured result with the documented metrics by model and split, and model selection should prioritize PR-AUC and positive-class recall/precision/F1 over accuracy. P5 selected `HistGradientBoostingClassifier` as the primary MVP model (D-016) without writing any artifact.

P6 trains and serializes the D-016 model once, at the start of the phase (D-017); after artifact creation, the Streamlit app only loads that local artifact and does not retrain, re-compare, or reload/re-split raw data. P6 is a local functional MVP, not a public deployment (see the Model Artifact Plan below and D-013).

Post-MVP candidates:

- XGBoost.
- Additional imbalance strategies.
- Optional MLP baseline if useful for comparison.

## Model Artifact Plan

The app should only load artifacts created by this project's offline training pipeline.

Initial plan:

- Use `joblib` for MVP model serialization, because it is the standard practical option for scikit-learn pipelines.
- Store lightweight metadata with the artifact, such as training date, feature order, model type, metrics, and package versions. P6 implemented this as a single joblib bundle holding both the fitted model and its metadata, so the two cannot drift apart.
- Do not load model artifacts from untrusted external sources.
- Evaluate `skops` before final portfolio packaging if safer model serialization becomes a priority.

Per D-017, serialization happens once, at the start of P6, using the D-016 selected model and the D-010 `joblib` format, paired with a local load/predict check before the app depends on the artifact. `models/*.joblib` stayed git-ignored through P6: the P6 app only needed to load the artifact from the local filesystem, not from a public deployment.

P7 resolved how the artifact reaches a deployed app: per D-013, the official artifact `models/diabetes_risk_model.joblib` (~263 KB) is version-controlled as a controlled Git exception, while all other files under `models/` remain git-ignored. The deployed app therefore loads the same local repository file with no network access, no runtime training, and no need for the raw CSV at serving time; the formal three-alternative comparison behind that choice is recorded in `docs/decisions.md`.

## Imbalance Strategy

Initial MVP strategies to consider:

- No balancing.
- `class_weight="balanced"` where supported.

The test set must never be balanced.

P5 may decide whether a simple imbalance-aware variant such as `class_weight="balanced"` belongs in the comparison. SMOTE, other resampling methods, threshold tuning, and calibration should stay out of P5 unless the backlog explicitly expands that scope.

## Evaluation Metrics

Primary metrics:

- ROC-AUC.
- PR-AUC / average precision.
- Recall for the positive class.
- Precision for the positive class.
- F1 for the positive class.
- Confusion matrix at selected threshold.

Secondary metrics:

- Accuracy, with explicit explanation of why it can be misleading.
- Brier score after probability calibration is introduced.

## Calibration Plan

Post-MVP calibration should compare:

- Uncalibrated probabilities.
- Sigmoid / Platt calibration.
- Isotonic calibration if enough calibration data is available.

Evaluation:

- Reliability diagram.
- Brier score.
- Before/after probability interpretation.

## Explainability Plan

Global explainability:

- Feature importance.
- SHAP beeswarm or bar plots generated offline.

Local explainability:

- SHAP waterfall or equivalent explanation for one prediction.

Required communication rule:

- SHAP explains model behavior, not causal health effects.

## Scenario Exploration Plan

Post-MVP, the app may include a model scenario explorer for selected modifiable inputs.

Rules:

- Only approved modifiable features should be editable in scenario mode.
- Non-modifiable or sensitive/contextual fields such as age, sex, education, and income should not be presented as improvement levers.
- Scenario results must be described as changes in model output, not as estimated medical benefit.
- The UI should avoid prescriptive wording such as "do this to reduce your risk by X%".
- Scenarios should use valid feature ranges and avoid unrealistic input combinations where possible.

## Fairness Analysis Plan

Evaluate key metrics by subgroup where sample sizes are adequate:

- `Sex`.
- Age groups.
- Income groups.

Potential metrics:

- Recall.
- False positive rate.
- Precision.
- Calibration by group where practical.

The purpose is to audit and communicate disparities, not to claim the model is fair.

## Testing Plan

The MVP should include focused pytest coverage for high-risk project behavior:

- Data schema validation: required columns, no unexpected target values, and expected feature ranges.
- Data quality checks: missing values, duplicate handling, and class distribution reporting.
- Split checks: stratified train/calibration/test splits preserve the selected target distribution within a small tolerance.
- Pipeline checks: preprocessing and model pipeline can fit on a small sample and produce valid probabilities.
- Model-comparison checks: tree-based candidates can fit and produce probabilities, comparison outputs have metrics by model and split, selection is deterministic, and P5 code does not use calibration rows or reload/re-split data ad hoc.
- Artifact checks: serialized model can be loaded locally and returns probabilities in `[0, 1]`, and saved metadata includes the feature order and the selected model identity.
- App-facing checks: the input-to-DataFrame helper preserves the training feature order, input validation rejects missing/out-of-range values, and the app-facing prediction helper returns a probability in `[0, 1]` without launching the Streamlit runtime.

Streamlit-specific smoke tests should be considered after the MVP app exists.

## Initial Risks

- The dataset is widely used, so project differentiation depends on methodological depth and communication quality.
- Self-reported labels can reflect diagnosis access and reporting bias.
- Probability outputs can be misleading if not calibrated.
- What-if simulations can be misread as causal advice if not framed carefully.
- Dependency drift can affect numerical results if package versions are not pinned.
- Serialized Python model artifacts can be unsafe if loaded from untrusted sources.
