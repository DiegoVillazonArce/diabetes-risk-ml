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
- Brier score and log loss once probability calibration work (P8) begins.

## Calibration and Threshold Analysis Plan (P8)

P8 compares uncalibrated, sigmoid (Platt), and isotonic probabilities for the frozen D-016 `HistGradientBoostingClassifier`. The calibration split has remained untouched since P3 and holds 25,368 rows at the preserved ~13.9% prevalence (~3,535 positives), enough calibration data for isotonic to be a serious candidate alongside sigmoid. The comparison can legitimately end with `calibration_method = none` -- retaining the uncalibrated output -- if no method qualifies under the pre-declared criteria; US-0607 then ships the version-2 artifact without a calibrator and the app keeps its uncalibrated wording.

### Leakage-Safe Evaluation Protocol

1. Train the selected `HistGradientBoostingClassifier` (D-016) on the train split only.
2. Freeze the base model: no refit, retuning, or reselection anywhere in P8.
3. Score the calibration split with the frozen model to obtain the uncalibrated baseline.
4. Apply stratified five-fold cross-fitting within the calibration split.
5. For each method, sigmoid and isotonic: fit the calibrator on four folds, predict the held-out fold, and combine the held-out predictions into complete out-of-fold calibrated probabilities.
6. Compare both methods and the uncalibrated baseline exclusively on those out-of-fold predictions, applying the operationalized criteria below.
7. Select the calibration method -- sigmoid, isotonic, or none -- and resolve D-018; no test data participates.
8. Analyze thresholds exclusively on the selected contract's out-of-fold probabilities.
9. Freeze the threshold scenarios and resolve D-019.
10. Refit the selected calibrator using the entire calibration split (skipped when `none` is selected; the per-fold calibrators are discarded either way).
11. Run the official P8 test evaluation of the frozen contract: one recorded evaluation, performed only after D-018 and D-019 are frozen. Test participates in no P8 decision; later runs (deterministic artifact regeneration, regression tests, full-suite repeats, deployment verification) may only repeat this evaluation as a deterministic regression check.
12. Record the official results; never select or modify the method or a threshold after observing them.

The cross-fitting operates on the frozen base model's output scores: the five folds partition the calibration split for calibrator fitting and out-of-fold prediction only, and no calibration row is ever used to retrain the `HistGradientBoostingClassifier` itself. Train and test rows are never used to fit a calibrator.

### Selection Criteria (D-018, operationalized and pre-declared)

Fixed during this refinement (2026-07-11), before any out-of-fold result exists, so selection cannot be tuned after peeking. All score comparisons use the out-of-fold predictions for the 25,368 calibration rows with paired differences and a fixed-seed bootstrap: 10,000 resamples of the calibration rows, with percentile 95% confidence intervals of the mean paired per-row score difference. The paired difference is defined per calibration row as `delta_i = loss_candidate_i - loss_reference_i`, where the reference is the uncalibrated baseline in the adoption rule and the other method in the pairwise choice. A candidate improves its reference only if the upper limit of the 95% confidence interval of the mean `delta` is below zero; between methods, the lower-loss method is superior when the interval excludes zero in the corresponding direction. Log loss uses exactly the same convention.

1. **Adoption rule against the uncalibrated baseline.** A calibration method is adoptable only if the upper limit of the 95% paired-bootstrap confidence interval of its Brier `delta` against the uncalibrated baseline is below zero. If neither sigmoid nor isotonic is adoptable, `calibration_method = none` is selected and the uncalibrated output is retained.
2. **Choice between two adoptable methods.** Prefer the method the paired-bootstrap Brier rule declares superior (interval excluding zero in its favor); if that interval includes zero, apply the identical rule to log loss as the tie-break; if that interval also includes zero, the methods are practically equivalent by this definition and sigmoid is selected for simplicity and strict ranking preservation.
3. **Ranking-preservation guard.** The selected method must not reduce out-of-fold ROC-AUC or PR-AUC by more than 0.005 absolute versus the uncalibrated baseline; a larger drop disqualifies it (the other method is considered if it passes, otherwise `none`). The 0.005 bound is a project-defined ranking-preservation guardrail fixed before observing P8 results, not a clinical or statistical standard; at the D-016 baseline levels it tolerates roughly a 0.6% relative drop in ROC-AUC and a 1.2% relative drop in PR-AUC, small enough to catch a relevant degradation without rejecting isotonic over minor ties.
4. **Reliability diagrams are visual diagnostics only**, reported for the baseline and both methods; they are never a subjective pass/fail criterion.

ECE may be mentioned as a secondary descriptive metric, but not as a selection criterion on its own, because it depends on the binning choice.

### Threshold Analysis Rules (D-019)

- Precision-recall curves and threshold tables (positive-class recall, precision, F1, false-positive and false-negative counts, confusion matrices) are computed from the out-of-fold probabilities of the D-018-selected probability contract on the calibration split only.
- Probability estimation stays explicitly separate from any decision layer: the app displays a probability with no threshold, and introducing decision labels would require an explicit, justified D-019 resolution -- it is not assumed.
- No threshold may be presented as a validated screening or diagnostic rule, and the analysis makes no clinical claims; the trade-offs are model behavior, not medical guidance.

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
