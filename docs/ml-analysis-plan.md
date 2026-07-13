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

## Explainability Protocol (P9)

P9's purpose is to explain globally and locally the final probabilistic contract selected in P8 without modifying its predictions and without presenting attributions as causal relationships, diagnoses, or medical recommendations. Implementation must provide two complementary communication levels: a simple, progressive explanation in Streamlit for people without AI expertise, and reproducible technical evidence in GitHub for academic and technical review.

### P8 Probability Contract to Explain

The explanation target is the deployed schema-version-2 artifact containing the frozen D-016 `HistGradientBoostingClassifier`. D-018 selected `calibration_method = none`, so the current serving path has no calibrator and `src.artifacts.predict_risk_probability()` returns the model's positive-class (`Diabetes_binary = 1`) `predict_proba` output. D-019 keeps that output probability-only, with no decision threshold or high/low-risk label.

P9 is a read-only explanation layer over this contract. It must not retrain, recalibrate, retune, or replace the model; change the artifact or thresholds; or change any probability returned by the app or the four public reference profiles. If a future phase changes the probability contract, the explainer's compatibility, output mapping, class selection, additivity, and numerical fidelity must be reevaluated before its explanations can be reused.

### Explanation Output and Positive Class (D-020)

The preferred explanation output is the exact positive-class probability served by `predict_risk_probability`, because that is the quantity users see. That preference is not an assumed SHAP capability: the Increment 1 spike must establish whether `TreeExplainer` can reproduce that probability faithfully and additively for the pinned stack and `HistGradientBoostingClassifier`.

D-020 must select and document one evidenced contract:

- Explain the served positive-class probability directly with `TreeExplainer` when the compatibility and additivity checks pass.
- Explain the raw model margin, with the output transformation and its communication to users made explicit and tested.
- Use a model-agnostic fallback when TreeExplainer cannot faithfully reproduce the served contract within the declared resource bounds.

No implementation may silently switch from probability to margin, from the positive class to another output, or from TreeExplainer to a fallback. The explainer configuration must record the explained model/output, model-output setting, feature order, expected-value shape, SHAP-value shape, and positive-class selection. For an analysis sample of `n` rows, the normalized project contract must yield a finite `n x 21` contribution matrix aligned exactly to `src.data.FEATURE_COLUMNS`.

### Global and Local Explanations

Global and local explanations answer different questions and both are required:

- **Global explanation:** summarize how strongly each feature contributes across the fixed analysis sample, using mean absolute SHAP values for the primary importance ranking. Publish an aggregate importance CSV, a bar plot, and a beeswarm plot generated offline. The beeswarm may show the direction and distribution of model contributions, but its narrative must not imply that the features cause the target.
- **Local explanation:** decompose one fixed prediction into a base value plus per-feature contributions under the selected D-020 output contract. P9 must produce reproducible local explanations for all four synthetic public reference profiles, with waterfall plots or an evidenced equivalent and a tabular representation showing the base value, final served estimate, and contribution of every feature.

Global mean absolute importance does not provide direction or establish an intervention effect; a local explanation does not generalize to the population. Neither explanation establishes medical mechanisms.

### Background and Analysis Sample Policy (D-021)

D-021 must freeze the origin, size, seed, sampling policy, privacy treatment, and intended use of both the explainer background and the global-analysis sample before the full analysis begins.

Initial proposal for the Increment 1 spike:

- Background: 256 rows sampled deterministically from train only.
- Global-analysis sample: up to 5,000 rows sampled deterministically and proportionally stratified by the target from calibration, preserving the calibration split's original positive-class prevalence within deterministic allocation/rounding.
- Random seed: the project seed, `42`.

The offline analysis background must never contain calibration or test rows. Calibration is reserved for the fixed global explanation sample because the model was not trained on those rows; test remains excluded from all P9 configuration and communication decisions. Test must not select the explainer, output space, background, sample, tolerance, narrative, feature emphasis, or visualization. The proposed sizes may change only during the compatibility spike for a recorded performance or memory constraint, never after observing SHAP results to favor a narrative.

Published evidence must identify split provenance and sampling rules without exposing row-level source data. Global CSV output must be aggregate. Local published outputs may contain only the already public synthetic reference profiles, never real train, calibration, or test rows. Real train rows used by the offline background must not be published, embedded in a deployable explainer, or shipped in any public runtime asset. A dynamic delivery option is privacy-eligible only if deployed code and assets expose no real background rows; otherwise P9 must validate a faithful aggregate or synthetic background under D-020/D-021 or reject that dynamic design.

### Compatibility, Fidelity, and Additivity

The spike must run against the frozen project stack before a SHAP version is added to `requirements.txt`: Python 3.12, NumPy 2.2.6, scikit-learn 1.7.1, and the frozen D-016 `HistGradientBoostingClassifier`. It must evaluate `TreeExplainer` first, record errors or unsupported output behavior, measure representative time and peak memory, and compare any necessary fallback without choosing it silently. The SHAP version is pinned only after the selected path passes these checks.

For each global-analysis row and each public reference profile, validate the SHAP identity in the D-020 output space:

`base_value + sum(feature_contributions) ~= explained_model_output`

The absolute additivity tolerance is fixed at `1e-4` for any D-020 alternative that explains the served positive-class probability directly. The spike tests that fixed guardrail; it does not tune or relax it. If a candidate cannot meet `1e-4`, direct-probability explanation is not accepted on that evidence. If D-020 selects a different mathematical output space, the spike must fix and justify a separate output-specific tolerance before the full analysis begins, and that tolerance must never be relaxed after observing inconvenient results. When the additive output is a raw margin rather than the served probability, both the additive margin identity and the explicit margin-to-probability relationship must be tested and communicated. Comparing values in different spaces is not evidence of fidelity.

Additional fidelity checks must prove that all values are finite, the positive class is correct, contribution columns exactly preserve `FEATURE_COLUMNS`, the final local estimate matches `predict_risk_probability` under the D-020 mapping, and the four P8 reference probabilities and recorded displays remain unchanged.

### Reproducibility

P9 evidence must record:

- Artifact identity, model class, `calibration_method`, explained output, positive class, and exact feature order.
- SHAP explainer type and complete relevant configuration.
- Background and analysis-sample split provenance, sizes, selection method, stratification policy, and seed.
- Python, NumPy, pandas, scikit-learn, SHAP, and plotting-library versions.
- Additivity tolerance and per-contract validation results.
- Commands or callable entry points used to regenerate the CSV files and plots.
- Representative execution time and memory measurements, plus any documented spike-driven size adjustment.

Running the offline analysis twice with the same artifact, data, versions, and seed must reproduce background/sample membership, mean absolute importance, plot inputs, and the four local contribution tables within predeclared numerical tolerances.

### Local Feature Representation

Technical evidence retains the exact model feature names and encoded values for auditability, while user-facing explanations translate them into labels already used by the app. Encoded variables must be described accurately: `Age`, for example, represents a BRFSS age group, not an exact age; `Education` and `Income` are ordinal survey categories; binary values are indicators; and the dataset is self-reported.

User-facing contribution text should use formulations such as "increased the model's estimate" and "decreased the model's estimate." It must not say or imply that a feature caused diabetes, prevented diabetes, diagnosed a condition, established a protective factor, or recommends an intervention.

### Simple Streamlit Communication

Subject to D-022's delivery choice, P9 must add a section similar to "How the model interprets this estimate" after a prediction. It should be visual, progressive, and written in everyday language for people without AI expertise:

- Begin with a short explanation that some entered variables pushed this model estimate up and others pushed it down relative to its reference value.
- Prioritize a compact visual and the most material contributions; avoid formulas, raw SHAP terminology, and unnecessary mathematics in the primary view.
- Offer an expander for limited additional context, including what the reference value means and why contributions do not represent causes.
- Keep the existing medical disclaimer visible and preserve probability-only behavior.
- Do not turn explanations into recommended actions, clinical interpretations, or a P10 scenario explorer.

D-022 must decide whether local explanations are dynamic, precomputed, or hybrid before Streamlit is modified. The evidence must cover runtime, memory, privacy, fidelity, maintenance, and non-technical user experience. A dynamic option may be accepted only if deployed code and assets do not expose or retain accessible real background rows. If the faithful dynamic path would ship real rows, it must use a separately validated aggregate or synthetic background that still passes D-020/D-021, or the dynamic option must be rejected. If dynamic explanations are selected, performance and timeout bounds must also be declared and tested; importing Streamlit must never fit a model, download data, or compute the global SHAP analysis. Because the simple Streamlit explanation is required in every D-022 outcome, P9 closure always requires redeployment and a successful public smoke test.

### Technical GitHub Communication

The future `docs/p9-explainability/report.md` must allow an academic or technical reviewer to audit and reproduce P9. It must document the methodology, base value, contribution definition, explainer configuration, background, global sample, seed, positive class, output mapping, additivity evidence, reproducibility results, versions, limitations, performance/memory evidence, privacy treatment, and plot-generation process. It should link the aggregate global-importance CSV, the local-contribution CSV for the four public profiles, the global bar and beeswarm plots, and the local waterfall plots or evidenced equivalent.

The simple Streamlit explanation and technical report describe the same D-020 contract but serve different audiences. The app must not expose real dataset rows, a full academic methods section, or unnecessary mathematical detail; the GitHub report must not replace technical evidence with simplified UI copy.

### Limitations and Interpretation Guardrails

SHAP explains the behavior of the fitted model under a chosen explainer, background, sample, and output contract; it does not explain real medical mechanisms. Specifically:

- A positive or negative SHAP contribution does not prove that a variable causes or prevents diabetes.
- SHAP values inherit model error, dataset bias, self-report limitations, correlations among features, background dependence, and any limits of the selected output transformation.
- Contributions for correlated features can redistribute attribution and should not be read as unique or independent effects.
- Global importance depends on the declared analysis sample and does not establish population-wide importance outside that scope.
- Local explanations describe one model output and must not be generalized into a diagnosis or recommendation.
- Explanations must never be converted into intervention advice or claims that changing an input will change a person's real medical risk; controlled scenario exploration belongs to P10 and is outside P9.
- Fairness conclusions do not follow from global or local SHAP results; subgroup auditing belongs to P12.

Planned implementation deliverables are `src/explainability.py`, `tests/test_explainability.py`, `docs/p9-explainability/report.md`, aggregate global and synthetic-profile local CSV files, global bar and beeswarm plots, local waterfall plots or the approved equivalent, controlled Streamlit changes only after D-022, and a SHAP dependency pin only after the compatibility spike. None of these deliverables is created during the P9 planning refinement.

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
