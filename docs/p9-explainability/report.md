# P9 SHAP Explainability Technical Report

## Executive Summary

P9 is locally implemented and validated as a read-only explanation layer over the frozen P8 probability contract. D-020 accepts direct explanation of the positive-class probability with SHAP 0.52.0 `TreeExplainer`; D-021 accepts one 256-row train-derived aggregate background for both offline analysis and Streamlit plus a fixed 5,000-row proportionally stratified calibration sample; and D-022 accepts a hybrid delivery strategy: dynamic cached local explanations in Streamlit and precomputed aggregate/global plus synthetic-reference evidence in GitHub. The model artifact, served probabilities, probability-only product behavior, and medical disclaimer are unchanged. P9 remains Ready until commit, push, redeploy, and public smoke verification are performed externally.

## Objective and Scope

The objective is to explain how the frozen model produces its final P8 estimate, globally and for individual inputs, without retraining, recalibrating, changing a threshold, or adding high/low labels. These attributions describe the fitted model under a declared background; they are not clinical findings, diagnoses, treatment guidance, or evidence that changing an input changes a person's health outcome.

## Artifact Identity

- File: `models/diabetes_risk_model.joblib`
- SHA-256: `957c14ff5a490bbc60822121a889f92ee2a6a20f797eef741a710d887ecc9216`
- Size: 269991 bytes
- Artifact schema: 2
- Model: `HistGradientBoostingClassifier` (D-016)
- Calibration: `none` (D-018)
- Served class: `Diabetes_binary = 1`
- Product contract: probability only, with no served threshold or category (D-019)

## Fixed Stack

| Component | Version |
|---|---|
| Python | 3.12.7 |
| NumPy | 2.2.6 |
| pandas | 2.3.1 |
| scikit-learn | 1.7.1 |
| SHAP | 0.52.0 |
| matplotlib | 3.10.3 |

SHAP 0.52.0 was pinned only after its Python 3.12 wheel installed without changing the existing NumPy, pandas, or scikit-learn pins and after the spike below passed.

## Compatibility Spike

`requested/effective` background counts are shown together because passing a 256-row DataFrame directly to SHAP creates an implicit `Independent(max_samples=100)` masker and silently retains only 100 rows. The accepted implementation constructs `Independent(..., max_samples=256)` explicitly and verifies 256 effective rows.

| Alternative | Status | Output | Background | Max error | Create (s) | Local (s) | Global (s) |
|---|---|---|---|---|---|---|---|
| tree_probability_real_background_implicit_masker | rejected | probability | 256/100 | not run | 0.0461 | not run | not run |
| tree_probability_real_background_explicit_256 | rejected | probability | 256/256 | 0.0000000118 | 2.3058 | 0.0054 | not run |
| tree_probability_safe_aggregate_background_256 | accepted | probability | 256/256 | 0.0000000132 | 3.0701 | 0.0160 | 19.39 |
| tree_raw_margin_safe_aggregate_background_256 | rejected | raw | 256/256 | 0.0000000893 | 2.3443 | 0.0041 | 16.40 |
| permutation_probability_safe_aggregate_background_256 | rejected | positive_class_probability_callable | 256/256 | 0.0000000000 | 0.0009 | 15.0076 | projected 779.21 |

All reported errors are absolute additive-identity errors. The model-agnostic global run was intentionally not executed after its deterministic 20-row benchmark projected beyond the predeclared 60-second limit; this is a measured projection, not an observed 5,000-row runtime.

## D-020 — Explanation Output Contract (Accepted)

The accepted output is the frozen model's direct positive-class probability. Configuration: `TreeExplainer`, `feature_perturbation="interventional"`, `model_output="probability"`, class `1`, exact `FEATURE_COLUMNS` order, scalar expected value, and a normalized `n x 21` contribution matrix. For every explained row:

`model reference estimate + sum(feature contributions) ≈ predict_risk_probability output`

The predeclared direct-probability tolerance is `1e-4`; the observed global maximum is `1.318595633e-08`. The raw-margin route was rejected despite numerical fidelity because its additive units are log-odds and require an additional logistic transformation. The model-agnostic probability fallback was rejected for runtime. No output, class, tolerance, or explainer changed after the full results were inspected.

## D-021 — Background and Global Sample (Accepted)

The accepted background has 256 requested and 256 effective rows. It is built only from the train features: train rows are stably sorted by the frozen model probability, divided into 256 bands, and each band is replaced by its feature-wise arithmetic mean. Every centroid aggregates 693–694 train rows. It contains no target, respondent identifier, split/source index, or real row; offline comparison found 0 exact matches with train. The deployable asset is `models/shap_background_v1.json`, with its own schema, feature order, deterministic construction provenance, project seed, model-artifact hash, and strict loader. The centroid algorithm itself uses no random operation; seed 42 governs the stratified global sample.

The global sample contains 5000 calibration rows: 697 positive and 4303 negative. Calibration prevalence is 0.1393487859; sample prevalence is 0.1394000000; absolute difference is 0.0000512141, within deterministic proportional rounding. The sample is selected with seed 42 and never published. Test is structurally absent from both builders.

The aggregate background is a privacy and deployment compromise: centroids may combine feature values in ways that no respondent reported, and sorting bands by model output can redistribute contributions compared with another valid background. That dependence is part of the explanation contract, not a claim that the background represents an average person.

## D-022 — Delivery Strategy (Accepted)

| Strategy | Evidence | Decision |
|---|---|---|
| Dynamic | Safe and faithful with the aggregate asset; cached warm local runtime meets the bound. | Viable component, but does not itself provide the fixed technical/global record. |
| Precomputed | Fast and simple, but cannot honestly explain an arbitrary submitted input. | Rejected as the app strategy. |
| Hybrid | Dynamic cached local explanation for the submitted input; precomputed aggregate global and four synthetic-profile evidence. | Accepted. |

Streamlit loads no raw CSV, calls no `prepare_data()`, trains no model, and does not derive the background at runtime. It creates and caches a `TreeExplainer` from the versioned aggregate asset under the official artifact hash, computes no global analysis, regenerates no technical plot, and accesses no test data. Widget values exist transiently in the active Streamlit session, but project code does not write or log them outside that session. A clear fallback leaves the unchanged probability visible if the safe background or explainer cannot load.

## Output, Class, and Explainer Configuration

- Output: `positive_class_probability` (`model_output="probability"`)
- Positive class: `1`; the fitted classes must be exactly `[0, 1]`
- Explainer: `TreeExplainer`
- Perturbation: `interventional`
- Background: explicit `shap.maskers.Independent`, max samples = 256
- Requested/effective background: 256/256
- Feature order: the exact 21-column `src.data.FEATURE_COLUMNS` contract
- Expected/base value shape: scalar
- Contribution shape: `n x 21`

The base value is called the **model reference estimate**. It is the explainer's expected model output over the declared aggregate background; it is not presented as the risk of an average person.

## Additivity and Mathematical Fidelity

| Scope | Rows | Base | Output | Max error | Mean error | Tolerance | Passes |
|---|---|---|---|---|---|---|---|
| global_calibration_sample_aggregate | 5000 | aggregate | aggregate | 0.0000000132 | 0.0000000025 | 0.0001000000 | True |
| low_risk_young_healthy | 1 | 0.1717210829 | 0.0030013847 | 0.0000000028 | 0.0000000028 | 0.0001000000 | True |
| high_risk_cardiac_smoker | 1 | 0.1717210829 | 0.6000009431 | 0.0000000113 | 0.0000000113 | 0.0001000000 | True |
| high_risk_poor_health | 1 | 0.1717210829 | 0.6999879501 | 0.0000000079 | 0.0000000079 | 0.0001000000 | True |
| high_risk_severe_obesity_cardiac | 1 | 0.1717210829 | 0.7990007167 | 0.0000000078 | 0.0000000078 | 0.0001000000 | True |

The probability served to users is compared directly with the additive reconstruction in the same mathematical space. No margin contribution is presented as a probability contribution.

## Reproducibility

- Project seed: 42; used by the global calibration sampler, not by the deterministic centroid construction
- Numerical reproducibility tolerance: `1e-10`
- Background repeat maximum difference: `0.0`
- Global sample repeat exact match: `True`
- Global contribution repeat maximum difference: `0.0`
- Global importance repeat maximum difference: `0.0`
- Local contribution repeat maximum difference: `0.0`

The artifact byte hash, dependency versions, feature order, background asset metadata, sample sizes, and tolerances are recorded in `configuration.json`. The evidence command fails if any additivity, privacy, size, or artifact-identity guard fails.

## Performance and Memory

- Explainer creation: 3.0701 s (limit 5.00 s)
- One warm local explanation: 0.0160 s (limit 0.25 s)
- Global 5,000-row explanation: 19.3882 s (limit 60.00 s)
- Repeated global run: 18.6533 s
- Approximate incremental process peak: 0.000 MiB (limit 512 MiB)
- Python-tracked peak during the accepted run: 4.212 MiB

Memory is an approximate process/`tracemalloc` observation, not a platform-independent allocator proof. The app loads only the model, SHAP package, and 256 x 21 aggregate asset; the offline 5,000-row analysis is never run in Streamlit.

## Global Importance

The primary ranking is mean absolute SHAP contribution over the fixed stratified calibration sample. It measures contribution magnitude in this model/sample/background contract; it does not provide a universal ranking or direction.

| Rank | Feature | Label | Mean absolute pp |
|---|---|---|---|
| 1 | GenHlth | General health (self-rated) | 5.2725 |
| 2 | BMI | Body mass index (BMI) | 4.2224 |
| 3 | HighBP | High blood pressure | 3.5904 |
| 4 | Age | Age group (BRFSS) | 3.0844 |
| 5 | HighChol | High cholesterol | 2.6558 |
| 6 | Income | Annual household income | 1.1737 |
| 7 | Sex | Sex | 0.8454 |
| 8 | MentHlth | Days of poor mental health in the past 30 days | 0.5725 |
| 9 | HeartDiseaseorAttack | Coronary heart disease or heart attack | 0.4138 |
| 10 | DiffWalk | Serious difficulty walking or climbing stairs | 0.3426 |

See `global_importance.csv`, `global_importance_bar.png`, and `global_beeswarm.png`. The CSV contains only 21 aggregate feature rows; no individual calibration row or row-level SHAP matrix is published.

## Four Synthetic Reference Profiles

| Profile | Probability | Display | Reference | Max error | Largest increase | Largest decrease |
|---|---|---|---|---|---|---|
| low_risk_young_healthy | 0.003001 | 0.3% | 0.171721 | 0.0000000028 | MentHlth | GenHlth |
| high_risk_cardiac_smoker | 0.600001 | 60.0% | 0.171721 | 0.0000000113 | BMI | Smoker |
| high_risk_poor_health | 0.699988 | 70.0% | 0.171721 | 0.0000000079 | GenHlth | PhysHlth |
| high_risk_severe_obesity_cardiac | 0.799001 | 79.9% | 0.171721 | 0.0000000078 | GenHlth | Income |

Each `waterfall_*.png` and the corresponding rows in `local_contributions.csv` use the exact synthetic inputs in `tests/reference_profiles.py`. Binary, age-group, education, and income codes are translated with the same pure label source used by Streamlit. The four served displays remain 0.3%, 60.0%, 70.0%, and 79.9%.

## Privacy

- Raw train, calibration, and test rows are not written under `docs/`, `models/`, or app assets.
- The background asset contains only 256 multi-row arithmetic means and strict non-match evidence.
- `global_importance.csv` is aggregate; `additivity_checks.csv` has one aggregate global row plus the four allowed synthetic-profile checks.
- `local_contributions.csv` contains only the four already-public synthetic profiles.
- Global plots communicate distributions without publishing the source feature matrix or per-row contribution table.
- Streamlit does not log or persist submitted values in project code.

## Limitations and Association vs. Causality

SHAP allocates a fitted model output under a chosen background. It does not establish medical causes, prevention, diagnosis, or an intervention effect. Contributions inherit model error, BRFSS self-report limitations, selection and measurement bias, correlations among features, and background dependence. Correlated inputs can share or redistribute attribution. A local explanation applies only to one model estimate; global importance applies only to the fixed calibration sample. Fairness conclusions require the separate P12 audit, and scenario exploration belongs to P10.

## Reproduction

From the pinned Python 3.12 environment and with the documented raw CSV present:

```text
python -m src.explainability
python -m pytest tests/test_explainability.py -v -p no:cacheprovider
python -m pytest tests/test_app.py tests/test_reference_profiles.py -v -p no:cacheprovider
python -m pytest tests -v -p no:cacheprovider
```

## Generated Files

- `report.md`
- `configuration.json`
- `spike_results.json`
- `global_importance.csv`
- `local_contributions.csv`
- `additivity_checks.csv`
- `global_importance_bar.png`
- `global_beeswarm.png`
- `waterfall_low_risk_young_healthy.png`
- `waterfall_high_risk_cardiac_smoker.png`
- `waterfall_high_risk_poor_health.png`
- `waterfall_high_risk_severe_obesity_cardiac.png`
