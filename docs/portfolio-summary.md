# Portfolio Summary

Tiered, evidence-backed descriptions of the diabetes-risk-ml project for
different audiences. Every figure below traces to versioned evidence in this
repository (linked per claim). The project is educational: it is not a medical
device, not clinically validated, and makes no fairness certification.

## One-sentence summary

An end-to-end, reproducible machine-learning portfolio project that estimates
self-reported diabetes/prediabetes risk from 21 BRFSS 2015 health indicators and
serves it through a deployed Streamlit app with local SHAP explanations, a
constrained what-if explorer, in-memory batch scoring, an offline fairness audit,
and a documented offline-to-serving architecture.

## CV bullets

- Built a full ML lifecycle on the BRFSS 2015 dataset (253,680 rows, ~13.9%
  positive): reproducible data contract and stratified splits, model comparison,
  and selection of a `HistGradientBoostingClassifier` (test PR-AUC ≈ 0.423,
  ROC-AUC ≈ 0.827) — see [decisions.md D-016](decisions.md) and
  [modeling code](../src/modeling.py).
- Added honest probability work without overclaiming: a pre-registered
  out-of-fold calibration comparison that selected *no* post-hoc calibrator
  (D-018) and a documented threshold trade-off analysis that keeps the product
  probability-only (D-019) — see [p8-calibration/report.md](p8-calibration/report.md).
- Shipped explainability and safe interactivity: SHAP local explanations
  (additivity error ≈ 1.3e-08), a one-field model-behavior scenario explorer, and
  a privacy-safe in-memory batch workflow (≤ 2 MiB, ≤ 1,000 rows) — see
  [p9-explainability/report.md](p9-explainability/report.md),
  [p10-scenarios/report.md](p10-scenarios/report.md), and
  [p11-batch/report.md](p11-batch/report.md).
- Engineered for trust and reproduction: a trusted repository-supplied model
  artifact with recorded SHA-256 and post-load semantic validation; 469 passing
  local tests (441 pass + 28 explicit skips in the final no-data clean clone);
  a least-privilege CI workflow; and a public Streamlit deployment — see
  [architecture.md](architecture.md).

## Short recruiter explanation

This project takes a public health-survey dataset and builds a complete, honest
machine-learning product around it. A gradient-boosting model is trained offline
and frozen; a deployed web app then lets anyone enter 21 answers and see an
estimated probability, along with a plain-language explanation of which answers
moved the estimate. The work emphasizes responsibility over hype: it shows a
probability rather than a diagnosis, documents where the model is uncertain,
includes a fairness audit that reports group differences honestly, and keeps
uploaded data in memory only. The whole pipeline is reproducible from a clean
checkout and is documented decision-by-decision.

## Technical interview narrative

The project is organized as reproducible phases with a decision log. Data flows
through one contract (`src/data.py`): fixed column order, validated ranges,
`uint8` downcasting, and a stratified 70/10/20 split that preserves prevalence
(duplicates kept, D-014). Model comparison (`src/modeling.py`) evaluates a Dummy
baseline, Logistic Regression (plus a class-weighted variant), and a
`HistGradientBoostingClassifier`, selecting the tree model on PR-AUC/ROC-AUC
(D-016).

Probability quality is handled carefully: the calibration method was chosen by a
pre-registered out-of-fold protocol with a paired-bootstrap adoption rule, and
because neither sigmoid nor isotonic beat the uncalibrated baseline, the contract
is `calibration_method = none` (D-018). A threshold trade-off analysis is
documented but the product stays probability-only (D-019). The serving contract
is a schema-version-2 `joblib` bundle validated on every load
(`validate_artifact_bundle`) against model class, hyperparameters, feature order,
classes, and recorded vs. runtime package versions.

The Streamlit app (`app/streamlit_app.py`) is a thin, offline-free consumer: it
loads the committed artifact (cached by SHA-256), never trains or reads the raw
CSV, and exposes individual prediction, SHAP explanations from a privacy-safe
256-centroid background, a whitelisted one-field scenario explorer, an in-memory
batch workflow, and a static architecture overview. Session state is transient
and hash-bound, so a failed rescore, an artifact change, or an upload replacement
cannot show a stale result. Offline, a report-first fairness audit
(`src/fairness.py`) measures subgroup probability/ranking/calibration/threshold
metrics with bootstrap intervals across 22 predeclared cohorts. Quality gates
include a 469-test local suite and a clean-clone CI workflow (D-034); the artifact trust
boundary and a `joblib`-vs-`skops` evaluation are documented (D-035).

## Principal results

All figures link to versioned evidence.

| Result | Value | Evidence |
|---|---|---|
| Selected model | `HistGradientBoostingClassifier` (D-016) | [decisions.md](decisions.md) |
| Test PR-AUC / ROC-AUC (selection) | ≈ 0.423 / ≈ 0.827 | [decisions.md D-016](decisions.md) |
| Calibration outcome | `none` (no method qualified) | [p8-calibration/report.md](p8-calibration/report.md) |
| Whole-test-cohort Brier / log loss | 0.0974 / 0.3144 | [p12-fairness/report.md](p12-fairness/report.md) |
| SHAP additivity max error | ≈ 1.3e-08 | [p9-explainability/report.md](p9-explainability/report.md) |
| Reference profile displays | 0.3%, 60.0%, 70.0%, 79.9% | [reference_profiles.py](../tests/reference_profiles.py) |
| Test suite | 469 passed locally; 441 passed + 28 CSV-gated skips in the final clean clone | [architecture.md](architecture.md) |
| Fairness audit scope | 22 cohorts, 50,736 test rows, 5,000 bootstrap resamples | [p12-fairness/report.md](p12-fairness/report.md) |

## Responsible limitations

- **Educational, not medical:** the output is a probability from a model trained
  on self-reported survey data — not a diagnosis, screening result, or medical
  advice (D-004).
- **Self-reported historical data:** BRFSS 2015 labels can reflect diagnosis
  access and reporting bias; the target is not clinically confirmed.
- **Descriptive fairness audit only:** the P12 audit reports group differences
  with uncertainty; it does not establish causes, discrimination, or fairness,
  and cannot judge whether any single estimate is fair (D-031). The processed
  data has only a binary `Sex` field and ordinal age/income groups, with no
  race/ethnicity variable.
- **Model-behavior explanations:** SHAP contributions and scenario deltas
  describe how this fitted model responds, not health mechanisms; they are not
  causal and not recommendations.
- **Trusted-artifact assumption:** the app loads only its reviewed,
  repository-committed artifact. Its SHA-256 is recorded and used to bind
  dependent state/assets, and its serving contract is validated only after
  deserialization; this is not a safe loader for untrusted artifacts
  (D-013, D-035).

## Future work (explicitly separate)

The following are deliberately **out of scope** for this project and would each
require their own planned phase:

- Fairness mitigation (reweighting, group thresholds) or additional demographic
  coverage.
- A `skops` artifact-format migration (evaluated and deferred in D-035).
- A lockfile-based dependency workflow (e.g. `uv`).
- Any clinical validation, richer datasets, or production-scale reliability work.
- Accounts, persistence, analytics, or external logging.
