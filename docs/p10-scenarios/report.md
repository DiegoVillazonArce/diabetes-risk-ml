# P10 Model Scenario Explorer Technical Report

## Status and Scope

P10 is **Done**. US-0605, US-0610, and US-0611 are complete. Reviewed implementation commit `fb50ed9` was pushed, the existing Streamlit deployment was updated, and mandatory public frontend verification passed on 2026-07-15.

The explorer is a constrained model-sensitivity comparison. It keeps the frozen D-016 `HistGradientBoostingClassifier`, schema-version-2 P8 artifact, `calibration_method = none`, D-019 probability-only policy, and P9 explanation contract unchanged. It compares one submitted profile with one hypothetical variant; it does not estimate an intervention, future health outcome, treatment effect, or achievable change in a person's real diabetes risk.

## Evidence Sources and Mapping

The working CSV is the cleaned 21-feature binary file published in the [Kaggle Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data). Its publisher states that the features are either BRFSS questions or calculated variables derived from participant responses. The semantic audit therefore uses the authoritative [CDC 2015 BRFSS data and documentation index](https://www.cdc.gov/brfss/annual_data/annual_2015.html), the [2015 BRFSS codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf), the [2015 calculated-variable specification](https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf), and the [calculated-variable summary matrix](https://www.cdc.gov/brfss/annual_data/2015/Summary_Matrix_15_version12.html).

The cleaned file maps CDC no/yes categories to project codes `0/1` and omits unknown/refused/missing responses. Project-valid domains are not redefined here: `src.data.VALUE_RANGES` remains the executable source of truth, and `src.feature_labels` remains the user-facing label/value source. `FEATURE_COLUMNS` remains the only ordering contract.

## D-023: Complete 21-Feature Semantic and Safety Audit

| Technical name | Exact BRFSS meaning represented by the cleaned feature | Project coding / valid values | Temporal nature | Reasonable reversibility | Risk if presented as an action | Decision and justification |
|---|---|---|---|---|---|---|
| `HighBP` | Adult has been told by a doctor, nurse, or other health professional that they have high blood pressure; the CDC calculated variable treats pregnancy-only and borderline responses as no. | `0` no, `1` yes | Historical diagnosis / current context | Not directly reversible as a present input choice | Very high: could imply changing a diagnosis or stopping care | **Exclude.** Diagnostic history is not a safe scenario control. |
| `HighChol` | Adult has had cholesterol checked and was told by a doctor, nurse, or other health professional that it was high. | `0` no, `1` yes | Historical diagnosis contingent on testing | Not directly reversible as a present input choice | Very high: could imply changing a diagnosis or test result | **Exclude.** Diagnostic/test history is not an action. |
| `CholCheck` | Blood cholesterol checked within the past five years; the source distinguishes checked, not checked in five years, and never checked before cleaned binary recoding. | `0` no check in past five years, `1` check in past five years | Historical health-care use | Changes only as time and care history change | High: can become a screening recommendation and reflects access | **Exclude.** It is a historical access/care event, not model-only behavior suitable for this UI. |
| `BMI` | Computed body mass index from reported weight in kilograms divided by reported height in metres squared; the cleaned file uses whole BMI units. | Integer `12`-`98` | Current-ish derived body measure | Can vary over time, but is not a discrete behavior or action | Very high: invites weight prescription, stigma, implausible jumps, or unsafe interpretation | **Exclude.** Candidate rejected: a body measure is not a controllable action, and the full model range is unsuitable as a neutral single-step scenario control. |
| `Smoker` | Respondent has smoked at least 100 cigarettes in their entire life (five packs); it is not current smoking status. | `0` no, `1` yes | Lifetime history | No: once true it cannot become false | Very high: a false-to-true/true-to-false switch misstates history and can be read as cessation advice | **Exclude.** Mandatory historical exclusion; present behavior cannot reverse this field. |
| `Stroke` | Respondent was ever told they had a stroke. | `0` no, `1` yes | Lifetime clinical event | No reasonable present reversal | Very high: implies changing a past medical event | **Exclude.** Historical diagnosis/event. |
| `HeartDiseaseorAttack` | Respondent ever reported coronary heart disease or myocardial infarction; CDC `_MICHD` combines either history. | `0` no, `1` yes | Lifetime clinical event | No reasonable present reversal | Very high: implies changing a past medical event | **Exclude.** Historical diagnosis/event. |
| `PhysActivity` | During the past month, outside the regular job, respondent participated in any physical activity or exercise, including examples such as walking, gardening, or running. | `0` no, `1` yes | Recent 30-day behavior | Yes, over a new observation window, without promising any health result | Moderate: can sound like exercise advice unless framed strictly as an input state | **Include.** The recent binary behavior is understandable and reasonably reversible; neutral model-behavior wording and no causal interpretation are mandatory. |
| `Fruits` | Calculated indicator that reported total fruit consumption (fruit plus fruit juice source items) was at least once per day versus less than once per day. | `0` less than once/day, `1` at least once/day | Recent/current dietary-frequency context derived from survey responses | Yes, over a new observation window | Moderate: can sound like dietary advice or a nutritional target | **Include.** The binary frequency state is understandable and reversible; the UI must show only the exact survey frequency and never prescribe intake. |
| `Veggies` | Calculated indicator that reported total vegetable consumption across the BRFSS vegetable source items was at least once per day versus less than once per day. | `0` less than once/day, `1` at least once/day | Recent/current dietary-frequency context derived from survey responses | Yes, over a new observation window | Moderate: can sound like dietary advice or a nutritional target | **Include.** Same constraints as `Fruits`; it is a survey-frequency scenario, not a health recommendation. |
| `HvyAlcoholConsump` | CDC heavy-drinking calculated variable: adult men reporting more than 14 drinks per week or adult women reporting more than 7 drinks per week, derived from past-30-day alcohol frequency and average drinks. | `0` does not meet the calculated threshold, `1` meets it | Recent calculated behavior, sex-dependent | Potentially, over a new observation window | Very high: a toggle can imply that one side is a medically safe limit, and its meaning depends on `Sex` | **Exclude.** Candidate rejected: sensitive sex-dependent threshold and high risk of implicit safety/recommendation claims outweigh UI value. |
| `AnyHealthcare` | Has any kind of health-care coverage, including insurance, prepaid plans, Medicare, or Indian Health Service. | `0` no, `1` yes | Current contextual/access variable | Sometimes changeable, but usually not a direct individual action | High: can blame the user for structural access and imply a recommendation outside model scope | **Exclude.** Health-care access/context. |
| `NoDocbcCost` | In the past 12 months, needed to see a doctor but could not because of cost. | `0` no, `1` yes | Historical 12-month access event | Not directly reversible for the recorded period | Very high: structural financial barrier, not a health action | **Exclude.** Historical access and socioeconomic context. |
| `GenHlth` | Self-rated general health: excellent, very good, good, fair, or poor. | `1` excellent, `2` very good, `3` good, `4` fair, `5` poor | Current subjective summary | Not a concrete action; may change but cannot be directly selected safely | Very high: invites users to declare an improved health outcome | **Exclude.** Subjective health assessment, not an intervention input. |
| `MentHlth` | Number of days in the past 30 days when mental health, including stress, depression, and emotional problems, was not good. | Integer `0`-`30` | Recent subjective outcome | Not a direct action and may be clinically sensitive | Very high: could imply choosing away mental-health symptoms | **Exclude.** Subjective/clinical outcome. |
| `PhysHlth` | Number of days in the past 30 days when physical health, including illness and injury, was not good. | Integer `0`-`30` | Recent subjective outcome | Not a direct action | Very high: could imply choosing away illness or injury | **Exclude.** Subjective health outcome. |
| `DiffWalk` | Has serious difficulty walking or climbing stairs. | `0` no, `1` yes | Current functional limitation | Not reasonably or safely reversible as a direct choice | Very high: ableist framing or implied treatment outcome | **Exclude.** Functional limitation, not an action. |
| `Sex` | Sex of respondent; the cleaned project encoding is female/male. | `0` female, `1` male | Demographic/contextual | Not a permitted improvement lever | Very high: sensitive immutable attribute and discriminatory framing | **Exclude.** Mandatory demographic exclusion. |
| `Age` | Reported age collapsed into 13 five-year groups: 18-24 through 75-79, then 80 or older. | Integer codes `1`-`13`, labeled by `AGE_GROUP_LABELS` | Current demographic context that progresses with time | Not reversible | Very high: immutable/progressive demographic lever | **Exclude.** Mandatory demographic exclusion. |
| `Education` | Highest grade or year of school completed, collapsed into six levels from no school/kindergarten through college graduate. | Integer codes `1`-`6`, labeled by `ORDINAL_VALUE_LABELS` | Historical socioeconomic context | May change over long periods, not a present health action | High: socioeconomic prescription and confounding | **Exclude.** Mandatory contextual exclusion. |
| `Income` | Annual household income from all sources, collapsed into eight bands from under $10,000 through $75,000 or more. | Integer codes `1`-`8`, labeled by `ORDINAL_VALUE_LABELS` | Current socioeconomic context | Potentially changeable but not a direct or equitable action | Very high: structural socioeconomic factor presented as self-improvement | **Exclude.** Mandatory contextual exclusion. |

### D-023 Resolution — Accepted

The accepted ordered whitelist is:

```text
PhysActivity, Fruits, Veggies
```

All three fields are binary and keep the `0/1` domains already defined by `VALUE_RANGES`. Labels and displayed values must come from `src.feature_labels`; P10 introduces no duplicate ranges, encodings, or general feature labels. Every other field is excluded. Inclusion authorizes only a hypothetical model-input state, never a recommended change.

## D-024: Deterministic Comparison Contract

**Status: Accepted after the pure engine and its focused tests passed, before Streamlit integration.**

`src/scenarios.py` freezes this contract:

- A comparison contains one original profile and one hypothetical profile.
- At most **one field** may be supplied in the `changes` mapping. It must be one of the three D-023 fields. An empty mapping is the valid no-change case.
- Input schema: a validated artifact bundle, an original mapping containing exactly the 21 `FEATURE_COLUMNS`, and a zero-or-one-entry change mapping.
- Output schema: immutable `ScenarioComparison` structure containing read-only `MappingProxyType` views over ordered, independent copies of the original and hypothetical profiles and effective differences (`ScenarioChange` old/new values), plus both probabilities and the signed percentage-point delta. Neither the frozen dataclass attributes nor any returned mapping can be modified by a caller.
- Both probabilities go through `predict_risk_probability`; no parallel scoring implementation exists.
- Sign convention: `delta_percentage_points = 100 * (hypothetical_probability - original_probability)`. Positive, negative, and zero are model-output directions only.
- Absolute numerical tolerance for contract tests: `1e-12`.
- No-change behavior: empty changes or a supplied value equal to the original yields identical profiles/probabilities, no effective changes, and delta exactly `0.0`.
- Reset behavior: `reset_scenario_profile` validates the submitted profile and returns a new mapping with exactly all 21 original values in `FEATURE_COLUMNS` order.
- Validation reuses `validate_input_values`; missing/extra original fields and missing, non-numeric, non-finite, fractional, or out-of-range scenario values are rejected. The engine separately rejects unknown and D-023-excluded change keys and more than one supplied field.
- The engine exposes no scenario search, optimization, ranking, preset, recommendation, threshold, high/low category, SHAP, data-read, fit, artifact-write, persistence, or external-logging path.

Focused evidence on Python 3.12.7 after review hardening: `tests/test_scenarios.py` collected and passed **44/44** tests in 4.10 seconds. Coverage includes caller-input immutability, attempted-write rejection for all returned mappings, exact reset/order, zero identity, every approved field, representative exclusions plus unknown keys, invalid value classes, direct P8-scoring equality, two-helper-call verification, signed positive/negative/zero deltas, determinism, finiteness, and source guards. The only warning was joblib's environment-specific physical-core detection fallback; it does not affect outputs.

## D-025: Streamlit Delivery and Communication

**Status: Accepted from the following pre-integration review.**

The review was completed before changing `app/streamlit_app.py`:

- **Placement and progressive disclosure:** render only after a valid submitted prediction. Preserve the existing original probability and P9 explanation first, then show a distinct “Model scenario explorer” section.
- **One controlled input:** a selector offers “No input change” plus only the three imported D-023 fields. A second binary selector uses `format_feature_value`; no code, range, or label is duplicated in the app.
- **Symmetric output:** show `Original estimate`, `Hypothetical scenario`, and signed `Difference` with equal metric treatment and no directional color, success/error state, arrow, threshold, or high/low category. The same sentence template handles positive, negative, and zero deltas.
- **Effective difference and reset:** show the exact label and old/new understandable value for the one effective change, or explicitly say none. “Reset to original” clears the scenario selection and reconstructs the exact validated 21-field baseline.
- **Communication:** state visibly that the comparison describes model behavior only, is not causal, does not forecast a real health effect from changing an input, and is not medical advice, diagnosis, or a recommendation. The existing medical disclaimer remains visible.
- **P9 separation:** call the existing explanation path only with the original submitted profile and probability. Do not call SHAP or any explanation helper with the hypothetical profile.
- **Failure behavior:** catch a scenario validation/scoring error inside the scenario section, show a controlled warning/details expander, and leave the already-rendered original probability, P9 explanation, and disclaimer unchanged. In contrast, if a newly submitted original profile cannot be scored, invalidate the prior saved result and its scenario state so no earlier probability can appear to belong to the failed submission.
- **Privacy/state:** retain the latest validated original profile only in Streamlit's active in-memory session so scenario-widget reruns have a stable baseline. Store the exact model-artifact SHA-256 with that result and verify it on every rerun; a mismatch invalidates the result and requires resubmission. Widget/baseline state is replaced by the next valid submission and is never written, externally logged, sent to analytics, or exposed as an artifact. This remains the same transient active-session boundary documented by P9.
- **Runtime:** the path loads no CSV and performs exactly two one-row serving calls. A 100-run measurement against the official artifact observed median `0.006492450` seconds, p95 `0.007815600` seconds, and maximum `0.634011100` seconds (the first cold call and joblib CPU-detection warning); subsequent calls remained well below interactive latency.
- **Closure boundary:** local/headless invalid paths remain required, while only the healthy path is exercised publicly so the deployed application is not deliberately broken. Implementation commit `fb50ed9` and the successful public verification completed US-0611 and P10 on 2026-07-15.

The integration implements this accepted policy in `app/streamlit_app.py`. A generation-scoped widget key resets browser-visible state on every valid new submission and on the explicit reset, while the saved original result remains stable for ordinary scenario reruns only when its stored SHA-256 matches the currently loaded artifact. `tests/test_app.py` covers progressive disclosure, the exact whitelist, neutral output, reset, new-submission reset, failed-original invalidation, artifact-hash invalidation, controlled scenario failure, original P9 preservation, disclaimer preservation, and prohibited runtime/language paths.

## Verification Record

All commands used `.venv\Scripts\python.exe`; every pytest run used `-p no:cacheprovider` and a `--basetemp` path under the operating-system temporary directory, outside the repository.

| Check | Result |
|---|---|
| `tests/test_scenarios.py` after review hardening | **44 passed** in 4.10 seconds; one joblib physical-core detection warning. |
| `tests/test_app.py tests/test_reference_profiles.py` after review hardening | **57 passed** in 12.62 seconds; 15 dependency/environment warnings. |
| `tests/test_artifacts.py tests/test_explainability.py tests/test_reference_profiles.py` | **125 passed** in 36.21 seconds; 15 dependency/environment warnings. |
| Complete suite after all code and review changes | **317 passed** in 48.82 seconds; 15 warnings (14 pre-existing Matplotlib/PyParsing deprecations and one environment-specific joblib physical-core fallback). |
| `python -m pip check` | Exit 0: `No broken requirements found.` |
| `git diff --check` | Exit 0; no whitespace errors. Git emitted only informational LF-to-CRLF working-copy warnings. |

### Reference Probability Regression

The official bundle returned the unchanged probabilities and displays:

| Reference profile | Probability | Display |
|---|---:|---:|
| `low_risk_young_healthy` | `0.0030013847190189` | **0.3%** |
| `high_risk_cardiac_smoker` | `0.6000009431177805` | **60.0%** |
| `high_risk_poor_health` | `0.6999879500512149` | **70.0%** |
| `high_risk_severe_obesity_cardiac` | `0.7990007166974580` | **79.9%** |

### Artifact Integrity

Neither reviewed artifact was regenerated or modified. Final SHA-256 values:

- `models/diabetes_risk_model.joblib`: `957c14ff5a490bbc60822121a889f92ee2a6a20f797eef741a710d887ecc9216`
- `models/shap_background_v1.json`: `73d1ff21e3c98ee79fa7d72758517047f13e5f454d7ff95edb1ee93812cca120`

### Local Browser Review

A real local Streamlit server at `http://localhost:8501/` was exercised through the in-app browser; this was not a public smoke test.

- The no-change state showed equal original/hypothetical values and `0.00 pp`, with no effective changes.
- On the unchanged 79.9% reference profile, `PhysActivity: No -> Yes` displayed **79.9% / 80.3% / +0.38 pp**; `Fruits: No -> Yes` displayed **79.9% / 79.0% / -0.90 pp**. Both used the same neutral hierarchy and sentence structure.
- Reset visibly returned the selector to “No input change,” restored **79.9% / 79.9% / 0.00 pp**, and stated that all 21 inputs were preserved.
- The original P9 reference estimate, chart/list content, scenario disclaimer, and medical disclaimer remained visible. Labels and long safety text wrapped without overlap or clipping at the default desktop viewport.
- An initial review found that a new form submission could retain a stale browser-visible scenario label even though the engine had reset. Generation-scoped widget keys fixed it; the browser retest showed “No input change,” and a dedicated headless regression now covers the transition.
- The controlled scenario-error fallback was exercised safely in the headless app test rather than by deliberately breaking the live local page. It preserved the original probability, P9 explanation, and disclaimer.
- Follow-up headless checks confirmed that a failed new original prediction clears the prior result and that changing the artifact SHA-256 invalidates the saved result before P9 or P10 renders.
- Browser console error log: empty.

### Public Deployment Verification

Reviewed implementation commit `fb50ed9` was pushed to `main` and deployed through the existing Streamlit Community Cloud application on 2026-07-15. The user confirmed the planned frontend case set. An independent public healthy-path check additionally confirmed:

- The public application started and produced a valid original estimate.
- The P9 explanation remained visible and tied to the original submitted profile.
- The P10 explorer appeared only after prediction and offered exactly `PhysActivity`, `Fruits`, and `Veggies`, plus the no-change option.
- Selecting an approved field produced the original/hypothetical comparison, effective-change text, neutral model-sensitivity language, and the existing medical disclaimer.
- The browser console reported no errors.

Invalid-artifact, failed-original, explanation-error, and scenario-error paths remain covered locally/headlessly instead of deliberately damaging the public deployment.

## Files and Closure

P10 created `src/scenarios.py`, `tests/test_scenarios.py`, and this report. It modified `app/streamlit_app.py`, `tests/test_app.py`, `tests/test_reference_profiles.py`, `README.md`, `docs/backlog.md`, `docs/decisions.md`, `docs/iteration-log.md`, `docs/ml-analysis-plan.md`, and `docs/roadmap.md`.

The implementation shipped in commit `fb50ed9` without regenerating either reviewed artifact. Following deployment and successful public verification, US-0611 and P10 moved to Done on 2026-07-15. P11-P13 remain Future.
