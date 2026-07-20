# P13 Portfolio Package

This directory holds the privacy-safe demo assets for the diabetes-risk-ml
project and the evidence behind the P13 product-polish and packaging phase.
It is a presentation layer over the frozen, publicly verified P0-P12 product; it
changes no model, probability, threshold, or artifact.

## Contents

| File | Purpose |
|---|---|
| `report.md` | This file: audits, decisions, evidence, and reproduction. |
| `assets_manifest.json` | Machine-readable manifest of every screenshot: name, purpose, viewport, synthetic source, date, dimensions, SHA-256, and a no-real-rows confirmation. |
| `01-landing-desktop.png` | Prediction-first landing (desktop). |
| `02-individual-result-desktop.png` | Individual estimate with the safe SHAP explanation (desktop). |
| `03-scenario-comparison-desktop.png` | One-field model-behavior scenario comparison (desktop). |
| `04-batch-workflow-desktop.png` | Valid-plus-mixed in-memory batch workflow (desktop). |
| `05-about-mobile.png` | About & architecture section (narrow/mobile viewport). |

## Audits and decisions (Increment 1)

P13 followed an evidence-before-implementation order. The full audit and spike
evidence is in [decisions.md](../decisions.md); in summary:

- **D-032 (information architecture / UX).** A desktop and narrow-viewport audit
  of the running app (via the accessibility tree) plus a headless `AppTest`
  state/timing spike confirmed the app already keeps individual prediction as the
  default, hides the explanation/scenario until a valid submit, preserves
  hash-bound state across navigation (switch cost 0.065-0.151 s), and reflows
  cleanly to one column at 375 px with no clipped labels, hidden disclaimer, or
  color-only meaning. The accepted change adds a third static `About &
  architecture` section to the existing navigation radio.
- **D-033 (publication contract).** The architecture and asset/claims inventory
  were reviewed before any asset was captured: GitHub carries the technical
  depth, Streamlit stays concise, assets use only synthetic inputs, and every
  claim traces to versioned evidence.
- **D-034 (clean-clone CI).** A `git archive HEAD` clean clone in a fresh
  python.org 3.12.1 virtualenv confirmed the exact CI command chain: pinned
  install, `pip check` clean, `compileall` OK, and `pytest` returning 420
  passed / 28 skipped / 0 failed in ~90 s.
- **D-035 (serialization).** A documented `joblib`-vs-`skops` comparison chose to
  retain the controlled `joblib` contract; a migration is deferred to a separate
  future phase, with no artifact regenerated in P13.

## Asset generation (Increment 3)

All screenshots were captured from the **final local app** (`app/streamlit_app.py`
served on `localhost`) through controlled Chromium browser sessions, using
**only synthetic inputs**:

- The individual and scenario screenshots use a synthetic higher-risk profile
  entered into the form (checkboxes plus BMI and age); no real dataset row is
  used.
- The batch screenshot uses a project-generated CSV built exclusively from
  `src.artifacts.example_input()` (a fixed synthetic case), with two rows
  deliberately made invalid to show mixed-validity handling. It contains no real
  BRFSS or user row.
- The About screenshot is a static informational page with no data.

The final About asset was recaptured after the links/session-privacy wording
polish at a 390 x 812 CSS-pixel viewport. Five overlapping viewport captures
were stitched using their verified scroll offsets, excluding the repeated fixed
browser toolbar from continuation segments; the resulting 390 x 3,163 PNG was
then inspected end to end for continuous text and duplicate/missing regions.

Each PNG was visually inspected for clipped text, overlapping labels,
inconsistent fonts, truncated controls, real data, usernames or local paths, and
console/terminal content before inclusion. The manifest records each file's
SHA-256, pixel dimensions, viewport, and a `contains_real_rows: false`
confirmation.

### Privacy guarantees

- No real BRFSS row, user upload, probability associated with a real/user row,
  target, or split index appears in any asset. Probabilities for the declared
  synthetic demonstrations are intentionally visible.
- The only inputs are the four public synthetic reference profiles, the
  `example_input()` synthetic case, and files built solely from them.
- `tests/test_portfolio.py` checks the manifest, file set, hashes, dimensions,
  text assets, and absence of data files or local-path/target leakage. The
  synthetic provenance of screenshot pixels is established by the documented
  capture inputs plus the recorded manual visual inspection; the test does not
  claim to classify image pixels as real or synthetic.

## Reproduction

The demo assets were captured from the local app (`app/streamlit_app.py` served
on `localhost`) through controlled Chromium sessions, entering only synthetic
inputs, and the manifest is derived from the resulting PNGs. Because screenshots are
display captures, byte-identical regeneration is not expected — the contract is
that every asset is synthetic-only, accessible, and free of real data and local
identifiers. `tests/test_portfolio.py` enforces the structural portion of that
contract (manifest/file-set integrity, SHA-256/dimension agreement, text/file
privacy guards); the documented capture provenance and visual inspection cover
the image content itself.

To re-verify the underlying product behavior shown in the assets:

```
python -m pytest tests/test_app.py tests/test_reference_profiles.py -p no:cacheprovider
python -m tests.reference_profiles   # prints the four synthetic reference cases
```

## Scope and boundaries

P13 is presentation only. It does not retrain, recalibrate, re-threshold, or
regenerate either official artifact; it introduces no accounts, persistence,
analytics, or external logging; and it makes no clinical, causal, or
fairness-certification claim. See [portfolio-summary.md](../portfolio-summary.md)
for the tiered narratives and [architecture.md](../architecture.md) for the
technical architecture.
