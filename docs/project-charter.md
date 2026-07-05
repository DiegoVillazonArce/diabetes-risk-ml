# Project Charter

## Project

**Name:** Diabetes Risk ML

**Type:** Academic portfolio project

**Primary goal:** Build an end-to-end supervised machine learning application that estimates self-reported diabetes or prediabetes risk from health survey indicators, using BRFSS 2015 data.

## Purpose

This project is designed to demonstrate the full lifecycle of a practical ML product:

- Data understanding and quality analysis.
- Reproducible preprocessing and train/calibration/test splits.
- Baseline modeling and model comparison.
- Imbalanced classification evaluation.
- Probability calibration.
- Explainability with SHAP.
- Fairness-oriented subgroup analysis.
- Streamlit deployment with a usable interface.

## Problem Statement

BRFSS contains self-reported health, behavior, access, and demographic indicators. The project uses these indicators to estimate diabetes-related risk as a probability, with formal calibration planned as part of the broader modeling roadmap, while clearly communicating that the result is educational and not diagnostic.

## Target Audience

- Portfolio reviewers and recruiters.
- Technical interviewers.
- Data science learners.
- Non-technical users exploring how ML can support risk awareness.

## In Scope

- Exploratory data analysis and data quality documentation.
- Binary diabetes/prediabetes risk modeling.
- Baseline and candidate model comparison.
- Correct handling of class imbalance.
- Evaluation with ROC-AUC, PR-AUC, recall, precision, F1, Brier score, and confusion matrices.
- Streamlit app for individual prediction.
- Post-MVP model scenario exploration for selected modifiable inputs, framed as model sensitivity rather than medical advice.
- Model limitations and medical disclaimer.
- Reproducible training and serialized model artifacts.

## Out of Scope

- Medical diagnosis.
- Clinical recommendation or treatment advice.
- Replacement for professional healthcare evaluation.
- Real-time model training inside Streamlit.
- Causal claims from SHAP explanations or scenario simulations.
- Prescriptive habit-change recommendations such as "doing X will reduce your real medical risk by Y%".

## Success Criteria

The MVP is successful when:

- The selected dataset and target definition are documented.
- The data split strategy prevents leakage and preserves the original test distribution.
- At least one baseline and two meaningful candidate models are evaluated.
- Metrics are reported with emphasis on imbalanced classification, not accuracy alone.
- A Streamlit MVP can load a trained artifact and return an estimated risk probability, with formal probability calibration planned as a later enhancement if it is not included in the first deployable version.
- The README explains the project clearly enough for a portfolio reviewer.

## Constraints

- The app must remain lightweight enough for Streamlit Community Cloud.
- Raw data remains out of git; large generated artifacts should be committed only when explicitly selected for deployment or documentation.
- The repository-facing documentation should be concise and written in English.
- The original Spanish blueprint remains a private internal planning reference, not the public project documentation.

## Disclaimer

This project is for educational and portfolio purposes only. Predictions do not constitute diagnosis, medical advice, or a clinical screening result. Diabetes diagnosis and treatment decisions must be made by qualified healthcare professionals.
