# RML Capstone Project — Group 4
## HMDA 2024 Mortgage Loan Fairness Evaluation

---

## Overview

This notebook implements a comprehensive **fairness audit** of a binary classification model trained on the 2024 Home Mortgage Disclosure Act (HMDA) Loan/Application Records (LAR) dataset. The goal is to predict whether a mortgage loan application will be **approved (1)** or **denied (0)** and evaluate whether the model produces fair outcomes across protected demographic groups.

The project applies principles from DNSC 6330: Responsible Machine Learning, including bias measurement, transparency, and governance frameworks grounded in U.S. anti-discrimination law.

---

## Dataset

The dataset file `2024_lar.txt` is not included in this repository due to its large size (~5GB).

**Download:** [HMDA 2024 LAR Data](https://www.consumerfinance.gov/data-research/hmda/)

**Data Dictionary:** [Public LAR Dictionary](https://www.consumerfinance.gov/data-research/hmda/resources/static/data-submission/public-lar-schema.pdf)

**After downloading:** Place `2024_lar.zip` in your Google Drive under `My Drive/` and update the zip path in the notebook accordingly.

**Dataset statistics:**
- Raw rows: ~12.2 million
- After filtering (action_taken 1, 2, 3): ~8.7 million rows
- Columns: 99

---

## Label Construction

| action_taken value | Meaning | Label |
|---|---|---|
| 1 | Loan originated | 1 (Approved) |
| 2 | Approved but not accepted | 1 (Approved) |
| 3 | Application denied | 0 (Denied) |
| 4, 5, 6, 7, 8 | Other (withdrawn, incomplete, etc.) | Filtered out |

---

## Environment Setup

This notebook is designed to run on **Google Colab Pro** with **A100 High RAM**.

### Requirements
- Google Colab Pro (A100 GPU + High RAM recommended)
- Google Drive with `2024_lar.zip` uploaded
- Python 3.10+

### Libraries Installed Automatically
```
duckdb
lightgbm
solas-ai-disparity
scikit-learn
pandas
numpy
matplotlib
scipy
```

### How to Run
1. Upload this notebook to Google Colab
2. Go to **Runtime → Change runtime type → A100 GPU + High RAM**
3. Run **Cell 0** to mount Google Drive
4. Run all cells **top to bottom** in order

---

## Feature Selection

### Model A — Fairness-Aware (Excludes Race)

| Feature | Type | Description |
|---|---|---|
| `income` | Numeric | Applicant annual income |
| `tract_minority_population_percent` | Numeric | % minority residents in census tract |
| `combined_loan_to_value_ratio` | Numeric | Loan amount / property value |
| `property_value` | Numeric | Estimated property value |
| `dti_group` | Categorical | Debt-to-income ratio group |
| `loan_term` | Categorical | Loan duration |
| `loan_type` | Categorical | Conventional, FHA, VA, etc. |
| `loan_purpose` | Categorical | Purchase, refinance, etc. |
| `applicant_credit_score_type` | Categorical | Credit scoring model used |

### Model B — Diagnostic (Includes Race)

All features from Model A plus `derived_race` — used for comparison only, not deployment.

---

## Modeling Design

Two model configurations and two algorithms are used, producing four models total:

| Model | Features | Algorithm | Purpose |
|---|---|---|---|
| `model_A` | No race | Logistic Regression | Fairness-aware baseline |
| `model_B` | With race | Logistic Regression | Diagnostic comparison |
| `model_A_gbt` | No race | LightGBM | Fairness-aware (flexible) |
| `model_B_gbt` | With race | LightGBM | Diagnostic comparison |

**Why LightGBM instead of sklearn GBT?**
LightGBM is a Gradient Boosted Tree implementation optimized for large datasets. It uses the same core algorithm as sklearn's GradientBoostingClassifier but is 10–20x faster at the scale of 12 million rows, making it the industry standard choice for large tabular data.

**Why two model configurations?**
Comparing Model A and Model B tests whether explicitly including race changes predictions. If results are similar, it suggests proxy variables in Model A are already carrying the racial signal, a key finding for fairness analysis.

---

## Notebook Structure

### Pre-Modeling
| Step | Description |
|---|---|
| 1. Data Inspection | Load data via DuckDB, inspect shape and columns |
| 2. Target Construction | Filter action_taken and create binary label |
| 3. Data Cleaning | Replace NA/Exempt, drop high missingness columns |
| 4. Baseline Fairness | Compare approval rates across race, sex, ethnicity before modeling |
| DTI Processing | Convert debt_to_income_ratio to numeric groups |

### Modeling
| Step | Description |
|---|---|
| 5. Feature Selection | Define featuresA and featuresB |
| 6. Train-Test Split | 80/20 stratified split |
| 7. Logistic Regression | Train Model A and B, compute FPR/FNR by race |
| 8. LightGBM | Train Model A and B, compute FPR/FNR by race |
| 9. Comparison | Compare LR vs GBT across Model A and B |
| 10. Save Models | Save all 4 models to Google Drive |
| 11. Interpretation | Key insights from modeling stage |

### Fairness Evaluation
| Step | Metric | Description |
|---|---|---|
| 1 | Results Table | Build predictions + attach demographics |
| 2 | Helper Functions | FPR, FNR, Accuracy per group |
| 3 | Equalized Odds (Race) | FPR/FNR by race, LR vs GBT |
| 4 | Equalized Odds (Sex) | FPR/FNR by sex |
| 5 | Equalized Odds (Ethnicity) | FPR/FNR by ethnicity |
| 6 | AIR | Adverse Impact Ratio, EEOC 80% Rule |
| 7 | Proxy Variable | tract_minority_population_percent quartile analysis |
| 8 | Intersectionality | Race × Sex subgroup analysis |
| 9 | SMD | Standardized Mean Difference on probability scores |
| 10 | ME + Z-Test | Marginal Effect with two-proportion z-test |
| 11 | Calibration | Calibration curves and Brier scores by race |
| 12 | Summary | Full summary across all dimensions |

---

## Fairness Metrics Reference

### 1. Equalized Odds (Hardt et al., 2016)
Equal TPR and FPR across groups.
- **FNR** = FN / (FN + TP) — under-approval rate for truly approved applicants
- **FPR** = FP / (FP + TN) — over-approval rate for truly denied applicants

### 2. Adverse Impact Ratio (AIR)
```
AIR = P(Ŷ=1 | group) / P(Ŷ=1 | reference group)
```
EEOC 80% Rule: AIR ≥ 0.80. Groups below 0.80 are flagged for adverse impact.

### 3. Marginal Effect (ME) + Z-Test
```
ME = P(Ŷ=1 | group) - P(Ŷ=1 | reference group)
```
Paired with two-proportion z-test for statistical significance. Both practical and statistical significance are reported (Hall et al., 2020).

### 4. Standardized Mean Difference (SMD)
```
SMD = (S̄_group - S̄_ref) / sqrt((σ²_group + σ²_ref) / 2)
```
Applied to continuous probability scores. Cohen's d thresholds: < 0.2 Negligible, ≥ 0.2 Small, ≥ 0.5 Medium, ≥ 0.8 Large.

### 5. Calibration
```
P(Y=1 | S=s, A=1) ≈ P(Y=1 | S=s, A=0)
```
Tests whether predicted probabilities carry equal meaning across groups. Evaluated via calibration curves and Brier scores.

### 6. Intersectionality (Crenshaw, 1989)
Race × Sex subgroup combinations. Reports worst-group AIR and FNR. Captures compounded disadvantage invisible in single axis analysis.

---

## Reference Groups

| Dimension | Reference Group |
|---|---|
| Race | White |
| Sex | Male |
| Ethnicity | Not Hispanic or Latino |
| Intersectional | White / Male |

---

## Key Findings

1. **Equalized Odds fails** — Racial disparities in FNR and FPR persist across both LR and GBT versions of Model A, confirming disparate impact even without the explicit race variable.

2. **AIR violations** — Multiple racial groups fall below the EEOC 80% threshold, constituting prima facie adverse impact under the burden shifting framework (Griggs v. Duke Power, 1971).

3. **ME is statistically significant** — Negative ME values for minority groups are not due to sampling noise, confirmed by two proportion z-tests.

4. **SMD reveals score-level bias** — Medium-to-large SMD values indicate that predicted probability distributions are shifted for certain groups before any threshold is applied.

5. **Calibration differs across groups** — The model is not equally well-calibrated for all racial groups, consistent with the Impossibility Theorem (Chouldechova, 2017).

6. **Intersectional disadvantage confirmed** — Groups such as Black/Female and AI/AN Female face compounded disadvantage not visible in single-axis analysis.

7. **Proxy discrimination active** — `tract_minority_population_percent` acts as a proxy for race. Approval rates decline monotonically from Q1 to Q4 (lowest to highest minority tract), confirming indirect bias.

8. **Consistent across models** — Results are robust across both LR and LightGBM, confirming findings are not model-specific.

---

