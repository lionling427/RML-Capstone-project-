# Transparency Analysis Part
**HMDA Capstone Project | Group 4 | Member: Hassan Alshamrani**

---

## Overview

This notebook covers the full **Transparency & Explainability** analysis for the 2024 HMDA Loan/Application Records dataset.  
The goal is to explain *why* the model approves or denies mortgage applications — and whether it treats different racial groups differently.

---

## What This Notebook Does

### Tools Applied
| Part | Tool | Purpose |
|---|---|---|
| Part 1 | ICE, c-ICE, d-ICE | How does income affect approval across racial groups? |
| Part 2 | LIME | Why was *this specific applicant* approved or denied? |
| Part 3 | DiCE | What would need to change to flip a denial to approval? |
| Part 4 | SHAP | Which features drive predictions globally and locally? |

### Methodology
Following the class methodology (analogous to Black vs. White defendant in COMPAS), we select:
- One **White** applicant
- One **Black or African American** applicant

Both with similar predicted probabilities (~0.4–0.6), so any differences in the explanations reflect **model behavior**, not differences in financial risk.

---

## Key Findings

1. **DTI ratio is the strongest predictor** — `dti_group_>60%` pushes strongly toward denial
2. **Income alone barely matters** — ICE and d-ICE plots show near-zero marginal effect of raw income
3. **`tract_minority_population_percent` acts as a proxy variable** — appears in SHAP for Black applicants, suggesting geographic discrimination risk (Redlining)
4. **SHAP vs. Confusion Matrix** — confusion matrix only shows accuracy; SHAP reveals *which features* drive denials and *for whom*
5. **LightGBM and LR agree on top features** — findings are robust and model-independent

---

## Technical Decisions

### Why LightGBM instead of sklearn GBT?
- sklearn GBT took **25+ minutes** on 500K rows and crashed on full data
- LightGBM is **10–20x faster**, uses less RAM, achieves same or better accuracy
- Same hyperparameters: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.7537 | 0.8087 |
| LightGBM | 0.8119 | 0.8549 |

### Why I used Google Colab Pro (A100 GPU)?
- Full dataset: **12.2M rows loaded**, **8.6M after filtering**
- Local MacBook Pro (16GB RAM) crashed during full data training
- Colab Pro A100 + High-RAM (~80GB) handled the full dataset comfortably

### Data Sizes
| Stage | Size | Reason |
|---|---|---|
| Training | ~7M rows (full) | Model trained on all available data |
| Test | ~1.7M rows (full) | Evaluation on all data |
| SHAP & LIME | 50K sample | Sufficient for accurate explanations — standard practice |

---

## Thanks All .. 
