# RML-Capstone-project-
## Data
The dataset file `2024_lar.txt` is not included in this repository due to its large size.
Please download the dataset separately and place it in the project folder before running the notebook.

## Current Progress and Approach

### What has been done

So far, the project has completed the **pre-modeling and modeling stages**.

In the pre-modeling stage:
- Data inspection, target construction, and cleaning were performed
- A baseline fairness check was conducted by comparing approval rates across groups (race, ethnicity, sex)

This initial analysis shows that **approval rates differ across racial groups**, indicating the presence of disparities before any modeling.

---

### Feature Selection

The selected features focus on variables that are directly related to loan decisions and financial risk, including:

- income  
- loan_amount  
- debt_to_income_ratio  
- loan_term  
- loan_type  
- loan_purpose  
- interest_rate  
- rate_spread  

The goal is to control for **legitimate financial and risk-related factors**, so that we can evaluate whether disparities still persist after accounting for these variables.

---

### Modeling Design

Two model configurations are constructed:

- **Model A**: excludes race (fairness-aware design)
- **Model B**: includes race (used for comparison and diagnosis)

Two modeling approaches are used:

- **Logistic Regression**: interpretable baseline model  
- **Gradient Boosted Tree**: more flexible model that captures non-linear relationships  

Using both models helps ensure that results are **robust and not dependent on a single modeling approach**.

---

### Current Findings

1. Approval rates differ across racial groups.

2. Income and Debt_To_Income do not explain these differences.

3. Including race in the model (Model B) does not significantly change:
   - accuracy  
   - false positive rate (FPR)  
   - false negative rate (FNR)  

   This suggests that the model does not rely heavily on the explicit race variable.

4. However, disparities still remain.

   This indicates that:
   - other variables may be acting as **proxy variables** for race  
   - bias may be embedded in the data structure rather than driven by a single feature  

5. Results are consistent across Logistic Regression and Gradient Boosted Tree, suggesting that the findings are **robust and not model-specific**.

---

### Next Steps

The remaining work will focus on deeper analysis of model behavior:

**Transparency (LIME / SHAP)**
- Identify which features drive model predictions  
- Detect potential proxy variables  

**Fairness**
- Further examine FPR and FNR across groups  
- Analyze disparities within subgroups  

**Robustness**
- Confirm consistency across models (LR vs GBT)  
- Test sensitivity to feature selection or modeling choices  

---

### Summary

Current results suggest that removing race from the model does not eliminate disparities.  
This highlights the importance of examining indirect sources of bias and understanding how data structure influences model outcomes.