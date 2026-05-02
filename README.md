# Lecture 5 — Privacy & Safety Audit (Adversarial ML)


## Scope

Audited our gradient-boosted classifier (`model_A_gbt`) against three attack classes from the NIST AI 100-2 taxonomy:

1. **PGD evasion** (deployment-time, integrity attack)
2. **Label-flip poisoning** (training-time, integrity + fairness attack)
3. **Membership inference** (deployment-time, privacy attack)

All experiments were executed on the **full HMDA dataset** (~6.93M training rows, ~1.73M test rows) using DuckDB + Parquet for the data load. No subsampling.

## What I built

### Setup (10.1, 10.2)
- Helper functions: `selection_rate_by_group`, `fpr_by_group`, `air` (Adverse Impact Ratio).
- Race groups: privileged = `White` (1.11M test rows), unprivileged = `Black or African American` (153k test rows).
- Clean-model baseline: AIR = 0.891, Train AUC = Test AUC = 0.854.

### Part A — PGD evasion (10.3)
- Implemented a **transfer attack**: trained a substitute logistic regression (attack tool only) on the same preprocessed feature space as `model_A_gbt`, then used PGD (`sign(w)`-step + L∞ projection) to perturb only the numeric features of the test set.
- Swept ε ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} (in feature-σ units).
- Scored perturbed inputs against the GBT target and recorded selection rate, FPR, and AIR by race.
- 3-panel plot: FPR by race, AIR under attack, ΔAIR vs baseline.

### Part B — Label-flip poisoning (10.4)
- `poison_label_flip()` flips approved → denied for a fraction of unprivileged-group training records.
- Sweep over poison rates {0%, 5%, 10%, 20%, 30%}.
- At each rate: rebuild a fresh GBT pipeline (same architecture: `n_estimators=100, max_depth=3`), retrain on the full poisoned 6.93M-row training set, re-evaluate on the clean test set.
- 3-panel poisoning degradation curve: AUC, FPR by race, AIR vs poison rate.

### Part C — Membership inference (10.5)
- Full Shokri et al. (2017) shadow-model pipeline.
- 5 shadow GBTs trained on the full training set with stratified 50/50 splits, same architecture as the target.
- Meta-classifier (decision tree, `max_depth=6`) trained on (max-confidence, member?) pairs from all 5 shadows.
- Applied meta-classifier to `model_A_gbt`'s confidence scores on its own train and test sets.
- ROC curve + confidence-gap histogram.

### Auto-summary (10.6)
- Findings table (severity-rated)
- Per-attack interpretation paragraphs
- Proactive + reactive mitigation recommendations
- Disparate-impact note on mitigation side effects

## Key results

| Attack | Result | Severity |
|---|---|---|
| **PGD evasion** | AIR worst = **0.489 at ε = 0.75**; 4/5ths line crossed at **ε = 0.25** | High (4/5ths violation under attack) |
| **Label-flip poisoning** | AUC flat at 0.854 across all rates; AIR linear decline 0.891 → 0.870 at 30% (115,861 labels flipped) | Medium (stealth pattern present) |
| **Membership inference** | MI AUC = **0.500** (= random); generalization gap = +0.000 | None (model does not leak) |

### Headline findings

- **PGD is the dominant fairness risk.** A modest perturbation (ε = 0.25, a quarter-σ shift in numerical features like income or LTV) is enough to push the deployed GBT into a 4/5ths-rule violation. This works even though GBT has no gradients — the substitute-then-transfer pattern is enough.
- **Poisoning shows the textbook "stealth zone".** AUC stays pinned at 0.854 across all five poison rates, while AIR descends linearly. An AUC-only production monitor would detect zero degradation despite ~115k labels flipped. This is the strongest argument in the audit for adding fairness metrics (selection-rate AIR by race) to production monitoring.
- **MI failure is a positive finding.** The shadow-model attack achieves exactly random performance (AUC = 0.500). This is consistent with the GBT's zero generalization gap (Train AUC = Test AUC = 0.854 to three decimals): there is no confidence asymmetry between training members and non-members for the attacker to exploit. The regularization choices (`max_depth=3`, `n_estimators=100`) on a large training corpus prevent memorization.

## Methodological notes

- **DuckDB + Parquet load.** Data is read once via DuckDB's `read_csv_auto` with `TRY_CAST` to numeric, written as Parquet, and reloaded into pandas. Subsequent runs skip the conversion. This is at the top of the notebook (cells 5–7) and was preserved unchanged through the audit.
- **Feature-typing fix.** Original notebook treated `loan_term` as categorical, which one-hot-encoded to ~470 columns and ballooned the preprocessed matrix to 502 columns (~6.5 GB dense). Moved `loan_term` to numeric (it is a count of months) — matrix shrinks to 36 columns, which lets the full data fit in memory without subsampling.
- **No subsampling.** All three attacks ran on the full HMDA training and test sets per project requirements.
- **Total runtime.** Section 10 takes ~4 hours end-to-end on a single machine (poisoning loop ~3.5 hours for 5 retrains, MI shadow training ~3 hours for 5 shadow GBTs). PGD itself is fast (~1 minute).

## Files

- `capstone_final_full_data.ipynb` — main notebook. My contribution is Section 10 (cells 96–117).
- `2024_lar.txt` — raw HMDA 2024 LAR data (not in repo; download from CFPB).
- `2024_lar_subset.parquet` — cached Parquet built by the DuckDB load cell. Auto-generated on first run.

## Recommended mitigations (from the audit)

- **Proactive (PGD):** validate input feature ranges at the API boundary. Reject applications whose numeric features deviate beyond plausible historical ranges (e.g., income, LTV, or loan term more than 3σ from training median). Highest-impact mitigation in this audit.
- **Proactive (poisoning):** track selection-rate AIR by protected group in production alongside AUC.
- **Proactive (privacy):** maintain current regularization. The MI-resistance observed here depends on the model not overfitting; resist tuning for marginal AUC by deepening trees or adding estimators.
- **Reactive:** monitor the empirical generalization gap on a held-out set as a routine health check.

## References

- NIST AI 100-2e2025 — *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*
- Shokri et al. (2017) — *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P
- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR
- Goodfellow, Shlens & Szegedy (2015) — *Explaining and Harnessing Adversarial Examples*, ICLR
- Bagdasaryan & Shmatikov (2019) — *Differential Privacy Has Disparate Impact on Model Accuracy*, NeurIPS
