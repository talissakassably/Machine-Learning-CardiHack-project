# Cardihack – Multimodal Risk Prediction (QWK 0.3626)

This repository contains my solution for the Cardihack challenge, focusing on predicting:
- **OUTCOME MACE** (ordinal, 3 classes)
- **OUTCOME SEVERITY** (binary)

The final approach combines:
- Polygenic Risk Score (PRS) construction from SNP data
- Classical clinical features
- Ensemble learning
- Consensus clustering for ordinal prediction

The best public leaderboard score achieved with this pipeline was **0.3626 **.

---

## Method Overview

### 1. Data Preprocessing
- Median imputation for missing values
- SNP features identified via column name pattern
- Separation of clinical and genetic variables

### 2. Polygenic Risk Score (PRS)
- LightGBM classifier trained on top SNPs (SNP1–SNP75)
- Feature importance used to select top 70 SNPs
- PRS computed as a weighted linear combination of SNP values

### 3. Outcome Severity Model
- Logistic Regression
- Input features: PRS, Age_Baseline, Genre
- Standard scaling applied
- Produces probability output

### 4. Outcome MACE Model (Ordinal)
An ensemble of:
- LightGBM Regressor
- ExtraTrees Regressor
- CatBoost Regressor

Each model predicts a continuous risk score.

### 5. Ordinal Prediction via Consensus KMeans
- Weighted ensemble of model scores
- KMeans clustering into 3 ordered risk groups
- Bootstrap consensus across multiple random seeds
- Weights optimized on validation set using Quadratic Weighted Kappa (QWK)

---

##  Final Performance
- **Best Validation QWK:** ~0.36
- **Public Leaderboard QWK:** 0.3626

---
Notes

This solution prioritizes stability and interpretability.

More complex approaches (deep learning, ordinal neural nets) were tested but did not outperform this pipeline under submission constraints.

The KMeans consensus step was critical for achieving the final score.


---


