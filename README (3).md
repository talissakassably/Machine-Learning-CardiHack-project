# Multimodal Risk Stratification for Hypertrophic Cardiomyopathy

This repository contains the complete pipeline developed for the Cardi-Hack Data Challenge organized by IHU ICAN on Trustii.

The objective is to predict:
- OUTCOME SEVERITY (binary classification)
- OUTCOME MACE (three-class ordinal risk)

using multimodal clinical, imaging, and genomic (SNP) data.

---

## 1. Method Overview

The pipeline is composed of four main components:

### 1.1 Polygenic Risk Score (PRS)
A LightGBM classifier is trained on biologically prioritized SNPs (SNP1–SNP75).  
Top features are selected based on feature importance, and a weighted Polygenic Risk Score is constructed.

### 1.2 Severity Model
A logistic regression model predicts the probability of severe outcome using:
- PRS
- Age at baseline
- Sex

### 1.3 MACE Ensemble
Three regression models are trained:
- LightGBM Regressor
- ExtraTrees Regressor
- CatBoost Regressor

Their predictions are combined using an optimized weighted ensemble.

### 1.4 Ordinal Risk Discretization
A bootstrap consensus K-Means clustering (K=3) converts continuous risk scores into stable ordinal classes corresponding to MACE severity.

---

## 2. Repository Structure

```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── main.py
├── submission_KMEANS_BOOTSTRAP.csv
└── README.md
```

---

## 3. Installation

Python 3.10+

```bash
pip install numpy pandas scikit-learn lightgbm catboost
```

---

## 4. Running the Pipeline

```bash
python main.py
```

The script performs:
1. Data loading and median imputation
2. PRS construction
3. Severity model training
4. MACE ensemble training
5. Weight optimization
6. Bootstrap K-Means clustering
7. Submission file generation

Output file:
```
submission_KMEANS_BOOTSTRAP.csv
```

---

## 5. Reproducibility

- Fixed random seeds
- Stratified validation splits
- Deterministic tree-based models

---

## 6. Computational Profile

| Stage | Typical Runtime (CPU) |
|------|-----------------------|
| PRS training | ~2 s |
| Ensemble training | ~5–7 s |
| Clustering | ~1 s |
| Inference | <1 s |

Memory usage: <1.5 GB  
GPU not required.

---

## 7. Authors

Master 1 GENIOMHE-AI  
Université d'Évry Paris-Saclay

---

## 8. License

Academic and research use only.
