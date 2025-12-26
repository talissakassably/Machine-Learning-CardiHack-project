# Cardiac Outcome Prediction

## Project Status
This project focuses on predicting two specific medical outcomes based on clinical and genetic data:

- **OUTCOME SEVERITY**
- **OUTCOME MACE (Major Adverse Cardiovascular Events)**

---

## 1. Description of the Data

The dataset consists of **clinical features** and **genetic markers (SNPs)**.

**Target Variables:**

- **OUTCOME SEVERITY:** Binary classification  
- **OUTCOME MACE:** Ordinal classification (3 levels: 0, 1, 2)

**Features:**  

- Demographic data: `Age_Baseline`, `Genre`  
- Clinical measurements: `Epaiss_max`, `Gradient`, `FEVG`, `TVNS`, `SYNCOPE`  
- Genetic markers: `SNP_xxx` series  

---

## 2. Tested Workflows

### Preprocessing
- **Imputation:** Missing values handled using median imputation via `SimpleImputer`.  
- **Scaling:** `StandardScaler` applied to SNP data before feature selection to ensure uniform contribution in linear models.  

### Feature Selection & Engineering
- **SNP Filtering:** Initially restrict genetic data to "priority SNPs" (index ≤ 75).  
- **Consensus Ranking:** A hybrid feature selection combining:
  - Logistic Regression (L1 or L2 coefficient importance)  
  - LightGBM (Gini/Gain importance)  
  - The top 40 SNPs are selected based on the normalized average of these two rankings.  
- **Polygenic Risk Score (PRS):** A custom PRS feature is engineered by calculating the dot product of the top 40 SNPs and their respective Logistic Regression coefficients.  

### Algorithms
Two distinct modeling paths are used for the two targets:

- **Severity Model:** `LGBMClassifier` with a custom `class_weight` (1.5 for class 0) to handle class imbalance.  
- **MACE Model:** Treated as an ordinal regression problem. `LGBMRegressor` predicts a continuous score, which is later discretized.  

### Postprocessing (Threshold Tuning)
To optimize the **Quadratic Weighted Kappa (QWK)** for the MACE outcome:

- A validation set (25%) is used to find thresholds `t1` and `t2`.  
- Continuous regressor output is converted to classes:  
  - 0 if score < t1  
  - 1 if t1 ≤ score < t2  
  - 2 if score ≥ t2  

---

## 3. Performance Metrics
- **Quadratic Weighted Cohen’s Kappa** is used to evaluate the alignment of predicted MACE categories with the actual values during threshold tuning.
