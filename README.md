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

- **Demographic data:** `Age_Baseline`, `Genre`  
- **Clinical measurements:** `Epaiss_max`, `Gradient`, `FEVG`, `TVNS`, `SYNCOPE`  
- **Genetic markers:** `SNP_xxx` series  

---

## 2. Tested Workflows

### Preprocessing
- **Imputation:** Missing values handled using median imputation via `SimpleImputer`.  
- **Scaling:** `StandardScaler` applied to SNP data before feature selection for uniform contribution in linear models.  

### Feature Selection & Engineering
- **SNP Filtering:** Genetic data initially restricted to "priority SNPs" (index ≤ 75).  
- **Consensus Ranking:** Combines:
  - **Logistic Regression:** Uses absolute coefficient values for linear importance.  
  - **Tree Method (LightGBM):** Uses Gini/Gain importance for non-linear contributions.  
- **Top Feature Selection:** Top 40 SNPs selected based on the normalized average ranking.  
- **Polygenic Risk Score (PRS):** Custom feature calculated as the dot product of top SNPs and their Logistic Regression coefficients.  

### Algorithms
Two modeling paths are used:

- **Severity Model (Logistic & Tree Hybrid):**
  - `LGBMClassifier` with custom `class_weight` (1.5 for class 0) to handle class imbalance.  
  - Incorporates `PRS` as a primary feature.  

- **MACE Model (Ordinal via Regression):**
  - Treated as an ordinal regression problem using `LGBMRegressor`.  
  - Predicts a continuous score, later discretized into categories.  

### Postprocessing (Threshold Tuning)
- Grid search on a validation set (25%) to find optimal thresholds `t1` and `t2`.  
- Continuous regressor outputs converted to classes:
  - **0** if score < t1  
  - **1** if t1 ≤ score < t2  
  - **2** if score ≥ t2  

---

## 3. Methodological Details

### Logistic Regression
- Provides **interpretability** and models **linear relationships** in high-dimensional genetic data.  
- Supplies weights for **Polygenic Risk Score (PRS)**.  
- Helps prioritize SNPs with strong, direct correlations to outcomes.  

### Tree Methods (LightGBM)
- Captures **non-linear interactions** and clinical complexities.  
- `class_weight` manages imbalances in the Severity model.  
- MACE regression treats ordinal stages (0,1,2) as ordered progression, optimizing **Quadratic Weighted Kappa**.  

---

## 4. Performance Metrics
- **Quadratic Weighted Cohen’s Kappa:** Primary metric for evaluating alignment of predicted MACE categories with actual outcomes during threshold tuning.
