# Cardi-Hack — Risk Prediction Pipeline (HCM Outcomes)

This repository contains a **reproducible end-to-end pipeline** to train models on the provided challenge dataset and generate a valid submission CSV.

The solution predicts two targets:

- **OUTCOME SEVERITY**: a probability score (continuous output).
- **OUTCOME MACE**: a 3-class label `{0, 1, 2}` produced via an ensemble + clustering-based discretization.

---

## 1) Project structure

Recommended structure (matches the scripts/notebook assumptions):

```
Cardihack/
├─ data/
│  ├─ train.csv
│  └─ test.csv
├─ main.py                      # training + inference + submission generation
├─ notebooks/
│  └─ BestScore_Notebook.ipynb  # same pipeline with explanations / experimentation
├─ submissions/
│  └─ submission_KMEANS_BOOTSTRAP.csv  # output (created after running)
├─ README.md
├─ poster.pdf # poster of our project
└─ requirements.txt             # optional, create from instructions below
```

> If your files are arranged differently, either move them into this structure or adjust file paths in `main.py`.

---

## 2) Environment setup

### Create a fresh virtual environment (recommended)

**Windows (PowerShell):**
```powershell
cd C:\path\to\Cardihack
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS/Linux:**
```bash
cd /path/to/Cardihack
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

Install the required packages:

```bash
pip install pandas numpy scikit-learn lightgbm catboost
```

> Notes  
> - `catboost` runs on CPU by default.  
> - `lightgbm` installation on Windows sometimes requires Visual C++ build tools. If you hit install issues, try:
>   - `pip install lightgbm --prefer-binary`

---

## 3) Data placement (required)

Place the competition files in:

```
data/train.csv
data/test.csv
```

The code expects the following columns in `train.csv`:
- `OUTCOME SEVERITY`
- `OUTCOME MACE`
- `ID` (or `trustii_id` depending on the competition export)

And the following identifier column in `test.csv`:
- `trustii_id`

If your dataset uses `ID` instead of `trustii_id` in the test set, update this line in `main.py`:

```python
test_ids = test["trustii_id"].copy()
```

to:

```python
test_ids = test["ID"].copy()
```

---

## 4) How to run the full pipeline (train + predict + submission)

From the repository root:

```bash
python main.py
```

### What this does

1. Loads `data/train.csv` and `data/test.csv`
2. Median-imputes missing values
3. Builds a **PRS (polygenic risk score)** using a LightGBM classifier on a prioritized set of SNPs
4. Trains a **severity model** (logistic regression) and produces `OUTCOME SEVERITY` probabilities for the test set
5. Trains three models for MACE (LightGBM + ExtraTrees + CatBoost)
6. Searches ensemble weights on a validation split
7. Converts ensemble scores into `{0,1,2}` classes via a **bootstrap-consensus KMeans discretization**
8. Writes a final submission file

### Output

The script writes:

- `submission_KMEANS_BOOTSTRAP.csv`

in the current working directory (or wherever you set it).  
A typical console run ends with:

```
 BEST QWK (val): <value>
 BEST WEIGHTS: (<w1>, <w2>, <w3>)
 submission_KMEANS_BOOTSTRAP.csv created
```

---

## 5) Submission format

The generated CSV follows:

| Column | Type | Description |
|---|---:|---|
| `trustii_id` | int | test sample identifier |
| `OUTCOME MACE` | int | predicted class in `{0,1,2}` |
| `OUTCOME SEVERITY` | float | predicted probability (0–1) |

Example (first rows):
```csv
trustii_id,OUTCOME MACE,OUTCOME SEVERITY
1,1,0.1268
2,0,0.3286
...
```

---

## 6) Method summary

### 6.1 Preprocessing
- **Median imputation** for all columns (`SimpleImputer(strategy="median")`)
- No feature removal beyond excluding target/ID columns

### 6.2 PRS (Polygenic Risk Score)
- SNP columns detected with: `SNP*`
- Prioritized SNPs: `SNP1` … `SNP75`
- LightGBM classifier is trained on `priority_snps` vs `OUTCOME SEVERITY`
- Top SNPs are selected by feature importance
- PRS is computed as a weighted sum:
  - `weights = sqrt(importance)` normalized to sum to 1
  - `PRS = X[top_snps] @ weights`

### 6.3 OUTCOME SEVERITY model
- Logistic regression on:
  - `PRS`, `Age_Baseline`, `Genre`
- Features standardized with `StandardScaler`
- Output: `predict_proba[:,1]` as continuous severity score

### 6.4 OUTCOME MACE model (ensemble + discretization)
- Three regressors are trained on selected features:
  - LightGBMRegressor
  - ExtraTreesRegressor
  - CatBoostRegressor
- The pipeline searches a small grid of ensemble weights `(w1,w2,w3)` on a held-out validation split
- Final continuous score is mapped to 3 ordered classes using:
  - Multiple KMeans runs (different seeds)
  - Majority-vote consensus of cluster labels
  - Cluster ordering by mean score to map → `{0,1,2}`

---

## 7) Reproducibility

- Seeds are set for:
  - train/validation split
  - LightGBM / ExtraTrees / CatBoost
  - KMeans consensus runs
- For tighter determinism across machines, also fix:
  - number of threads for BLAS/OpenMP
  - catboost deterministic settings (optional)

---

## 8) Troubleshooting

### 8.1 “KeyError: trustii_id”
Your test file likely uses `ID` instead.
Fix:
```python
test_ids = test["ID"].copy()
```

### 8.2 “LightGBM install failed on Windows”
Try:
```bash
pip install lightgbm --prefer-binary
```

### 8.3 “CatBoost is slow”
Reduce:
```python
iterations=1200
```
to e.g. `600` for a faster run.

### 8.4 “Permission denied writing submission”
Run from a writable directory, or change the output path:
```python
submission.to_csv("submissions/submission_KMEANS_BOOTSTRAP.csv", index=False)
```
(make sure the folder exists)

---

## 9) Run via notebook

Open:

`notebooks/BestScore_Notebook.ipynb`

and run all cells top-to-bottom.  
The notebook mirrors `main.py` but is structured for experimentation and visualization.

---

## 10) Notes

This solution prioritizes stability and interpretability.

More complex approaches (deep learning, ordinal neural nets) were tested but did not outperform this pipeline under submission constraints.

The KMeans consensus step was critical for achieving the final score.

---

