import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from collections import Counter

# =========================
# 1. LOAD DATA
# =========================

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

test_ids = test["trustii_id"].copy()

X = train.drop(columns=["OUTCOME SEVERITY", "OUTCOME MACE", "ID"], errors="ignore")
y_sev  = train["OUTCOME SEVERITY"].values
y_mace = train["OUTCOME MACE"].values

X_test = test[X.columns]

# =========================
# 2. IMPUTATION
# =========================

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# =========================
# 3. PRS (BEST VERSION – UNCHANGED)
# =========================

snp_cols = [c for c in X.columns if c.startswith("SNP")]
priority_snps = [c for c in snp_cols if int(c[3:]) <= 75]

prs_model = lgb.LGBMClassifier(
    n_estimators=900,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbose=-1
)

prs_model.fit(X_imp[priority_snps], y_sev)

importance = pd.Series(prs_model.feature_importances_, index=priority_snps)
top_snps = importance.sort_values(ascending=False).head(70).index.tolist()

weights = np.sqrt(importance[top_snps])
weights /= weights.sum()

X_imp["PRS"] = X_imp[top_snps].values @ weights.values
X_test_imp["PRS"] = X_test_imp[top_snps].values @ weights.values

# =========================
# 4. FEATURES
# =========================

sev_features = ["PRS", "Age_Baseline", "Genre"]

mace_features = ["PRS"] + top_snps + [
    "Age_Baseline", "Genre",
    "Epaiss_max", "Gradient",
    "FEVG", "TVNS", "SYNCOPE"
]

# =========================
# 5. SEVERITY MODEL (UNCHANGED)
# =========================

scaler = StandardScaler()
Xs = scaler.fit_transform(X_imp[sev_features])
Xs_test = scaler.transform(X_test_imp[sev_features])

sev_model = LogisticRegression(
    solver="liblinear",
    class_weight={0: 1.5, 1: 1.0},
    max_iter=4000
)

sev_model.fit(Xs, y_sev)
sev_pred = sev_model.predict_proba(Xs_test)[:, 1]

# =========================
# 6. MACE MODELS
# =========================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_imp[mace_features],
    y_mace,
    test_size=0.25,
    stratify=y_mace,
    random_state=42
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=1600,
    learning_rate=0.015,
    num_leaves=63,
    min_child_samples=25,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_tr, y_tr)

et_model = ExtraTreesRegressor(
    n_estimators=800,
    min_samples_leaf=5,
    max_features=0.6,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_tr, y_tr)

cb_model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    verbose=False,
    random_seed=42
)
cb_model.fit(X_tr, y_tr)

# =========================
# 7. WEIGHT SEARCH + BOOTSTRAP KMEANS
# =========================

best_qwk = -1
best_weights = None

lgb_val = lgb_model.predict(X_val)
et_val  = et_model.predict(X_val)
cb_val  = cb_model.predict(X_val)

def consensus_kmeans(scores, n_runs=25):
    all_preds = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=3, random_state=seed, n_init=10)
        clusters = km.fit_predict(scores.reshape(-1, 1))

        order = (
            pd.DataFrame({"c": clusters, "s": scores})
            .groupby("c")["s"].mean()
            .sort_values()
            .index
        )
        mapping = {order[0]: 0, order[1]: 1, order[2]: 2}
        preds = np.vectorize(mapping.get)(clusters)
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    final = np.apply_along_axis(
        lambda x: Counter(x).most_common(1)[0][0],
        axis=0,
        arr=all_preds
    )
    return final

for w1 in np.arange(0.3, 0.6, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 <= 0:
            continue

        scores = w1 * lgb_val + w2 * et_val + w3 * cb_val
        preds = consensus_kmeans(scores)

        qwk = cohen_kappa_score(y_val, preds, weights="quadratic")
        if qwk > best_qwk:
            best_qwk = qwk
            best_weights = (w1, w2, w3)

print("🔥 BEST QWK (val):", round(best_qwk, 4))
print("🔥 BEST WEIGHTS:", best_weights)

# =========================
# 8. FINAL TEST PREDICTION
# =========================

test_scores = (
    best_weights[0] * lgb_model.predict(X_test_imp[mace_features]) +
    best_weights[1] * et_model.predict(X_test_imp[mace_features]) +
    best_weights[2] * cb_model.predict(X_test_imp[mace_features])
)

mace_pred = consensus_kmeans(test_scores)

# =========================
# 9. SUBMISSION
# =========================

submission = pd.DataFrame({
    "trustii_id": test_ids,
    "OUTCOME MACE": mace_pred,
    "OUTCOME SEVERITY": sev_pred
})

submission.to_csv("submission_KMEANS_BOOTSTRAP.csv", index=False)
print(" submission_KMEANS_BOOTSTRAP.csv created")
