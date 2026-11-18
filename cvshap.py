import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr, ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

SEED = 42
TOP_K = 7
TARGET = "Solubility"

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("curated-solubility-dataset.csv")
df = df.drop(columns=["ID"])

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("category").cat.codes

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df[TARGET].describe())
print("Skewness:", round(df[TARGET].skew(), 3), "Kurtosis:", round(df[TARGET].kurt(), 3))

X_full = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------------------------------------------------------
# 2. Feature-selection methods
# ------------------------------------------------------------------
def rank_rf(X, y):
    m = RandomForestRegressor(random_state=SEED).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)

def rank_xgb(X, y):
    m = XGBRegressor(random_state=SEED).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)

def rank_shap(model, X, y):
    model.fit(X, y)
    Xs = X.sample(n=min(100, len(X)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)

def rank_rf_shap(X, y):
    return rank_shap(RandomForestRegressor(random_state=SEED), X, y)

def rank_xgb_shap(X, y):
    return rank_shap(XGBRegressor(random_state=SEED), X, y)

def rank_gmm(X, y):
    scores = {}
    for c in X.columns:
        v = X[[c]].values
        g = GaussianMixture(n_components=2, random_state=SEED).fit(v)
        mu = g.means_.ravel()
        sd = np.sqrt(g.covariances_.ravel())
        scores[c] = abs(mu[0] - mu[1]) / (sd.sum() + 1e-12)
    return pd.Series(scores).sort_values(ascending=False)

def rank_hvgs(X, y):
    mean = X.mean().abs() + 1e-12
    disp = X.var() / mean
    return disp.sort_values(ascending=False)

def rank_spearman(X, y):
    s = {c: abs(spearmanr(X[c], y)[0]) for c in X.columns}
    return pd.Series(s).fillna(0).sort_values(ascending=False)

def rank_dea(X, y):
    high = y > y.median()
    pv = {}
    for c in X.columns:
        _, p = ttest_ind(X.loc[high, c], X.loc[~high, c], equal_var=False)
        pv[c] = p
    return pd.Series(pv).fillna(1.0).sort_values(ascending=True)

METHODS = {
    "RF":        (rank_rf,       "RF"),
    "RF-SHAP":   (rank_rf_shap,  "RF"),
    "XGB":       (rank_xgb,      "XGB"),
    "XGB-SHAP":  (rank_xgb_shap, "XGB"),
    "GMM":       (rank_gmm,      "RF"),
    "HVGS":      (rank_hvgs,     "RF"),
    "Spearman":  (rank_spearman, "RF"),
    "DEA":       (rank_dea,      "RF"),
}

def make_model(kind):
    return RandomForestRegressor(random_state=SEED) if kind == "RF" else XGBRegressor(random_state=SEED)

# ------------------------------------------------------------------
# 3. Leakage-free CV
# ------------------------------------------------------------------
class TopKSelector(BaseEstimator, TransformerMixin):
    def __init__(self, rank_fn=None, k=TOP_K):
        self.rank_fn = rank_fn
        self.k = k
    def fit(self, X, y):
        self.cols_ = list(self.rank_fn(X, y).index[: self.k])
        return self
    def transform(self, X):
        return X[self.cols_]

def cv_r2(rank_fn, model_kind, X, y):
    pipe = Pipeline([
        ("sel", TopKSelector(rank_fn=rank_fn, k=TOP_K)),
        ("model", make_model(model_kind)),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    mean = scores.mean()
    ci = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores))
    return mean, mean - ci, mean + ci

# ------------------------------------------------------------------
# NEW: exact position rate
# Compare bottom 6 of top-7 (ranks 2..7) with top-6 of the reduced
# dataset, position by position. Rate = (# identical positions) / 6.
# ------------------------------------------------------------------
def exact_position_rate(top7, top6):
    bottom6_of_top7 = top7[1:]            # drop the #1 feature -> 6 features
    n = len(top6)
    matches = sum(a == b for a, b in zip(bottom6_of_top7, top6))
    return matches / n

# ------------------------------------------------------------------
# 4. Run all methods
# ------------------------------------------------------------------
rows = []
for name, (rank_fn, model_kind) in METHODS.items():
    print(f"Running {name} ...")

    ranking_full = rank_fn(X_full, y)
    top7 = list(ranking_full.index[:TOP_K])

    mean, lo, hi = cv_r2(rank_fn, model_kind, X_full, y)

    top_feature = ranking_full.index[0]
    X_red = X_full.drop(columns=[top_feature])
    ranking_red = rank_fn(X_red, y)
    top6 = list(ranking_red.index[:6])

    epr = exact_position_rate(top7, top6)                 # NEW
    n_exact = int(round(epr * 6))                         # NEW

    rows.append({
        "method": name,
        "CV7 R2": f"{mean:.3g}",
        "95%CI": f"[{lo:.3g}, {hi:.3g}]",
        "top 7 features": ", ".join(top7),
        "top 6 features": ", ".join(top6),
        "exact position rate": f"{epr:.3g}",              # NEW
        "exact position (n/6)": f"{n_exact}/6",           # NEW
    })

result = pd.DataFrame(rows)
result.to_csv("result.csv", index=False)
print("\n", result.to_string(index=False))