import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr, ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

SEED = 42
TOP_K = 7
TARGET = "Solubility"

# ------------------------------------------------------------------
# 1. Load data, drop ID, encode string columns via dtype (category codes)
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
# 2. Feature-selection methods: each returns features ranked best -> worst
# ------------------------------------------------------------------
def rank_rf(X, y):
    m = RandomForestRegressor(random_state=SEED).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)

def rank_xgb(X, y):
    m = XGBRegressor(random_state=SEED).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)

def rank_shap(model, X, y):
    model.fit(X, y)
    Xs = X.sample(n=min(100, len(X)), random_state=SEED)   # 100 random instances
    sv = shap.TreeExplainer(model).shap_values(Xs)
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)

def rank_rf_shap(X, y):
    return rank_shap(RandomForestRegressor(random_state=SEED), X, y)

def rank_xgb_shap(X, y):
    return rank_shap(XGBRegressor(random_state=SEED), X, y)

def rank_gmm(X, y):
    # Fit a 2-component GMM per feature; score = separation of component means
    # relative to the pooled std (larger = more bimodal / informative structure)
    scores = {}
    for c in X.columns:
        v = X[[c]].values
        g = GaussianMixture(n_components=2, random_state=SEED).fit(v)
        mu = g.means_.ravel()
        sd = np.sqrt(g.covariances_.ravel())
        scores[c] = abs(mu[0] - mu[1]) / (sd.sum() + 1e-12)
    return pd.Series(scores).sort_values(ascending=False)

def rank_hvgs(X, y):
    # Highly variable gene selection: rank by dispersion (variance / mean)
    mean = X.mean().abs() + 1e-12
    disp = X.var() / mean
    return disp.sort_values(ascending=False)

def rank_spearman(X, y):
    s = {c: abs(spearmanr(X[c], y)[0]) for c in X.columns}
    s = pd.Series(s).fillna(0)
    return s.sort_values(ascending=False)

def rank_dea(X, y):
    # Differential expression analysis: split samples into low/high target groups
    # (median split), then t-test per feature; rank by smallest p-value
    high = y > y.median()
    pv = {}
    for c in X.columns:
        _, p = ttest_ind(X.loc[high, c], X.loc[~high, c], equal_var=False)
        pv[c] = p
    pv = pd.Series(pv).fillna(1.0)
    return pv.sort_values(ascending=True)

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
# 3. Leakage-free CV: feature selection is performed inside each training fold
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
# 4. Run all methods
# ------------------------------------------------------------------
rows = []
for name, (rank_fn, model_kind) in METHODS.items():
    print(f"Running {name} ...")

    # Selection on the full feature set -> top 7 (reported features)
    ranking_full = rank_fn(X_full, y)
    top7 = list(ranking_full.index[:TOP_K])

    # Cross-validated R2 with selection inside folds (no leakage)
    mean, lo, hi = cv_r2(rank_fn, model_kind, X_full, y)

    # Remove the highest-ranked feature -> reduced dataset, re-fit & re-select top 6
    top_feature = ranking_full.index[0]
    X_red = X_full.drop(columns=[top_feature])
    ranking_red = rank_fn(X_red, y)
    top6 = list(ranking_red.index[:6])

    rows.append({
        "method": name,
        "CV7 R2": f"{mean:.3g}",
        "95%CI": f"[{lo:.3g}, {hi:.3g}]",
        "top 7 features": ", ".join(top7),
        "top 6 features": ", ".join(top6),
    })

result = pd.DataFrame(rows)
result.to_csv("result.csv", index=False)
print("\n", result.to_string(index=False))