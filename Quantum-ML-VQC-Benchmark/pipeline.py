import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import chi2, binom

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
RESULTS_DIR = "/home/claude/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATASET CONFIGS
# ---------------------------------------------------------------------------
CONFIGS = {
    "Diabetes": {
        "path": "/content/diabetes.csv",
        "target": "Outcome",
        "num": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
        "cat": [],
        "zero_as_missing": ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"],
    },
    "Heart Disease": {
        "path": "/content/heart.csv",
        "target": "target",
        "num": ["age","trestbps","chol","thalach","oldpeak"],
        "cat": ["sex","cp","fbs","restecg","exang","slope","ca","thal"],
        "zero_as_missing": [],
    },
    "Liver Disease": {
        "path": "/content/Indian Liver Patient Dataset (ILPD).csv",
        "target": "is_patient",
        "num": ["age","tot_bilirubin","direct_bilirubin","tot_proteins","albumin","ag_ratio","sgpt","sgot","alkphos"],
        "cat": ["gender"],
        "zero_as_missing": [],
    },
    "CKD": {
        "path": "/content/kidney_disease.csv",
        "target": "classification",
        "num": ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"],
        "cat": ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"],
        "zero_as_missing": [],
    },
}

# ---------------------------------------------------------------------------
# 2. LOAD + CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(name, cfg):
    df = pd.read_csv(cfg["path"])
    df.columns = [c.strip() for c in df.columns]
    df = df.replace(["?", "\t?", "\t", " ", "nan", ""], np.nan)

    if name == "Diabetes":
        for c in cfg["zero_as_missing"]:
            df[c] = df[c].replace(0, np.nan)
        y = df[cfg["target"]].astype(int)

    elif name == "Heart Disease":
        y = df[cfg["target"]].astype(int)

    elif name == "Liver Disease":
        # gender text -> keep as categorical; target already 1/2 -> map to 1=patient,0=healthy
        y = df[cfg["target"]].map({1: 1, 2: 0}).astype(int)

    elif name == "CKD":
        for c in cfg["num"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in cfg["cat"]:
            df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        df[cfg["target"]] = df[cfg["target"]].astype(str).str.strip().str.lower()
        y = df[cfg["target"]].map({"ckd": 1, "notckd": 0}).astype(int)

    X = df[cfg["num"] + cfg["cat"]].copy()
    for c in cfg["num"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y, df

DATA = {}
for name, cfg in CONFIGS.items():
    X, y, raw = load_and_clean(name, cfg)
    DATA[name] = dict(X=X, y=y, raw=raw, cfg=cfg)
    print(name, X.shape, "positives:", int(y.sum()), "/", len(y))

# ---------------------------------------------------------------------------
# 3. DATA QUALITY / CKD LEAKAGE DIAGNOSTICS
# ---------------------------------------------------------------------------
quality_rows = []
for name, d in DATA.items():
    X, y, raw = d["X"], d["y"], d["raw"]
    dup_full = raw.duplicated().sum()
    dup_features = X.duplicated().sum()
    # duplicate feature rows with conflicting labels
    conflict = 0
    if dup_features > 0:
        tmp = X.copy()
        tmp["_y"] = y.values
        grp = tmp.groupby(list(X.columns), dropna=False)["_y"].nunique()
        conflict = int((grp > 1).sum())
    quality_rows.append({
        "Dataset": name,
        "N": len(X),
        "Missing values (total)": int(X.isnull().sum().sum()),
        "Missing values (%)": round(100 * X.isnull().sum().sum() / (X.shape[0]*X.shape[1]), 2),
        "Full-row duplicates": int(dup_full),
        "Feature-only duplicates": int(dup_features),
        "Duplicate groups w/ conflicting label": conflict,
        "Class balance (pos %)": round(100 * y.mean(), 1),
    })
quality_df = pd.DataFrame(quality_rows)
quality_df.to_csv(f"{RESULTS_DIR}/data_quality_and_leakage.csv", index=False)
print("\n=== Data quality / leakage diagnostics ===")
print(quality_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE BUILDER (leakage-safe: fit only inside each fold)
# ---------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    return ColumnTransformer(transformers)

MODELS = {
    "LogisticRegression": (LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
                            {"clf__C": [0.1, 1.0, 10.0]}),
    "SVM": (SVC(probability=True, random_state=RANDOM_SEED),
            {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["rbf", "linear"]}),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_SEED),
                      {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 8]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=RANDOM_SEED),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]}),
}

# ---------------------------------------------------------------------------
# 5. NESTED CV EVALUATION (pre-PCA and post-PCA[4])  -- also stash per-sample
#    predictions across outer folds so we can run McNemar later
# ---------------------------------------------------------------------------
def nested_cv(X, y, num_cols, cat_cols, use_pca, n_outer=5, n_inner=3, n_components=4):
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_SEED)
    fold_rows = []
    oof_preds = {m: np.full(len(y), -1, dtype=int) for m in MODELS}
    pca_variance_rows = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        for model_name, (est, grid) in MODELS.items():
            pre = build_preprocessor(num_cols, cat_cols)
            steps = [("pre", pre)]
            if use_pca:
                steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)))
            steps.append(("clf", est))
            pipe = Pipeline(steps)

            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_SEED)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_
            pred = best.predict(X_te)
            proba = best.predict_proba(X_te)[:, 1] if hasattr(best, "predict_proba") else pred

            oof_preds[model_name][te_idx] = pred

            row = {
                "Model": model_name, "Fold": fold_i,
                "Accuracy": accuracy_score(y_te, pred),
                "F1": f1_score(y_te, pred, zero_division=0),
                "Precision": precision_score(y_te, pred, zero_division=0),
                "Recall": recall_score(y_te, pred, zero_division=0),
                "AUC": roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else np.nan,
                "Best_Params": json.dumps(gs.best_params_),
            }
            fold_rows.append(row)

            # capture PCA variance explained (once per fold, model-independent -> just do for first model)
            if use_pca and model_name == list(MODELS.keys())[0]:
                pre_only = build_preprocessor(num_cols, cat_cols)
                Xt = pre_only.fit_transform(X_tr)
                pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(Xt)
                pca_variance_rows.append({
                    "Fold": fold_i,
                    **{f"PC{i+1}_var": v for i, v in enumerate(pca.explained_variance_ratio_)},
                    "Cumulative_4PC_var": pca.explained_variance_ratio_.sum(),
                })

    return pd.DataFrame(fold_rows), oof_preds, pd.DataFrame(pca_variance_rows) if use_pca else None

all_fold_results = []
oof_store = {}   # dataset -> {"prepca": {...}, "postpca": {...}, "y": array}
pca_variance_store = {}

for name, d in DATA.items():
    X, y, cfg = d["X"], d["y"], d["cfg"]
    print(f"\n=== Nested CV: {name} (pre-PCA, original features) ===")
    fr_pre, oof_pre, _ = nested_cv(X, y, cfg["num"], cfg["cat"], use_pca=False)
    fr_pre["Dataset"] = name; fr_pre["Feature_Space"] = "original"
    all_fold_results.append(fr_pre)

    print(f"=== Nested CV: {name} (post-PCA, 4 components, matched to VQC input) ===")
    fr_post, oof_post, pca_var = nested_cv(X, y, cfg["num"], cfg["cat"], use_pca=True)
    fr_post["Dataset"] = name; fr_post["Feature_Space"] = "pca4"
    all_fold_results.append(fr_post)

    oof_store[name] = {"prepca": oof_pre, "postpca": oof_post, "y": y.values}
    pca_variance_store[name] = pca_var

fold_results_df = pd.concat(all_fold_results, ignore_index=True)
fold_results_df.to_csv(f"{RESULTS_DIR}/classical_ml_fold_results.csv", index=False)

summary_df = (fold_results_df
              .groupby(["Dataset", "Feature_Space", "Model"])[["Accuracy","F1","Precision","Recall","AUC"]]
              .agg(["mean","std"]))
summary_df.columns = ["_".join(c) for c in summary_df.columns]
summary_df = summary_df.reset_index()
summary_df.to_csv(f"{RESULTS_DIR}/classical_ml_summary.csv", index=False)
print("\n=== Classical ML summary (mean +/- sd across 5 outer folds) ===")
print(summary_df.to_string(index=False))

pca_var_all = pd.concat([v.assign(Dataset=k) for k, v in pca_variance_store.items()], ignore_index=True)
pca_var_all.to_csv(f"{RESULTS_DIR}/pca_variance_explained.csv", index=False)
print("\n=== PCA (4-component) variance explained, per fold ===")
print(pca_var_all.to_string(index=False))

import pickle
with open(f"{RESULTS_DIR}/oof_store.pkl", "wb") as f:
    pickle.dump(oof_store, f)

print("\nDone with classical phase.")
