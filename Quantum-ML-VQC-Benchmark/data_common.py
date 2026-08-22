import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42

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


def get_all_data():
    data = {}
    for name, cfg in CONFIGS.items():
        X, y, raw = load_and_clean(name, cfg)
        data[name] = dict(X=X, y=y, raw=raw, cfg=cfg)
    return data


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
