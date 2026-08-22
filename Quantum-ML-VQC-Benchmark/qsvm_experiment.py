import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from vqc2 import quantum_kernel_matrix, scale_to_angles
from data_common import get_all_data, build_preprocessor, RANDOM_SEED

RESULTS_DIR = "/home/claude/results"
DATA = get_all_data()
N_QUBITS = 4
C_GRID = [0.1, 1.0, 10.0]

rows = []
oof_qsvm = {}

for name, d in DATA.items():
    X, y, cfg = d["X"], d["y"], d["cfg"]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), np.nan)
    t0 = time.time()

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values

        pre = build_preprocessor(cfg["num"], cfg["cat"])
        Xtr_t = pre.fit_transform(X_tr); Xte_t = pre.transform(X_te)
        if hasattr(Xtr_t, "toarray"):
            Xtr_t = Xtr_t.toarray(); Xte_t = Xte_t.toarray()
        pca = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED).fit(Xtr_t)
        Xtr_p = pca.transform(Xtr_t); Xte_p = pca.transform(Xte_t)
        Xtr_ang, xmin, xmax = scale_to_angles(Xtr_p)
        Xte_ang, _, _ = scale_to_angles(Xte_p, xmin, xmax)
        Xte_ang = np.clip(Xte_ang, 0, np.pi)

        Ktr = quantum_kernel_matrix(Xtr_ang, Xtr_ang, N_QUBITS)

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        best_c, best_score = C_GRID[0], -1
        for c in C_GRID:
            scores = []
            for itr, ite in inner.split(Xtr_ang, y_tr):
                Ktr_inner = Ktr[np.ix_(itr, itr)]
                Kte_inner = Ktr[np.ix_(ite, itr)]
                clf = SVC(kernel="precomputed", C=c, probability=False)
                clf.fit(Ktr_inner, y_tr[itr])
                scores.append(accuracy_score(y_tr[ite], clf.predict(Kte_inner)))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score, best_c = mean_score, c

        clf = SVC(kernel="precomputed", C=best_c, probability=True)
        clf.fit(Ktr, y_tr)
        Kte = quantum_kernel_matrix(Xte_ang, Xtr_ang, N_QUBITS)
        pred = clf.predict(Kte)
        proba = clf.predict_proba(Kte)[:, 1]

        oof_pred[te_idx] = pred
        oof_proba[te_idx] = proba

        rows.append({
            "Dataset": name, "Fold": fold_i, "Best_C": best_c,
            "Accuracy": accuracy_score(y_te, pred),
            "F1": f1_score(y_te, pred, zero_division=0),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "AUC": roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else np.nan,
        })

    oof_qsvm[name] = {"pred": oof_pred, "proba": oof_proba, "y": y.values}
    print(f"{name} QSVM done in {time.time()-t0:.1f}s")

df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS_DIR}/qsvm_fold_results.csv", index=False)
summary = df.groupby("Dataset")[["Accuracy","F1","Precision","Recall","AUC"]].agg(["mean","std"])
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary.to_csv(f"{RESULTS_DIR}/qsvm_summary.csv", index=False)
print(summary.to_string(index=False))

import pickle
with open(f"{RESULTS_DIR}/oof_qsvm.pkl", "wb") as f:
    pickle.dump(oof_qsvm, f)
print("QSVM baseline complete and saved.")
