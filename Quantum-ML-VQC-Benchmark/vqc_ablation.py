import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from vqc2 import train_vqc, predict_proba, scale_to_angles
from data_common import get_all_data, build_preprocessor, RANDOM_SEED

RESULTS_DIR = "/home/claude/results"
DATA = get_all_data()
N_QUBITS = 4
N_LAYERS = 2

rows = []
for name, d in DATA.items():
    X, y, cfg = d["X"], d["y"], d["cfg"]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    tr_idx, te_idx = next(iter(outer.split(X, y)))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx].values.astype(float), y.iloc[te_idx].values.astype(float)

    pre = build_preprocessor(cfg["num"], cfg["cat"])
    Xtr_t = pre.fit_transform(X_tr); Xte_t = pre.transform(X_te)
    if hasattr(Xtr_t, "toarray"):
        Xtr_t = Xtr_t.toarray(); Xte_t = Xte_t.toarray()
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED).fit(Xtr_t)
    Xtr_p = pca.transform(Xtr_t); Xte_p = pca.transform(Xte_t)
    Xtr_ang, xmin, xmax = scale_to_angles(Xtr_p)
    Xte_ang, _, _ = scale_to_angles(Xte_p, xmin, xmax)
    Xte_ang = np.clip(Xte_ang, 0, np.pi)

    for entangle in [True, False]:
        t0 = time.time()
        weights, hist = train_vqc(Xtr_ang, y_tr, n_layers=N_LAYERS, n_qubits=N_QUBITS, entangle=entangle,
                                   epochs=40, batch_size=32, lr=0.05, seed=RANDOM_SEED)
        proba = predict_proba(Xte_ang, weights, N_QUBITS, entangle=entangle)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        auc = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else np.nan
        rows.append({"Dataset": name, "Entangled": entangle, "Accuracy": acc, "F1": f1, "AUC": auc,
                      "Final_train_loss": hist[-1]})
        print(f"{name} entangled={entangle}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} ({time.time()-t0:.1f}s)")

pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/vqc_entanglement_ablation.csv", index=False)
print("Entanglement ablation complete and saved.")
