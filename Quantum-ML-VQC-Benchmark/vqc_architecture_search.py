import sys, os, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from vqc2 import train_vqc, predict_proba, scale_to_angles
from data_common import get_all_data, build_preprocessor, RANDOM_SEED

RESULTS_DIR = "/home/claude/results"
DATA = get_all_data()

QUBIT_GRID = [2, 4, 6]
LAYER_GRID = [1, 2, 3]
SEARCH_EPOCHS = 15
SEARCH_FOLDS = 3

target_names = [sys.argv[1]] if len(sys.argv) > 1 else list(DATA.keys())
QUBIT_GRID = [int(q) for q in sys.argv[2].split(",")] if len(sys.argv) > 2 else QUBIT_GRID

out_path = f"{RESULTS_DIR}/vqc_architecture_search.csv"
rows = []
if os.path.exists(out_path):
    rows = pd.read_csv(out_path).to_dict("records")

for name in target_names:
    d = DATA[name]
    X, y, cfg = d["X"], d["y"], d["cfg"]
    t0 = time.time()

    for n_qubits in QUBIT_GRID:
        for n_layers in LAYER_GRID:
            fold_accs, fold_f1s, fold_aucs = [], [], []
            skf = StratifiedKFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            for tr_idx, te_idx in skf.split(X, y):
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te = y.iloc[tr_idx].values.astype(float), y.iloc[te_idx].values.astype(float)

                pre = build_preprocessor(cfg["num"], cfg["cat"])
                Xtr_t = pre.fit_transform(X_tr); Xte_t = pre.transform(X_te)
                if hasattr(Xtr_t, "toarray"):
                    Xtr_t = Xtr_t.toarray(); Xte_t = Xte_t.toarray()

                pca = PCA(n_components=n_qubits, random_state=RANDOM_SEED).fit(Xtr_t)
                Xtr_p = pca.transform(Xtr_t); Xte_p = pca.transform(Xte_t)

                Xtr_ang, xmin, xmax = scale_to_angles(Xtr_p)
                Xte_ang, _, _ = scale_to_angles(Xte_p, xmin, xmax)
                Xte_ang = np.clip(Xte_ang, 0, np.pi)

                weights, hist = train_vqc(Xtr_ang, y_tr, n_layers=n_layers, n_qubits=n_qubits,
                                           epochs=SEARCH_EPOCHS, batch_size=32, lr=0.05, seed=RANDOM_SEED)
                proba = predict_proba(Xte_ang, weights, n_qubits)
                pred = (proba >= 0.5).astype(int)
                fold_accs.append(accuracy_score(y_te, pred))
                fold_f1s.append(f1_score(y_te, pred, zero_division=0))
                fold_aucs.append(roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else np.nan)

            n_params = n_layers * n_qubits * 3
            row = {
                "Dataset": name, "n_qubits": n_qubits, "n_layers": n_layers, "n_params": n_params,
                "Mean_Accuracy": float(np.mean(fold_accs)), "Mean_F1": float(np.mean(fold_f1s)),
                "Mean_AUC": float(np.nanmean(fold_aucs)),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"{name} q={n_qubits} L={n_layers} params={n_params}: "
                  f"acc={row['Mean_Accuracy']:.4f} f1={row['Mean_F1']:.4f} auc={row['Mean_AUC']:.4f}")

    print(f"[{name} architecture search done in {time.time()-t0:.1f}s, saved]\n")

print("Architecture search complete for:", target_names)
