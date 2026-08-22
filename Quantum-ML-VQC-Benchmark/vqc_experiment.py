import pickle, json, time, sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA

from vqc import train_vqc, predict_proba, scale_to_angles
from data_common import get_all_data, build_preprocessor, RANDOM_SEED

RESULTS_DIR = "/home/claude/results"
DATA = get_all_data()

N_OUTER = 5
N_LAYERS = 2
EPOCHS = 40
BATCH_SIZE = 32
LR = 0.05

# Which datasets to run this invocation (CLI args), default = all
target_names = sys.argv[1:] if len(sys.argv) > 1 else list(DATA.keys())

vqc_fold_rows = []
oof_vqc = {}

# load any previously saved partial results so we don't lose them
partial_fold_path = f"{RESULTS_DIR}/vqc_fold_results.csv"
partial_oof_path = f"{RESULTS_DIR}/oof_vqc.pkl"
if os.path.exists(partial_fold_path):
    prev_df = pd.read_csv(partial_fold_path)
    vqc_fold_rows = prev_df.to_dict("records")
if os.path.exists(partial_oof_path):
    with open(partial_oof_path, "rb") as f:
        oof_vqc = pickle.load(f)

for name in target_names:
    d = DATA[name]
    X, y, cfg = d["X"], d["y"], d["cfg"]
    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), np.nan)

    t0 = time.time()
    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values.astype(float), y.iloc[te_idx].values.astype(float)

        pre = build_preprocessor(cfg["num"], cfg["cat"])
        Xtr_t = pre.fit_transform(X_tr)
        Xte_t = pre.transform(X_te)
        if hasattr(Xtr_t, "toarray"):
            Xtr_t = Xtr_t.toarray(); Xte_t = Xte_t.toarray()

        pca = PCA(n_components=4, random_state=RANDOM_SEED).fit(Xtr_t)
        Xtr_p = pca.transform(Xtr_t)
        Xte_p = pca.transform(Xte_t)

        Xtr_ang, xmin, xmax = scale_to_angles(Xtr_p)
        Xte_ang, _, _ = scale_to_angles(Xte_p, xmin, xmax)
        Xte_ang = np.clip(Xte_ang, 0, np.pi)  # test set may fall outside train min/max range

        weights, history = train_vqc(Xtr_ang, y_tr, n_layers=N_LAYERS, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, lr=LR, seed=RANDOM_SEED + fold_i)
        proba_te = predict_proba(Xte_ang, weights)
        pred_te = (proba_te >= 0.5).astype(int)

        oof_pred[te_idx] = pred_te
        oof_proba[te_idx] = proba_te

        row = {
            "Dataset": name, "Fold": fold_i,
            "Accuracy": accuracy_score(y_te, pred_te),
            "F1": f1_score(y_te, pred_te, zero_division=0),
            "Precision": precision_score(y_te, pred_te, zero_division=0),
            "Recall": recall_score(y_te, pred_te, zero_division=0),
            "AUC": roc_auc_score(y_te, proba_te) if len(set(y_te)) > 1 else np.nan,
            "Final_train_loss": history[-1],
        }
        vqc_fold_rows.append(row)
        print(f"{name} fold {fold_i}: acc={row['Accuracy']:.4f} f1={row['F1']:.4f} auc={row['AUC']:.4f} "
              f"(train_loss={history[-1]:.4f})")

    oof_vqc[name] = {"pred": oof_pred, "proba": oof_proba, "y": y.values}
    print(f"{name} total VQC time: {time.time()-t0:.1f}s\n")

    # save incrementally after each dataset finishes, so a later timeout doesn't lose this work
    vqc_fold_df = pd.DataFrame(vqc_fold_rows)
    vqc_fold_df.to_csv(partial_fold_path, index=False)
    with open(partial_oof_path, "wb") as f:
        pickle.dump(oof_vqc, f)
    print(f"[saved incremental results after {name}]\n")

vqc_fold_df = pd.DataFrame(vqc_fold_rows)
if set(DATA.keys()).issubset(set(vqc_fold_df["Dataset"].unique())):
    vqc_summary = (vqc_fold_df.groupby("Dataset")[["Accuracy","F1","Precision","Recall","AUC"]]
                   .agg(["mean","std"]))
    vqc_summary.columns = ["_".join(c) for c in vqc_summary.columns]
    vqc_summary = vqc_summary.reset_index()
    vqc_summary.to_csv(f"{RESULTS_DIR}/vqc_summary.csv", index=False)
    print("\n=== VQC summary (mean +/- sd across 5 outer folds) ===")
    print(vqc_summary.to_string(index=False))
    print("\nAll datasets complete.")
else:
    done = vqc_fold_df["Dataset"].unique().tolist()
    remaining = [n for n in DATA.keys() if n not in done]
    print(f"Done so far: {done}. Remaining: {remaining}")
