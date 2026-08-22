import sys, os, time
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
SEEDS = [42, 43, 44, 45, 46]
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]

target_names = sys.argv[1:] if len(sys.argv) > 1 else list(DATA.keys())

seed_path = f"{RESULTS_DIR}/vqc_seed_stability.csv"
noise_path = f"{RESULTS_DIR}/vqc_noise_robustness.csv"
seed_rows = pd.read_csv(seed_path).to_dict("records") if os.path.exists(seed_path) else []
noise_rows = pd.read_csv(noise_path).to_dict("records") if os.path.exists(noise_path) else []
# drop any existing rows for datasets we're about to recompute, to avoid duplicates
seed_rows = [r for r in seed_rows if r["Dataset"] not in target_names]
noise_rows = [r for r in noise_rows if r["Dataset"] not in target_names]

for name in target_names:
    d = DATA[name]
    X, y, cfg = d["X"], d["y"], d["cfg"]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    tr_idx, te_idx = next(iter(outer.split(X, y)))  # fold 0, representative
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

    t0 = time.time()
    seed_accs = []
    first_weights = None
    for seed in SEEDS:
        weights, hist = train_vqc(Xtr_ang, y_tr, n_layers=N_LAYERS, n_qubits=N_QUBITS,
                                   epochs=40, batch_size=32, lr=0.05, seed=seed)
        proba = predict_proba(Xte_ang, weights, N_QUBITS)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        seed_accs.append(acc)
        seed_rows.append({"Dataset": name, "Seed": seed, "Accuracy": acc, "F1": f1})
        if seed == 42:
            first_weights = weights
        print(f"{name} seed={seed}: acc={acc:.4f} f1={f1:.4f}")

    print(f"{name} seed-stability: mean={np.mean(seed_accs):.4f} sd={np.std(seed_accs):.4f} "
          f"range=[{min(seed_accs):.4f},{max(seed_accs):.4f}] ({time.time()-t0:.1f}s)\n")

    # noisy-simulation inference using the seed=42 trained model
    for p in NOISE_LEVELS:
        accs_this_p = []
        n_trials = 5 if p > 0 else 1
        for trial in range(n_trials):
            proba = predict_proba(Xte_ang, first_weights, N_QUBITS, noise_p=p, seed=1000 + trial)
            pred = (proba >= 0.5).astype(int)
            accs_this_p.append(accuracy_score(y_te, pred))
        noise_rows.append({
            "Dataset": name, "Depolarizing_p": p,
            "Mean_Accuracy": float(np.mean(accs_this_p)),
            "SD_Accuracy": float(np.std(accs_this_p)) if len(accs_this_p) > 1 else 0.0,
        })
        print(f"{name} noise_p={p}: acc={np.mean(accs_this_p):.4f} (sd={np.std(accs_this_p):.4f})")

    pd.DataFrame(seed_rows).to_csv(f"{RESULTS_DIR}/vqc_seed_stability.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(f"{RESULTS_DIR}/vqc_noise_robustness.csv", index=False)
    print(f"[{name} robustness analysis saved]\n")

print("Robustness analysis complete for:", target_names)
