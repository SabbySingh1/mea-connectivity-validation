#!/usr/bin/env python3
"""
Stability Lasso-VAR connectivity inference on Dale simulation.
Replicates the core of UoI-Lasso without MPI/pyuoi dependencies:
- Run Lasso on n_boots random subsamples of the data
- Keep only connections that appear in >= stability_threshold fraction of boots
This bootstrap stability selection kills false positives (same principle as UoI-Lasso).
"""
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

SPIKES_PATH      = "/private/tmp/dale_sim/truthDale/daleN200_test.spikes.npz"
TRUTH_PATH       = "/private/tmp/dale_sim/truthDale/daleN200_test.simTruth.npz"
STATE            = 0
DT_S             = 0.010
N_BOOTS          = 24
SUBSAMPLE_FRAC   = 0.8
STABILITY_THRESH = 0.75   # connection must appear in 75% of bootstraps

spk   = np.load(SPIKES_PATH, allow_pickle=True)
truth = np.load(TRUTH_PATH,  allow_pickle=True)

Y      = spk["spikes"][STATE].astype(np.float64)
E_true = truth["E_true"].astype(bool)
T, N   = Y.shape
duration = T * DT_S

print(f"Neurons: {N}, Time bins: {T}, Duration: {duration:.0f}s", flush=True)
print(f"GT connections: {E_true.sum()} ({E_true.mean():.1%} density)", flush=True)
print(f"Bootstraps: {N_BOOTS}, Subsample: {SUBSAMPLE_FRAC:.0%}, Stability threshold: {STABILITY_THRESH:.0%}", flush=True)

gt_pairs = {(i, j) for i in range(N) for j in range(N) if i != j and E_true[i, j]}

X = Y[:-1]  # (T-1, N)
Z = Y[1:]   # (T-1, N)
n_samples = X.shape[0]
n_sub = int(n_samples * SUBSAMPLE_FRAC)

rng = np.random.default_rng(42)
# Count how many boots each connection appears in
vote_matrix = np.zeros((N, N), dtype=np.int32)

for boot in range(N_BOOTS):
    if boot % 4 == 0:
        print(f"Bootstrap {boot}/{N_BOOTS}...", flush=True)
    idx = rng.choice(n_samples, size=n_sub, replace=False)
    X_sub = X[idx]
    Z_sub = Z[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    for j in range(N):
        lasso = LassoCV(cv=3, n_jobs=4, max_iter=1000)
        lasso.fit(X_scaled, Z_sub[:, j])
        coef = lasso.coef_
        for i in range(N):
            if i != j and coef[i] != 0:
                vote_matrix[i, j] += 1

# Keep connections that appear in >= stability_threshold fraction of boots
A_stable = vote_matrix / N_BOOTS
detected = {(i, j) for i in range(N) for j in range(N)
            if i != j and A_stable[i, j] >= STABILITY_THRESH}

TP = gt_pairs & detected
FP = detected - gt_pairs
FN = gt_pairs - detected
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

print(f"\n=== Stability Lasso-VAR (UoI-style) ===")
print(f"GT: {len(gt_pairs)}  Detected: {len(detected)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

np.savez("/private/tmp/stability_lasso_dale_results.npz",
         A_stable=A_stable, precision=prec, recall=rec, f1=f1,
         TP=len(TP), FP=len(FP), FN=len(FN))
print("Done.", flush=True)
