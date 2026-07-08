#!/usr/bin/env python3
"""
VAR connectivity inference on Dale simulation using sklearn Lasso.
Same approach as pyuoi UoI-Lasso but without MPI dependency.
For each target neuron j, fit Lasso regression:
    Y[t, j] ~ sum_i (A[i,j] * Y[t-1, i])
Nonzero A[i,j] = inferred connection from i to j.
Cross-validated alpha selection per neuron.
"""
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

SPIKES_PATH = "/private/tmp/dale_sim/truthDale/daleN200_test.spikes.npz"
TRUTH_PATH  = "/private/tmp/dale_sim/truthDale/daleN200_test.simTruth.npz"
STATE = 0
DT_S  = 0.010

spk   = np.load(SPIKES_PATH, allow_pickle=True)
truth = np.load(TRUTH_PATH,  allow_pickle=True)

Y      = spk["spikes"][STATE].astype(np.float64)  # (T, N)
E_true = truth["E_true"].astype(bool)              # (N, N)

T, N = Y.shape
print(f"Neurons: {N}, Time bins: {T}, Duration: {T*DT_S:.0f}s", flush=True)
print(f"GT connections: {E_true.sum()} ({E_true.mean():.1%} density)", flush=True)

gt_pairs = {(i, j) for i in range(N) for j in range(N) if i != j and E_true[i, j]}

# Lag-1 design matrix
X = Y[:-1]  # (T-1, N) predictors
Z = Y[1:]   # (T-1, N) targets

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFitting Lasso-VAR for each neuron (cross-validated alpha)...", flush=True)
A_hat = np.zeros((N, N))

for j in range(N):
    if j % 20 == 0:
        print(f"  Neuron {j}/{N}", flush=True)
    lasso = LassoCV(cv=5, n_jobs=4, max_iter=2000)
    lasso.fit(X_scaled, Z[:, j])
    A_hat[:, j] = lasso.coef_

# Detect nonzero connections
detected = {(i, j) for i in range(N) for j in range(N)
            if i != j and A_hat[i, j] != 0}

TP = gt_pairs & detected
FP = detected - gt_pairs
FN = gt_pairs - detected
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

print(f"\n=== Lasso-VAR ===")
print(f"GT: {len(gt_pairs)}  Detected: {len(detected)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

# Build full N×N connectivity matrix (1=detected, 0=not)
conn_matrix = np.zeros((N, N), dtype=np.int8)
detected_list = np.array(sorted(detected), dtype=np.int32)
tp_list = np.array(sorted(TP), dtype=np.int32)
fp_list = np.array(sorted(FP), dtype=np.int32)
fn_list = np.array(sorted(FN), dtype=np.int32)
node_idx = {n: i for i, n in enumerate(sorted({n for pair in detected for n in pair}))}
for pre, post in detected:
    if pre in node_idx and post in node_idx:
        conn_matrix[node_idx[pre], node_idx[post]] = 1

np.savez("/private/tmp/lasso_var_dale_results.npz",
         A_hat=A_hat,
         conn_matrix=conn_matrix,
         detected_pairs=detected_list,
         tp_pairs=tp_list,
         fp_pairs=fp_list,
         fn_pairs=fn_list,
         precision=prec, recall=rec, f1=f1,
         TP=len(TP), FP=len(FP), FN=len(FN))
print("Done.", flush=True)
