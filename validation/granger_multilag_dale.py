#!/usr/bin/env python3
"""
Granger causality and multi-lag Lasso-VAR on Dale simulation.

Granger causality: neuron A Granger-causes B if knowing A's past
significantly improves prediction of B's future beyond B's own past alone.
Tested via F-test comparing full model (all neurons) vs reduced model (B only).

Multi-lag VAR: extend lag-1 to lag-1+2+3, capturing connections that
operate over multiple 10ms steps rather than just one.
"""
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import f as f_dist

SPIKES_PATH = "/private/tmp/dale_sim/truthDale/daleN200_test.spikes.npz"
TRUTH_PATH  = "/private/tmp/dale_sim/truthDale/daleN200_test.simTruth.npz"
STATE = 0; DT_S = 0.010; ALPHA = 0.05

spk   = np.load(SPIKES_PATH, allow_pickle=True)
truth = np.load(TRUTH_PATH,  allow_pickle=True)
Y      = spk["spikes"][STATE].astype(np.float64)
E_true = truth["E_true"].astype(bool)
T, N   = Y.shape
print(f"Neurons: {N}, Time bins: {T}, Duration: {T*DT_S:.0f}s", flush=True)
gt_pairs = {(i, j) for i in range(N) for j in range(N) if i != j and E_true[i, j]}
print(f"GT connections: {len(gt_pairs)} ({E_true.mean():.1%} density)", flush=True)

# ── Method 1: Granger Causality (pairwise F-test) ────────────────────────────
print("\n=== Granger Causality (lag-1, pairwise F-test) ===", flush=True)
MAX_LAG = 1
X_full = Y[MAX_LAG:-1] if MAX_LAG > 1 else Y[:-1]  # predictors at t-1
Z      = Y[MAX_LAG:]                                  # targets at t

detected_granger = set()
for j in range(N):
    if j % 40 == 0: print(f"  Neuron {j}/{N}", flush=True)
    y_j = Z[:, j]
    # Reduced model: only B's own history
    X_red = X_full[:, j:j+1]
    # Full model: all neurons
    X_full_j = X_full

    # Fit both via OLS (closed form)
    def ols_rss(X, y):
        X_ = np.hstack([X, np.ones((len(X), 1))])
        coef, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
        resid = y - X_ @ coef
        return np.sum(resid**2), len(coef)

    rss_red,  k_red  = ols_rss(X_red,    y_j)
    rss_full, k_full = ols_rss(X_full_j, y_j)
    n_obs = len(y_j)

    # F-test: (RSS_red - RSS_full) / (k_full - k_red) / (RSS_full / (n - k_full))
    df1 = k_full - k_red
    df2 = n_obs - k_full
    if df1 <= 0 or df2 <= 0 or rss_full < 1e-10:
        continue
    F = ((rss_red - rss_full) / df1) / (rss_full / df2)
    p = 1 - f_dist.cdf(F, df1, df2)

    # Bonferroni correction
    if p < ALPHA / (N * (N-1)):
        # Identify which neurons significantly contribute
        # Use coefficient magnitude from full model
        X_ = np.hstack([X_full_j, np.ones((len(X_full_j), 1))])
        coef, _, _, _ = np.linalg.lstsq(X_, y_j, rcond=None)
        for i in range(N):
            if i != j and abs(coef[i]) > 0:
                detected_granger.add((i, j))

TP = gt_pairs & detected_granger
FP = detected_granger - gt_pairs
FN = gt_pairs - detected_granger
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_granger)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

# ── Method 2: Multi-lag Lasso-VAR (lags 1,2,3) ───────────────────────────────
print("\n=== Multi-lag Lasso-VAR (lags 1+2+3) ===", flush=True)
MAX_LAG = 3
# Build design matrix with lags 1,2,3: shape (T-MAX_LAG, N*MAX_LAG)
X_multi = np.hstack([Y[MAX_LAG-lag:-lag if lag > 0 else None] for lag in range(1, MAX_LAG+1)])
Z_multi = Y[MAX_LAG:]
print(f"Design matrix: {X_multi.shape}", flush=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

A_hat = np.zeros((N * MAX_LAG, N))
for j in range(N):
    if j % 40 == 0: print(f"  Neuron {j}/{N}", flush=True)
    lasso = LassoCV(cv=5, n_jobs=4, max_iter=2000)
    lasso.fit(X_scaled, Z_multi[:, j])
    A_hat[:, j] = lasso.coef_

# Connection detected if ANY lag has nonzero coefficient for that pair
detected_multilag = set()
for i in range(N):
    for j in range(N):
        if i == j: continue
        # Check all lag coefficients for i->j
        coefs = [A_hat[i + lag*N, j] for lag in range(MAX_LAG)]
        if any(c != 0 for c in coefs):
            detected_multilag.add((i, j))

TP2 = gt_pairs & detected_multilag
FP2 = detected_multilag - gt_pairs
FN2 = gt_pairs - detected_multilag
prec2 = len(TP2)/(len(TP2)+len(FP2)) if (len(TP2)+len(FP2)) > 0 else 0
rec2  = len(TP2)/(len(TP2)+len(FN2)) if (len(TP2)+len(FN2)) > 0 else 0
f12   = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) > 0 else 0
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_multilag)}")
print(f"TP: {len(TP2)}  FP: {len(FP2)}  FN: {len(FN2)}")
print(f"Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}", flush=True)

np.savez("/private/tmp/granger_multilag_dale_results.npz",
         granger_f1=f1, granger_prec=prec, granger_rec=rec,
         multilag_f1=f12, multilag_prec=prec2, multilag_rec=rec2)
print("\nDone.", flush=True)
