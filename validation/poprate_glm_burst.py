#!/usr/bin/env python3
"""
Population-rate GLM for bursty organoid-like MEA data.

For each target neuron i, fits a Poisson GLM:
    log(rate_i(t)) = sum_j w_ij * y_j(t-1) + alpha_i * pop_rate(t-1) + beta_i

The pop_rate term absorbs shared burst drive. Residual w_ij captures
true pairwise connectivity beyond what bursting explains.

Uses L1 (Lasso) regularization via sklearn LogisticRegression (Poisson
approximation via binary binning) or direct PoissonRegressor.
"""
import numpy as np
from sklearn.linear_model import PoissonRegressor, LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

SPIKES_PATH = "/private/tmp/mea-connectivity-validation/data/sim_hdmea_burst_spikes.npz"
CONN_PATH   = "/private/tmp/mea-connectivity-validation/data/sim_hdmea_burst_connectivity.npz"
DT_S = 0.005  # 5ms bins

# ── Load data ─────────────────────────────────────────────────────────────────
spk  = np.load(SPIKES_PATH, allow_pickle=True)
conn = np.load(CONN_PATH,   allow_pickle=True)

times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]

duration    = times_s[-1] - times_s[0]
all_nodes   = np.unique(ids)
valid_nodes = np.array([n for n in all_nodes if np.sum(ids == n)/duration >= 0.5])
print(f"Units: {len(valid_nodes)}, Duration: {duration:.1f}s", flush=True)

# Bin spikes into 5ms bins
t0      = times_s[0]
n_bins  = int(np.ceil(duration / DT_S))
node_map = {n: i for i, n in enumerate(valid_nodes)}
N = len(valid_nodes)

Y = np.zeros((n_bins, N), dtype=np.float32)
mask = np.isin(ids, valid_nodes)
for t, uid in zip(times_s[mask], ids[mask]):
    b = min(int((t - t0) / DT_S), n_bins - 1)
    n = node_map.get(uid)
    if n is not None:
        Y[b, n] += 1

T = n_bins
print(f"Bins: {T} ({T*DT_S:.1f}s), mean rate: {Y.mean()/DT_S:.2f} Hz", flush=True)

# ── Ground truth ──────────────────────────────────────────────────────────────
pre_gt  = conn["pre_gid"].astype(int)
post_gt = conn["post_gid"].astype(int)
valid_set = set(valid_nodes.tolist())
gt_pairs = {(p, q) for p, q in zip(pre_gt, post_gt) if p in valid_set and q in valid_set}
print(f"GT connections (valid nodes): {len(gt_pairs)}", flush=True)

# ── Population rate ───────────────────────────────────────────────────────────
pop_rate = Y.sum(axis=1, keepdims=True)  # (T, 1) total population spikes per bin

# ── Method 1: Lasso-VAR with population rate covariate ───────────────────────
print("\n=== Population-Rate Lasso-VAR ===", flush=True)
# Design matrix: [Y(t-1), pop_rate(t-1)]  -> predict Y(t)
X_lag = Y[:-1]                      # (T-1, N)  individual neuron lag
P_lag = pop_rate[:-1]               # (T-1, 1)  population lag
X_full = np.hstack([X_lag, P_lag])  # (T-1, N+1)
Z = Y[1:]                           # (T-1, N)  targets

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

A_hat = np.zeros((N + 1, N))
for j in range(N):
    if j % 20 == 0:
        print(f"  Neuron {j}/{N}", flush=True)
    lasso = LassoCV(cv=5, n_jobs=4, max_iter=3000, precompute=False)
    lasso.fit(X_scaled, Z[:, j])
    A_hat[:, j] = lasso.coef_

# Detect connections: nonzero coefficient for neuron i predicting neuron j
# (only first N columns correspond to individual neurons)
detected_lasso = set()
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if A_hat[i, j] != 0:
            ni, nj = valid_nodes[i], valid_nodes[j]
            detected_lasso.add((ni, nj))

TP = gt_pairs & detected_lasso
FP = detected_lasso - gt_pairs
FN = gt_pairs - detected_lasso
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_lasso)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

# ── Method 2: Pearson on population-rate-residuals ────────────────────────────
print("\n=== Pearson on Pop-Rate Residuals ===", flush=True)
# Regress out population rate from each neuron's activity, then compute Pearson
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# For each neuron, regress Y_i ~ pop_rate, take residuals
pop_z = (pop_rate - pop_rate.mean()) / (pop_rate.std() + 1e-10)
residuals = np.zeros_like(Y)
for i in range(N):
    y_i = Y[:, i]
    # OLS: y_i = a * pop_rate + b
    cov = np.cov(pop_z[:, 0], y_i)
    beta = cov[0, 1] / (cov[0, 0] + 1e-10)
    residuals[:, i] = y_i - beta * pop_z[:, 0]

# Compute pairwise Pearson on residuals
pvals = []
pairs_list = []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        r, p = pearsonr(residuals[:, i], residuals[:, j])
        pvals.append(p)
        pairs_list.append((valid_nodes[i], valid_nodes[j]))

pvals = np.array(pvals)
reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

detected_pearson = {pairs_list[k] for k in range(len(pairs_list)) if reject[k]}

TP2 = gt_pairs & detected_pearson
FP2 = detected_pearson - gt_pairs
FN2 = gt_pairs - detected_pearson
prec2 = len(TP2)/(len(TP2)+len(FP2)) if (len(TP2)+len(FP2)) > 0 else 0
rec2  = len(TP2)/(len(TP2)+len(FN2)) if (len(TP2)+len(FN2)) > 0 else 0
f12   = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) > 0 else 0
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_pearson)}")
print(f"TP: {len(TP2)}  FP: {len(FP2)}  FN: {len(FN2)}")
print(f"Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}", flush=True)

np.savez("/private/tmp/poprate_glm_burst_results.npz",
         lasso_f1=f1, lasso_prec=prec, lasso_rec=rec,
         pearson_resid_f1=f12, pearson_resid_prec=prec2, pearson_resid_rec=rec2)
print("\nDone.", flush=True)
