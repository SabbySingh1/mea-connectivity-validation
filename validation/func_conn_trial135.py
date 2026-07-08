#!/usr/bin/env python3
"""
Functional connectivity on trial_135 professor data.
Methods: Pearson correlation + STTC on binned spike trains.
Ground truth: provided connectivity matrix (pre_gid, post_gid).
"""
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

SPIKES_PATH = "/Users/sabadnoorsingh/Downloads/trial_135_rerun_300s_spikes.npz"
CONN_PATH   = "/Users/sabadnoorsingh/Downloads/trial_135_rerun_300s_connectivity.npz"
BIN_S       = 0.020   # 20ms bins for functional connectivity
ALPHA       = 0.001   # significance threshold

# ── Load data ────────────────────────────────────────────────────────────────
spk     = np.load(SPIKES_PATH, allow_pickle=True)
times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]
t_start  = times_s[0]
duration = times_s[-1] - times_s[0]

conn    = np.load(CONN_PATH, allow_pickle=True)
pre_gt  = conn["pre_gid"].astype(int)
post_gt = conn["post_gid"].astype(int)
gt_pairs = set(zip(pre_gt.tolist(), post_gt.tolist()))

all_nodes   = np.unique(ids)
valid_nodes = np.array([n for n in all_nodes if np.sum(ids == n)/duration >= 0.5])
valid_set   = set(valid_nodes.tolist())
gt_valid    = {(p, q) for p, q in gt_pairs if p in valid_set and q in valid_set}

print(f"Units: {len(valid_nodes)}, Duration: {duration:.1f}s", flush=True)
print(f"GT connections (valid nodes): {len(gt_valid)}", flush=True)
print(f"Connection probability: {len(gt_valid)/(len(valid_nodes)*(len(valid_nodes)-1)):.1%}", flush=True)

# ── Bin spike trains ──────────────────────────────────────────────────────────
n_bins = int(np.ceil(duration / BIN_S))
node_idx = {int(n): i for i, n in enumerate(valid_nodes)}
spike_matrix = np.zeros((len(valid_nodes), n_bins), dtype=np.float32)

mask = np.isin(ids, valid_nodes)
for t, uid in zip(times_s[mask], ids[mask]):
    b = min(int((t - t_start) / BIN_S), n_bins - 1)
    spike_matrix[node_idx[uid], b] += 1

print(f"Spike matrix: {spike_matrix.shape}", flush=True)

# ── Method 1: Pearson correlation (numpy corrcoef — fast) ────────────────────
print("\nRunning Pearson correlation...", flush=True)
C = np.corrcoef(spike_matrix)  # (N, N) correlation matrix

# Bonferroni threshold via Fisher z-transform
n_pairs = len(valid_nodes) * (len(valid_nodes) - 1)
from scipy.stats import norm
z_thresh = norm.ppf(1 - ALPHA / n_pairs)
# r to z: z = arctanh(r), var = 1/(n_bins-3)
r_thresh = np.tanh(z_thresh / np.sqrt(n_bins - 3))
print(f"  Pearson r threshold (Bonferroni): {r_thresh:.4f}", flush=True)

detected_pearson = set()
N = len(valid_nodes)
for i in range(N):
    for j in range(N):
        if i == j: continue
        if C[i, j] >= r_thresh:
            detected_pearson.add((int(valid_nodes[i]), int(valid_nodes[j])))

TP = gt_valid & detected_pearson
FP = detected_pearson - gt_valid
FN = gt_valid - detected_pearson
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
print(f"\n=== Pearson Correlation ===")
print(f"GT: {len(gt_valid)}  Detected: {len(detected_pearson)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

# ── Method 2: STTC (Spike Time Tiling Coefficient) ───────────────────────────
print("\nRunning STTC (dt=50ms)...", flush=True)
DT = 0.050  # 50ms coincidence window

def compute_sttc(t1, t2, dt, T):
    """Directed STTC: proportion of t2 spikes within dt of t1 spikes."""
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    # P_A: proportion of recording covered by windows around t1
    windows = np.minimum(t1 + dt, T) - np.maximum(t1 - dt, 0)
    P_A = np.sum(windows) / T
    # P_B: proportion of t2 spikes within dt of any t1 spike
    count = 0
    for sp in t2:
        if np.any(np.abs(t1 - sp) <= dt):
            count += 1
    P_B = count / len(t2)
    if P_A == 1.0:
        return P_B
    return (P_B - P_A) / (1 - P_A)

# Precompute spike times per unit
spike_times = {int(n): times_s[ids == n] for n in valid_nodes}
T = duration

detected_sttc = set()
STTC_THRESH = 0.1  # standard threshold from literature

for k, i in enumerate(range(N)):
    if k % 50 == 0:
        print(f"  STTC: unit {k}/{N}", flush=True)
    t1 = spike_times[int(valid_nodes[i])]
    for j in range(N):
        if i == j: continue
        t2 = spike_times[int(valid_nodes[j])]
        sttc = compute_sttc(t1, t2, DT, T)
        if sttc >= STTC_THRESH:
            detected_sttc.add((int(valid_nodes[i]), int(valid_nodes[j])))

TP2 = gt_valid & detected_sttc
FP2 = detected_sttc - gt_valid
FN2 = gt_valid - detected_sttc
prec2 = len(TP2)/(len(TP2)+len(FP2)) if (len(TP2)+len(FP2)) > 0 else 0
rec2  = len(TP2)/(len(TP2)+len(FN2)) if (len(TP2)+len(FN2)) > 0 else 0
f12   = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) > 0 else 0
print(f"\n=== STTC (dt=50ms, threshold=0.1) ===")
print(f"GT: {len(gt_valid)}  Detected: {len(detected_sttc)}")
print(f"TP: {len(TP2)}  FP: {len(FP2)}  FN: {len(FN2)}")
print(f"Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}", flush=True)

pearson_matrix = np.zeros((N, N), dtype=np.int8)
for pre, post in detected:
    pi = np.where(valid_nodes == pre)[0]
    pj = np.where(valid_nodes == post)[0]
    if len(pi) and len(pj):
        pearson_matrix[pi[0], pj[0]] = 1

sttc_matrix = np.zeros((N, N), dtype=np.int8)
for pre, post in detected_sttc:
    pi = np.where(valid_nodes == pre)[0]
    pj = np.where(valid_nodes == post)[0]
    if len(pi) and len(pj):
        sttc_matrix[pi[0], pj[0]] = 1

np.savez("/private/tmp/func_conn_trial135_results.npz",
         pearson_f1=f1, pearson_prec=prec, pearson_rec=rec,
         pearson_conn_matrix=pearson_matrix,
         pearson_detected_pairs=np.array(sorted(detected), dtype=np.int32),
         pearson_tp_pairs=np.array(sorted(TP), dtype=np.int32),
         pearson_fp_pairs=np.array(sorted(FP), dtype=np.int32),
         pearson_fn_pairs=np.array(sorted(FN), dtype=np.int32),
         sttc_f1=f12, sttc_prec=prec2, sttc_rec=rec2,
         sttc_conn_matrix=sttc_matrix,
         sttc_detected_pairs=np.array(sorted(detected_sttc), dtype=np.int32),
         sttc_tp_pairs=np.array(sorted(TP2), dtype=np.int32),
         sttc_fp_pairs=np.array(sorted(FP2), dtype=np.int32),
         sttc_fn_pairs=np.array(sorted(FN2), dtype=np.int32))
print("\nDone. Results saved.", flush=True)
