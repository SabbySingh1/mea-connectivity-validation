#!/usr/bin/env python3
"""
Functional connectivity on Dale VAR simulation.
Methods: Pearson correlation (FDR corrected) + STTC.
"""
import numpy as np
from scipy.stats import norm, t as t_dist

SPIKES_PATH = "/private/tmp/dale_sim/truthDale/daleN200_test.spikes.npz"
TRUTH_PATH  = "/private/tmp/dale_sim/truthDale/daleN200_test.simTruth.npz"
STATE       = 0
DT_S        = 0.010
ALPHA_FDR   = 0.05
STTC_DT     = 0.050
STTC_THRESH = 0.1

spk   = np.load(SPIKES_PATH, allow_pickle=True)
truth = np.load(TRUTH_PATH,  allow_pickle=True)
spike_matrix = spk["spikes"][STATE].astype(np.float32)  # (T, N)
E_true       = truth["E_true"].astype(bool)

T, N = spike_matrix.shape
duration = T * DT_S
print(f"Neurons: {N}, Duration: {duration:.0f}s, Rate: {spike_matrix.mean()/DT_S:.2f} Hz", flush=True)

gt_pairs = {(i, j) for i in range(N) for j in range(N) if i != j and E_true[i, j]}
print(f"GT connections: {len(gt_pairs)} ({E_true.mean():.1%} density)", flush=True)

# ── Pearson with FDR correction ───────────────────────────────────────────────
print("\nRunning Pearson + FDR...", flush=True)
C = np.corrcoef(spike_matrix.T)  # (N, N)

# Convert r to p-value using t-distribution
pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
r_vals = np.array([C[i, j] for i, j in pairs])
# t-statistic: t = r * sqrt(T-2) / sqrt(1-r^2)
t_stat = r_vals * np.sqrt(T - 2) / np.sqrt(np.clip(1 - r_vals**2, 1e-10, None))
p_vals = 2 * t_dist.sf(np.abs(t_stat), df=T-2)

# Benjamini-Hochberg FDR
sorted_idx = np.argsort(p_vals)
m = len(p_vals)
bh_thresh = np.zeros(m)
for k, idx in enumerate(sorted_idx):
    bh_thresh[idx] = (k + 1) / m * ALPHA_FDR
sig = p_vals <= bh_thresh

detected_pearson = set()
for sig_flag, (i, j), r in zip(sig, pairs, r_vals):
    if sig_flag and r > 0:
        detected_pearson.add((i, j))

TP = gt_pairs & detected_pearson
FP = detected_pearson - gt_pairs
FN = gt_pairs - detected_pearson
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
print(f"\n=== Pearson + FDR (alpha={ALPHA_FDR}) ===")
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_pearson)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

# ── STTC ─────────────────────────────────────────────────────────────────────
print("\nRunning STTC...", flush=True)
spike_times = {}
for n in range(N):
    bin_indices = np.where(spike_matrix[:, n] > 0)[0]
    ts = []
    for b in bin_indices:
        count = int(spike_matrix[b, n])
        ts.extend([b * DT_S + k * DT_S/max(count,1) for k in range(count)])
    spike_times[n] = np.array(ts)

def sttc_directed(t1, t2, dt, T):
    if len(t1) == 0 or len(t2) == 0: return 0.0
    P_A = np.sum(np.minimum(t1+dt, T) - np.maximum(t1-dt, 0)) / T
    P_B = sum(1 for sp in t2 if np.any(np.abs(t1-sp) <= dt)) / len(t2)
    return (P_B - P_A) / (1 - P_A) if P_A < 1.0 else P_B

detected_sttc = set()
for i in range(N):
    if i % 40 == 0: print(f"  STTC: {i}/{N}", flush=True)
    t1 = spike_times[i]
    for j in range(N):
        if i == j: continue
        if sttc_directed(t1, spike_times[j], STTC_DT, duration) >= STTC_THRESH:
            detected_sttc.add((i, j))

TP2 = gt_pairs & detected_sttc
FP2 = detected_sttc - gt_pairs
FN2 = gt_pairs - detected_sttc
prec2 = len(TP2)/(len(TP2)+len(FP2)) if (len(TP2)+len(FP2)) > 0 else 0
rec2  = len(TP2)/(len(TP2)+len(FN2)) if (len(TP2)+len(FN2)) > 0 else 0
f12   = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) > 0 else 0
print(f"\n=== STTC (dt=50ms, threshold=0.1) ===")
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_sttc)}")
print(f"TP: {len(TP2)}  FP: {len(FP2)}  FN: {len(FN2)}")
print(f"Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}", flush=True)

# Connectivity matrices
pearson_matrix = np.zeros((N, N), dtype=np.int8)
for i, j in detected_pearson:
    pearson_matrix[i, j] = 1

sttc_matrix = np.zeros((N, N), dtype=np.int8)
for i, j in detected_sttc:
    sttc_matrix[i, j] = 1

np.savez("/private/tmp/func_conn_dale_fdr_results.npz",
         pearson_f1=f1, pearson_prec=prec, pearson_rec=rec,
         pearson_conn_matrix=pearson_matrix,
         pearson_detected_pairs=np.array(sorted(detected_pearson), dtype=np.int32),
         pearson_tp_pairs=np.array(sorted(TP), dtype=np.int32),
         pearson_fp_pairs=np.array(sorted(FP), dtype=np.int32),
         pearson_fn_pairs=np.array(sorted(FN), dtype=np.int32),
         sttc_f1=f12, sttc_prec=prec2, sttc_rec=rec2,
         sttc_conn_matrix=sttc_matrix,
         sttc_detected_pairs=np.array(sorted(detected_sttc), dtype=np.int32),
         sttc_tp_pairs=np.array(sorted(TP2), dtype=np.int32),
         sttc_fp_pairs=np.array(sorted(FP2), dtype=np.int32),
         sttc_fn_pairs=np.array(sorted(FN2), dtype=np.int32))
print("\nDone.", flush=True)
