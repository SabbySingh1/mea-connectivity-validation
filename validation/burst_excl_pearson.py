#!/usr/bin/env python3
"""
Burst exclusion + Pearson correlation connectivity inference.
Based on Boddeti et al. 2024 (Frontiers in Network Physiology).

Steps:
1. Detect network burst windows using population firing rate threshold
2. Remove all spikes within burst windows
3. Bin remaining inter-burst spikes into 5ms bins
4. Compute pairwise Pearson correlation across bins
5. Threshold at p<0.001 (Bonferroni corrected) for significance
6. Evaluate against ground truth (1-5ms monosynaptic connections)
"""
import sys, argparse, itertools
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--spikes", required=True)
parser.add_argument("--conn",   required=True)
parser.add_argument("--out",    required=True)
args = parser.parse_args()

# ── Load data ────────────────────────────────────────────────────────────────
spk     = np.load(args.spikes, allow_pickle=True)
times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]

conn    = np.load(args.conn, allow_pickle=True)
pre_gt  = conn["pre_gid"].astype(int)
post_gt = conn["post_gid"].astype(int)

duration    = times_s[-1] - times_s[0]
t_start     = times_s[0]
all_nodes   = np.unique(ids)
valid_nodes = np.array([n for n in all_nodes if np.sum(ids == n)/duration >= 0.5])
print(f"Units: {len(valid_nodes)}, Duration: {duration:.1f}s", flush=True)

# GT pairs
gt_pairs = {}
for p, q, d in zip(pre_gt, post_gt, conn["delay"].astype(float)):
    key = (p, q)
    if key not in gt_pairs or d < gt_pairs[key]: gt_pairs[key] = d
valid_set = set(valid_nodes.tolist())
gt_valid  = {k for k, v in gt_pairs.items() if k[0] in valid_set and k[1] in valid_set}
print(f"GT connections (valid nodes): {len(gt_valid)}", flush=True)

# ── Step 1: Detect network bursts via population rate ────────────────────────
BIN_S       = 0.010   # 10ms bins for burst detection
BURST_HZ    = 5.0     # population rate threshold (spikes/s/neuron) to call burst
MERGE_S     = 0.100   # merge burst windows within 100ms
MIN_DUR_S   = 0.050   # minimum burst duration 50ms

n_bins  = int(np.ceil(duration / BIN_S))
pop_rate = np.zeros(n_bins)
mask_valid = np.isin(ids, valid_nodes)
t_valid = times_s[mask_valid]

bin_idx = np.floor((t_valid - t_start) / BIN_S).astype(int)
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
np.add.at(pop_rate, bin_idx, 1)
pop_rate /= (BIN_S * len(valid_nodes))  # spikes/s/neuron

# Smooth population rate
pop_rate_smooth = uniform_filter1d(pop_rate, size=5)

# Threshold
in_burst = pop_rate_smooth >= BURST_HZ
burst_bins = np.where(in_burst)[0]

# Build burst windows
burst_windows = []
if len(burst_bins) > 0:
    starts = [burst_bins[0]]
    ends   = []
    for i in range(1, len(burst_bins)):
        if burst_bins[i] - burst_bins[i-1] > MERGE_S / BIN_S:
            ends.append(burst_bins[i-1])
            starts.append(burst_bins[i])
    ends.append(burst_bins[-1])
    for s, e in zip(starts, ends):
        t_s = t_start + s * BIN_S
        t_e = t_start + (e + 1) * BIN_S
        if t_e - t_s >= MIN_DUR_S:
            burst_windows.append((t_s, t_e))

print(f"Detected {len(burst_windows)} burst windows", flush=True)

# ── Step 2: Remove burst spikes ───────────────────────────────────────────────
def in_any_burst(t, windows):
    for ws, we in windows:
        if ws <= t <= we:
            return True
    return False

if len(burst_windows) > 0:
    burst_arr = np.array(burst_windows)
    keep = np.ones(len(times_s), dtype=bool)
    for ws, we in burst_arr:
        keep &= ~((times_s >= ws) & (times_s <= we))
    times_nb = times_s[keep]
    ids_nb   = ids[keep]
else:
    times_nb = times_s
    ids_nb   = ids

n_inter = np.sum(np.isin(ids_nb, valid_nodes))
n_total = np.sum(np.isin(ids, valid_nodes))
print(f"Inter-burst spikes: {n_inter}/{n_total} ({100*n_inter/n_total:.1f}%)", flush=True)

# ── Step 3: Bin inter-burst spikes (5ms bins) ─────────────────────────────────
BIN_S2  = 0.005
n_bins2 = int(np.ceil(duration / BIN_S2))

spike_matrix = np.zeros((len(valid_nodes), n_bins2), dtype=np.float32)
node_idx = {n: i for i, n in enumerate(valid_nodes)}

mask_valid2 = np.isin(ids_nb, valid_nodes)
t2 = times_nb[mask_valid2]
id2 = ids_nb[mask_valid2]

for t, uid in zip(t2, id2):
    b = min(int((t - t_start) / BIN_S2), n_bins2 - 1)
    spike_matrix[node_idx[uid], b] += 1

print(f"Spike matrix shape: {spike_matrix.shape}", flush=True)

# ── Step 4: Pairwise Pearson correlation ──────────────────────────────────────
# Bonferroni-corrected p-value threshold
n_pairs   = len(valid_nodes) * (len(valid_nodes) - 1)
alpha_raw = 0.001
alpha_corr = alpha_raw / n_pairs
print(f"Running {n_pairs} pairwise Pearson correlations...", flush=True)

detected = set()
pairs = list(itertools.permutations(range(len(valid_nodes)), 2))
for k, (i, j) in enumerate(pairs):
    if k % 5000 == 0:
        print(f"  {k}/{len(pairs)}", flush=True)
    xi = spike_matrix[i]
    xj = spike_matrix[j]
    if xi.std() < 1e-9 or xj.std() < 1e-9:
        continue
    r, p = pearsonr(xi, xj)
    if p < alpha_corr and r > 0:
        detected.add((valid_nodes[i], valid_nodes[j]))

# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
TP = gt_valid & detected
FP = detected - gt_valid
FN = gt_valid - detected
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

print(f"\n=== Burst Exclusion + Pearson ===")
print(f"GT: {len(gt_valid)}  Detected: {len(detected)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

conn_matrix = np.zeros((len(valid_nodes), len(valid_nodes)), dtype=np.int8)
node_idx2 = {n: i for i, n in enumerate(valid_nodes)}
for pre, post in detected:
    if pre in node_idx2 and post in node_idx2:
        conn_matrix[node_idx2[pre], node_idx2[post]] = 1

np.savez(args.out, precision=prec, recall=rec, f1=f1,
         n_gt=len(gt_valid), n_detected=len(detected),
         TP=len(TP), FP=len(FP), FN=len(FN),
         conn_matrix=conn_matrix,
         detected_pairs=np.array(sorted(detected), dtype=np.int32),
         tp_pairs=np.array(sorted(TP), dtype=np.int32),
         fp_pairs=np.array(sorted(FP), dtype=np.int32),
         fn_pairs=np.array(sorted(FN), dtype=np.int32))
print(f"Saved to {args.out}", flush=True)
