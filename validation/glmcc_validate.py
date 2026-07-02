#!/usr/bin/env python3
"""
Run GLMCC on a simulated or real spike dataset and evaluate against ground truth.

Usage:
    python validation/glmcc_validate.py
    python validation/glmcc_validate.py --spikes path/to/spikes.npz --conn path/to/conn.npz

Spike .npz must contain: spkt_s (spike times in seconds), spkid (unit IDs)
Connectivity .npz must contain: pre_gid, post_gid, delay (ms)
"""
import argparse, importlib.util, sys, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
from scipy.stats import norm

# ── Locate repo root and load spycon files bundled in this repo ───────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_module("spycon.spycon_result",    REPO_ROOT / "spycon_result.py")
_load_module("spycon.spycon_inference", REPO_ROOT / "spycon_inference.py")

def _load_class(modname, filename, classname):
    mod = _load_module(modname, REPO_ROOT / filename)
    return getattr(mod, classname)

GLMCC = _load_class("sci_glmcc", "sci_glmcc.py", "GLMCC")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--spikes", default=str(REPO_ROOT / "data" / "sim_hdmea_burst_spikes.npz"))
parser.add_argument("--conn",   default=str(REPO_ROOT / "data" / "sim_hdmea_burst_connectivity.npz"))
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

SPIKES_PATH = Path(args.spikes)
CONN_PATH   = Path(args.conn)

if not SPIKES_PATH.exists():
    print(f"ERROR: spikes file not found: {SPIKES_PATH}")
    print("Run simulation/generate_brian2_hdmea_burst.py first to generate data.")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────────────────────
spk     = np.load(SPIKES_PATH, allow_pickle=True)
times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]
times_ms = times_s * 1000.0

conn     = np.load(CONN_PATH, allow_pickle=True)
pre_gt   = conn["pre_gid"].astype(int)
post_gt  = conn["post_gid"].astype(int)
delay_gt = conn["delay"].astype(float)

gt_pairs = {}
for p, q, d in zip(pre_gt, post_gt, delay_gt):
    key = (p, q)
    if key not in gt_pairs or d < gt_pairs[key]:
        gt_pairs[key] = d

duration    = times_s[-1] - times_s[0]
all_nodes   = np.unique(ids)
valid_nodes = [n for n in all_nodes if np.sum(ids == n) / duration >= 0.5]
pairs       = list(itertools.combinations(valid_nodes, 2))

print(f"Units: {len(valid_nodes)}, Pairs: {len(pairs)}")
print(f"Ground truth connections: {len(gt_pairs)}")
print(f"Delay range: {delay_gt.min():.2f} - {delay_gt.max():.2f} ms")

PARAMS = {"binsize": 1e-3, "ccg_tau": 50e-3, "syn_delay": 3e-3,
          "tau": [10e-3, 10e-3], "beta": 4000, "alpha": 0.001, "deconv_ccg": False}
glmcc = GLMCC(params=PARAMS)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running GLMCC on all pairs (workers={args.workers})...")
    with mp.Pool(args.workers) as pool:
        raw = pool.starmap(glmcc._test_connection_pair,
                           zip(repeat(times_ms), repeat(ids), pairs))

    threshold = norm.ppf(1 - 0.5 * 0.001)
    detected = {}
    for (p, q), (z1, w1, z2, w2, _) in zip(pairs, raw):
        if abs(z1) > threshold:
            detected[(p, q)] = w1
        if abs(z2) > threshold:
            detected[(q, p)] = w2

    print(f"GLMCC detected: {len(detected)} ({100*len(detected)/(len(pairs)*2):.1f}%)")

    valid_set = set(valid_nodes)
    gt_valid  = {k: v for k, v in gt_pairs.items() if k[0] in valid_set and k[1] in valid_set}
    gt_set    = set(gt_valid.keys())
    det_set   = set(detected.keys())

    TP = gt_set & det_set
    FP = det_set - gt_set
    FN = gt_set - det_set

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    recall    = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== GLMCC vs Ground Truth ===")
    print(f"GT connections (valid nodes): {len(gt_set)}")
    print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    bins = [0.1, 1, 2, 3, 4, 5, 10]
    print(f"\n=== TPR by delay bin ===")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        gt_bin = {k for k, d in gt_valid.items() if lo <= d < hi}
        tp_bin = gt_bin & det_set
        tpr = len(tp_bin) / len(gt_bin) if gt_bin else 0
        print(f"  {lo}-{hi}ms: {len(tp_bin)}/{len(gt_bin)} (TPR={tpr:.2f})")
