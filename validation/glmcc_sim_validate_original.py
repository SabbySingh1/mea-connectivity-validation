#!/usr/bin/env python3
"""
Run GLMCC on the original trial_135 simulated dataset and plot GT vs detected.

Usage:
    python validation/glmcc_sim_validate_original.py \
        --spikes path/to/spikes.npz --conn path/to/conn.npz

Note: the original trial_135 dataset has ground truth delays of 200-1345ms which
fall outside GLMCC's ±50ms search window, so results will be poor by design.
"""
import argparse, importlib.util, sys, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

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

parser = argparse.ArgumentParser()
parser.add_argument("--spikes",  required=True, help="Path to spikes .npz file")
parser.add_argument("--conn",    required=True, help="Path to connectivity .npz file")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

SPIKES_PATH = Path(args.spikes)
CONN_PATH   = Path(args.conn)
OUT_DIR     = REPO_ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

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

PARAMS = {"binsize": 1e-3, "ccg_tau": 50e-3, "syn_delay": 3e-3,
          "tau": [10e-3, 10e-3], "beta": 4000, "alpha": 0.001, "deconv_ccg": False}
glmcc = GLMCC(params=PARAMS)

if __name__ == "__main__":
    print(f"Running GLMCC (workers={args.workers})...")
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

    bins = [0.1, 1, 5, 10, 50, 200, 1345]
    print(f"\n=== TPR by delay bin ===")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        gt_bin = {k for k, d in gt_valid.items() if lo <= d < hi}
        tp_bin = gt_bin & det_set
        tpr = len(tp_bin) / len(gt_bin) if gt_bin else 0
        print(f"  {lo}-{hi}ms: {len(tp_bin)}/{len(gt_bin)} (TPR={tpr:.2f})")

    node_idx = {uid: i for i, uid in enumerate(valid_nodes)}
    n = len(valid_nodes)
    gt_mat  = np.zeros((n, n))
    det_mat = np.zeros((n, n))
    for (p, q) in gt_set:
        gt_mat[node_idx[p], node_idx[q]] = 1
    for (p, q), w in detected.items():
        if p in node_idx and q in node_idx:
            det_mat[node_idx[p], node_idx[q]] = w

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(gt_mat, cmap="Reds", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    axes[0].set_title(f"Ground Truth\n{len(gt_set)} connections", fontsize=11)
    axes[0].set_xlabel("Post-synaptic unit index")
    axes[0].set_ylabel("Pre-synaptic unit index")

    vmax = np.abs(det_mat).max() or 1
    im = axes[1].imshow(det_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        interpolation="nearest", aspect="auto")
    axes[1].set_title(f"GLMCC Detected\n{len(detected)} significant", fontsize=11)
    axes[1].set_xlabel("Post-synaptic unit index")
    axes[1].set_ylabel("Pre-synaptic unit index")
    plt.colorbar(im, ax=axes[1], label="Weight (+=exc, -=inh)")

    fig.suptitle(f"GLMCC vs Ground Truth\n"
                 f"TP={len(TP)}  FP={len(FP)}  FN={len(FN)}  "
                 f"Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "sim_gt_vs_glmcc_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")
