#!/usr/bin/env python3
import importlib.util, sys, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

SPIKES_PATH = Path("/Users/sabadnoorsingh/Downloads/trial_135_rerun_300s_spikes.npz")
CONN_PATH   = Path("/Users/sabadnoorsingh/Downloads/trial_135_rerun_300s_connectivity.npz")
OUT_DIR     = Path("/private/tmp")

_spycon_root = Path(importlib.util.find_spec("spycon").origin).parent
for _mod_name in ["spycon_result", "spycon_inference"]:
    _spec = importlib.util.spec_from_file_location(f"spycon.{_mod_name}", _spycon_root / f"{_mod_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"spycon.{_mod_name}"] = _mod
    _spec.loader.exec_module(_mod)

def _load_class(modname, filename, classname):
    spec = importlib.util.spec_from_file_location(modname, _spycon_root / "coninf" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, classname)

GLMCC = _load_class("sci_glmcc", "sci_glmcc.py", "GLMCC")

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
    print("Running GLMCC on all pairs...")
    with mp.Pool(4) as pool:
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

    valid_set   = set(valid_nodes)
    gt_valid    = {k: v for k, v in gt_pairs.items() if k[0] in valid_set and k[1] in valid_set}
    gt_set      = set(gt_valid.keys())
    det_set     = set(detected.keys())

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

    # Plot GT vs GLMCC
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
    axes[0].set_title(f"Ground Truth\n{len(gt_set)} connections (all delays)", fontsize=11)
    axes[0].set_xlabel("Post-synaptic unit index")
    axes[0].set_ylabel("Pre-synaptic unit index")

    vmax = np.abs(det_mat).max() or 1
    im = axes[1].imshow(det_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        interpolation="nearest", aspect="auto")
    axes[1].set_title(f"GLMCC Detected\n{len(detected)} significant", fontsize=11)
    axes[1].set_xlabel("Post-synaptic unit index")
    axes[1].set_ylabel("Pre-synaptic unit index")
    plt.colorbar(im, ax=axes[1], label="Weight (+=exc, -=inh)")

    fig.suptitle(f"GLMCC vs Ground Truth — Simulated Data\n"
                 f"TP={len(TP)}  FP={len(FP)}  FN={len(FN)}  "
                 f"Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "sim_gt_vs_glmcc_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")
