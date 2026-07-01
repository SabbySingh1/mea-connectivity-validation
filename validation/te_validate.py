#!/usr/bin/env python3
import importlib.util, sys, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
from scipy.stats import norm

SPIKES_PATH = Path("/Users/sabadnoorsingh/Downloads/trial_135_rerun_300s_spikes.npz")
CONN_PATH   = Path("/Users/sabadnoorsingh/Downloads/trial_135_rerun_300s_connectivity.npz")

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

TE_PyInform = _load_class("sci_pyinform", "sci_pyinform.py", "TE_PyInform")

spk     = np.load(SPIKES_PATH, allow_pickle=True)
times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]

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

# TE params — binsize=1ms, k=5 (5ms history window), matches monosynaptic range
PARAMS = {
    "binsize": 1e-3,
    "k": 5,
    "num_surrogates": 30,
    "alpha": 1e-2,
    "jitter": False,
    "jitter_factor": 7.0,
}
te = TE_PyInform(params=PARAMS)

if __name__ == "__main__":
    print("Running TE on all pairs...")
    with mp.Pool(4) as pool:
        raw = pool.starmap(te._test_connection_pair,
                           zip(repeat(times_s), repeat(ids), pairs))

    threshold = norm.ppf(1 - 0.5 * PARAMS["alpha"])
    detected = set()
    for (p, q), (te1, z1, te2, z2, _) in zip(pairs, raw):
        if z1 > threshold:
            detected.add((p, q))
        if z2 > threshold:
            detected.add((q, p))

    print(f"TE detected: {len(detected)} ({100*len(detected)/(len(pairs)*2):.1f}%)")

    valid_set = set(valid_nodes)
    gt_valid  = {k: v for k, v in gt_pairs.items() if k[0] in valid_set and k[1] in valid_set}
    gt_set    = set(gt_valid.keys())

    TP = gt_set & detected
    FP = detected - gt_set
    FN = gt_set - detected

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    recall    = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== TE vs Ground Truth ===")
    print(f"GT connections (valid nodes): {len(gt_set)}")
    print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    bins = [0.1, 1, 5, 10, 50, 200, 1345]
    print(f"\n=== TPR by delay bin ===")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        gt_bin = {k for k, d in gt_valid.items() if lo <= d < hi}
        tp_bin = gt_bin & detected
        tpr = len(tp_bin) / len(gt_bin) if gt_bin else 0
        print(f"  {lo}-{hi}ms: {len(tp_bin)}/{len(gt_bin)} (TPR={tpr:.2f})")
