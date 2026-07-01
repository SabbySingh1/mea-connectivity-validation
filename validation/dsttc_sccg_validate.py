#!/usr/bin/env python3
import importlib.util, sys, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
from scipy.stats import norm

SPIKES_PATH = Path("/private/tmp/sim_hdmea_spikes.npz")
CONN_PATH   = Path("/private/tmp/sim_hdmea_connectivity.npz")
METHOD      = sys.argv[1]  # "dsttc" or "sccg"

_spycon_root = Path("/private/tmp/sandbox_venv_x86/lib/python3.9/site-packages/spycon")
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

if METHOD == "dsttc":
    Cls = _load_class("sci_dsttc", "sci_dsttc.py", "directed_STTC")
    PARAMS = {"delta_t": 7e-3, "alpha": 1e-3}
else:
    Cls = _load_class("sci_sccg", "sci_sccg.py", "Smoothed_CCG")
    PARAMS = {"binsize": 0.4e-3, "ccg_tau": 50e-3, "alpha": 1e-3}

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

print(f"[{METHOD.upper()}] Units: {len(valid_nodes)}, Pairs: {len(pairs)}")
print(f"[{METHOD.upper()}] GT connections: {len(gt_pairs)}")

model = Cls(params=PARAMS)

if __name__ == "__main__":
    print(f"[{METHOD.upper()}] Running...")
    with mp.Pool(8) as pool:
        raw = pool.starmap(model._test_connection_pair,
                           zip(repeat(times_s), repeat(ids), pairs))

    threshold = norm.ppf(1 - 0.5 * PARAMS["alpha"])
    detected = set()
    for (p, q), result in zip(pairs, raw):
        # result is (stat_AB, z_AB, stat_BA, z_BA, pair) for both methods
        z_AB = result[1]
        z_BA = result[3]
        if z_AB > threshold:
            detected.add((p, q))
        if z_BA > threshold:
            detected.add((q, p))

    print(f"[{METHOD.upper()}] Detected: {len(detected)} ({100*len(detected)/(len(pairs)*2):.1f}%)")

    valid_set = set(valid_nodes)
    gt_valid  = {k: v for k, v in gt_pairs.items() if k[0] in valid_set and k[1] in valid_set}
    gt_set    = set(gt_valid.keys())
    det_set   = set(detected)

    TP = gt_set & det_set
    FP = det_set - gt_set
    FN = gt_set - det_set

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    recall    = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== {METHOD.upper()} vs Ground Truth ===")
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
