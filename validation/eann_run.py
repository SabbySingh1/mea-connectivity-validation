#!/usr/bin/env python3
"""
Run spycon eANN (ensemble ANN) on sim_cdkl5 and sim_hdmea_burst datasets.
eANN combines outputs of CI, sCCG, GLMCC, dSTTC, GLMPP into a neural network
trained on simulated ground-truth data (Donner et al., PLOS CB 2024).
"""
import sys, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, "/private/tmp/spycon")

import spycon
from spycon.coninf.setup_ensemble import get_eann_coninf_dict
from spycon.coninf.sci_ensemble import NNEnsemble as eANN

parser = argparse.ArgumentParser()
parser.add_argument("--spikes", required=True)
parser.add_argument("--conn",   required=True)
parser.add_argument("--out",    required=True)
parser.add_argument("--cores",  type=int, default=4)
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
all_nodes   = np.unique(ids)
valid_nodes = np.array([n for n in all_nodes if np.sum(ids == n)/duration >= 0.5])
print(f"Units: {len(valid_nodes)}, Duration: {duration:.1f}s", flush=True)

# Filter spike data to valid nodes
mask    = np.isin(ids, valid_nodes)
times_s = times_s[mask]
ids     = ids[mask]

# Ground truth
gt_pairs = set(zip(pre_gt, post_gt))
valid_set = set(valid_nodes.tolist())
gt_valid  = {(p, q) for p, q in gt_pairs if p in valid_set and q in valid_set}
print(f"GT connections (valid nodes): {len(gt_valid)}", flush=True)

# ── Run base methods + eANN ──────────────────────────────────────────────────
print("Setting up base methods...", flush=True)
coninf_dict, model_name = get_eann_coninf_dict(num_cores=args.cores)

# Run each base method
results_dict = {}
for name, method in coninf_dict.items():
    print(f"  Running {name}...", flush=True)
    result = method.infer_connectivity(times_s, ids)
    results_dict[name] = result
    print(f"  {name} done", flush=True)

# Run eANN
print("Running eANN...", flush=True)
eann = eANN(params={
    "model_path": "/private/tmp/spycon/data/nn_models/",
    "name": "eANN",
    "con_inf_dict": coninf_dict,
})
eann_result = eann.infer_connectivity(
    times_s, ids,
    spycon_result_dict=results_dict
)

# ── Evaluate ─────────────────────────────────────────────────────────────────
# edges contains pairs above threshold
detected = set()
if len(eann_result.edges) > 0:
    for edge in eann_result.edges:
        detected.add((int(edge[0]), int(edge[1])))

TP = gt_valid & detected
FP = detected - gt_valid
FN = gt_valid - detected
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

print(f"\n=== eANN Results ===")
print(f"GT: {len(gt_valid)}  Detected: {len(detected)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

# Save
np.savez(args.out, precision=prec, recall=rec, f1=f1,
         n_gt=len(gt_valid), n_detected=len(detected),
         TP=len(TP), FP=len(FP), FN=len(FN))
print(f"Results saved to {args.out}", flush=True)
