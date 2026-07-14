#!/usr/bin/env python3
"""
Run undirected STTC (Cutts & Eglen 2014) on a spike dataset and evaluate
against ground truth connectivity.

Unlike DSTTC, this evaluates undirectedly: a pair (A,B) is detected if either
the A→B or B→A z-score exceeds threshold. A pair is a TP if A→B or B→A
exists in the ground truth. This matches standard usage in the organoid MEA
literature (MEA-NAP, published CDKL5/Folic Acid MEA papers).

Parameters tuned for bursty in-vitro organoid data:
  - delta_t=10ms: standard value in published organoid MEA literature
  - num_surrogates=200: sufficient for alpha=0.05 (95th percentile)
  - jitter=False: circular shift surrogates — preserves burst structure,
    more principled null for bursty data than spike-centered jitter
  - alpha=0.05: standard threshold; Bonferroni correction applied for
    number of pairs

Usage:
    python validation/sttc_validate.py
    python validation/sttc_validate.py --spikes path/to/spikes.npz --conn path/to/conn.npz
"""
import argparse, importlib.util, sys, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
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

DSTTC = _load_class("sci_dsttc", "sci_dsttc.py", "directed_STTC")

parser = argparse.ArgumentParser()
parser.add_argument("--spikes",        default=str(REPO_ROOT / "data" / "sim_hdmea_burst_spikes.npz"))
parser.add_argument("--conn",          default=str(REPO_ROOT / "data" / "sim_hdmea_burst_connectivity.npz"))
parser.add_argument("--workers",       type=int,   default=8)
parser.add_argument("--delta-t",       type=float, default=10.0,  help="Coincidence window in ms (default 10ms)")
parser.add_argument("--surrogates",    type=int,   default=200,   help="Number of circular-shift surrogates")
parser.add_argument("--alpha",         type=float, default=0.05,  help="Significance level before Bonferroni")
args = parser.parse_args()

PARAMS = {
    "delta_t":       args.delta_t * 1e-3,
    "num_surrogates": args.surrogates,
    "jitter":        False,   # circular shift — correct for bursty data
    "alpha":         args.alpha,
}

# ── Load spikes ───────────────────────────────────────────────────────────────
spk     = np.load(args.spikes, allow_pickle=True)
times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]

conn     = np.load(args.conn, allow_pickle=True)
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

print(f"[STTC] Units: {len(valid_nodes)}, Pairs: {len(pairs)}")
print(f"[STTC] GT connections: {len(gt_pairs)}")
print(f"[STTC] delta_t={args.delta_t}ms  surrogates={args.surrogates}  alpha={args.alpha}")

model = DSTTC(params=PARAMS)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[STTC] Running (workers={args.workers})...")
    with mp.Pool(args.workers) as pool:
        raw = pool.starmap(model._test_connection_pair,
                           zip(repeat(times_s), repeat(ids), pairs))

    # Undirected detection: pair (A,B) is significant if either direction
    # exceeds threshold. Bonferroni correction for number of pairs.
    bonf_alpha = args.alpha / len(pairs)
    threshold  = norm.ppf(1 - bonf_alpha)

    detected_pairs = set()  # undirected: store as frozenset {A,B}
    for (p, q), result in zip(pairs, raw):
        # result = (zval_BA, STTC_BA, zval_AB, STTC_AB, pair)
        zval_BA = result[0]
        zval_AB = result[2]
        if max(zval_BA, zval_AB) > threshold:
            detected_pairs.add(frozenset([p, q]))

    print(f"[STTC] Detected pairs: {len(detected_pairs)} ({100*len(detected_pairs)/len(pairs):.1f}%)")

    # ── Evaluate undirectedly ─────────────────────────────────────────────────
    # Build undirected GT: a pair {A,B} is positive if A→B OR B→A in ground truth
    valid_set  = set(valid_nodes)
    gt_valid   = {k: v for k, v in gt_pairs.items() if k[0] in valid_set and k[1] in valid_set}
    gt_undirected = set(frozenset(k) for k in gt_valid.keys())

    TP = gt_undirected & detected_pairs
    FP = detected_pairs - gt_undirected
    FN = gt_undirected - detected_pairs

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    recall    = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== STTC vs Ground Truth (undirected) ===")
    print(f"GT pairs with any connection: {len(gt_undirected)}")
    print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"\nBonferroni threshold: z > {threshold:.2f}  (alpha={args.alpha} / {len(pairs)} pairs)")
