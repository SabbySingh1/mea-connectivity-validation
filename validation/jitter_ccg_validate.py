#!/usr/bin/env python3
"""
Jitter-corrected CCG connectivity inference.

For each pair of neurons:
  1. Compute raw CCG (±50ms, 1ms bins)
  2. Generate N_SURR surrogate CCGs by jittering each spike ±JITTER_MS
     (destroys 1-5ms synaptic peak but preserves burst co-firing structure)
  3. Corrected CCG = raw - mean(surrogates)
  4. Test for a significant peak in the 1-5ms or -5 to -1ms window
     z-score vs. the std of corrected CCG outside the peak window

Jitter window (25ms) > synaptic delay (1-5ms) so the peak is destroyed in surrogates.
Jitter window (25ms) << burst duration (2s) so burst structure is preserved.

Usage:
    python validation/jitter_ccg_validate.py
    python validation/jitter_ccg_validate.py --spikes path/to/spikes.npz --conn path/to/conn.npz
"""
import argparse, itertools, multiprocessing as mp
from itertools import repeat
from pathlib import Path
import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--spikes",  default=str(REPO_ROOT / "data" / "sim_cdkl5_spikes.npz"))
parser.add_argument("--conn",    default=str(REPO_ROOT / "data" / "sim_cdkl5_connectivity.npz"))
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

SPIKES_PATH = Path(args.spikes)
CONN_PATH   = Path(args.conn)

# ── Parameters ────────────────────────────────────────────────────────────────
WIN_MS    = 50       # CCG half-window in ms
BIN_MS    = 1        # CCG bin size in ms
JITTER_MS = 25       # jitter amplitude — larger than synaptic delay, smaller than burst
N_SURR    = 200      # number of surrogate CCGs per pair
ALPHA     = 0.001    # significance threshold
SYN_LO    = 1        # monosynaptic window low (ms)
SYN_HI    = 6        # monosynaptic window high (ms, exclusive)

N_BINS  = 2 * WIN_MS // BIN_MS + 1          # total CCG bins
LAGS    = np.arange(-WIN_MS, WIN_MS+1, BIN_MS)  # lag values in ms
EXC_IDX = np.where((LAGS >= SYN_LO) & (LAGS < SYN_HI))[0]   # excitatory peak bins
INH_IDX = np.where((LAGS > -SYN_HI) & (LAGS <= -SYN_LO))[0] # inhibitory peak bins
BG_IDX  = np.where((np.abs(LAGS) > SYN_HI) & (np.abs(LAGS) <= WIN_MS))[0]

# ── Load data ─────────────────────────────────────────────────────────────────
spk     = np.load(SPIKES_PATH, allow_pickle=True)
times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]

conn     = np.load(CONN_PATH,   allow_pickle=True)
pre_gt   = conn["pre_gid"].astype(int)
post_gt  = conn["post_gid"].astype(int)

duration    = times_s[-1] - times_s[0]
all_nodes   = np.unique(ids)
valid_nodes = [n for n in all_nodes if np.sum(ids == n) / duration >= 0.5]
pairs       = list(itertools.combinations(valid_nodes, 2))

print(f"Units: {len(valid_nodes)}  Pairs: {len(pairs)}")

gt_set    = set(zip(pre_gt, post_gt))
valid_set = set(valid_nodes)
gt_valid  = {(p, q) for p, q in gt_set if p in valid_set and q in valid_set}
print(f"GT connections: {len(gt_valid)}")

# ── Pre-build per-neuron spike arrays in ms ───────────────────────────────────
spike_dict = {}
for uid in valid_nodes:
    spike_dict[uid] = times_s[ids == uid] * 1000.0  # ms

# ── CCG computation ───────────────────────────────────────────────────────────
def compute_ccg(t_pre, t_post):
    """Fast CCG using searchsorted."""
    ccg = np.zeros(N_BINS, dtype=np.float64)
    for tp in t_pre:
        lo = np.searchsorted(t_post, tp - WIN_MS, 'left')
        hi = np.searchsorted(t_post, tp + WIN_MS, 'right')
        for tq in t_post[lo:hi]:
            lag = tq - tp
            b   = int(round(lag)) + WIN_MS
            if 0 <= b < N_BINS and b != WIN_MS:  # exclude lag=0
                ccg[b] += 1
    return ccg

def jitter_ccg_pair(args):
    uid_pre, uid_post, seed = args
    rng    = np.random.default_rng(seed)
    t_pre  = spike_dict[uid_pre]
    t_post = spike_dict[uid_post]

    raw = compute_ccg(t_pre, t_post)

    # Surrogate: jitter PRE spike times, keep post fixed
    surr_sum = np.zeros(N_BINS, dtype=np.float64)
    for _ in range(N_SURR):
        t_jit = t_pre + rng.uniform(-JITTER_MS, JITTER_MS, len(t_pre))
        t_jit = np.sort(t_jit)
        surr_sum += compute_ccg(t_jit, t_post)
    surr_mean = surr_sum / N_SURR

    corrected = raw - surr_mean

    # Test excitatory direction (pre→post, lag 1-5ms)
    peak_exc  = corrected[EXC_IDX].max() if len(EXC_IDX) else 0
    bg_std    = corrected[BG_IDX].std() + 1e-9
    bg_mean   = corrected[BG_IDX].mean()
    z_exc     = (peak_exc - bg_mean) / bg_std

    # Test inhibitory direction
    peak_inh  = corrected[INH_IDX].min() if len(INH_IDX) else 0
    z_inh     = (bg_mean - peak_inh) / bg_std  # negative peak → positive z

    return (uid_pre, uid_post, z_exc, z_inh, corrected)

if __name__ == '__main__':
    threshold = norm.ppf(1 - ALPHA)
    print(f"Running jitter-CCG (workers={args.workers}, surrogates={N_SURR}, "
          f"jitter=±{JITTER_MS}ms, threshold z={threshold:.2f})...")

    work = [(p, q, i) for i, (p, q) in enumerate(pairs)]

    with mp.Pool(args.workers) as pool:
        results = pool.map(jitter_ccg_pair, work)

    detected = set()
    for uid_pre, uid_post, z_exc, z_inh, _ in results:
        if z_exc > threshold:
            detected.add((uid_pre, uid_post))
        if z_inh > threshold:
            detected.add((uid_post, uid_pre))

    TP = gt_valid & detected
    FP = detected  - gt_valid
    FN = gt_valid  - detected

    p  = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
    r  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0

    print(f"\n=== Jitter-CCG vs Ground Truth ===")
    print(f"GT connections: {len(gt_valid)}  Detected: {len(detected)}")
    print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
    print(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
