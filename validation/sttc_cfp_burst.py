#!/usr/bin/env python3
"""
STTC with circular shift surrogates (MEA-NAP method) and
Conditional Firing Probability (CFP) on sim_hdmea_burst.

STTC circular shift: compute real STTC, then shift one spike train
by random circular offsets 1000 times to build null distribution.
Connection detected if real STTC > 95th percentile of null.

CFP: probability of neuron j firing within window tau of neuron i spike.
Threshold via circular shift surrogates same way.
"""
import numpy as np
from statsmodels.stats.multitest import multipletests

SPIKES_PATH = "/private/tmp/mea-connectivity-validation/data/sim_hdmea_burst_spikes.npz"
CONN_PATH   = "/private/tmp/mea-connectivity-validation/data/sim_hdmea_burst_connectivity.npz"
N_SURR      = 1000
ALPHA       = 0.05
DT          = 0.025   # 25ms tiling window for STTC
TAU_CFP     = 0.050   # 50ms window for CFP
SEED        = 42

# ── Load data ─────────────────────────────────────────────────────────────────
spk  = np.load(SPIKES_PATH, allow_pickle=True)
conn = np.load(CONN_PATH,   allow_pickle=True)

times_s = spk["spkt_s"].astype(float)
ids     = spk["spkid"].astype(int)
order   = np.argsort(times_s)
times_s, ids = times_s[order], ids[order]

duration    = times_s[-1] - times_s[0]
t0          = times_s[0]
all_nodes   = np.unique(ids)
valid_nodes = np.array([n for n in all_nodes if np.sum(ids == n)/duration >= 0.5])
N = len(valid_nodes)
node_map = {n: i for i, n in enumerate(valid_nodes)}
print(f"Units: {N}, Duration: {duration:.1f}s", flush=True)

# Per-neuron spike arrays
spikes = []
mask = np.isin(ids, valid_nodes)
t_filt = times_s[mask] - t0
id_filt = ids[mask]
for i, node in enumerate(valid_nodes):
    spikes.append(t_filt[id_filt == node])

# Ground truth
pre_gt  = conn["pre_gid"].astype(int)
post_gt = conn["post_gid"].astype(int)
valid_set = set(valid_nodes.tolist())
gt_pairs = {(p, q) for p, q in zip(pre_gt, post_gt) if p in valid_set and q in valid_set}
print(f"GT connections: {len(gt_pairs)}", flush=True)

rng = np.random.default_rng(SEED)


def sttc(sp_a, sp_b, dt, T):
    """Spike Time Tiling Coefficient between two spike trains."""
    if len(sp_a) == 0 or len(sp_b) == 0:
        return 0.0
    # P_A: proportion of spikes in A within dt of a spike in B
    ta = 0.0
    for t in sp_b:
        ta += min(t + dt, T) - max(t - dt, 0.0)
    ta = min(ta, T) / T

    tb = 0.0
    for t in sp_a:
        tb += min(t + dt, T) - max(t - dt, 0.0)
    tb = min(tb, T) / T

    # Count spikes in A within dt of spikes in B
    pa = 0
    for t in sp_a:
        if np.any(np.abs(sp_b - t) <= dt):
            pa += 1
    pa = pa / len(sp_a)

    pb = 0
    for t in sp_b:
        if np.any(np.abs(sp_a - t) <= dt):
            pb += 1
    pb = pb / len(sp_b)

    denom_a = 1 - ta * pa
    denom_b = 1 - tb * pb
    if abs(denom_a) < 1e-10 or abs(denom_b) < 1e-10:
        return 0.0
    return 0.5 * ((pa - ta) / denom_a + (pb - tb) / denom_b)


def cfp(sp_pre, sp_post, tau, T):
    """Conditional Firing Probability: fraction of pre spikes followed by
    a post spike within tau ms."""
    if len(sp_pre) == 0 or len(sp_post) == 0:
        return 0.0
    count = 0
    for t in sp_pre:
        if np.any((sp_post > t) & (sp_post <= t + tau)):
            count += 1
    return count / len(sp_pre)


def circular_shift(sp, T, rng):
    """Circularly shift spike train by a random offset in [0.1*T, 0.9*T]."""
    shift = rng.uniform(0.1 * T, 0.9 * T)
    return np.mod(sp + shift, T)


# ── Precompute fast STTC using numpy vectorized version ──────────────────────
def sttc_fast(sp_a, sp_b, dt, T):
    if len(sp_a) == 0 or len(sp_b) == 0:
        return 0.0
    # ta: total time within dt of any spike in b
    b_sorted = np.sort(sp_b)
    intervals_b = np.minimum(b_sorted + dt, T) - np.maximum(b_sorted - dt, 0.0)
    # merge overlapping intervals
    starts = np.maximum(b_sorted - dt, 0.0)
    ends   = np.minimum(b_sorted + dt, T)
    order  = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    merged_len = 0.0
    cur_s, cur_e = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged_len += cur_e - cur_s
            cur_s, cur_e = s, e
    merged_len += cur_e - cur_s
    ta = merged_len / T

    starts_a = np.maximum(np.sort(sp_a) - dt, 0.0)
    ends_a   = np.minimum(np.sort(sp_a) + dt, T)
    order_a  = np.argsort(starts_a)
    starts_a, ends_a = starts_a[order_a], ends_a[order_a]
    merged_a = 0.0
    cur_s, cur_e = starts_a[0], ends_a[0]
    for s, e in zip(starts_a[1:], ends_a[1:]):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged_a += cur_e - cur_s
            cur_s, cur_e = s, e
    merged_a += cur_e - cur_s
    tb = merged_a / T

    # pa: fraction of spikes in a within dt of any spike in b
    idx = np.searchsorted(b_sorted, sp_a)
    close = np.zeros(len(sp_a), dtype=bool)
    for k, (t, ix) in enumerate(zip(sp_a, idx)):
        for jj in [ix - 1, ix]:
            if 0 <= jj < len(b_sorted) and abs(b_sorted[jj] - t) <= dt:
                close[k] = True
                break
    pa = close.mean()

    a_sorted = np.sort(sp_a)
    idx2 = np.searchsorted(a_sorted, sp_b)
    close2 = np.zeros(len(sp_b), dtype=bool)
    for k, (t, ix) in enumerate(zip(sp_b, idx2)):
        for jj in [ix - 1, ix]:
            if 0 <= jj < len(a_sorted) and abs(a_sorted[jj] - t) <= dt:
                close2[k] = True
                break
    pb = close2.mean()

    denom_a = 1 - ta * pa
    denom_b = 1 - tb * pb
    if abs(denom_a) < 1e-10 or abs(denom_b) < 1e-10:
        return 0.0
    return 0.5 * ((pa - ta) / denom_a + (pb - tb) / denom_b)


def cfp_fast(sp_pre, sp_post, tau):
    if len(sp_pre) == 0 or len(sp_post) == 0:
        return 0.0
    sp_post_s = np.sort(sp_post)
    count = 0
    for t in sp_pre:
        idx = np.searchsorted(sp_post_s, t, side='right')
        if idx < len(sp_post_s) and sp_post_s[idx] <= t + tau:
            count += 1
    return count / len(sp_pre)


# ── Method 1: STTC with circular shift surrogates ────────────────────────────
print(f"\n=== STTC with Circular Shift Surrogates (dt={DT*1000:.0f}ms, {N_SURR} surrogates) ===", flush=True)
T = duration

detected_sttc = set()
total_pairs = N * (N - 1)
done = 0

for i in range(N):
    if i % 20 == 0:
        print(f"  Neuron {i}/{N}", flush=True)
    sp_i = spikes[i]
    for j in range(N):
        if i == j:
            continue
        sp_j = spikes[j]
        real_val = sttc_fast(sp_i, sp_j, DT, T)

        # Surrogate distribution: shift sp_j
        surr_vals = np.zeros(N_SURR)
        for s in range(N_SURR):
            sp_j_shift = circular_shift(sp_j, T, rng)
            surr_vals[s] = sttc_fast(sp_i, sp_j_shift, DT, T)

        threshold = np.percentile(surr_vals, 95)
        if real_val > threshold:
            detected_sttc.add((valid_nodes[i], valid_nodes[j]))

TP = gt_pairs & detected_sttc
FP = detected_sttc - gt_pairs
FN = gt_pairs - detected_sttc
prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0
rec  = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_sttc)}")
print(f"TP: {len(TP)}  FP: {len(FP)}  FN: {len(FN)}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}", flush=True)

# ── Method 2: CFP with circular shift surrogates ──────────────────────────────
print(f"\n=== CFP with Circular Shift Surrogates (tau={TAU_CFP*1000:.0f}ms, {N_SURR} surrogates) ===", flush=True)

detected_cfp = set()
for i in range(N):
    if i % 20 == 0:
        print(f"  Neuron {i}/{N}", flush=True)
    sp_i = spikes[i]
    for j in range(N):
        if i == j:
            continue
        sp_j = spikes[j]
        real_val = cfp_fast(sp_i, sp_j, TAU_CFP)

        surr_vals = np.zeros(N_SURR)
        for s in range(N_SURR):
            sp_j_shift = circular_shift(sp_j, T, rng)
            surr_vals[s] = cfp_fast(sp_i, sp_j_shift, TAU_CFP)

        threshold = np.percentile(surr_vals, 95)
        if real_val > threshold:
            detected_cfp.add((valid_nodes[i], valid_nodes[j]))

TP2 = gt_pairs & detected_cfp
FP2 = detected_cfp - gt_pairs
FN2 = gt_pairs - detected_cfp
prec2 = len(TP2)/(len(TP2)+len(FP2)) if (len(TP2)+len(FP2)) > 0 else 0
rec2  = len(TP2)/(len(TP2)+len(FN2)) if (len(TP2)+len(FN2)) > 0 else 0
f12   = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) > 0 else 0
print(f"GT: {len(gt_pairs)}  Detected: {len(detected_cfp)}")
print(f"TP: {len(TP2)}  FP: {len(FP2)}  FN: {len(FN2)}")
print(f"Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}", flush=True)

np.savez("/private/tmp/sttc_cfp_burst_results.npz",
         sttc_f1=f1, sttc_prec=prec, sttc_rec=rec,
         cfp_f1=f12, cfp_prec=prec2, cfp_rec=rec2)
print("\nDone.", flush=True)
