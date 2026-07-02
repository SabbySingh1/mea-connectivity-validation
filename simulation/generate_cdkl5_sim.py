#!/usr/bin/env python3
"""
Generate a simulated spike dataset matching CDKL5 MEA statistics.
Ground truth connectivity uses 1-5ms synaptic delays (monosynaptic range).
Outputs:
  sim_cdkl5_spikes.npz       — spike times + unit IDs
  sim_cdkl5_connectivity.npz — pre/post/delay/weight ground truth
"""
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# --- match real data stats ---
DURATION    = 299.0      # seconds
N_UNITS     = 129
# log-normal fit to real distribution (mean=2.24, median=0.95, max=19.98)
RATE_MU     = np.log(0.95)
RATE_SIGMA  = 1.1

# --- connectivity ---
CONN_PROB   = 0.08       # ~8% connection probability (cortical-like)
DELAY_MIN   = 1.0        # ms
DELAY_MAX   = 5.0        # ms
SYN_WEIGHT  = 0.04       # fraction of baseline rate added post-synaptically
EXC_FRAC    = 0.8        # 80% excitatory, 20% inhibitory

# ── Step 1: draw firing rates ──────────────────────────────────────────────
rates = RNG.lognormal(RATE_MU, RATE_SIGMA, N_UNITS)
rates = np.clip(rates, 0.06, 19.98)

# ── Step 2: draw connectivity ──────────────────────────────────────────────
pre_list, post_list, delay_list, weight_list = [], [], [], []
for i in range(N_UNITS):
    for j in range(N_UNITS):
        if i == j:
            continue
        if RNG.random() < CONN_PROB:
            delay  = RNG.uniform(DELAY_MIN, DELAY_MAX)
            is_exc = RNG.random() < EXC_FRAC
            w      = SYN_WEIGHT if is_exc else -SYN_WEIGHT
            pre_list.append(i)
            post_list.append(j)
            delay_list.append(delay)
            weight_list.append(w)

pre_gt    = np.array(pre_list,   dtype=int)
post_gt   = np.array(post_list,  dtype=int)
delay_gt  = np.array(delay_list, dtype=float)
weight_gt = np.array(weight_list, dtype=float)
print(f"Ground truth connections: {len(pre_gt)}")

# ── Step 3: generate spike trains with synaptic influence ──────────────────
# Each unit fires as an inhomogeneous Poisson process.
# For each connection, post-synaptic unit gets a rate bump at spike+delay.

# baseline spikes first (homogeneous Poisson)
spike_times_dict = {}
for uid in range(N_UNITS):
    n_spikes = RNG.poisson(rates[uid] * DURATION)
    spk = np.sort(RNG.uniform(0, DURATION, n_spikes))
    spike_times_dict[uid] = spk.tolist()

# inject synaptic spikes: for each connection, for each pre spike,
# add a probabilistic post spike at spike_time + delay (in seconds)
for p, q, d_ms, w in zip(pre_gt, post_gt, delay_gt, weight_gt):
    d_s = d_ms / 1000.0
    pre_spikes = np.array(spike_times_dict[p])
    if len(pre_spikes) == 0:
        continue
    post_times = pre_spikes + d_s
    post_times = post_times[post_times < DURATION]
    # 8% coupling probability — enough for a detectable CCG peak, low enough not to inflate rates
    n_add = RNG.binomial(len(post_times), 0.08)
    if n_add > 0:
        chosen = RNG.choice(post_times, size=n_add, replace=False)
        spike_times_dict[q] = sorted(spike_times_dict[q] + chosen.tolist())

# ── Step 4: flatten to arrays ──────────────────────────────────────────────
all_spkt, all_ids = [], []
for uid, spks in spike_times_dict.items():
    all_spkt.extend(spks)
    all_ids.extend([uid] * len(spks))

all_spkt = np.array(all_spkt, dtype=float)
all_ids  = np.array(all_ids,  dtype=int)
order    = np.argsort(all_spkt)
all_spkt, all_ids = all_spkt[order], all_ids[order]

# ── Step 5: verify stats ───────────────────────────────────────────────────
dur = all_spkt[-1] - all_spkt[0]
sim_rates = np.array([np.sum(all_ids == u) / dur for u in range(N_UNITS)])
print(f"Sim duration: {dur:.1f} s")
print(f"Sim units: {N_UNITS}")
print(f"Sim rate — mean: {sim_rates.mean():.2f}  std: {sim_rates.std():.2f}  "
      f"min: {sim_rates.min():.2f}  max: {sim_rates.max():.2f}")
print(f"Sim percentiles [10,25,50,75,90]: {np.percentile(sim_rates, [10,25,50,75,90]).round(2)}")
print(f"Total spikes: {len(all_spkt)}")

# ── Step 6: save ───────────────────────────────────────────────────────────
spk_out  = OUT_DIR / "sim_cdkl5_spikes.npz"
conn_out = OUT_DIR / "sim_cdkl5_connectivity.npz"

np.savez(spk_out,  spkt_s=all_spkt, spkid=all_ids)
np.savez(conn_out, pre_gid=pre_gt, post_gid=post_gt,
         delay=delay_gt, weight=weight_gt)

print(f"\nSaved spikes:       {spk_out}")
print(f"Saved connectivity: {conn_out}")
