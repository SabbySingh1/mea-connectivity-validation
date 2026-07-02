#!/usr/bin/env python3
"""
Simulate a realistic E/I network using Brian2, matched to CDKL5 MEA statistics.
- 129 neurons (80% E, 20% I)
- Leaky integrate-and-fire with adaptation (produces bursting)
- Distance-dependent connectivity
- Synaptic delays 1-5ms (monosynaptic range)
- 300s recording duration
Outputs:
  sim_brian2_spikes.npz       — spike times + unit IDs
  sim_brian2_connectivity.npz — pre/post/delay/weight ground truth
"""
import numpy as np
from pathlib import Path
import brian2 as b2

b2.start_scope()
b2.set_device('runtime')
b2.defaultclock.dt = 0.1 * b2.ms

OUT_DIR  = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)
RNG      = np.random.default_rng(42)

# --- network parameters ---
N        = 129
N_E      = 103   # ~80% excitatory
N_I      = 26    # ~20% inhibitory
DURATION = 300   * b2.second

# LIF + adaptation parameters (Brette & Gerstner 2005 AdEx simplified)
# Tuned to produce low mean firing rates with occasional bursts (CDKL5-like)
tau_m    = 20  * b2.ms    # membrane time constant
tau_ada  = 200 * b2.ms    # adaptation time constant
V_rest   = -70 * b2.mV
V_thresh = -50 * b2.mV
V_reset  = -65 * b2.mV
R        = 500 * b2.Mohm  # input resistance
a_ada    = 0.5 * b2.nS    # subthreshold adaptation
b_ada    = 20  * b2.pA    # spike-triggered adaptation

# Synaptic parameters
tau_E    = 5   * b2.ms
tau_I    = 10  * b2.ms
g_E      = 0.15 * b2.nS
g_I      = 0.25 * b2.nS
E_E      = 0   * b2.mV
E_I      = -80 * b2.mV

# Background drive — direct current injection (more controllable than Poisson)
# ~50pA needed to reach threshold from rest; heterogeneous to match rate distribution

# --- neuron model ---
eqs = '''
dv/dt  = (-(v - V_rest) - R*w + R*(g_e*(E_E - v) + g_i*(E_I - v)) + R*I_bg) / tau_m : volt
dw/dt  = (a_ada*(v - V_rest) - w) / tau_ada : amp
dg_e/dt = -g_e / tau_E : siemens
dg_i/dt = -g_i / tau_I : siemens
I_bg   : amp
'''

neurons = b2.NeuronGroup(N, eqs,
                          threshold='v > V_thresh',
                          reset='v = V_reset; w += b_ada',
                          namespace=dict(V_rest=V_rest, V_thresh=V_thresh,
                                         V_reset=V_reset, R=R, tau_m=tau_m,
                                         tau_ada=tau_ada, a_ada=a_ada, b_ada=b_ada,
                                         tau_E=tau_E, tau_I=tau_I,
                                         E_E=E_E, E_I=E_I),
                          method='euler')

# initialise
neurons.v     = V_rest + RNG.uniform(0, 5, N) * b2.mV
neurons.w     = 0 * b2.pA
neurons.g_e   = 0 * b2.nS
neurons.g_i   = 0 * b2.nS

# heterogeneous background drive (log-normal, matching real firing rate spread)
bg_rates = RNG.lognormal(np.log(0.95), 1.1, N)
bg_rates = np.clip(bg_rates, 0.1, 20.0)
# Map log-normal rate distribution to drive current
# LIF threshold: need ~50pA to reach threshold; scale by relative rate
# median unit (0.95 Hz) gets 50pA, max unit (20 Hz) gets ~1000pA
I_min, I_max = 48.0, 85.0
r_min, r_max = bg_rates.min(), bg_rates.max()
I_drive = I_min + (I_max - I_min) * (bg_rates - r_min) / (r_max - r_min + 1e-9)
neurons.I_bg = I_drive * b2.pA

# --- recurrent connectivity with 1-5ms delays ---
# place neurons on a 2D grid for distance-dependent connectivity
grid_side = int(np.ceil(np.sqrt(N)))
xs = (np.arange(N) % grid_side).astype(float)
ys = (np.arange(N) // grid_side).astype(float)

pre_list, post_list, delay_list, weight_list = [], [], [], []

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
        # connection probability decays with distance
        p_conn = 0.15 * np.exp(-dist / 3.0)
        if RNG.random() < p_conn:
            delay = RNG.uniform(1.0, 5.0)   # ms
            pre_list.append(i)
            post_list.append(j)
            delay_list.append(delay)
            weight_list.append(1.0 if i < N_E else -1.0)

pre_arr   = np.array(pre_list,   dtype=int)
post_arr  = np.array(post_list,  dtype=int)
delay_arr = np.array(delay_list, dtype=float)
weight_arr= np.array(weight_list, dtype=float)
print(f"Recurrent connections: {len(pre_arr)}")

# E→all synapses
exc_mask = pre_arr < N_E
inh_mask = ~exc_mask

S_E = b2.Synapses(neurons, neurons, on_pre='g_e += g_E',
                   namespace=dict(g_E=g_E))
S_E.connect(i=pre_arr[exc_mask].tolist(), j=post_arr[exc_mask].tolist())
S_E.delay = delay_arr[exc_mask] * b2.ms

S_I = b2.Synapses(neurons, neurons, on_pre='g_i += g_I',
                   namespace=dict(g_I=g_I))
S_I.connect(i=pre_arr[inh_mask].tolist(), j=post_arr[inh_mask].tolist())
S_I.delay = delay_arr[inh_mask] * b2.ms

# --- record spikes ---
spike_mon = b2.SpikeMonitor(neurons)

print("Running simulation (300s)...")
b2.run(DURATION, report='text', report_period=30*b2.second)

# --- extract spike trains ---
spkt_s = np.array(spike_mon.t / b2.second)
spkid  = np.array(spike_mon.i, dtype=int)
order  = np.argsort(spkt_s)
spkt_s, spkid = spkt_s[order], spkid[order]

duration = spkt_s[-1] - spkt_s[0]
sim_rates = np.array([np.sum(spkid == u) / duration for u in range(N)])
print(f"\nSim duration: {duration:.1f} s")
print(f"Units: {N}")
print(f"Rate — mean: {sim_rates.mean():.2f}  std: {sim_rates.std():.2f}  "
      f"min: {sim_rates.min():.2f}  max: {sim_rates.max():.2f}")
print(f"Percentiles [10,25,50,75,90]: {np.percentile(sim_rates,[10,25,50,75,90]).round(2)}")
print(f"Total spikes: {len(spkt_s)}")

# --- save ---
spk_out  = OUT_DIR / "sim_brian2_spikes.npz"
conn_out = OUT_DIR / "sim_brian2_connectivity.npz"
np.savez(spk_out,  spkt_s=spkt_s, spkid=spkid)
np.savez(conn_out, pre_gid=pre_arr, post_gid=post_arr,
         delay=delay_arr, weight=weight_arr)
print(f"\nSaved: {spk_out}")
print(f"Saved: {conn_out}")
