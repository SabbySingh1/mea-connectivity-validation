#!/usr/bin/env python3
"""
HD-MEA simulation matched to real CDKL5 burst dynamics.

Target statistics (real CDKL5 R59X well000, 300s recording):
  - 129 neurons, real XY positions
  - Mean rate: 2.24 Hz, std: 3.77 Hz
  - ~18 network bursts / 300s (mean IBI ~17s)
  - ~100% of spikes fall within network bursts (ISI<50ms at population level)

Network burst mechanism:
  - Burst gate TimedArray: 1 during burst windows, 0 between
  - Between bursts: each neuron receives I_base (sub-threshold, 15pA) → silence
  - During bursts:  each neuron receives I_burst (neuron-specific, supra-threshold)
    scaled proportionally to its real firing rate → preserves rate heterogeneity

Outputs:
  sim_hdmea_burst_spikes.npz
  sim_hdmea_burst_connectivity.npz
"""
import numpy as np
from pathlib import Path
import brian2 as b2

b2.start_scope()
b2.set_device('runtime')
b2.defaultclock.dt = 0.1 * b2.ms

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)
RNG     = np.random.default_rng(42)

# ── Real unit data from metrics_curated (unit_id, rate_Hz, loc_x_um, loc_y_um) ──
UNITS_RAW = [
    (5,   2.706, -1.215,   915.333),
    (7,   0.308,  0.000,  1051.894),
    (11,  0.535, 132.370,  864.860),
    (14,  3.301,  77.537,  907.086),
    (21,  0.609, 289.499,  137.733),
    (25,  0.716, 311.589,  297.676),
    (26,  0.542, 222.024,  371.036),
    (31,  0.144, 273.494, 1312.764),
    (32,  2.532, 273.866, 1350.426),
    (43,  0.110, 208.150, 1886.917),
    (48,  0.870, 371.527,   73.508),
    (49,  1.766, 352.779,   95.238),
    (50, 16.849, 380.296,   92.135),
    (52,  2.294, 368.661,  113.472),
    (53,  1.532, 367.986,  292.494),
    (55,  1.147, 450.142,  518.192),
    (56,  1.719, 443.641,  513.264),
    (57,  0.552, 446.623,  501.495),
    (59, 14.806, 439.815,  512.499),
    (60,  5.023, 445.099,  527.887),
    (63,  1.007, 459.314,  991.997),
    (66,  1.194, 465.959, 1017.846),
    (68,  0.502, 462.628, 1015.468),
    (79,  3.569, 464.579, 1485.350),
    (81,  0.522, 418.384, 1589.903),
    (83,  0.575, 410.940, 1594.981),
    (84,  7.732, 395.203, 1567.990),
    (86,  0.234, 420.946, 1679.296),
    (87,  0.234, 415.032, 1698.989),
    (95,  0.441, 595.288,  433.271),
    (96,  1.171, 566.194,  411.715),
    (101, 0.334, 601.536,  778.101),
    (105, 1.043, 584.808,  908.208),
    (106, 0.672, 595.864,  937.871),
    (107, 0.619, 593.852,  928.313),
    (122, 1.314, 527.451, 1599.791),
    (123, 4.398, 513.532, 1585.513),
    (128, 2.074, 530.934, 1874.403),
    (135, 1.064, 711.771, 1133.659),
    (145, 3.047, 836.359, 1988.276),
    (151, 0.649, 1096.081,   33.523),
    (152, 1.030, 1102.046,   57.927),
    (156, 3.261, 1110.493,  431.212),
    (162, 1.696, 1066.812, 1008.973),
    (164, 0.699, 1133.417, 1668.592),
    (168, 0.639, 1306.692,  384.077),
    (169, 0.241, 1293.917,  400.849),
    (173, 0.900, 1194.338,  685.467),
    (179, 0.602, 1244.587, 1326.972),
    (181, 3.435, 1211.677, 1322.928),
    (182, 1.689, 1286.877, 1438.804),
    (183, 1.127, 1276.190, 1449.266),
    (186, 0.217, 1239.821, 1765.909),
    (187, 1.094, 1248.926, 1799.562),
    (192, 0.488, 1312.983, 1899.224),
    (199, 0.127, 1465.347,  544.034),
    (202, 1.181, 1385.454,  816.809),
    (203, 1.097, 1365.511,  797.805),
    (211, 0.569, 1534.488,  701.313),
    (213, 1.254, 1538.835,  708.370),
    (215, 2.652, 1531.477, 1555.459),
    (216, 7.512, 1527.536, 1579.756),
    (217, 0.870, 1543.126, 1574.149),
    (219, 0.448, 1704.905,  693.087),
    (223, 0.515, 1763.513, 1929.902),
    (226, 0.736, 1758.984, 1945.212),
    (227, 2.592, 1780.276, 1932.679),
    (235, 1.304, 1835.832,  100.333),
    (239, 1.254, 1799.231, 1074.637),
    (242, 1.445, 1812.107, 1112.693),
    (245, 1.582, 1802.844, 1099.572),
    (247, 0.538, 1877.661, 1344.931),
    (254, 1.716, 1898.300, 1600.072),
    (257, 0.622, 1891.300, 1615.375),
    (261, 1.151, 1865.966, 1770.539),
    (262, 6.612, 1878.481, 1742.098),
    (266, 1.622, 1830.775, 1858.777),
    (267, 0.515, 1854.854, 1857.351),
    (271, 3.100, 1838.946, 1995.814),
    (272, 0.950, 2209.810,  440.216),
    (273,19.462, 2210.927,  428.486),
    (276, 0.699, 2121.838,  651.000),
    (277, 1.652, 2155.453,  684.152),
    (284, 1.134, 2389.160,  219.815),
    (285, 5.314, 2433.054,  202.705),
    (291, 0.656, 2426.847,  964.702),
    (292, 2.435, 2406.844,  977.982),
    (293, 0.425, 2402.270, 1005.357),
    (298, 8.783, 2447.861,  219.313),
    (303, 0.886, 2509.679, 1528.484),
    (305, 0.525, 2539.227, 1561.590),
    (315, 0.806, 2540.528, 2071.397),
    (318,17.278, 2713.417,  138.169),
    (322, 0.334, 2658.225, 1093.585),
    (323, 0.920, 2669.679, 1082.834),
    (325, 0.244, 2674.661, 1112.259),
    (327, 0.094, 2802.344, 1411.340),
    (328,19.983, 2791.612, 1429.387),
    (332, 1.562, 3051.040,  993.606),
    (333,14.716, 3033.043,  981.555),
    (334, 0.910, 3069.213, 1352.415),
    (338, 2.552, 2932.595, 1444.699),
    (354, 1.522, 3305.915,  231.339),
    (355, 0.331, 3307.846,  234.948),
    (356, 0.719, 3280.560,  379.085),
    (358, 0.304, 3279.990,  388.157),
    (368, 1.090, 3394.152, 1002.475),
    (371, 0.478, 3390.856, 1385.523),
    (372, 0.753, 3420.274, 1372.973),
    (375, 2.856, 3366.985, 1415.871),
    (376, 0.813, 3435.589, 1766.651),
    (379, 0.468, 3489.403,  177.731),
    (383, 0.087, 3520.418, 1319.112),
    (385, 0.057, 3034.553, 1687.263),
    (388, 0.572, 3444.144, 1780.058),
    (390, 2.669, 3652.784,  152.706),
    (393, 1.860, 3648.599,  178.031),
    (395, 0.060, 3711.519,  514.585),
    (396, 1.351, 3693.591,  719.529),
    (398, 0.351, 3707.976,  731.778),
    (402, 2.204, 3686.268, 1404.735),
    (411, 0.445, 3773.166,  296.139),
    (412, 0.258, 3763.255,  299.615),
    (413, 9.237, 3769.485,  270.807),
    (418, 0.328, 3747.584,  517.677),
    (420, 0.221, 3832.243,  921.294),
    (421, 0.194, 3734.070,  954.708),
    (424, 8.863, 3779.197, 1288.914),
    (426, 0.331, 3794.616, 1318.600),
]

N = len(UNITS_RAW)
unit_ids   = np.array([u[0] for u in UNITS_RAW])
real_rates = np.array([u[1] for u in UNITS_RAW])
xs         = np.array([u[2] for u in UNITS_RAW])
ys         = np.array([u[3] for u in UNITS_RAW])

print(f"Units: {N}")
print(f"MEA extent: x={xs.min():.0f}-{xs.max():.0f} um  y={ys.min():.0f}-{ys.max():.0f} um")
print(f"Real rate: mean={real_rates.mean():.2f}  std={real_rates.std():.2f} Hz")

# ── E/I assignment ────────────────────────────────────────────────────────────
is_exc = RNG.random(N) < 0.8
N_E = is_exc.sum()
N_I = (~is_exc).sum()
print(f"E: {N_E}  I: {N_I}")

# ── Distance-dependent recurrent connectivity ─────────────────────────────────
COND_VEL  = 300.0   # um/ms = 0.3 m/s
SYN_DELAY = 1.0     # ms

pre_list, post_list, delay_list, weight_list = [], [], [], []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
        if is_exc[i] and is_exc[j]:
            p0, lam = 0.15, 500.0
        elif is_exc[i] and not is_exc[j]:
            p0, lam = 0.20, 500.0
        elif not is_exc[i] and is_exc[j]:
            p0, lam = 0.30, 200.0
        else:
            p0, lam = 0.15, 200.0
        if RNG.random() < p0 * np.exp(-dist / lam):
            delay = np.clip(dist / COND_VEL + SYN_DELAY, 1.0, 5.0)
            pre_list.append(i)
            post_list.append(j)
            delay_list.append(delay)
            weight_list.append(1.0 if is_exc[i] else -1.0)

pre_arr    = np.array(pre_list,    dtype=int)
post_arr   = np.array(post_list,   dtype=int)
delay_arr  = np.array(delay_list,  dtype=float)
weight_arr = np.array(weight_list, dtype=float)
print(f"Recurrent connections: {len(pre_arr)}")
print(f"Delay range: {delay_arr.min():.2f} - {delay_arr.max():.2f} ms")

# ── Burst timing ──────────────────────────────────────────────────────────────
# Real: 18 bursts in 300s, mean IBI=17.39s
# Burst duration: total_time - 17×IBI = 300 - 17×17.39 ≈ 4.4s → ~244ms/burst
# We use 400ms burst duration to get adequate spike counts per burst
N_BURSTS     = 18
BURST_DUR_S  = 0.28    # 280ms per burst (calibrated for mean rate ~2.24 Hz)

burst_onsets = []
t = 5.0
while len(burst_onsets) < N_BURSTS and t < 296.0:
    burst_onsets.append(t)
    ibi = max(RNG.exponential(17.0), 1.5)
    t += ibi
burst_onsets = np.array(burst_onsets[:N_BURSTS])
print(f"\nBurst onsets ({len(burst_onsets)} bursts), IBI: "
      f"mean={np.diff(burst_onsets).mean():.1f}s  std={np.diff(burst_onsets).std():.1f}s")

# Gate signal: 1.0 during burst, 0.0 between
DT_TA   = 0.001
n_steps = int(300.0 / DT_TA) + 2
gate_trace = np.zeros(n_steps)
for t0 in burst_onsets:
    i0 = int(t0 / DT_TA)
    i1 = min(n_steps, int((t0 + BURST_DUR_S) / DT_TA))
    gate_trace[i0:i1] = 1.0

burst_gate_ta = b2.TimedArray(gate_trace, dt=DT_TA * b2.second)

# ── Per-neuron burst drive (preserves real rate heterogeneity) ────────────────
# During burst, neuron i fires at rate proportional to real_rates[i].
# We want mean in-burst rate ≈ real_rate / burst_fraction:
#   burst_fraction = N_BURSTS * BURST_DUR_S / 300 = 18*0.4/300 = 2.4%
# Mean target in-burst rate = 2.24 Hz / 0.024 = 93 Hz (for mean neuron)
# Map real_rates linearly to I_burst in [I_burst_min, I_burst_max]
# such that mean → 93 Hz during burst. Calibrated empirically.
#
# LIF neuron (R=500MOhm, tau_m=20ms, V_thresh=-50mV, V_reset=-65mV, V_rest=-70mV):
#   With b_ada=20pA and f=93Hz, adaptation load = b * f * tau_ada = 20e-12 * 93 * 0.2 = 372pA
#   So need R*I >> 20mV + 500MOhm * 372pA = 20mV + 186mV → I >> 406pA (too high)
#   With tau_ada=200ms and 400ms burst, adaptation builds but decays partially.
#   In practice: aim for ~80-100 Hz from lower I by accepting adaptation limits firing.
#   Empirically calibrated: I_burst = 130pA → ~80 Hz during burst (after adaptation).

# LIF rate calibration (b_ada=0, I_base=15pA, R=500MOhm, tau_m=20ms):
#   T_isi = tau_m * ln((0.5*I_total-5)/(0.5*I_total-20))  [mV units: R*I(pA)=0.5*I mV]
#   I_threshold = 40 pA (total), I_BASE=15 pA → I_burst_min = 26 pA for slowest neurons
#   Target mean rate = 2.24 Hz / burst_fraction = 2.24/0.024 ≈ 93 Hz during burst
#   → I_total ≈ 201 pA → I_burst_mean ≈ 186 pA
#   Linear mapping: I_burst = I_min + (rate/rate_max) * (I_max - I_min)
I_BURST_MIN  =  26.0   # pA — just above threshold (40pA total): lowest-rate neurons
I_BURST_MAX  = 500.0   # pA — highest-rate neurons (20 Hz → ~820 Hz in burst)
I_BASE       =  15.0   # pA sub-threshold base (neurons silent between bursts)

# Linear mapping preserves relative rate ordering and produces realistic range
I_burst_arr = I_BURST_MIN + (real_rates / real_rates.max()) * (I_BURST_MAX - I_BURST_MIN)
print(f"\nPer-neuron burst drive: min={I_burst_arr.min():.1f}  mean={I_burst_arr.mean():.1f}  "
      f"max={I_burst_arr.max():.1f} pA")

# ── LIF + adaptation model ────────────────────────────────────────────────────
DURATION = 300 * b2.second
tau_m    = 20  * b2.ms
tau_ada  = 200 * b2.ms
V_rest   = -70 * b2.mV
V_thresh = -50 * b2.mV
V_reset  = -65 * b2.mV
R        = 500 * b2.Mohm
a_ada    = 0.5 * b2.nS
b_ada    = 0   * b2.pA   # no adaptation — burst termination is handled by gate signal
tau_E    = 5   * b2.ms
tau_I    = 10  * b2.ms
g_E      = 0.15 * b2.nS
g_I      = 0.25 * b2.nS
E_E      =  0  * b2.mV
E_I      = -80 * b2.mV

# Total drive = I_base (always) + I_burst * gate(t) (only during burst)
# Between bursts: I_base = 15pA < threshold (40pA needed) → silence
# During bursts:  I_base + I_burst (neuron-specific) → heterogeneous firing rates
eqs = '''
dv/dt  = (-(v - V_rest) - R*w + R*(g_e*(E_E - v) + g_i*(E_I - v)) + R*(I_base + I_burst * burst_gate_ta(t))) / tau_m : volt
dw/dt  = (a_ada*(v - V_rest) - w) / tau_ada : amp
dg_e/dt = -g_e / tau_E : siemens
dg_i/dt = -g_i / tau_I : siemens
I_base  : amp
I_burst : amp
'''

neurons = b2.NeuronGroup(N, eqs,
    threshold='v > V_thresh',
    reset='v = V_reset; w += b_ada',
    namespace=dict(V_rest=V_rest, V_thresh=V_thresh, V_reset=V_reset,
                   R=R, tau_m=tau_m, tau_ada=tau_ada, a_ada=a_ada, b_ada=b_ada,
                   tau_E=tau_E, tau_I=tau_I, E_E=E_E, E_I=E_I,
                   burst_gate_ta=burst_gate_ta),
    method='euler')

neurons.v       = V_rest + RNG.uniform(0, 5, N) * b2.mV
neurons.w       = 0 * b2.pA
neurons.g_e     = 0 * b2.nS
neurons.g_i     = 0 * b2.nS
neurons.I_base  = I_BASE       * b2.pA              # same for all (sub-threshold)
neurons.I_burst = I_burst_arr  * b2.pA              # neuron-specific

# ── Recurrent synapses ────────────────────────────────────────────────────────
exc_mask = weight_arr > 0
inh_mask = ~exc_mask

S_E = b2.Synapses(neurons, neurons, on_pre='g_e += g_E', namespace=dict(g_E=g_E))
S_E.connect(i=pre_arr[exc_mask].tolist(), j=post_arr[exc_mask].tolist())
S_E.delay = delay_arr[exc_mask] * b2.ms

S_I = b2.Synapses(neurons, neurons, on_pre='g_i += g_I', namespace=dict(g_I=g_I))
S_I.connect(i=pre_arr[inh_mask].tolist(), j=post_arr[inh_mask].tolist())
S_I.delay = delay_arr[inh_mask] * b2.ms

spike_mon = b2.SpikeMonitor(neurons)

print("\nRunning simulation (300s)...")
b2.run(DURATION, report='text', report_period=30*b2.second)

# ── Results ───────────────────────────────────────────────────────────────────
spkt_s = np.array(spike_mon.t / b2.second)
spkid  = np.array(spike_mon.i, dtype=int)
order  = np.argsort(spkt_s)
spkt_s, spkid = spkt_s[order], spkid[order]

rec_dur   = spkt_s[-1] - spkt_s[0] if len(spkt_s) > 0 else 300.0
sim_rates = np.array([np.sum(spkid == u) / rec_dur for u in range(N)])

print(f"\n=== Simulation Results ===")
print(f"Duration: {rec_dur:.1f} s  |  Total spikes: {len(spkt_s)}")
print(f"Rate — mean: {sim_rates.mean():.2f}  std: {sim_rates.std():.2f}  "
      f"min: {sim_rates.min():.2f}  max: {sim_rates.max():.2f}  [target: 2.24±3.77 Hz]")
print(f"Percentiles [10,25,50,75,90]: {np.percentile(sim_rates,[10,25,50,75,90]).round(2)}")

if len(spkt_s) > 100:
    isi_all   = np.diff(spkt_s)
    frac_5ms  = (isi_all < 0.005).mean()
    frac_20ms = (isi_all < 0.020).mean()
    frac_50ms = (isi_all < 0.050).mean()
    print(f"\nISI fractions (pooled population):")
    print(f"  ISI<5ms:  {100*frac_5ms:.1f}%  (real: 78.7%)")
    print(f"  ISI<20ms: {100*frac_20ms:.1f}%  (real: 98.3%)")
    print(f"  ISI<50ms: {100*frac_50ms:.1f}%  (real: 100%)")

    bin_sz = 0.010
    bins   = np.arange(0, 300, bin_sz)
    pop_rate, _ = np.histogram(spkt_s, bins=bins)
    thr_cnt     = max(3, int(N * 0.05))
    in_burst    = pop_rate > thr_cnt
    n_bursts_sim = np.sum(np.diff(in_burst.astype(int)) > 0)
    print(f"\nNetwork bursts (>5% neurons / 10ms bin): {n_bursts_sim}  (target: {N_BURSTS})")

# ── Save ──────────────────────────────────────────────────────────────────────
spk_out  = OUT_DIR / "sim_hdmea_burst_spikes.npz"
conn_out = OUT_DIR / "sim_hdmea_burst_connectivity.npz"
np.savez(spk_out,  spkt_s=spkt_s, spkid=spkid, unit_ids=unit_ids,
         loc_x=xs, loc_y=ys)
np.savez(conn_out, pre_gid=pre_arr, post_gid=post_arr,
         delay=delay_arr, weight=weight_arr,
         is_exc=is_exc.astype(int))
print(f"\nSaved: {spk_out}")
print(f"Saved: {conn_out}")
print(f"Ground truth connections: {len(pre_arr)}")
