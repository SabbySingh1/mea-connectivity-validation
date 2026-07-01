#!/usr/bin/env python3
"""
Simulate HD-MEA network matched to real CDKL5 unit locations and firing rates.
- Real XY positions from metrics_curated.xlsx (loc_x, loc_y in micrometers)
- 80/20 E/I split
- Distance-dependent connectivity (exponential decay, type-specific rules)
- Synaptic delays from distance: delay = dist/conduction_velocity + synaptic_delay
  → naturally 1-5ms for distances 50-1200um at 0.3m/s
- LIF + adaptation neurons
"""
import numpy as np
from pathlib import Path
import brian2 as b2

b2.start_scope()
b2.set_device('runtime')
b2.defaultclock.dt = 0.1 * b2.ms

OUT_DIR = Path("/private/tmp")
RNG     = np.random.default_rng(42)

# ── Real unit data from metrics_curated ──────────────────────────────────────
# unit_id, firing_rate, loc_x (um), loc_y (um)
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
xs         = np.array([u[2] for u in UNITS_RAW])   # micrometers
ys         = np.array([u[3] for u in UNITS_RAW])   # micrometers

print(f"Units: {N}")
print(f"MEA extent: x={xs.min():.0f}-{xs.max():.0f}um  y={ys.min():.0f}-{ys.max():.0f}um")

# ── E/I assignment (80/20, randomly assigned) ────────────────────────────────
is_exc = RNG.random(N) < 0.8
N_E = is_exc.sum()
N_I = (~is_exc).sum()
print(f"E: {N_E}  I: {N_I}")

# ── Distance-dependent connectivity ─────────────────────────────────────────
# Connection probability: p = p0 * exp(-dist / lambda)
# E cells reach further (lambda=500um), I cells more local (lambda=200um)
# Type-specific p0:
#   E→E: 0.15,  E→I: 0.20,  I→E: 0.30,  I→I: 0.15
COND_VEL  = 0.0003   # m/s = 0.3 mm/ms → for distances in um: vel = 0.0003 um/us = 0.3 um/us
                     # delay(ms) = dist(um) / 300(um/ms) + 1.0ms synaptic
SYN_DELAY = 1.0      # ms fixed synaptic transmission delay

pre_list, post_list, delay_list, weight_list = [], [], [], []

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)  # micrometers

        if is_exc[i] and is_exc[j]:      # E→E
            p0, lam = 0.15, 500.0
        elif is_exc[i] and not is_exc[j]: # E→I
            p0, lam = 0.20, 500.0
        elif not is_exc[i] and is_exc[j]: # I→E
            p0, lam = 0.30, 200.0
        else:                              # I→I
            p0, lam = 0.15, 200.0

        p_conn = p0 * np.exp(-dist / lam)
        if RNG.random() < p_conn:
            # delay from axonal conduction distance + synaptic delay
            delay = dist / 300.0 + SYN_DELAY   # ms (300 um/ms = 0.3 m/s)
            delay = np.clip(delay, 1.0, 5.0)    # keep within monosynaptic range
            pre_list.append(i)
            post_list.append(j)
            delay_list.append(delay)
            weight_list.append(1.0 if is_exc[i] else -1.0)

pre_arr    = np.array(pre_list,    dtype=int)
post_arr   = np.array(post_list,   dtype=int)
delay_arr  = np.array(delay_list,  dtype=float)
weight_arr = np.array(weight_list, dtype=float)
print(f"Connections: {len(pre_arr)}")
print(f"Delay range: {delay_arr.min():.2f} - {delay_arr.max():.2f} ms")

# ── LIF + adaptation model ───────────────────────────────────────────────────
DURATION = 300 * b2.second
tau_m    = 20  * b2.ms
tau_ada  = 200 * b2.ms
V_rest   = -70 * b2.mV
V_thresh = -50 * b2.mV
V_reset  = -65 * b2.mV
R        = 500 * b2.Mohm
a_ada    = 0.5 * b2.nS
b_ada    = 20  * b2.pA
tau_E    = 5   * b2.ms
tau_I    = 10  * b2.ms
g_E      = 0.15 * b2.nS
g_I      = 0.25 * b2.nS
E_E      =  0  * b2.mV
E_I      = -80 * b2.mV

eqs = '''
dv/dt  = (-(v - V_rest) - R*w + R*(g_e*(E_E - v) + g_i*(E_I - v)) + R*I_bg) / tau_m : volt
dw/dt  = (a_ada*(v - V_rest) - w) / tau_ada : amp
dg_e/dt = -g_e / tau_E : siemens
dg_i/dt = -g_i / tau_I : siemens
I_bg   : amp
'''

neurons = b2.NeuronGroup(N, eqs,
    threshold='v > V_thresh', reset='v = V_reset; w += b_ada',
    namespace=dict(V_rest=V_rest, V_thresh=V_thresh, V_reset=V_reset,
                   R=R, tau_m=tau_m, tau_ada=tau_ada, a_ada=a_ada, b_ada=b_ada,
                   tau_E=tau_E, tau_I=tau_I, E_E=E_E, E_I=E_I),
    method='euler')

neurons.v   = V_rest + RNG.uniform(0, 5, N) * b2.mV
neurons.w   = 0 * b2.pA
neurons.g_e = 0 * b2.nS
neurons.g_i = 0 * b2.nS

# Map real firing rates → drive current (48-85 pA range tuned to match ~2Hz mean)
r_min, r_max = real_rates.min(), real_rates.max()
I_min, I_max = 48.0, 85.0
I_drive = I_min + (I_max - I_min) * (real_rates - r_min) / (r_max - r_min + 1e-9)
neurons.I_bg = I_drive * b2.pA

# ── Recurrent synapses ───────────────────────────────────────────────────────
exc_mask = weight_arr > 0
inh_mask = ~exc_mask

S_E = b2.Synapses(neurons, neurons, on_pre='g_e += g_E', namespace=dict(g_E=g_E))
S_E.connect(i=pre_arr[exc_mask].tolist(), j=post_arr[exc_mask].tolist())
S_E.delay = delay_arr[exc_mask] * b2.ms

S_I = b2.Synapses(neurons, neurons, on_pre='g_i += g_I', namespace=dict(g_I=g_I))
S_I.connect(i=pre_arr[inh_mask].tolist(), j=post_arr[inh_mask].tolist())
S_I.delay = delay_arr[inh_mask] * b2.ms

spike_mon = b2.SpikeMonitor(neurons)

print("Running simulation (300s)...")
b2.run(DURATION, report='text', report_period=30*b2.second)

# ── Extract and save ─────────────────────────────────────────────────────────
spkt_s = np.array(spike_mon.t / b2.second)
spkid  = np.array(spike_mon.i, dtype=int)
order  = np.argsort(spkt_s)
spkt_s, spkid = spkt_s[order], spkid[order]

duration   = spkt_s[-1] - spkt_s[0]
sim_rates  = np.array([np.sum(spkid == u) / duration for u in range(N)])
print(f"\nSim duration: {duration:.1f} s")
print(f"Units: {N}")
print(f"Rate — mean: {sim_rates.mean():.2f}  std: {sim_rates.std():.2f}  "
      f"min: {sim_rates.min():.2f}  max: {sim_rates.max():.2f}")
print(f"Percentiles [10,25,50,75,90]: {np.percentile(sim_rates,[10,25,50,75,90]).round(2)}")
print(f"Total spikes: {len(spkt_s)}")

# Save with original unit IDs mapped back
spk_out  = OUT_DIR / "sim_hdmea_spikes.npz"
conn_out = OUT_DIR / "sim_hdmea_connectivity.npz"
np.savez(spk_out,  spkt_s=spkt_s, spkid=spkid, unit_ids=unit_ids,
         loc_x=xs, loc_y=ys)
np.savez(conn_out, pre_gid=pre_arr, post_gid=post_arr,
         delay=delay_arr, weight=weight_arr,
         is_exc=is_exc.astype(int))
print(f"\nSaved: {spk_out}")
print(f"Saved: {conn_out}")
