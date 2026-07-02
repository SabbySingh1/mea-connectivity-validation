#!/usr/bin/env python3
"""
Generate a simulated spike dataset matching CDKL5 MEA statistics exactly.

Uses the real per-neuron firing rates and XY positions from CDKL5 R59X well000
so every neuron in the simulation corresponds to a real recorded unit.
Connectivity uses 1-5ms synaptic delays (monosynaptic range for local MEA).
Spike trains are Poisson (asynchronous) — no network bursting — so that
GLMCC/sCCG/DSTTC can be validated under the conditions they were designed for.

Outputs:
  data/sim_cdkl5_spikes.npz       — spike times + unit IDs
  data/sim_cdkl5_connectivity.npz — pre/post/delay/weight ground truth
"""
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# ── Real per-neuron data from CDKL5 R59X well000 (unit_id, rate_Hz, x_um, y_um) ─
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

N        = len(UNITS_RAW)
unit_ids = np.array([u[0] for u in UNITS_RAW])
rates    = np.array([u[1] for u in UNITS_RAW])   # exact real firing rates
xs       = np.array([u[2] for u in UNITS_RAW])   # real electrode X positions (um)
ys       = np.array([u[3] for u in UNITS_RAW])   # real electrode Y positions (um)

DURATION  = 300.0   # seconds — matches real recording length
EXC_FRAC  = 0.8     # 80% excitatory (standard cortical ratio)
DELAY_MIN = 1.0     # ms — monosynaptic range for local MEA
DELAY_MAX = 5.0     # ms
SYN_PROB  = 0.08    # 8% coupling probability per pre spike (detectable CCG peak)

print(f"Units: {N}")
print(f"Real rates — mean: {rates.mean():.2f}  std: {rates.std():.2f}  "
      f"min: {rates.min():.3f}  max: {rates.max():.2f} Hz")

# ── Step 1: distance-dependent connectivity ────────────────────────────────────
# Connection probability decays exponentially with electrode distance.
# lambda=1000um gives ~15% at 0um, ~5% at 1000um, matches cortical local connectivity.
COND_VEL = 300.0   # um/ms (0.3 m/s — unmyelinated axon)

pre_list, post_list, delay_list, weight_list = [], [], [], []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        dist   = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
        p_conn = 0.15 * np.exp(-dist / 1000.0)
        if RNG.random() < p_conn:
            delay  = np.clip(dist / COND_VEL + 1.0, DELAY_MIN, DELAY_MAX)
            is_exc = RNG.random() < EXC_FRAC
            w      = 0.04 if is_exc else -0.04
            pre_list.append(i)
            post_list.append(j)
            delay_list.append(delay)
            weight_list.append(w)

pre_gt    = np.array(pre_list,   dtype=int)
post_gt   = np.array(post_list,  dtype=int)
delay_gt  = np.array(delay_list, dtype=float)
weight_gt = np.array(weight_list, dtype=float)
print(f"Ground truth connections: {len(pre_gt)}")
print(f"Delay range: {delay_gt.min():.2f} - {delay_gt.max():.2f} ms")

# ── Step 2: generate Poisson spike trains (asynchronous, no bursting) ──────────
# Each neuron fires independently at its real rate — gives GLMCC the asynchronous
# baseline it needs. Synaptic influence adds a small rate bump at spike+delay.
spike_times_dict = {}
for uid in range(N):
    n_spikes = RNG.poisson(rates[uid] * DURATION)
    spk = np.sort(RNG.uniform(0, DURATION, n_spikes))
    spike_times_dict[uid] = spk.tolist()

# Inject synaptic spikes: for each connection, each pre spike has SYN_PROB chance
# of causing an extra post spike at exactly spike_time + delay (ms→s)
for p, q, d_ms, w in zip(pre_gt, post_gt, delay_gt, weight_gt):
    d_s        = d_ms / 1000.0
    pre_spikes = np.array(spike_times_dict[p])
    if len(pre_spikes) == 0:
        continue
    post_times = pre_spikes + d_s
    post_times = post_times[post_times < DURATION]
    n_add = RNG.binomial(len(post_times), SYN_PROB)
    if n_add > 0:
        chosen = RNG.choice(post_times, size=n_add, replace=False)
        spike_times_dict[q] = sorted(spike_times_dict[q] + chosen.tolist())

# ── Step 3: flatten and sort ───────────────────────────────────────────────────
all_spkt, all_ids = [], []
for uid, spks in spike_times_dict.items():
    all_spkt.extend(spks)
    all_ids.extend([uid] * len(spks))

all_spkt = np.array(all_spkt, dtype=float)
all_ids  = np.array(all_ids,  dtype=int)
order    = np.argsort(all_spkt)
all_spkt, all_ids = all_spkt[order], all_ids[order]

# ── Step 4: verify stats ───────────────────────────────────────────────────────
dur       = all_spkt[-1] - all_spkt[0]
sim_rates = np.array([np.sum(all_ids == u) / dur for u in range(N)])
print(f"\nSim duration: {dur:.1f} s")
print(f"Sim rate — mean: {sim_rates.mean():.2f}  std: {sim_rates.std():.2f}  "
      f"min: {sim_rates.min():.3f}  max: {sim_rates.max():.2f} Hz  [target: 2.24±3.77]")
print(f"Total spikes: {len(all_spkt)}")

# ── Step 5: save ───────────────────────────────────────────────────────────────
spk_out  = OUT_DIR / "sim_cdkl5_spikes.npz"
conn_out = OUT_DIR / "sim_cdkl5_connectivity.npz"

np.savez(spk_out,  spkt_s=all_spkt, spkid=all_ids, unit_ids=unit_ids, loc_x=xs, loc_y=ys)
np.savez(conn_out, pre_gid=pre_gt, post_gid=post_gt,
         delay=delay_gt, weight=weight_gt)

print(f"\nSaved: {spk_out}")
print(f"Saved: {conn_out}")
