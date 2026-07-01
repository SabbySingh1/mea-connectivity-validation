# MEA Connectivity Validation

Validation pipeline for spike connectivity inference methods on simulated HD-MEA data matched to CDKL5 R59X recordings.

---

## Methods Evaluated

| Method | Type | Delay Sensitivity | E/I Classification |
|--------|------|-------------------|--------------------|
| **GLMCC** | Monosynaptic | 1–5ms only | Yes (signed weight) |
| **sCCG** | Functional | Any peak in ±50ms CCG | No |
| **DSTTC** | Functional | Any co-firing within delta_t | No |
| **TE** | Functional (information-theoretic) | k-bin history window | No |

All methods are loaded from the [spycon](https://github.com/christiando/spycon) package. The `__init__.py` is bypassed using `importlib` to avoid importing `TE_IDTXL` which requires `idtxl` (fails to build on most systems).

---

## Key Findings

### On the provided simulated dataset (`trial_135_rerun_300s`)
- Ground truth connections had delays of **0.1–1345ms** (majority 200–1345ms)
- GLMCC, sCCG detect within a ±50ms CCG window → **completely blind to 200–1345ms connections**
- GLMCC: 5,317 detections (2.3%) — mostly false positives from burst correlations
- sCCG: 0 detections — flat CCG within ±50ms window
- DSTTC: still running (very slow on 480 units / 114,960 pairs)

### On HD-MEA simulated dataset (Brian2, matched to CDKL5)
- 129 units, real XY positions, 80/20 E/I, distance-dependent connectivity
- All 280 ground truth connections have delays **1–5ms** (monosynaptic range)
- Results:

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| GLMCC  | 0.218 | 0.967 | 0.355 |
| sCCG   | 0.000 | 0.000 | 0.000 |
| DSTTC  | 0.000 | 0.000 | 0.000 |

- GLMCC has near-perfect recall (0.967) but low precision (748 FPs) due to burst-driven spurious CCG peaks
- sCCG and DSTTC detect nothing — synaptic coupling too weak to produce a statistically detectable signal with non-parametric tests at realistic firing rates (~2 Hz mean)

---

## GLMCC Parameters

```python
PARAMS = {
    "binsize":    1e-3,    # 1ms — CCG bin width
    "ccg_tau":   50e-3,   # 50ms — CCG half-window (±50ms total)
    "syn_delay":  3e-3,   # 3ms — center of monosynaptic range (1–5ms)
    "tau":  [10e-3, 10e-3], # 10ms — exponential kernel width (tuned from default 1ms)
    "beta":      4000,    # smoothness penalty on background model
    "alpha":     1e-3,    # significance threshold (z > 3.29 to call connection)
    "deconv_ccg": False,
}
```

**Parameter notes:**
- `syn_delay=3ms` — center of 1–5ms monosynaptic window, not an enforced filter. The GLM kernel is most sensitive near 3ms and decays toward the edges.
- `tau=10ms` — tuned from default 1ms via parameter sweep. Wider tau improved TPR at 1–5ms from 0.85→1.00.
- `ccg_tau=50ms` — biological convention for monosynaptic detection, not spycon-specific. Comes from 1990s CCG literature.
- `alpha=1e-3` — can be tightened (e.g. 1e-5) to reduce false positives at the cost of some recall.

---

## Simulation Details

### HD-MEA simulation (`generate_brian2_hdmea.py`)
- **129 neurons** at real XY positions from `metrics_curated.xlsx` (CDKL5 R59X well000)
- **80/20 E/I** assigned randomly
- **Distance-dependent connectivity:**
  - E→E: p = 0.15 × exp(−d/500μm)
  - E→I: p = 0.20 × exp(−d/500μm)
  - I→E: p = 0.30 × exp(−d/200μm)
  - I→I: p = 0.15 × exp(−d/200μm)
- **Synaptic delays:** d(μm) / 300μm·ms⁻¹ + 1ms → naturally 1–5ms for distances 0–1200μm
- **Neuron model:** LIF + spike-triggered adaptation (AdEx simplified)
  - tau_m=20ms, tau_ada=200ms, b_ada=20pA
  - Drive current mapped from real firing rate distribution (48–85pA)
- **Firing rates:** mean 2.04 Hz, std 2.11 (real data: mean 2.24, std 3.77)
- **Duration:** 300s

### Poisson simulation (`generate_cdkl5_sim.py`)
- Simpler baseline: homogeneous Poisson spike trains + injected synaptic spikes
- Log-normal rate distribution matched to CDKL5
- 1,286 ground truth connections, all 1–5ms delays

---

## Running

### Requirements
```
spycon
numpy
scipy
brian2
```

Spycon is loaded via `importlib` to bypass the broken `__init__.py`:
```python
_spycon_root = Path("/path/to/spycon/package")
```
Update this path in each script to match your environment.

### Generate HD-MEA simulation
```bash
python3 simulation/generate_brian2_hdmea.py
# outputs: sim_hdmea_spikes.npz, sim_hdmea_connectivity.npz
```

### Run validation
```bash
# GLMCC
python3 validation/glmcc_validate.py

# sCCG or DSTTC
python3 validation/dsttc_sccg_validate.py sccg
python3 validation/dsttc_sccg_validate.py dsttc
```

---

## Why GLMCC for CDKL5 Data

CDKL5 is a channelopathy that disrupts inhibitory interneuron function, directly affecting E/I balance. GLMCC is the only method that:
1. Detects **direct monosynaptic connections** specifically (not indirect or co-firing)
2. Returns **signed weights** (positive = excitatory, negative = inhibitory)
3. Can quantify E/I ratio at the single-connection level

DSTTC and sCCG detect functional co-firing — useful for network-level analysis but cannot distinguish direct synapses from shared common input.

---
