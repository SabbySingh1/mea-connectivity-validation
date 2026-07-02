# MEA Connectivity Validation

Validation pipeline for spike connectivity inference methods (GLMCC, sCCG, DSTTC) on simulated HD-MEA data matched to CDKL5 R59X mouse recordings.

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/SabbySingh1/mea-connectivity-validation
cd mea-connectivity-validation
bash setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Generate simulation data
python simulation/generate_brian2_hdmea_burst.py
# → outputs data/sim_hdmea_burst_spikes.npz and data/sim_hdmea_burst_connectivity.npz

# 4. Run validation methods
python validation/glmcc_validate.py
python validation/dsttc_sccg_validate.py sccg
python validation/dsttc_sccg_validate.py dsttc
```

No path configuration needed — all scripts auto-detect the repo root and load spycon from the bundled source files.

---

## Custom data

All validation scripts accept `--spikes` and `--conn` arguments:

```bash
python validation/glmcc_validate.py \
    --spikes path/to/your_spikes.npz \
    --conn   path/to/your_connectivity.npz
```

**Spike file** (`.npz`) must contain:
- `spkt_s` — spike times in seconds
- `spkid` — integer unit ID for each spike

**Connectivity file** (`.npz`) must contain:
- `pre_gid`, `post_gid` — integer unit IDs
- `delay` — synaptic delay in milliseconds

---

## Methods

| Method | Type | Delay Sensitivity | E/I Classification |
|--------|------|-------------------|--------------------|
| **GLMCC** | Monosynaptic | 1–5ms only | Yes (signed weight) |
| **sCCG** | Functional | Any peak in ±50ms CCG | No |
| **DSTTC** | Functional | Any co-firing within delta_t | No |

### GLMCC Parameters
```python
binsize   = 1ms       # CCG bin width
ccg_tau   = 50ms      # CCG half-window (±50ms total)
syn_delay = 3ms       # center of monosynaptic range (1–5ms)
tau       = 10ms      # exponential kernel width (tuned from default 1ms)
beta      = 4000      # smoothness penalty on background model
alpha     = 0.001     # significance threshold
```

`tau=10ms` was tuned from the spycon default of `1ms` via parameter sweep — it improved TPR at 1–5ms delays significantly. All other parameters are spycon defaults.

### Why DSTTC has a scale limit

DSTTC loops through every individual spike and compares it against every spike from every other neuron. With 480 neurons and a 300-second recording this requires roughly 41 billion comparisons — it ran for over 17 hours without completing. DSTTC is only practical for datasets with fewer than ~150 units.

---

## Simulation (`generate_brian2_hdmea_burst.py`)

Generates a realistic HD-MEA network matched to real CDKL5 R59X statistics:

| Parameter | Simulation | Real CDKL5 |
|-----------|------------|------------|
| Units | 129 | 129 |
| Mean rate | 2.23 Hz | 2.24 Hz |
| Rate std | 2.86 Hz | 3.77 Hz |
| Network bursts / 300s | 18 | 18 |
| % spikes in bursts | ~100% | ~100% |
| GT connection delays | 1–5ms | — |
| GT connections | 280 | — |

**Key design decisions:**
- Real XY unit positions from `metrics_curated.xlsx` (CDKL5 R59X well000)
- 80/20 E/I split, distance-dependent connectivity (exponential decay)
- Burst dynamics: shared gate signal drives network into synchrony ~18 times per 300s, with per-neuron drive scaled to real firing rates
- Synaptic delays: `distance / 0.3 m/s + 1ms` → naturally 1–5ms for local MEA distances

---

## Results on burst simulation

| Method | Detected | Precision | Recall | F1 | Notes |
|--------|----------|-----------|--------|----|-------|
| GLMCC | 2,209 | 0.021 | 0.202 | 0.038 | Burst background drives false positives |
| sCCG | 0 | 0.000 | 0.000 | 0.000 | Burst background masks synaptic peak |
| DSTTC | 0 | 0.000 | 0.000 | 0.000 | Same as sCCG |

All methods struggle because synchronized network bursting creates strong co-firing correlations between every neuron pair, masking the true 1–5ms synaptic signal. This is the fundamental challenge of CDKL5 data where E/I imbalance drives hyper-synchronous bursting.

---

## spycon source files

The algorithm source files (`sci_glmcc.py`, `sci_sccg.py`, `sci_dsttc.py`, `sci_pyinform.py`, `spycon_inference.py`, `spycon_result.py`) are included directly from the [spycon](https://github.com/christiando/spycon) package **without modification**. They are bundled here so the repo runs without a separate spycon installation. The only tuning applied was passing `tau=[10e-3, 10e-3]` as a runtime parameter to GLMCC in the validation scripts — the source files themselves are unmodified.

The spycon `__init__.py` is intentionally excluded because it eagerly imports `TE_IDTXL` which requires `idtxl` (difficult to build on most systems). Scripts instead load only the needed modules via `importlib`.

---

## Why GLMCC for CDKL5

CDKL5 is a channelopathy that disrupts inhibitory interneuron function, directly affecting E/I balance. GLMCC is the only method that:
1. Detects **direct monosynaptic connections** specifically (not indirect co-firing)
2. Returns **signed weights** — positive = excitatory, negative = inhibitory
3. Can quantify the E/I ratio at the single-connection level

sCCG and DSTTC detect functional co-firing, which cannot distinguish a direct synapse from shared common input or burst-driven correlation.
