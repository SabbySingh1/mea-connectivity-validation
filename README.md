# MEA Connectivity Validation

Validation pipeline for spike connectivity inference methods on HD-MEA data
matched to CDKL5 R59X mouse recordings. **GLMCC and sCCG are the two
actively-used methods.** DSTTC was tested and dropped (F1=0.000 on bursty
data, and its cost scales with the square of spike count — impractical
above ~150 units). The repo also contains ~10 additional exploratory
methods tried on other datasets — see `validation/README.md`.

**Primary benchmark: `trial_157`** (419 neurons, 87,571 directed pairs,
55,575 ground-truth synaptic connections, delays 0.10–9.33ms, 12.7× burst
ratio). Included in `data/`.

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/SabbySingh1/mea-connectivity-validation
cd mea-connectivity-validation
bash setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run the validated sCCG result on trial_157 (F1=0.460)
python validation/dsttc_sccg_validate.py sccg \
    --spikes data/trial_157_spikes.npz \
    --conn   data/trial_157_connectivity.npz \
    --alpha  0.001

# 4. Run GLMCC on trial_157
python validation/glmcc_validate.py \
    --spikes data/trial_157_spikes.npz \
    --conn   data/trial_157_connectivity.npz
```

`trial_157`'s data files are already included in `data/` — no generation
step needed. No path configuration needed either — all scripts
auto-detect the repo root and load the bundled spycon source files.

**Python version note**: use Python 3.9+. On macOS, the system `python3`
may not have a working `pip`/`venv` — if `python3 -m venv venv` fails, try
the Python bundled with Xcode Command Line Tools instead:
`/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9`

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

## Core methods

| Method | Type | Delay Sensitivity | E/I Classification |
|--------|------|-------------------|--------------------|
| **GLMCC** | Monosynaptic | Searches 1–5ms, picks best fit per pair | Yes (signed weight) |
| **sCCG** | Functional | Any peak within `syn_window` | No |

*(DSTTC — functional, any co-firing within a fixed window, no E/I output —
was tested and dropped. See "Why DSTTC was dropped" below.)*

### GLMCC parameters (validated on trial_157)

```python
binsize          = 1ms
ccg_tau          = 100ms       # CCG half-window (±100ms total)
tau              = [4ms, 4ms]  # exponential kernel width
beta             = 1000        # smoothness penalty on background model
alpha            = 0.001
deconv_ccg       = True        # Spivak 2022 deconvolution
syn_delay_range  = [1, 2, 3, 4, 5]  # ms — scanned per pair, best fit selected
```

**Bug history**: the delay search was previously hardcoded to always test
a single fixed value (3ms), making the intended 1–5ms grid search dead
code — every pair was tested at exactly 3ms regardless of its true delay,
regardless of what config was passed in. This is fixed in the current
`glmcc_validate.py` — `syn_delay_range` is now genuinely scanned per pair.
If you're working from an older copy of this script, confirm this list is
actually being iterated, not just declared.

### sCCG parameters (validated on trial_157)

```python
binsize     = 1ms
syn_window  = (0.1ms, 9.33ms)  # matches trial_157's actual GT delay range
gauss_std   = 5ms
ccg_tau     = 100ms
deconv_ccg  = True              # removes burst-driven autocorrelation structure
alpha       = 0.001             # best of a 0.01 / 0.005 / 0.001 sweep
```

**Bug history**: `sci_sccg.py` had two bugs fixed directly in its source
(not settable via parameters):
1. A NaN bug — `log(-expm1(x))` returns NaN when the Poisson CDF
   saturates for strongly-connected pairs, and `NaN > threshold` silently
   evaluates `False` — this was dropping the strongest true positives
   with no error raised.
2. An array-size mismatch (`ccg_bins_eff = ccg_bins`) that caused an
   IndexError crash whenever `deconv_ccg=True`.

The validation script itself also previously had a third bug — it
compared the connection *weight* against a Gaussian z-score threshold,
instead of the *log p-value* against the correct Poisson threshold
`-log(alpha / bonf_corr)`. All three are fixed in the current
`dsttc_sccg_validate.py` and `sci_sccg.py`.

### Why DSTTC was dropped

DSTTC loops through every individual spike and compares it against every
spike from every other neuron. With 480 neurons and a 300-second
recording this requires roughly 41 billion comparisons — it ran for over
17 hours without completing. Combined with scoring F1=0.000 on bursty
simulated data, it was dropped from the active method set. Only
practical for datasets with fewer than ~150 units, if used at all.

---

## Results on trial_157 (419 units, 87,571 pairs, 55,575 GT connections)

A naive classifier that declares every pair connected scores F1≈0.377 on
this dataset — any method below that is worse than doing nothing.

| Method | Precision | Recall | F1 | Notes |
|--------|-----------|--------|----|----|
| **sCCG** (α=0.001, fixed + tuned) | 0.306 | 0.926 | **0.460** | Above naive baseline; recall strong across all delay bins (89–95% TPR), precision limited by residual burst-driven false positives |
| GLMCC (bugged, hardcoded 3ms delay) | 0.245 | 0.640 | 0.355 | Preliminary/superseded — delay-search bug now fixed, full re-run pending |

Both methods' core limitation on bursty data: network-wide synchronous
bursting creates strong co-firing correlation between every neuron pair
regardless of true connectivity, which both methods use as their core
detection signal — so bursting directly attacks the thing they're
looking for.

---

## Repo structure

```
data/         simulated spike + connectivity .npz files (trial_157 included; others are output of simulation/ scripts)
simulation/   scripts that generate synthetic ground-truth spike data
validation/   scripts that run a connectivity method and score it against ground truth
sci_*.py      spycon algorithm source — sci_sccg.py and sci_glmcc.py have been
              patched (see bug history above); others unmodified
```

## Simulation scripts

The `simulation/` folder has four generators for producing additional
synthetic datasets beyond trial_157 — use `generate_brian2_hdmea_burst.py`
unless you specifically need one of the others:

| Script | Bursting? | Matched to real CDKL5 data? | When to use |
|--------|-----------|------------------------------|-------------|
| **generate_brian2_hdmea_burst.py** | Yes | Yes (rates, positions, burst timing) | Closest match to real recordings among the local generators. |
| `generate_brian2_hdmea.py` | No | Yes (positions, rates) | Clean/non-bursty version of the same network, for isolating the effect of bursting. |
| `generate_brian2_sim.py` | Yes (from adaptation) | No — generic E/I network | Earlier prototype, kept for reference. |
| `generate_cdkl5_sim.py` | No (Poisson, asynchronous) | Yes (rates, positions) | Sanity-check dataset — validates methods under ideal, non-bursty conditions before testing on bursty data. |

### generate_brian2_hdmea_burst.py

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

**Note**: results on this dataset (below) predate the sCCG/GLMCC bug
fixes described above — they show the pre-fix behavior, not current
validated performance. Use trial_157 (above) for current validated
numbers.

| Method | Detected | Precision | Recall | F1 | Notes |
|--------|----------|-----------|--------|----|-------|
| GLMCC | 2,209 | 0.021 | 0.202 | 0.038 | Pre-fix — hardcoded delay bug active |
| sCCG | 0 | 0.000 | 0.000 | 0.000 | Pre-fix — NaN/threshold bugs active |
| DSTTC | 0 | 0.000 | 0.000 | 0.000 | Dropped from active method set |

---

## spycon source files

The algorithm source files (`sci_glmcc.py`, `sci_sccg.py`, `sci_dsttc.py`,
`sci_pyinform.py`, `spycon_inference.py`, `spycon_result.py`) are based on
the [spycon](https://github.com/christiando/spycon) package, bundled here
so the repo runs without a separate spycon installation. **`sci_sccg.py`
and `sci_glmcc.py` have been patched** — see the bug histories under
"Core methods" above. The other source files are unmodified.

The spycon `__init__.py` is intentionally excluded because it eagerly
imports `TE_IDTXL` which requires `idtxl` (difficult to build on most
systems). Scripts instead load only the needed modules via `importlib`.

---

## Why GLMCC for CDKL5

CDKL5 is a channelopathy that disrupts inhibitory interneuron function,
directly affecting E/I balance. GLMCC is the only method that:
1. Detects **direct monosynaptic connections** specifically (not indirect co-firing)
2. Returns **signed weights** — positive = excitatory, negative = inhibitory
3. Can quantify the E/I ratio at the single-connection level

sCCG detects functional co-firing, which cannot distinguish a direct
synapse from shared common input or burst-driven correlation.

---

## More methods

`validation/` also contains a larger set of exploratory scripts tried
against other datasets (Lasso-VAR, Granger causality, population-rate
GLM, STTC/CFP with circular-shift surrogates, eANN, Transfer Entropy, and
more) — see `validation/README.md` for what each one does and why.

## Converting data to NWB format

`convert_to_nwb.py` converts spike `.npz` files (from any dataset in this
repo, or real recordings) into [NWB](https://www.nwb.org/) format for use
with NERSC and shared pipeline tooling. Run `python convert_to_nwb.py --help`
for usage.
