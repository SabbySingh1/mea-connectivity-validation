# Connectivity Validation Results — Master Log

Project goal: evaluate connectivity inference methods (GLMCC, sCCG) on
Mouse Primary Neuron and Cerebral Organoid HD MEA data. Not chasing a
perfect method — the point is to rigorously show where and why current
methods fail on bursty data, establish an honest benchmark, and pick the
best 2-3 methods to tune and run on real recordings.

**Core problem across every dataset**: network-wide synchronous bursting
makes every pair of neurons look correlated, even when they're not
connected. Both methods key off elevated co-firing, so bursting directly
attacks their signal.

**Note**: DSTTC was also tested early on but is no longer in use going
forward — dropped after F1=0.000 on bursty data and its O(N²·spikes²) cost
made it impractical above ~100-150 units. Kept only in the historical
section below for completeness.

---

## Results Table

| Dataset | Method | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| trial_157 (Kaustubh sim) | sCCG (α=0.001, tuned) | 0.306 | 0.926 | **0.460** | Best sCCG result; above naive baseline (0.377) |
| trial_157 | sCCG (α=0.005) | 0.279 | 0.965 | 0.432 | |
| trial_157 | sCCG (α=0.010) | 0.266 | 0.976 | 0.418 | |
| trial_157 | GLMCC (bugged, delay hardcoded to 3ms) | 0.245 | 0.640 | 0.355 | Preliminary only — superseded once NERSC job finishes |
| Jan stationary sim (200n, 2ms, 1 state) | sCCG (tuned: window 1-4ms, deconv off, gauss_std=3ms, α=0.05) | 0.131 | 0.692 | **0.220** | Best of 3 param sets tried; still below naive baseline (0.414) |
| Jan stationary sim | sCCG (default params) | 0.135 | 0.430 | 0.206 | |
| Jan stationary sim | sCCG (tight window) | 0.463 | 0.107 | 0.173 | Highest precision, but recall collapses |
| Jan 2-state sim (200n, 2ms, non-stationary) | sCCG | 0.126 | 1.000 | 0.224 | This session's run — see below |
| Dale VAR (older Jan sim, 300s, statistical) | GLMCC | — | — | 0.015 | Wrong kind of connection — no ms-scale timing signature |
| Dale VAR | STTC | 0.129 | 0.134 | 0.131 | Best of the Dale VAR methods, still poor |
| Dale VAR | Pearson+FDR / Granger / Lasso-VAR variants | — | — | 0.000–0.078 | All failed — see historical section |
| trial_135 (professor's 1st sim dataset) | STTC | 0.342 | 0.485 | 0.401 | Flagged as unusable — 39% GT density unrealistic |

---

## The Two Methods (for reference)

**GLMCC** — cross-correlogram + GLM background fit, looks for a sharp 1-5ms
peak (excitatory) or trough (inhibitory). Only method that returns signed
E/I weights, making it the most biologically relevant for CDKL5 work. Best
performer on bursty data, though still poor in absolute terms.

**sCCG** — cross-correlogram vs. a Poisson null model. Simpler than GLMCC,
works at any timescale, but can't tell a direct synapse apart from shared
input or burst-driven correlation. No E/I output.

---

## Ground Truth: trial_157 (Kaustubh's simulated data)

**Why this dataset**: the lab's separate deep-learning organoid simulation
project needs to know which connectivity methods actually work, so this
benchmark (where ground truth is known exactly) feeds into validating
whether those DL-generated networks are biologically realistic.

**Specs**: 419 neurons, 87,571 directed pairs, 55,575 ground-truth synaptic
connections, delays 0.10–9.33ms (73.1% fall in the 1-5ms monosynaptic
range), burst ratio 12.7× (population rate during bursts vs. baseline).

**Naive baseline**: a classifier that calls every pair connected scores
F1=0.377 on this dataset. Any method below that is worse than doing
nothing.

### sCCG on trial_157 — bugs fixed, then tuned

Three bugs were silently zeroing out all detections before fixing:
1. **NaN bug** — `log(-expm1(x))` returns NaN when the Poisson CDF
   saturates for strongly-connected pairs, and `NaN > threshold` is always
   `False`, so the strongest true positives were silently dropped.
2. **Array size mismatch** — `ccg_bins_eff = ccg_bins` caused an IndexError
   crash whenever `deconv_ccg=True`.
3. **Wrong detection logic** — the validation script checked the
   connection *weight* instead of the *log p-value*, and used a Gaussian
   z-score threshold instead of the correct Poisson threshold
   `−log(α / bonf_corr)`.

Tuned parameters (based on Spivak et al. 2022, English et al. 2017):
`deconv_ccg=True`, `gauss_std=5ms` (was 10ms), `syn_window=0.1–9.33ms`
(was 0.8–5.8ms), `binsize=1ms` (was 0.4ms), `ccg_tau=100ms` (was 50ms),
`alpha=0.001` (best of the sweep).

**Result**: F1=0.460, Precision=0.306, Recall=0.926 (best α). TPR by delay:
0.1-1ms=95%, 1-5ms=93%, 5-10ms=89%.

**Why**: recall is strong across every delay range — sCCG reliably finds
real synaptic peaks. Precision is capped at ~0.31 because burst-driven
co-firing (12.7× burst ratio) still creates false positives even after
deconvolution removes some of the burst-driven autocorrelation structure.
F1=0.460 clears the naive baseline (0.377) but not by much — most of the
gain is from recall, not from actually distinguishing real synapses from
burst noise.

### GLMCC on trial_157 — one critical bug, then fixed

**Bug**: the synaptic delay search was hardcoded so `mode="sim"` always
tested only `syn_delay=3ms` — the 1-5ms delay grid search was dead code.
Any pair with a true delay other than 3ms was essentially guaranteed to be
missed.

**Bugged/preliminary result** (before fix): F1=0.355, Precision=0.245,
Recall=0.640. Delay breakdown confirms the bug directly — TPR was 91% at
exactly 3ms, but only 75% at 1ms, 61% at 2ms, 48% at 4ms, and 33% at 5ms.
The pattern falls off almost exactly with distance from the hardcoded 3ms.

**Fixed version**: delay search now scans 1-5ms in 1ms steps and picks the
best fit by log-posterior. Params: `deconv_ccg=True`, `tau=[4ms,4ms]`,
`ccg_tau=100ms`, `beta=1000`. This is the version that was queued to run
on NERSC as a chunked job (10-way array split across 87,571 pairs). That
NERSC job hit a SLURM policy error ("Job request does not match any
supported policy") on the `shared` partition/QOS combination and was
parked mid-session — **status needs to be checked**, since this result
would replace the bugged 0.355 number above.

### Other methods tried on bursty data (all failed)

Jitter-CCG, Lasso-VAR, Granger causality, population-rate GLM, and STTC
with circular-shift surrogates were all tested. None produced usable
results. Best of these was Jitter-CCG at F1=0.022 — its ±25ms jitter
window isn't wide enough to fully capture burst dynamics.

---

## Jan's Simulated Data (Dale network, Poisson GLM)

**What it is**: Jan Balewski (LBL) shared two simulated datasets built on a
Dale-principle network — 200 recurrently-connected neurons, excitatory
neurons only make excitatory connections, inhibitory only inhibitory. Each
neuron's firing probability at each timestep depends on the weighted sum
of all neurons' spikes at the previous timestep, plus a bias. Connectivity
is a weight matrix A where `A[i,j]` = synaptic strength from neuron j to
neuron i.

**Why the data as-shipped can't be used directly**:
1. **Time resolution**: Jan's data ships in 10ms bins. Both sCCG and GLMCC
   need to see timing differences of 1-5ms to find monosynaptic
   connections — anything finer than 10ms is invisible inside a single
   bin, and this can't be fixed by widening the detection window (that
   just floods the signal with indirect-correlation noise and makes
   Bonferroni correction too strict).
2. **Non-stationarity**: the original ground-truth file has no fixed
   connectivity matrix at all — just a time series of which "state" the
   network was in per bin. Both methods assume connections are fixed for
   the whole recording; when connectivity actually switches states, the
   CCG becomes a blurred mixture of both and clean detection breaks down.

**Fix**: Jan's own pipeline (`gen_daleMatrices3c.py` +
`gen_nonStationarySpikes3c.py`) exposes both knobs. `--Boffsets` controls
number of states (`--Boffsets 0` = single/stationary state).
`--step_size 0.002` (2ms) is the finest resolution his code allows
(`assert step_size > 0.001`) — and conveniently, at 2ms resolution his
one-step-lag GLM gives every connection exactly a 2ms delay, landing
squarely in the 1-5ms detection window both methods look for.

### Jan stationary sim (this project's earlier run — single state, 2ms)

**Generated**: 200 neurons (140 excitatory), spectral radius 0.9, 2ms step,
single bias vector, 300,000 steps (600s). Oracle state-agreement score was
1.000 with zero transitions — confirmed fully stationary. Converted: 951,647
spikes, 5,205 ground-truth connections, all at exactly 2ms delay, 13%
connection density.

sCCG has 5 tunable params: `binsize` (resolution vs. noise trade-off),
`syn_window` (lag range tested — wider catches more but adds indirect-
correlation noise and stricter Bonferroni correction), `gauss_std`
(baseline smoothing), `deconv_ccg` (removes burst-driven autocorrelation —
not needed here since Jan's network isn't strongly bursty), `alpha`
(significance threshold).

| Params | Precision | Recall | F1 |
|---|---|---|---|
| Default (window 0.1-9.33ms, deconv on, gauss_std=5ms, α=0.001) | 0.135 | 0.430 | 0.206 |
| **Tuned** (window 1-4ms, deconv off, gauss_std=3ms, α=0.05) | 0.131 | 0.692 | **0.220** |
| Tight (window 1-3ms, binsize=2ms, deconv off, α=0.001) | 0.463 | 0.107 | 0.173 |

Best F1=0.220 — still **below** the naive baseline of 0.414 on this
dataset.

**Why tuning didn't help — this is not a parameter problem**: Jan's
network is 13% connected with spectral radius 0.9, i.e. highly recurrent.
Neurons that aren't directly wired together still show elevated co-firing
through polysynaptic paths (if A→B and A→C, then B and C look correlated
even with no direct B-C connection). sCCG can't tell a direct 2ms synapse
apart from this kind of indirect network-driven correlation, so it
generates heavy false positives no matter where the threshold is set.

**This is a different failure mode than CDKL5 bursting.** On CDKL5 data,
bursting inflates co-firing across all pairs. On Jan's network, it's
*polysynaptic connectivity* creating the same inflated-correlation
pattern. Both violate sCCG's core assumption (independent Poisson firing),
just via different mechanisms.

### Jan 2-state sim (this session's run — non-stationary, 2ms)

**Generated this session**: same generator, but `--Boffsets 0 5` (2
states: low-activity ~7.7 Hz and high-activity ~148 Hz), state switches
roughly every 1 second (229 transitions over 600s). Ground truth: 5,017
directed connections. This tests the non-stationarity failure mode
directly, on top of the polysynaptic-correlation problem already seen in
the stationary version.

**sCCG**: Precision=0.126, Recall=1.000, F1=0.224. Detected 39,800 out of
39,800 possible pairs — essentially called everything connected. Recall
is perfect because every real synapse still produces a detectable bump;
precision collapses because the ~20× rate swing between states makes every
pair active in the same state look correlated, on top of the polysynaptic
effect already present in the stationary version.

**GLMCC**: ran locally this session (background job, `caffeinate`-protected
against sleep interruption). *(Results pending — update this section once
the job's final numbers are confirmed/re-checked.)*

---

## Real CDKL5 organoid data (no ground truth)

Since there's no known wiring for real recordings, precision/recall/F1
can't be computed — only the fraction of pairs each method calls
significant, with no way to know if that's accurate:

- GLMCC: 2.8% of pairs significant (85 excitatory, 185 inhibitory — the
  inhibitory count being higher than excitatory is anomalous, likely burst
  artifacts creating false inhibitory signatures)
- sCCG: 33.8% of pairs significant

---

## Minimum data / non-stationary handling — answers from earlier discussion

**Minimum data needed**: no fixed threshold — determined by spike count
per neuron. Both GLMCC and sCCG build a CCG over the full recording, so
they need enough spike pairs for a statistically stable histogram —
typically a few hundred spikes per neuron, which at 1-5 Hz MEA firing
rates means at least 5-10 minutes of recording. The validation scripts
already filter out units below 0.5 Hz.

**Can we exclude burst times?** Yes, worth trying — detect burst epochs
(e.g. MEA-NAP burst detection), mask them out, run GLMCC/sCCG on
inter-burst intervals only. Risk: if most spikes happen during bursts (as
in CDKL5 cultures), the remaining inter-burst spike trains may be too
sparse (<0.1-0.2 Hz) for a reliable CCG. Untested on real data so far —
depends on measuring actual inter-burst firing rates.

**Burst-level connectivity** (a different question from monosynaptic
connectivity): burst leader analysis (rank neurons by which fire first in
each burst — consistent early-firers are likely hub/driver neurons),
within-burst directed STTC/CCG (condition on burst-window spikes only),
or population-rate GLM (model each neuron's burst participation as a
function of recent population activity). None of these recover 1-5ms
synaptic weights, but they recover burst-level functional ordering, which
may be the more meaningful quantity when the phenotype of interest is the
bursting itself.

**HD MEA preprocessing needed before running GLMCC/sCCG**: yes, two
essential steps. (1) Remove any unit firing below 0.5 Hz — not enough
spike pairs for a stable CCG baseline. (2) Remove any unit with >0.5% of
inter-spike intervals under 1ms — real neurons can't fire twice within
their refractory period, so violations mean the spike sorter merged two+
neurons into one unit, and its CCG would be an uninterpretable mixture.
Beyond that, standard Kilosort/Phy output (spike times + unit IDs) needs
no reformatting.

---

## Historical / exploratory results (older data, mostly superseded or abandoned)

### Dale VAR (Jan Balewski's earlier, different simulation — statistical VAR model, 200 neurons, 300s)

Built for state estimation, not connectivity — connection weights turned
out too weak relative to noise for almost every method:

| Method | Precision | Recall | F1 | Note |
|---|---|---|---|---|
| GLMCC | — | — | 0.015 | No ms-scale timing signature to find |
| STTC | 0.129 | 0.134 | 0.131 | Best of this group, still poor |
| Pearson + FDR | 0.000 | 0.000 | 0.000 | Connected/unconnected pairs statistically identical |
| Granger causality | 0.125 | 0.057 | 0.078 | Many false positives, missed most true connections |
| Lasso-VAR (lag-1) | 0.119 | 0.044 | 0.064 | TP=212, FP=1571, FN=4586 |
| Multi-lag Lasso-VAR | 0.116 | 0.037 | 0.056 | Worse than lag-1 — extra lags added noise |
| Stability Lasso-VAR | 0.111 | 0.007 | 0.013 | TP=33, FP=263, FN=4765 — bootstrap filter too aggressive |

### trial_135 (Ben-Shalom Lab, 533 neurons, 300s, 39% GT connection density)

Flagged in the old project README as an **unusable benchmark** — real
biological networks aren't 39% connected, so scores here don't mean much
even when "high."

- STTC: Precision=0.342, Recall=0.485, F1=0.401 — noted as meaningless
  given the unrealistic density.
- sCCG/TE window sweep (1ms-1000ms): representative numbers — at 1ms
  window, sCCG FPR=0.292 (TPR 0.1-1ms=0.27, 1-10ms=0.00, 10-50ms=0.19,
  50-200ms=0.20, 200-1345ms=0.27); TE FPR=0.124. FPR climbed sharply as
  window size grew while TPR also climbed — a mechanical sensitivity
  trade-off, not evidence of better detection. **This sweep crashed
  partway through** (a dependency script was deleted mid-run during a
  later repo cleanup) — incomplete. (A DSTTC pass was also run in this
  old sweep, but DSTTC is no longer part of the method set going forward.)
- GLMCC was **never actually completed** on trial_135 — a script exists
  for it, but its own docstring notes trial_135's ground-truth delays
  (200-1345ms) fall far outside GLMCC's ±50ms search window, so it's
  expected to fail by design. Exists to document the limitation, not as a
  working benchmark.

---

## Dataset provenance — confirmed

- **trial_135** = professor's **first** simulated dataset (Ben-Shalom Lab,
  533 neurons, 39% GT density — abandoned as unrealistic).
- **trial_157** = professor's **second** simulated dataset, shared via
  Kaustubh (419 neurons, 87,571 pairs — the current main benchmark).
- **"Dale VAR" / original Jan data** = run directly from Jan's GitHub
  pipeline (`BouchardLab/pyuoi`, `uoi-var` branch), **not** pulled from
  NERSC. Output was written locally to `/private/tmp/dale_sim/` at the
  time, but `/private/tmp` isn't persistent and that directory no longer
  exists — only the result numbers above survive, not the raw spike/truth
  files. If we need to rerun anything against this exact dataset, it would
  have to be regenerated from the pipeline again (parameters used aren't
  fully known — likely an earlier pipeline version than
  `nonStation_ver3c_states_EM_FDR`, since that's the one used for the 2ms
  work this session).
- **Jan's NERSC-hosted files** (`daleN200_55e5a6_ff089c`,
  `daleN200_74e6d6_e2b7d7`, 10ms bins, 2-state) — these are **separate**
  from the "Dale VAR" run above. Per Jan's email (pasted below), these are
  the 2 simulated datasets he formatted and shared specifically, alongside
  2 real Canine recordings, for the analysis he presented to Mandar.

### From Jan's email (for reference)

> I'll share data whose analysis I presented 2 weeks ago: 2 simulated sets
> and 2 real data sets. Code: `github.com/BouchardLab/pyuoi/tree/uoi-var/causal_net/nonStation_ver3c_states_EM_FDR`

Files at `/global/cfs/cdirs/m2043/causal_inference/2026-06-formatted-data/spikesData/`:

| File | Type | Size |
|---|---|---|
| `Canine_260324_r21_w0_1hz.spikes.npz` + `.bioExp.npz` | **Real** canine recording | 2.7MB |
| `Canine_260324_r23_w0_1hz.spikes.npz` + `.bioExp.npz` | **Real** canine recording | 2.3MB |
| `daleN200_55e5a6_ff089c.spikes.npz` + `.prismTruth.npz` | Simulated (2-state, 10ms) | 6.3MB |
| `daleN200_74e6d6_e2b7d7.spikes.npz` + `.prismTruth.npz` | Simulated (2-state, 10ms) | 14.3MB |

**⚠️ The two real Canine datasets have not been touched at all** — not
downloaded, not inspected, no method run on them. Since they're real
recordings there's no ground truth, but they'd be useful as another
no-ground-truth sanity check alongside the real CDKL5 organoid results
already in this doc. Let me know if you want these pulled down.

Jan also referenced a Google Slides deck with his own analysis
(link in his email) — I haven't opened it; say the word if you want me to
pull anything from it.

---

## ⚠️ Still open

1. **The professor's `000066.tar`** pulled from NERSC this session is
   still uninspected — need the extraction output to know its format and
   whether it's trial_135, trial_157, something newer, or unrelated.
2. **NERSC GLMCC job on trial_157**: script bug fixed (missing
   `--qos=shared`, wrong account, invalid partition/node combo) — waiting
   on you to resubmit from your open Perlmutter terminal and confirm it
   runs. Also still need confirmation that `sim_hdmea_burst_spikes.npz` on
   NERSC is actually trial_157 (419 neurons) and not a different/older
   burst simulation.
3. **Jan 2-state GLMCC (this session, local)**: still running — will fill
   in final numbers once done.
4. **Real Canine recordings**: untouched, per above — download and inspect
   if wanted.
