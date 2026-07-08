# Connectivity Validation Scripts
Ben-Shalom Lab — Sabadnoor Singh, Summer 2025

Scripts for benchmarking spike train connectivity inference methods on simulated ground truth data.
All scripts save results as .npz files with precision, recall, and F1 score.

See `connectivity_results.md` in the root for full results summary.

---

## Datasets Used
- **sim_cdkl5** — Donner et al. 2024 (spycon). Clean non-bursty simulation. 123 neurons, 300s, ~8% connection density.
- **sim_burst** — Donner et al. 2024 (spycon). Bursty in vitro MEA simulation. 114 neurons, 257s, 89% of spikes in bursts.
- **Dale VAR** — Jan Balewski. VAR model for state estimation. 200 neurons, 300s, 12.5% connection density.
- **trial_135** — Ben-Shalom Lab. 533 neurons, 300s. Abandoned — 39% connection density is unrealistic.

---

## Scripts

### burst_excl_pearson.py
**Method:** Pearson correlation after removing burst periods.
**Datasets:** sim_cdkl5, sim_burst
**Result:** F1=0.352 on sim_cdkl5, F1=0.000 on sim_burst (no inter-burst spikes to analyze)

### func_conn_dale_fdr.py
**Method:** Pearson correlation (FDR corrected) + STTC on Dale VAR data.
**Dataset:** Dale VAR (state 0 only, 4.88 Hz)
**Result:** Pearson F1=0.000, STTC F1=0.131

### func_conn_trial135.py
**Method:** Pearson correlation + STTC on trial_135.
**Dataset:** trial_135 (unusable benchmark)
**Result:** Pearson F1=0.362, STTC F1=0.401 — scores meaningless due to 39% GT density

### granger_multilag_dale.py
**Method:** Granger causality (pairwise F-test) + Multi-lag Lasso-VAR (lags 1+2+3).
**Dataset:** Dale VAR
**Result:** Granger F1=0.078, Multi-lag Lasso-VAR F1=0.056

### lasso_var_dale.py
**Method:** Lasso-VAR (lag-1) using LassoCV per neuron.
**Dataset:** Dale VAR
**Result:** F1=0.064, Precision=0.119, Recall=0.044

### stability_lasso_dale.py
**Method:** Bootstrap stability selection Lasso-VAR (24 bootstraps, 75% threshold).
**Dataset:** Dale VAR
**Result:** F1=0.013 — too aggressive, eliminated true positives

### poprate_glm_burst.py
**Method:** Population-rate Lasso-VAR (novel). Adds population firing rate as covariate to absorb burst drive. Also includes Pearson on population-rate residuals.
**Dataset:** sim_burst
**Result:** Lasso F1=0.036, Pearson residuals F1=0.035 — high recall but terrible precision

### sttc_cfp_burst.py
**Method:** STTC with circular shift surrogates (MEA-NAP standard method) + CFP (Conditional Firing Probability) with circular shift surrogates.
**Dataset:** sim_burst
**Result:** STTC F1=0.035, CFP pending

### eann_run.py
**Method:** eANN ensemble neural network (Donner et al. 2024). Combines CI, sCCG, dSTTC, GLMCC, GLMPP outputs.
**Datasets:** sim_cdkl5, sim_burst
**Result:** Crashed — API incompatibility with installed spycon version. See spycon GitHub for original results.

---

## Key Finding
No existing method reliably detects connectivity in heavily bursty organoid-like data (sim_burst).
GLMCC performs best (F1=0.195) but is still poor. The core problem is network-wide bursting causing
all neuron pairs to appear correlated, which no current method fully accounts for.
