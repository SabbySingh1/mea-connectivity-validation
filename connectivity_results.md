# Connectivity Method Results
Ben-Shalom Lab — Sabadnoor Singh, Summer 2025

---

## Definitions

- **Precision:** of all the connections a method detected, what fraction were actually real (in the ground truth). Low precision = lots of false alarms.
- **Recall:** of all the real connections that exist, what fraction did the method actually find. Low recall = missed most real connections.
- **F1 score:** a single number combining precision and recall (their harmonic mean). Ranges 0 to 1 — 1 is perfect, 0 is complete failure. Used because a method can look good on precision alone (by barely detecting anything) or on recall alone (by flagging almost everything) — F1 punishes both extremes and only rewards methods that are good at both.
- **Ground truth:** the actual, known set of connections in a dataset. Only available for simulated data, where the network was built by hand — real recordings have no ground truth, so we can't compute F1 on them, only report what percentage of pairs a method flagged as connected.

---

## GLMCC
**Datasets:** sim_cdkl5, sim_burst, Dale VAR, Real CDKL5 organoid

- **sim_cdkl5** — F1=0.660, Precision=0.626, Recall=0.698. Best result of any method. Works well on clean non-bursty data with short synaptic delays.
- **sim_burst** — F1=0.195. Failed. Burst co-firing creates spurious peaks in the cross-correlogram that look like synaptic connections.
- **Dale VAR** — F1=0.015. Failed. GLMCC looks for 1-5ms synaptic peaks but Dale connections are statistical regression weights with no millisecond-scale timing signature.
- **Real CDKL5** — No ground truth, cannot compute F1. Found 2.8% of pairs significant (85 excitatory, 185 inhibitory). Inhibitory count higher than excitatory is anomalous — likely burst artifacts creating false inhibitory signatures.

---

## Jitter-Corrected CCG
**Datasets:** sim_cdkl5, sim_burst

- **sim_cdkl5** — F1=0.408. Moderate performance on clean data.
- **sim_burst** — F1=0.022. Failed. The ±25ms jitter window is too wide — burst co-firing persists across the jitter, so the surrogate and real CCG look the same.

---

## Pearson Correlation (with Bonferroni correction)
**Datasets:** sim_cdkl5

- **sim_cdkl5** — F1=0.352, Precision=0.401, Recall=0.314, TP=191, FP=285, FN=418. Moderate on clean data but misses many connections.

---

## Pearson Correlation (Burst Exclusion)
**Datasets:** sim_burst, sim_cdkl5

- **sim_burst** — F1=0.000, TP=0, FP=0, FN=228. Completely failed. 89% of spikes are inside bursts — after removing burst periods there are almost no spikes left to analyze.
- **sim_cdkl5** — Not the right method for this dataset.

---

## dSTTC (Directed Spike Time Tiling Coefficient)
**Datasets:** sim_burst, Real CDKL5 organoid

- **sim_burst** — F1=0.000. Completely failed. Every neuron pair looks correlated because they all fire together in bursts.
- **Real CDKL5** — No ground truth. Found 26.9% of pairs significant. No reference to know if this is accurate.

---

## sCCG (Smoothed Cross-Correlogram)
**Datasets:** sim_cdkl5, sim_burst, Real CDKL5 organoid

- **sim_cdkl5** — F1=0.000. Failed. The Poisson assumption (neurons fire independently) is violated even in non-bursty data.
- **sim_burst** — F1=0.000. Completely failed. Same reason, worse with bursting.
- **Real CDKL5** — No ground truth. Found 33.8% of pairs significant. Cannot trust.

---

## STTC (Spike Time Tiling Coefficient)
**Datasets:** Dale VAR, trial_135

- **Dale VAR** — F1=0.131, Precision=0.129, Recall=0.134. Poor. Dale connections are too weak to produce detectable co-firing.
- **trial_135** — F1=0.401, Precision=0.342, Recall=0.485. Unusable benchmark — 39% ground truth connection density makes all scores meaningless.

---

## Pearson FDR (Benjamini-Hochberg correction)
**Datasets:** Dale VAR

- **Dale VAR** — F1=0.000, Precision=0.000, Recall=0.000. Complete failure. Connected and unconnected neuron pairs had identical correlation distributions — no detectable signal.

---

## Granger Causality
**Datasets:** Dale VAR

- **Dale VAR** — F1=0.078, Precision=0.125, Recall=0.057. Failed. Pairwise F-test approach detects many false positives and misses most true connections. VAR connection weights too weak to detect reliably.

---

## Lasso-VAR (Lag-1)
**Datasets:** Dale VAR

- **Dale VAR** — F1=0.064, Precision=0.119, Recall=0.044, TP=212, FP=1571, FN=4586. Failed. Mathematically the correct method for VAR-generated data, but connection weights are too weak relative to noise to recover reliably.

---

## Multi-lag Lasso-VAR (Lags 1+2+3)
**Datasets:** Dale VAR

- **Dale VAR** — F1=0.056, Precision=0.116, Recall=0.037. Worse than lag-1 alone. Adding more lags introduced more noise than signal.

---

## Stability Lasso-VAR
**Datasets:** Dale VAR

- **Dale VAR** — F1=0.013, Precision=0.111, Recall=0.007, TP=33, FP=263, FN=4765. Worst result. Bootstrap stability selection was too aggressive — eliminated almost all true positives trying to reduce false positives.

---

## Population-Rate Lasso-VAR (Novel method)
**Datasets:** sim_burst

- **sim_burst** — F1=0.036, Precision=0.018, Recall=0.829. Failed. Found most real connections (high recall) but flagged almost everything as connected (terrible precision). Adding population rate as a covariate did not sufficiently remove burst drive from the signal.

---

## Pearson on Population-Rate Residuals (Novel method)
**Datasets:** sim_burst

- **sim_burst** — F1=0.035, Precision=0.018, Recall=0.873. Failed. Same problem as above — regressing out population rate left too much burst-driven correlation in the residuals.

---

## eANN (Ensemble ANN)
**Datasets:** sim_cdkl5, sim_burst

- Both runs crashed due to API error. Results pending after fix.
- Note: Donner et al. 2024 (the authors) already validated eANN on these exact datasets and reported strong performance (APS=0.88 on bursty regime). Our runs would only reproduce their results.

---

## STTC with Circular Shift Surrogates (MEA-NAP method)
**Datasets:** sim_burst

- Currently running. Results pending.

---

## CFP with Circular Shift Surrogates
**Datasets:** sim_burst

- Currently running after STTC completes. Results pending.

---

## Summary

The only method that produced meaningful results on bursty data was GLMCC (F1=0.195 on sim_burst), and even that is poor. Every other method failed on bursty data. The core problem is that network-wide bursting causes all neuron pairs to appear correlated, and no current method fully accounts for this. Results on Dale VAR data were consistently poor because the simulation was designed for state estimation, not connectivity inference.
