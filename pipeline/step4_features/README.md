# Step 4 – Feature Extraction

**Scripts:** `features.py` (library, also used for latency tests), `run_features.py` (runner)  
**Run:** `python -m pipeline.step4_features.run_features [--zc-threshold 0.01] [--show]`

| | |
|---|---|
| Input | `artifacts/step3_windows.npz` |
| Output | `artifacts/step4_features.npz`: `F` `(M, 12)`, `y`, `subject`, `rep_uid`, `feature_names`, `zc_threshold` |

## Feature set (per channel, window of $N = 200$ samples)

| Feature | Formula | What it captures | Cost |
|---|---|---|---|
| MAV | $\frac{1}{N}\sum_i \lvert x_i\rvert$ | Contraction level | N adds |
| RMS | $\sqrt{\frac{1}{N}\sum_i x_i^2}$ | Signal power; related to force | N MACs + 1 sqrt |
| WL | $\sum_{i=1}^{N-1} \lvert x_{i+1}-x_i\rvert$ | Combined amplitude × frequency ("waveform complexity") | N subs |
| ZC | $\sum_i \mathbb{1}[x_i x_{i+1}<0 \wedge \lvert x_i-x_{i+1}\rvert \ge \theta]$ | Rough frequency content | N compares |

**Order (fixed, and identical in C):** `[MAV_ch1..3, RMS_ch1..3, WL_ch1..3, ZC_ch1..3]`, i.e. feature index $= 3f + c$.

Everything is computed in a single $O(N)$ pass per channel: about 2400 operations per decision. That is tiny even on a Cortex-M0.

## Choosing parameters

* **ZC threshold θ (default 0.01, in dataset units):** without a threshold, noise around zero at rest produces many false crossings. θ should sit just above the noise floor. A rule of thumb is ≈ 3 × the RMS of filtered Rest windows. Compare the Rest row of the printed table with the active gestures and adjust with `--zc-threshold`. The value is saved and exported to C automatically.
* **Why time-domain only:** frequency-domain features (MNF/MDF) need an FFT and DWT needs a filter bank, while these four need one pass per channel. For a 3-channel, 5-gesture problem, TD features usually carry most of the discriminative information.

## Pitfalls

* **MAV, RMS and WL scale with amplitude units.** The dataset is in volts after an unknown acquisition gain. Your MCU reads ADC counts through a different gain. Step 9 converts counts to volts, but the gains still differ, so plan to **recalibrate the Min-Max profile** (Step 5) with your own recordings.
* **The 1D-CNN alternative skips this step.** It uses `W` from Step 3 directly.
