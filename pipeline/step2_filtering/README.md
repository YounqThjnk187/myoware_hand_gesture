# Step 2 – Signal Preprocessing / Filtering

**Scripts:** `filters.py` (shared library), `run_filtering.py` (runner)  
**Run:** `python -m pipeline.step2_filtering.run_filtering [--show]`

| | |
|---|---|
| Input | `artifacts/step1_clean.npz` |
| Output | `artifacts/step2_filtered.npz` (same keys, `X` shape `(reps, 3, 4200)`), `artifacts/figures/step2_*.png` |

## Filter chain

A single cascade of second-order sections (SOS, biquads) is used. Its coefficients are the ones exported to C in Step 9.

| Stage | Design | Purpose |
|---|---|---|
| Notch | `iirnotch(50 Hz, Q=30)`, −3 dB bandwidth $f_0/Q \approx 1.7$ Hz | Removes mains interference; the dataset is battery powered, but your MCU setup may not be |
| Bandpass | `butter(N=4, [20, 450] Hz, bandpass)`, giving 4 biquads (8 poles) | Below 20 Hz: motion artifacts and drift. Above 450 Hz: noise, and anything close to Nyquist (500 Hz) |

Notch transfer function:

$$H_{notch}(z)=\frac{1-2\cos\omega_0 z^{-1}+z^{-2}}{1-2r\cos\omega_0 z^{-1}+r^2 z^{-2}},\qquad \omega_0 = 2\pi\frac{50}{1000},\ r\approx 1-\frac{\pi f_0/Q}{f_s}$$

Each biquad is run in Direct-Form II Transposed. This is exactly what the firmware does:

$$y = b_0x + s_1,\quad s_1 = b_1x - a_1y + s_2,\quad s_2 = b_2x - a_2y$$

## Design decisions

* **Causal `sosfilt`, not `sosfiltfilt`.** Zero-phase filtering looks cleaner, but it needs future samples. The MCU cannot do that. If you train on zero-phase output and deploy with causal filtering, the features no longer match. So training uses the same causal filter as the device.
* **SOS instead of one `(b, a)` polynomial.** A 10th-order polynomial loses precision in float32 and can become unstable. Biquads stay stable. The script prints the largest pole radius, which must be < 1.
* **The first 300 samples are dropped.** The notch (Q = 30) rings with a time constant of about $1/(\pi \cdot 1.7\,\text{Hz}) \approx 0.19$ s, so its start-up transient is removed. On the MCU this happens once at power-up (see `EMG_TRANSIENT_SAMPLES` in Step 9).
* **What "4th order" means here.** SciPy's `butter(4, …, 'bandpass')` produces 8 poles, i.e. a 4th-order high-pass plus a 4th-order low-pass. Use `N=2` if your reference means 4 poles in total.

## What the script prints

* the SOS matrix, the largest pole radius, and the median group delay in the 50–150 Hz band (this delay counts toward latency);
* the percentage of power below 20 Hz, at 48–52 Hz, and in 20–450 Hz, before and after filtering.

## Pitfalls

* This only works at a 1000 Hz sampling rate. At 100 Hz (the old Arduino sketch) a 20–450 Hz band cannot exist.
* Use the MyoWare **RAW** output. The ENV/RECT outputs are already rectified and smoothed, so there is no bandpass content left to filter.
