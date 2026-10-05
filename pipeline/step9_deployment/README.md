# Step 9 – Embedded Deployment & C/C++ Packaging

**Scripts:** `export_c.py` (Python → C header), `firmware/emg_realtime/` (Arduino sketch + portable C/C++ pipeline)  
**Run:** `python -m pipeline.step9_deployment.export_c [--model lda]`, then open `firmware/emg_realtime/emg_realtime.ino` in the Arduino IDE and upload it.

| | |
|---|---|
| Input | `artifacts/models/lda.joblib`, `artifacts/normalizer_profile.json`, `artifacts/step4_features.npz` (ZC threshold), filter design from Step 2 |
| Output | `firmware/emg_realtime/emg_model.h` (generated; all constants) |

## Files

| File | Role |
|---|---|
| `emg_model.h` | **Generated.** Sampling / window constants, SOS coefficients, EMA α, ZC threshold, Min-Max `min`/`max`, LDA weights `W[5][12]` and bias `b[5]`, class names |
| `emg_pipeline.h/.cpp` | Portable, Arduino-independent: EMA baseline → biquad cascade → ring buffer → MAV/RMS/WL/ZC → Min-Max + clip → $\arg\max_k (Wz+b)_k$ |
| `emg_realtime.ino` | Samples A0–A2 every 1000 µs using `micros()` with drift-free scheduling. Prints `label,compute_us` every 100 ms |

## Mapping from Python to firmware

| Python step | Firmware equivalent |
|---|---|
| Step 1: per-repetition mean removal | `baseline += α (x − baseline)`, α = $1-e^{-2\pi\cdot 1/1000}$ (1 Hz EMA) |
| Step 2: `sosfilt(sos, x)` | `biquad_cascade()`, DF-II-T with the same coefficients; one state per channel |
| Step 2: drop the first 300 samples | no decisions until `EMG_TRANSIENT_SAMPLES + EMG_WINDOW_SIZE` samples |
| Step 3: 200 / 100 windows | ring buffer of 200 samples; `emg_push_sample` returns 1 every 100 samples |
| Step 4: `extract_features` | one pass per channel, same feature order and ZC rule |
| Step 5: `transform(clip=True)` | same formula, span = 0 → 1, clipped to [0, 1] |
| Step 7: `lda.predict` | `argmax(W z + b)`. `export_c.py` checks in float32 that this matches sklearn on the test set |

## Resources (3 channels, 5 SOS, 12 features, 5 classes)

* **RAM:** about 2.5 KB of pipeline state, almost all of it the 3 × 200 float ring buffer. This does **not fit an AVR UNO R3** (2 KB total). Use an UNO R4, ESP32, STM32 or similar.
* **Flash:** under 1 KB of constants.
* **CPU per sample:** 3 channels × 5 biquads ≈ 75 MACs.
* **CPU per decision:** about 2 400 operations for features plus 60 MACs for LDA. That is well under 1 ms on a Cortex-M4F at 48 MHz, compared with a 100 ms budget. The `compute_us` column is the measured value.

## Hardware checklist

1. Use the MyoWare 2.0 **RAW** (ENV off) output on each sensor, and give all three a common ground and reference electrode.
2. Place the electrodes to reproduce the **dataset's** channels (Ch1 anterior forearm, Ch2 posterior forearm, Ch3 anterior radius), in that order. See the Step 1 findings.
3. Set `ADC_VREF` / `ADC_MAX` for your board: UNO R4 5 V / 1023; ESP32 3.3 V / 4095 with `analogReadResolution(12)`.
4. Verify the timing: the sampling rate must really be 1000 Hz. Three `analogRead`s take about 60 µs on an R4, which is fine; on an AVR they take about 340 µs, which is tight.

## Domain shift: required before trusting the output

The dataset amplitudes are volts after *their* amplifier gain. Your ADC counts × `ADC_TO_SIGNAL` pass through MyoWare's gain instead. MAV, RMS and WL therefore land in a different range, and a profile fitted on the dataset will clip or compress them.

**Fix:** record a short calibration session on your own arm with `1_collect_data.py` (updated for 3 channels at 1 kHz). Run Steps 1–6 on it, or at least refit `normalizer_profile.json`, then run `export_c.py` again. Ideally, retrain LDA on your own data too. It trains in seconds.

## Other model types

* **RF/SVM:** these have no `coef_`, so `export_c.py` refuses them. Convert them with [emlearn](https://github.com/emlearn/emlearn) (RF → C if/else trees) or `micromlgen`. Check the flash size first: 100 unbounded trees can reach hundreds of KB.
* **1D-CNN:** train in Keras, quantize to int8 with the TFLite converter, and run it with TensorFlow Lite Micro. Keep this Step's C front-end (filter + ring buffer) and feed the `(3, 200)` window to the interpreter instead of computing features.
