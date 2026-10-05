# myoware_hand_gesture

## RAW EMG acquisition

The raw Arduino sketch in `myo notfiltered/myo2.ino` sends packets in the existing format:

```text
timestamp_ms,raw_adc
```

It uses the existing analog pin, 115200 baud rate, and 100 Hz sampling schedule. Upload that sketch, close Arduino Serial Monitor/Plotter, then run:

```powershell
python -m pip install -r requirements.txt
python 1_collect_data.py
```

Use a standard Windows Python installation with Tcl/Tk enabled; `tkinter` is part of Python on Windows but is not installed by pip.

Enter the Arduino COM port in the window, then use `START RECORDING` and `STOP RECORDING`. Each start/stop interval creates a separate file under:

```text
recordings/session_YYYYMMDD_HHMMSS/trial_YYYYMMDD_HHMMSS.csv
```

Each CSV contains the device timestamp, a sequential valid-packet index, and the unchanged decoded ADC value:

```text
timestamp,sample_index,raw_emg
```

The graph is monitoring-only. Serial reading and CSV writing run in a worker thread; malformed packets are skipped and counted. The existing `2_plot_report.py` remains a separate post-processing script for the older labeled CSV format.

Before each recording, set `Rest before recording (s)` and wait for the countdown. Use this time to relax the forearm and switch to the next gesture. Samples received during the countdown are discarded from the serial buffer, so the new CSV begins only when recording starts.

## How to use the sEMG gesture-recognition pipeline

The `pipeline/` package trains a 5-gesture classifier on the Kaggle dataset `dataset/kaggle/DS1_EMG_SIGNALS_AVERAGE` (3 channels, 1000 Hz) and exports it to C for a microcontroller. Each step lives in its own folder, and each folder has a `README.md` explaining the theory, the parameters and the pitfalls.

| Step | Folder | Command | Main output (in `artifacts/`) |
|---|---|---|---|
| 1 Data inspection & cleaning | [pipeline/step1_data_inspection](pipeline/step1_data_inspection/README.md) | `python -m pipeline.step1_data_inspection.inspect_data` | `step1_clean.npz`, `step1_quality_report.csv` |
| 2 Notch + bandpass filtering | [pipeline/step2_filtering](pipeline/step2_filtering/README.md) | `python -m pipeline.step2_filtering.run_filtering` | `step2_filtered.npz` |
| 3 Sliding windows (200 ms / 100 ms) | [pipeline/step3_windowing](pipeline/step3_windowing/README.md) | `python -m pipeline.step3_windowing.run_windowing` | `step3_windows.npz` |
| 4 Features (MAV, RMS, WL, ZC × 3) | [pipeline/step4_features](pipeline/step4_features/README.md) | `python -m pipeline.step4_features.run_features` | `step4_features.npz` |
| 5 Min-Max normalization | [pipeline/step5_normalization](pipeline/step5_normalization/README.md) | *(fitted inside Step 6, on the training split only)* | `normalizer_profile.json` |
| 6 Stratified 70/15/15 split | [pipeline/step6_splitting](pipeline/step6_splitting/README.md) | `python -m pipeline.step6_splitting.split_and_normalize [--by repetition\|subject]` | `step6_dataset.npz` |
| 7 Training + 5-fold CV | [pipeline/step7_training](pipeline/step7_training/README.md) | `python -m pipeline.step7_training.train` | `models/*.joblib`, `step7_summary.json` |
| 8 Evaluation + latency | [pipeline/step8_evaluation](pipeline/step8_evaluation/README.md) | `python -m pipeline.step8_evaluation.evaluate` | `step8_results.json` |
| 9 C export + firmware | [pipeline/step9_deployment](pipeline/step9_deployment/README.md) | `python -m pipeline.step9_deployment.export_c` | `pipeline/step9_deployment/firmware/emg_realtime/emg_model.h` |

### 1. Setup

Use a 64-bit Python ≥ 3.8 with the dataset `.mat` files in place.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Run

Run every command from the repository root, so that `pipeline` can be imported.

```powershell
# Everything, Steps 1 -> 9
python -m pipeline.run_all

# Or resume from a step (earlier artifacts must exist), e.g. re-train and re-export only
python -m pipeline.run_all --from-step 7
```

Each step also runs on its own with the commands in the table. Add `--show` to steps 1–4, 7 and 8 to open the figures; they are always saved to `artifacts/figures/`. All tunable constants are in [pipeline/config.py](pipeline/config.py): filter band, window size, ZC threshold, split ratios, latency budget.

Useful options:

* `split_and_normalize --by subject`: tests on 3 unseen subjects. This is the realistic estimate for a new user.
* `train --models lda svm --max-tune-samples 10000`: trains fewer models and tunes on a smaller sample, which is faster.
* `run_features --zc-threshold 0.02`: sets the ZC noise threshold. It is exported to C automatically.

### 3. Deploy to the microcontroller

1. Run Step 9. It writes `emg_model.h` next to the sketch.
2. Open `pipeline/step9_deployment/firmware/emg_realtime/emg_realtime.ino` in the Arduino IDE. Use a board with ≥ 8 KB RAM, e.g. an UNO R4 or ESP32.
3. Connect three MyoWare 2.0 **RAW** outputs to A0/A1/A2. Place them like the dataset: anterior forearm, posterior forearm, anterior radius.
4. Upload and open Serial Monitor at 115200 baud. Every 100 ms it prints `gesture,compute_us`.

### 4. Before trusting results on your own hardware

* **Sampling rate:** the existing `myo/`, `myo2/` sketches sample at 50–100 Hz with envelope/smoothing filters. The pipeline needs **1000 Hz raw** data. Use the Step 9 sketch as the acquisition reference.
* **Channel placement:** in the dataset, Ch3 dominates Thumb flexion and Finger extension (see the Step 1 output). Electrodes placed on FCR / EDC / brachioradialis will produce a different pattern.
* **Amplitude scale:** refit the Min-Max profile (and ideally LDA) on a short calibration recording from your own sensors, then re-run Step 9. See the "Domain shift" section in [pipeline/step9_deployment/README.md](pipeline/step9_deployment/README.md).