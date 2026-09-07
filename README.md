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