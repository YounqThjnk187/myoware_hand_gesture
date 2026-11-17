"""Recognize the gesture in one 3-channel raw sEMG recording (1 kHz).

Inputs (pick one):
  --mov K --trial N      trial N (1-400) of dataset file MovK.mat
  --csv FILE             CSV with columns channel_1, channel_2, channel_3 (1 kHz). Supported layouts:
                         csv_data/MovK.csv (filter with --trial), csv_data/tdms/P{k}.csv (filter with --group),
                         or any single recording.
"""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from emg_pipeline import config
from emg_pipeline.dataset import load_mov, subject_lookup, subject_of
from emg_pipeline.processing import Calibration
from emg_pipeline.recognizer import recognize

CHANNEL_COLS = ["channel_1", "channel_2", "channel_3"]


def load_input(args):
    """Return (raw (3, time), true label or None, subject or None)."""
    if args.mov is not None:
        X = load_mov(args.mov)
        if not 1 <= args.trial <= len(X):
            raise SystemExit("Mov{} has trials 1-{}".format(args.mov, len(X)))
        return X[args.trial - 1], args.mov - 1, subject_of(args.mov, args.trial, subject_lookup())

    df = pd.read_csv(args.csv)
    missing = [c for c in CHANNEL_COLS if c not in df.columns]
    if missing:
        raise SystemExit("{} is missing columns {}".format(args.csv, missing))
    filters = (("trial", args.trial), ("group", args.group))
    for col, value in filters:
        if value is not None:
            if col not in df.columns:
                raise SystemExit("{} has no '{}' column".format(args.csv, col))
            df = df[df[col] == value]
    for col, value in filters:
        if value is None and col in df.columns and df[col].nunique() > 1:
            raise SystemExit("{} contains several recordings; pass --{}".format(args.csv, col))
    if df.empty:
        raise SystemExit("no rows selected")
    if "sample" in df.columns:
        df = df.sort_values("sample")
    true_label, subject = None, args.subject
    if "mov" in df.columns and df["mov"].iloc[0] > 0:
        true_label = int(df["mov"].iloc[0]) - 1
    mov_file = re.fullmatch(r"Mov([1-5])\.csv", Path(args.csv).name)
    if mov_file and args.trial is not None:
        true_label = int(mov_file.group(1)) - 1
        if subject is None:
            subject = subject_of(true_label + 1, args.trial, subject_lookup())
    return df[CHANNEL_COLS].to_numpy(dtype=np.float64).T, true_label, subject


def segments(decisions, gestures):
    """Collapse per-window decisions into [start_s, end_s, name]; window k ends at TRANSIENT + k*STEP + WINDOW."""
    out = []
    for k, d in enumerate(decisions):
        name = "(uncertain)" if d is None else gestures[d]
        t_end = (config.TRANSIENT + k * config.STEP + config.WINDOW) / config.FS
        if out and out[-1][2] == name:
            out[-1][1] = t_end
        else:
            out.append([t_end, t_end, name])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mov", type=int, choices=range(1, 6))
    src.add_argument("--csv")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--group", help="TDMS group name, e.g. 'Group Name #45'")
    parser.add_argument("--subject", type=int, help="subject id (1-20) whose calibration to use for --csv input")
    parser.add_argument("--save-input", help="write the recording that was classified as CSV")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    args = parser.parse_args(argv)
    if args.mov is not None and args.trial is None:
        parser.error("--mov needs --trial")

    bundle = joblib.load(args.model)
    gestures = bundle["gestures"]
    raw, true_label, subject = load_input(args)

    if subject is not None and subject in bundle["calibrations"]:
        calib, src = Calibration(bundle["calibrations"][subject]), "subject {}".format(subject)
    else:
        calib, src = Calibration(bundle["global_calibration"]), "global (unknown or held-out subject)"
    if args.mov is not None and (args.mov, args.trial) not in set(map(tuple, bundle["test_trials"])):
        print("Note: Mov{} trial {} was used for training; held-out examples are listed in {}".format(
            args.mov, args.trial, config.REPORT_PATH.name))
    if args.save_input:
        out = pd.DataFrame(raw.T, columns=CHANNEL_COLS)
        out.insert(0, "sample", np.arange(len(out)))
        out.to_csv(args.save_input, index=False, float_format="%.7g")
        print("Saved input -> {}".format(args.save_input))

    print("Input: {} samples x {} channels at {} Hz ({:.2f} s)".format(
        raw.shape[1], raw.shape[0], config.FS, raw.shape[1] / config.FS))
    print("Calibration: {}  scale {}".format(src, np.round(calib.scale, 4).tolist()))

    result = recognize(bundle["model"], raw, calib)
    print("Model: {} (split by {})  windows: {} x {} ms, step {} ms, vote {}, min proba {}".format(
        bundle["model_name"].upper(), bundle["split"], len(result["decisions"]),
        config.WINDOW * 1000 // config.FS, config.STEP * 1000 // config.FS, config.VOTE_N, config.MIN_PROBA))

    print("\nTimeline of stable decisions:")
    for start, end, name in segments(result["decisions"], gestures):
        print("  {:5.2f} - {:5.2f} s  {}".format(start, end, name))

    mean_p = result["proba"].mean(axis=0)
    print("\nMean probability per gesture:")
    for k in np.argsort(-mean_p):
        print("  {:20s} {:.3f}".format(gestures[bundle["model"].classes_[k]], mean_p[k]))

    if result["gesture"] is None:
        print("\nRecognized gesture: none (no window above the confidence threshold)")
    else:
        print("\nRecognized gesture: {}  ({:.0%} of decisions)".format(gestures[result["gesture"]], result["share"]))
    if true_label is not None:
        print("True gesture:       {}  -> {}".format(
            gestures[true_label], "CORRECT" if result["gesture"] == true_label else "WRONG"))


if __name__ == "__main__":
    main()
