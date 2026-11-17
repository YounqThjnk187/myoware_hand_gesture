"""Recognize the gesture in one 3-channel recording.

Inputs (pick one):
  --mov K --trial N    raw 1 kHz trial N from csv_data/MovK.csv (simulated into a KIT0012 envelope)
  --csv FILE           either seq,a0,a1,a2 (KIT0012 ADC at 200 Hz, the Uno logger format)
                       or channel_1,channel_2,channel_3 raw 1 kHz sEMG (optionally filtered by --trial)
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from envelope_pipeline import config
from envelope_pipeline.dataset import CHANNEL_COLS, load_raw_trials, subject_of
from envelope_pipeline.kit0012_sim import raw_to_envelope, to_adc
from envelope_pipeline.preprocess import Calibration
from envelope_pipeline.recognizer import recognize

ADC_COLS = ["a0", "a1", "a2"]


def load_input(args, bundle):
    """Return (adc (3, time) at FS, true label or None, subject or None)."""
    if args.mov is not None:
        _, X = load_raw_trials(args.mov - 1, [args.trial])
        return to_adc(raw_to_envelope(X[0]), bundle["adc_scale"]), args.mov - 1, int(subject_of(args.trial))

    df = pd.read_csv(args.csv)
    if all(c in df.columns for c in ADC_COLS):
        if "seq" in df.columns:
            gaps = np.diff(df["seq"].to_numpy())
            if np.any(gaps != 1):
                print("Warning: {} missing/out-of-order samples according to seq".format(int(np.sum(gaps != 1))))
        return df[ADC_COLS].to_numpy(dtype=np.float64).T, None, None
    if all(c in df.columns for c in CHANNEL_COLS):
        if "trial" in df.columns:
            if args.trial is None and df["trial"].nunique() > 1:
                raise SystemExit("{} contains several trials; pass --trial".format(args.csv))
            if args.trial is not None:
                df = df[df["trial"] == args.trial]
        if "sample" in df.columns:
            df = df.sort_values("sample")
        raw = df[CHANNEL_COLS].to_numpy(dtype=np.float64).T
        return to_adc(raw_to_envelope(raw), bundle["adc_scale"]), None, None
    raise SystemExit("CSV needs columns {} or {}".format(ADC_COLS, CHANNEL_COLS))


def segments(decisions, gestures):
    """Collapse per-window decisions into (start_s, end_s, name) runs; window k ends at (k*STEP+WINDOW)/FS."""
    out = []
    for k, d in enumerate(decisions):
        name = "(uncertain)" if d is None else gestures[d]
        t_end = (k * config.STEP + config.WINDOW) / config.FS
        if out and out[-1][2] == name:
            out[-1][1] = t_end
        else:
            out.append([t_end, t_end, name])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mov", type=int, choices=range(1, 6), help="gesture file csv_data/MovK.csv")
    src.add_argument("--csv", help="input CSV file")
    parser.add_argument("--trial", type=int, help="trial number (1-400)")
    parser.add_argument("--calib", help="calibration JSON {baseline: [3], mvc: [3]} in ADC counts")
    parser.add_argument("--save-input", help="also write the envelope fed to the model as seq,a0,a1,a2 CSV")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    args = parser.parse_args(argv)
    if args.mov is not None and args.trial is None:
        parser.error("--mov needs --trial")

    bundle = joblib.load(args.model)
    gestures = bundle["gestures"]
    adc, true_label, subject = load_input(args, bundle)

    if args.calib:
        calib, calib_src = Calibration.load(args.calib), args.calib
    elif subject is not None and subject in bundle["calibrations"]:
        calib, calib_src = Calibration.from_dict(bundle["calibrations"][subject]), "subject {} (from training)".format(subject)
    else:
        calib, calib_src = Calibration.from_dict(bundle["global_calibration"]), "global (median over training subjects)"

    if args.mov is not None and (args.mov, args.trial) not in set(map(tuple, bundle["test_trials"])):
        print("Note: Mov{} trial {} was used for training; pick a held-out trial from {}".format(
            args.mov, args.trial, config.REPORT_PATH.name))
    if args.save_input:
        out = pd.DataFrame(adc.T.astype(int), columns=ADC_COLS)
        out.insert(0, "seq", np.arange(len(out)))
        out.to_csv(args.save_input, index=False)
        print("Saved input envelope -> {}".format(args.save_input))

    print("Input: {} samples x {} channels at {} Hz ({:.2f} s), ADC range per channel min {} max {}".format(
        adc.shape[1], adc.shape[0], config.FS, adc.shape[1] / config.FS,
        adc.min(axis=1).astype(int).tolist(), adc.max(axis=1).astype(int).tolist()))
    print("Calibration: {}  baseline {}  MVC {}".format(
        calib_src, np.round(calib.baseline, 1).tolist(), np.round(calib.mvc, 1).tolist()))

    result = recognize(bundle["model"], adc, calib)
    n = len(result["decisions"])
    print("Model: {}  windows: {} ({} ms, step {} ms, vote {} / min proba {})".format(
        bundle["model_name"].upper(), n, 1000 * config.WINDOW // config.FS, 1000 * config.STEP // config.FS,
        config.VOTE_N, config.MIN_PROBA))

    print("\nTimeline of stable decisions:")
    for start, end, name in segments(result["decisions"], gestures):
        print("  {:5.2f} - {:5.2f} s  {}".format(start, end, name))

    mean_p = result["proba"].mean(axis=0)
    print("\nMean probability per gesture:")
    for k in np.argsort(-mean_p):
        print("  {:20s} {:.3f}".format(gestures[bundle["model"].classes_[k]], mean_p[k]))

    if result["gesture"] is None:
        print("\nRecognized gesture: none (all windows below confidence threshold)")
    else:
        print("\nRecognized gesture: {}  ({:.0%} of decisions)".format(gestures[result["gesture"]], result["share"]))
    if true_label is not None:
        print("True gesture:       {}  -> {}".format(
            gestures[true_label], "CORRECT" if result["gesture"] == true_label else "WRONG"))


if __name__ == "__main__":
    main()
