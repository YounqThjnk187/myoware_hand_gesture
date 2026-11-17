"""Load Kaggle trials from csv_data/MovK.csv and turn them into simulated KIT0012 ADC streams."""

import numpy as np
import pandas as pd

from envelope_pipeline import config
from envelope_pipeline.kit0012_sim import raw_to_envelope, to_adc

CHANNEL_COLS = ["channel_1", "channel_2", "channel_3"]


def load_raw_trials(label, trials=None):
    """Return (trial ids, X (n_trials, 3, time)) for gesture index `label`."""
    path = config.RAW_CSV_DIR / "Mov{}.csv".format(label + 1)
    dtypes = {"sample": np.int32, "trial": np.int32}
    dtypes.update({c: np.float32 for c in CHANNEL_COLS})
    df = pd.read_csv(path, dtype=dtypes)
    if trials is not None:
        df = df[df["trial"].isin(trials)]
    df = df.sort_values(["trial", "sample"], kind="stable")
    ids, counts = np.unique(df["trial"].to_numpy(), return_counts=True)
    if len(ids) == 0:
        raise ValueError("no trials {} in {}".format(trials, path))
    if np.any(counts != counts[0]):
        raise ValueError("trials in {} have different lengths".format(path))
    X = df[CHANNEL_COLS].to_numpy().reshape(len(ids), counts[0], config.N_CHANNELS)
    return ids, np.ascontiguousarray(X.transpose(0, 2, 1))


def subject_of(trial_ids):
    return (np.asarray(trial_ids) - 1) // config.TRIALS_PER_SUBJECT


def load_envelope_dataset(rebuild=False):
    """Dict with E (trials, 3, time) int16 ADC at FS, y, trial, subject, scale. Cached as npz."""
    if config.DATASET_CACHE.exists() and not rebuild:
        with np.load(config.DATASET_CACHE) as d:
            return {k: d[k] for k in d.files}

    envs, ys, trials = [], [], []
    for label, name in enumerate(config.GESTURES):
        print("Loading Mov{}.csv ({}) and simulating KIT0012 envelope...".format(label + 1, name))
        ids, X = load_raw_trials(label)
        envs.append(raw_to_envelope(X).astype(np.float32))
        ys.append(np.full(len(ids), label, dtype=np.int64))
        trials.append(ids)
    env = np.concatenate(envs)
    scale = config.SIM_ADC_PEAK / np.percentile(env, 99.5, axis=(0, 2))
    trial = np.concatenate(trials)
    data = {
        "E": to_adc(env, scale),
        "y": np.concatenate(ys),
        "trial": trial,
        "subject": subject_of(trial),
        "scale": scale,
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.DATASET_CACHE, **data)
    return data
