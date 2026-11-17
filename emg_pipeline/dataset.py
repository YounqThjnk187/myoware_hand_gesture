"""Step 1: load the 2000 labelled trials (raw 1 kHz, 3 channels) with gesture and subject ids."""

import numpy as np
import pandas as pd
import scipy.io

from emg_pipeline import config


def load_mov(mov):
    """Raw trials of csv/mat file MovK: (400, 3, 4500)."""
    name = "Mov{}".format(mov)
    return scipy.io.loadmat(str(config.DATA_DIR / "{}.mat".format(name)))[name]


def subject_lookup():
    """{(mov, trial): subject} from the TDMS index, or None if tdms_extract has not been run."""
    if not config.TRIAL_INDEX.exists():
        return None
    idx = pd.read_csv(config.TRIAL_INDEX)
    return {(int(m), int(t)): int(s) for m, t, s in zip(idx["mov"], idx["trial"], idx["subject"])}


def conflicting_trials():
    """(mov, trial) pairs whose recording appears in two MovK files with different labels."""
    if not config.TRIAL_INDEX.exists():
        return set()
    idx = pd.read_csv(config.TRIAL_INDEX)
    dup = idx[idx.duplicated(["subject", "group"], keep=False)]
    return set(zip(dup["mov"].astype(int), dup["trial"].astype(int)))


def subject_of(mov, trial, lookup=None):
    if lookup is not None and (mov, trial) in lookup:
        return lookup[(mov, trial)]
    return (trial - 1) // config.TRIALS_PER_SUBJECT + 1


def load_dataset(rebuild=False):
    """Dict with X (n, 3, 4500) float32, y (0-4), mov, trial, subject (1-20); conflicting duplicates removed."""
    if config.DATASET_CACHE.exists() and not rebuild:
        with np.load(config.DATASET_CACHE) as d:
            data = {k: d[k] for k in d.files}
    else:
        data = _load_from_mat()
    bad = conflicting_trials()
    keep = np.array([(int(m), int(t)) not in bad for m, t in zip(data["mov"], data["trial"])])
    if not keep.all():
        print("Dropped {} trials that duplicate another trial with a different label".format(int((~keep).sum())))
    return {k: v[keep] for k, v in data.items()}


def _load_from_mat():
    lookup = subject_lookup()
    if lookup is None:
        print("Warning: {} not found, assuming 20 consecutive trials per subject "
              "(run python -m emg_pipeline.tdms_extract)".format(config.TRIAL_INDEX))
    xs, movs, trials = [], [], []
    for mov in range(1, len(config.GESTURES) + 1):
        X = load_mov(mov)
        xs.append(X.astype(np.float32))
        movs.append(np.full(len(X), mov))
        trials.append(np.arange(1, len(X) + 1))
    mov, trial = np.concatenate(movs), np.concatenate(trials)
    data = {
        "X": np.concatenate(xs),
        "y": mov - 1,
        "mov": mov,
        "trial": trial,
        "subject": np.array([subject_of(int(m), int(t), lookup) for m, t in zip(mov, trial)]),
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.DATASET_CACHE, **data)
    return data
