"""Step 0: map every MovK trial to its subject/TDMS group and optionally export P*.tdms to CSV.

Each TDMS group is one ~8.1 s repetition; the MovK trial is a 4.5 s slice of it. Groups that match
no MovK trial were not kept by the dataset authors and are exported with an empty label.
"""

import argparse

import numpy as np
import pandas as pd
import scipy.io
from nptdms import TdmsFile

from emg_pipeline import config

KEY_LEN = 4


def read_subject(k):
    """Return [(group name, array (3, n))] for P{k}.tdms."""
    out = []
    with TdmsFile.open(str(config.DATA_DIR / "P{}.tdms".format(k))) as f:
        for g in f.groups():
            chans = g.channels()
            if len(chans) == config.N_CHANNELS and len(chans[0]) > 0:
                out.append((g.name, np.stack([c[:] for c in chans])))
    return out


def key_of(x):
    return tuple(np.round(x[:, :KEY_LEN].ravel(), 9))


def locate(trial, groups):
    """Search every group for the trial start (slow path)."""
    for gi, (_, data) in enumerate(groups):
        n = trial.shape[1]
        for i in np.flatnonzero(np.isclose(data[0, : data.shape[1] - n + 1], trial[0, 0], atol=1e-9)):
            if np.allclose(data[:, i : i + KEY_LEN], trial[:, :KEY_LEN], atol=1e-9):
                return gi, int(i)
    return None


def build_index(subjects):
    print("Reading TDMS files...")
    groups = {k: read_subject(k) for k in subjects}
    rows = []
    for mov in range(1, len(config.GESTURES) + 1):
        X = scipy.io.loadmat(str(config.DATA_DIR / "Mov{}.mat".format(mov)))["Mov{}".format(mov)]
        n = X.shape[2]
        # Fast path: in P1 every trial starts 101 samples before the end of a 4.5 s tail.
        fast = {}
        for k, gs in groups.items():
            for gi, (_, data) in enumerate(gs):
                off = data.shape[1] - n - 101
                if off >= 0:
                    fast[key_of(data[:, off:])] = (k, gi, off)
        missing = 0
        for t in range(X.shape[0]):
            hit = fast.get(key_of(X[t]))
            if hit is None:
                guess = t // config.TRIALS_PER_SUBJECT + 1
                order = [guess] + [k for k in groups if k != guess]
                for k in order:
                    if k in groups:
                        found = locate(X[t], groups[k])
                        if found:
                            hit = (k, found[0], found[1])
                            break
            if hit is None:
                missing += 1
                continue
            k, gi, off = hit
            rows.append((mov, t + 1, k, groups[k][gi][0], off, n))
        print("  Mov{}: {}/{} trials located".format(mov, X.shape[0] - missing, X.shape[0]))
    return pd.DataFrame(rows, columns=["mov", "trial", "subject", "group", "start", "length"]), groups


def export_csv(k, groups, index):
    # A few TDMS groups appear in two MovK files with different labels; export those unlabelled.
    labels = index[index["subject"] == k].drop_duplicates("group", keep=False).set_index("group")
    parts = []
    for name, data in groups[k]:
        n = data.shape[1]
        df = pd.DataFrame(data.T, columns=["channel_1", "channel_2", "channel_3"])
        df.insert(0, "sample", np.arange(n))
        df.insert(0, "group", name)
        if name in labels.index:
            r = labels.loc[name]
            df["mov"], df["trial"] = int(r["mov"]), int(r["trial"])
            df["in_trial"] = ((df["sample"] >= r["start"]) & (df["sample"] < r["start"] + r["length"])).astype(int)
        else:
            df["mov"], df["trial"], df["in_trial"] = 0, 0, 0
        parts.append(df)
    path = config.TDMS_CSV_DIR / "P{}.csv".format(k)
    pd.concat(parts).to_csv(path, index=False, float_format="%.7g")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", type=int, nargs="*", metavar="K",
                        help="also write csv_data/tdms/P{K}.csv (~40 MB each); no K = all subjects")
    parser.add_argument("--rebuild", action="store_true", help="rebuild trial_index.csv even if it exists")
    args = parser.parse_args(argv)
    all_subjects = list(range(1, config.N_SUBJECTS + 1))

    groups = {}
    if config.TRIAL_INDEX.exists() and not args.rebuild:
        index = pd.read_csv(config.TRIAL_INDEX)
        print("Using existing {}".format(config.TRIAL_INDEX))
    else:
        index, groups = build_index(all_subjects)
        config.TDMS_CSV_DIR.mkdir(parents=True, exist_ok=True)
        index.to_csv(config.TRIAL_INDEX, index=False)
        print("Saved {}".format(config.TRIAL_INDEX))
    print("Trials per subject: {}".format(index.groupby("subject").size().to_dict()))
    assumed = (index["trial"] - 1) // config.TRIALS_PER_SUBJECT + 1
    print("Trials whose subject differs from the 20-per-subject assumption: {}".format(int((assumed != index["subject"]).sum())))

    if args.export_csv is not None:
        for k in args.export_csv or all_subjects:
            if k not in groups:
                groups[k] = read_subject(k)
            print("Saved {}".format(export_csv(k, groups, index)))


if __name__ == "__main__":
    main()
