"""Step 1: load Mov1-Mov5 with the correct axes, check signal quality, drop bad repetitions."""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from pipeline import config


def load_raw(data_dir=config.DATA_DIR):
    """Return X (reps, channels, time), gesture label, subject id and unique repetition id."""
    xs, ys, subjects = [], [], []
    for label in range(len(config.GESTURES)):
        name = "Mov{}".format(label + 1)
        x = scipy.io.loadmat(str(data_dir / "{}.mat".format(name)))[name]
        if x.ndim != 3 or x.shape[1] != config.N_CHANNELS or x.shape[0] > x.shape[2]:
            raise ValueError(
                "{} has shape {}; expected (repetitions, 3, time)".format(name, x.shape)
            )
        n_reps = x.shape[0]
        xs.append(x.astype(np.float32))
        ys.append(np.full(n_reps, label, dtype=np.int64))
        subjects.append(np.arange(n_reps) // config.REPS_PER_SUBJECT)
    X = np.concatenate(xs)
    return X, np.concatenate(ys), np.concatenate(subjects), np.arange(len(X))


def quality_report(X, y):
    """Per-repetition quality metrics and rejection flags."""
    finite = np.isfinite(X).all(axis=(1, 2))
    dc = X.mean(axis=2, dtype=np.float64)
    rms = X.std(axis=2, dtype=np.float64)

    # Clipping = repeated identical samples in the top 10 % of each channel's amplitude range.
    peak = np.nanmax(np.abs(X), axis=(0, 2), keepdims=True)
    plateau = (np.diff(X, axis=2) == 0) & (np.abs(X[..., 1:]) > 0.9 * peak)
    clip_ratio = plateau.mean(axis=2)

    n_blocks = X.shape[2] // config.FLAT_BLOCK
    blocks = X[..., : n_blocks * config.FLAT_BLOCK].reshape(
        X.shape[0], X.shape[1], n_blocks, config.FLAT_BLOCK
    )
    flat = (blocks.std(axis=-1) < config.FLAT_EPS).any(axis=(1, 2))

    # Robust z-score on log-RMS, per gesture and channel (RMS is right-skewed across subjects).
    log_rms = np.log(rms + 1e-12)
    z = np.zeros_like(log_rms)
    for label in np.unique(y):
        m = y == label
        med = np.median(log_rms[m], axis=0)
        mad = np.median(np.abs(log_rms[m] - med), axis=0) + 1e-12
        z[m] = 0.6745 * (log_rms[m] - med) / mad

    reasons = {
        "non_finite": ~finite,
        "clipping": (clip_ratio > config.CLIP_RATIO_LIMIT).any(axis=1),
        "flatline": flat,
        "rms_outlier": (np.abs(z) > config.RMS_Z_LIMIT).any(axis=1),
    }
    bad = np.zeros(len(X), dtype=bool)
    for flag in reasons.values():
        bad |= flag
    return {"dc": dc, "rms": rms, "clip_ratio": clip_ratio, "z": z, "reasons": reasons, "bad": bad}


def print_summary(report, y):
    print("\nRejected repetitions per gesture:")
    header = "{:22s}".format("Gesture") + "".join(
        "{:>13s}".format(k) for k in report["reasons"]
    ) + "{:>10s}".format("total")
    print(header)
    for label, name in enumerate(config.GESTURES):
        m = y == label
        row = "{:22s}".format(name)
        row += "".join("{:13d}".format(int(v[m].sum())) for v in report["reasons"].values())
        row += "{:7d}/{}".format(int(report["bad"][m].sum()), int(m.sum()))
        print(row)

    print("\nMedian RMS per gesture and channel (which channel responds to which gesture):")
    print("{:22s}".format("Gesture") + "".join("{:>24s}".format(c) for c in config.CHANNEL_NAMES))
    for label, name in enumerate(config.GESTURES):
        med = np.median(report["rms"][y == label], axis=0)
        print("{:22s}".format(name) + "".join("{:24.4f}".format(v) for v in med))
    print("\nMax |DC offset|: {:.4f}".format(np.abs(report["dc"]).max()))


def write_report_csv(path, report, y, subject, rep_uid):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rep_uid", "gesture", "subject"]
            + ["dc_ch{}".format(c + 1) for c in range(config.N_CHANNELS)]
            + ["rms_ch{}".format(c + 1) for c in range(config.N_CHANNELS)]
            + ["max_clip_ratio", "max_abs_z"]
            + list(report["reasons"])
            + ["rejected"]
        )
        for i in range(len(y)):
            writer.writerow(
                [int(rep_uid[i]), config.GESTURES[y[i]], int(subject[i])]
                + ["{:.6g}".format(v) for v in report["dc"][i]]
                + ["{:.6g}".format(v) for v in report["rms"][i]]
                + ["{:.4g}".format(report["clip_ratio"][i].max()),
                   "{:.3f}".format(np.abs(report["z"][i]).max())]
                + [int(v[i]) for v in report["reasons"].values()]
                + [int(report["bad"][i])]
            )


def plot_inspection(X, y, report):
    t = np.arange(X.shape[2]) / config.FS
    n_g = len(config.GESTURES)

    fig, axes = plt.subplots(n_g, 1, sharex=True, figsize=(12, 2.2 * n_g), constrained_layout=True)
    for label, ax in enumerate(axes):
        idx = np.flatnonzero((y == label) & ~report["bad"])[0]
        for c in range(config.N_CHANNELS):
            ax.plot(t, X[idx, c], lw=0.5, label=config.CHANNEL_NAMES[c])
        ax.set_title("{} - repetition {} (raw)".format(config.GESTURES[label], idx), loc="left")
        ax.set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(config.figure("step1_raw_examples.png"), dpi=120)

    fig, axes = plt.subplots(1, config.N_CHANNELS, figsize=(14, 4), sharey=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        ax.boxplot([report["rms"][y == g, c] for g in range(n_g)])
        ax.set_xticks(range(1, n_g + 1))
        ax.set_xticklabels(config.GESTURES, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_title("{} - RMS per repetition".format(config.CHANNEL_NAMES[c]))
    fig.savefig(config.figure("step1_rms_boxplot.png"), dpi=120)

    # 100 ms RMS envelope, median over repetitions: shows whether the gesture is held for the whole record.
    block = 100
    n_blocks = X.shape[2] // block
    env = np.sqrt(
        (X[..., : n_blocks * block].reshape(X.shape[0], X.shape[1], n_blocks, block) ** 2).mean(-1)
    )
    tb = (np.arange(n_blocks) + 0.5) * block / config.FS
    fig, axes = plt.subplots(1, config.N_CHANNELS, figsize=(14, 4), sharey=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        for g in range(n_g):
            ax.plot(tb, np.median(env[(y == g) & ~report["bad"], c], axis=0), label=config.GESTURES[g])
        ax.set_title("{} - median RMS envelope".format(config.CHANNEL_NAMES[c]))
        ax.set_xlabel("Time (s)")
    axes[0].legend(fontsize=8)
    fig.savefig(config.figure("step1_rms_envelope.png"), dpi=120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args(argv)

    X, y, subject, rep_uid = load_raw()
    print("Loaded X {} (repetitions, channels, time) at {} Hz".format(X.shape, config.FS))

    report = quality_report(X, y)
    print_summary(report, y)
    write_report_csv(config.artifact("step1_quality_report.csv"), report, y, subject, rep_uid)
    plot_inspection(X, y, report)

    keep = ~report["bad"]
    X_clean = X[keep] - report["dc"][keep][..., None].astype(np.float32)
    np.savez(
        config.artifact("step1_clean.npz"),
        X=X_clean, y=y[keep], subject=subject[keep], rep_uid=rep_uid[keep],
    )
    print("\nKept {}/{} repetitions -> {}".format(keep.sum(), len(keep), config.artifact("step1_clean.npz")))

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
