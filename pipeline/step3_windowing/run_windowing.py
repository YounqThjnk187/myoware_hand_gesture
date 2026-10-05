"""Step 3: cut filtered repetitions into 200 ms windows with a 100 ms stride."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from pipeline import config
from pipeline.step2_filtering.filters import design_filters, group_delay_ms
from pipeline.step3_windowing.windowing import segment_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args(argv)

    data = np.load(config.artifact("step2_filtered.npz"))
    X = data["X"]
    W, y, subject, rep_uid = segment_dataset(X, data["y"], data["subject"], data["rep_uid"])

    per_rep = W.shape[0] // X.shape[0]
    size_ms = 1000.0 * config.WINDOW_SIZE / config.FS
    step_ms = 1000.0 * config.WINDOW_STEP / config.FS
    gd = group_delay_ms(design_filters())
    print("Windows: {} total, {} per repetition, shape {}".format(W.shape[0], per_rep, W.shape[1:]))
    print("Window {:.0f} ms, stride {:.0f} ms, overlap {:.0f} %".format(
        size_ms, step_ms, 100 * (1 - config.WINDOW_STEP / config.WINDOW_SIZE)))
    print("Decision rate: {:.0f} Hz. Worst-case wait for a new decision: {:.0f} ms".format(1000 / step_ms, step_ms))
    print("Window centre lags the newest sample by {:.0f} ms; filter group delay adds ~{:.1f} ms".format(size_ms / 2, gd))
    for label, name in enumerate(config.GESTURES):
        print("  {:22s} {:6d} windows".format(name, int((y == label).sum())))

    t = np.arange(X.shape[-1]) / config.FS
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.plot(t, X[0, 0], lw=0.5, color="k")
    for k in range(4):
        start = k * config.WINDOW_STEP / config.FS
        ax.axvspan(start, start + size_ms / 1000, alpha=0.15, color="C{}".format(k))
    ax.set_xlim(0, 1.0)
    ax.set_title("First 4 windows of repetition 0, Ch1 (200 ms, 50 % overlap)")
    ax.set_xlabel("Time (s)")
    fig.savefig(config.figure("step3_windows.png"), dpi=120)

    np.savez(config.artifact("step3_windows.npz"), W=W, y=y, subject=subject, rep_uid=rep_uid)
    print("Saved {}".format(config.artifact("step3_windows.npz")))

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
