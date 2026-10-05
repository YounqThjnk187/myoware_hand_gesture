"""Step 4: extract the 12-D time-domain feature vector from every window."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from pipeline import config
from pipeline.step4_features.features import extract_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zc-threshold", type=float, default=config.ZC_THRESHOLD)
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args(argv)

    data = np.load(config.artifact("step3_windows.npz"))
    y = data["y"]
    F = extract_features(data["W"], threshold=args.zc_threshold)
    print("Feature matrix {} ({})".format(F.shape, ", ".join(config.FEATURE_NAMES)))

    print("\nPer-gesture median of each feature:")
    print("{:22s}".format("Gesture") + "".join("{:>10s}".format(n) for n in config.FEATURE_NAMES))
    for label, name in enumerate(config.GESTURES):
        med = np.median(F[y == label], axis=0)
        print("{:22s}".format(name) + "".join("{:10.4g}".format(v) for v in med))

    n_f = F.shape[1]
    fig, axes = plt.subplots(len(config.FEATURES), config.N_CHANNELS, figsize=(14, 12), constrained_layout=True)
    for k in range(n_f):
        ax = axes[k // config.N_CHANNELS, k % config.N_CHANNELS]
        ax.boxplot([F[y == g, k] for g in range(len(config.GESTURES))], showfliers=False)
        ax.set_title(config.FEATURE_NAMES[k], fontsize=9)
        ax.set_xticks(range(1, len(config.GESTURES) + 1))
        ax.set_xticklabels([g.split()[0] for g in config.GESTURES], fontsize=7)
    fig.savefig(config.figure("step4_feature_boxplots.png"), dpi=120)

    np.savez(
        config.artifact("step4_features.npz"),
        F=F, y=y, subject=data["subject"], rep_uid=data["rep_uid"],
        feature_names=np.array(config.FEATURE_NAMES), zc_threshold=args.zc_threshold,
    )
    print("\nSaved {}".format(config.artifact("step4_features.npz")))

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
