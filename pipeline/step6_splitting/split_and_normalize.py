"""Step 6: stratified 70/15/15 split by repetition (or subject), then fit Min-Max on train only."""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from pipeline import config
from pipeline.step5_normalization.normalizer import MinMaxNormalizer


def split_groups(y, rep_uid, subject, by, ratios, seed):
    """Return (train, val, test) boolean masks over windows. Whole groups go to one split."""
    if by == "repetition":
        groups = rep_uid
        uid, first = np.unique(rep_uid, return_index=True)
        labels = y[first]
        train_g, rest_g, _, rest_y = train_test_split(
            uid, labels, train_size=ratios[0], stratify=labels, random_state=seed
        )
        val_g, test_g = train_test_split(
            rest_g, train_size=ratios[1] / (ratios[1] + ratios[2]), stratify=rest_y, random_state=seed
        )
    else:
        # Every subject performs every gesture, so a subject split stays class-balanced.
        groups = subject
        perm = np.random.RandomState(seed).permutation(np.unique(subject))
        n_train = int(round(ratios[0] * len(perm)))
        n_val = int(round(ratios[1] * len(perm)))
        train_g, val_g, test_g = perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]
    return np.isin(groups, train_g), np.isin(groups, val_g), np.isin(groups, test_g)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by", choices=("repetition", "subject"), default="repetition",
        help="repetition = user-dependent evaluation; subject = unseen-user evaluation",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args(argv)

    data = np.load(config.artifact("step4_features.npz"))
    F, y, subject, rep_uid = data["F"], data["y"], data["subject"], data["rep_uid"]

    masks = split_groups(y, rep_uid, subject, args.by, config.SPLIT_RATIOS, args.seed)
    names = ("train", "val", "test")
    for i in range(3):
        for j in range(i + 1, 3):
            shared = np.intersect1d(rep_uid[masks[i]], rep_uid[masks[j]])
            assert shared.size == 0, "repetitions shared between {} and {}".format(names[i], names[j])

    normalizer = MinMaxNormalizer(feature_names=config.FEATURE_NAMES).fit(F[masks[0]])
    normalizer.save(config.artifact("normalizer_profile.json"))

    print("Split by {} (seed {}):".format(args.by, args.seed))
    print("{:22s}".format("Gesture") + "".join("{:>8s}".format(n) for n in names))
    for label, gname in enumerate(config.GESTURES):
        print("{:22s}".format(gname) + "".join("{:8d}".format(int((y[m] == label).sum())) for m in masks))
    print("{:22s}".format("Total windows") + "".join("{:8d}".format(int(m.sum())) for m in masks))
    print("{:22s}".format("Share") + "".join("{:7.1f}%".format(100.0 * m.sum() / len(y)) for m in masks))

    out = {"split_by": args.by, "feature_names": np.array(config.FEATURE_NAMES)}
    for name, m in zip(names, masks):
        out["X_" + name] = normalizer.transform(F[m])
        out["y_" + name] = y[m]
        out["rep_" + name] = rep_uid[m]
        out["subject_" + name] = subject[m]
    np.savez(config.artifact("step6_dataset.npz"), **out)
    print("Saved {} and {}".format(config.artifact("step6_dataset.npz"), config.artifact("normalizer_profile.json")))


if __name__ == "__main__":
    main()
