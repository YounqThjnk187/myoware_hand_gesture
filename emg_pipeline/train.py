"""Step 6: train and compare classifiers, evaluate on held-out data, save the model bundle.

--split trial    : 20 % of the repetitions of every (subject, gesture) are held out (user-dependent)
--split subject  : TEST_SUBJECTS people are held out entirely and use the global calibration (new user)
"""

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from emg_pipeline import config  # noqa: E402
from emg_pipeline.dataset import load_dataset  # noqa: E402
from emg_pipeline.processing import Calibration, extract_features, filter_emg, sliding_windows  # noqa: E402
from emg_pipeline.recognizer import vote  # noqa: E402


def build_models():
    seed = config.RANDOM_STATE
    return {
        "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "rf": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=seed)),
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=seed)),
    }


def split(data, mode, seed):
    idx = np.arange(len(data["y"]))
    if mode == "trial":
        strata = data["subject"] * 10 + data["y"]
        train, test = train_test_split(idx, test_size=config.TEST_FRACTION, stratify=strata, random_state=seed)
    else:
        subjects = np.unique(data["subject"])
        held = np.random.RandomState(seed).choice(subjects, config.TEST_SUBJECTS, replace=False)
        test_mask = np.isin(data["subject"], held)
        train, test = idx[~test_mask], idx[test_mask]
    return np.sort(train), np.sort(test)


def features_for(idx, filtered, subject, calibs, global_calib):
    feats = []
    for i in idx:
        calib = calibs.get(int(subject[i]), global_calib)
        feats.append(extract_features(sliding_windows(calib.apply(filtered[i]))))
    counts = [len(f) for f in feats]
    return np.concatenate(feats), np.repeat(idx, counts)


def subsample(n, n_max, rng):
    return np.arange(n) if n <= n_max else np.sort(rng.choice(n, n_max, replace=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("trial", "subject"), default="trial")
    parser.add_argument("--models", nargs="+", default=["lda", "knn", "rf", "mlp"], choices=list(build_models()),
                        help="svm is available but slow (probability calibration)")
    parser.add_argument("--deploy", default="best", help="model saved for prediction; 'best' = highest CV F1")
    parser.add_argument("--rebuild", action="store_true", help="reload the .mat files instead of the cache")
    args = parser.parse_args(argv)
    if args.deploy != "best" and args.deploy not in args.models:
        parser.error("--deploy must be 'best' or one of --models")

    data = load_dataset(rebuild=args.rebuild)
    y, subject = data["y"], data["subject"]
    print("Dataset: {} trials x {} channels x {} samples at {} Hz, {} subjects".format(
        *data["X"].shape, config.FS, len(np.unique(subject))))

    print("Filtering (notch {:g} Hz + bandpass {}-{} Hz)...".format(config.NOTCH_HZ, *config.BANDPASS_HZ))
    filtered = filter_emg(data["X"])
    train_idx, test_idx = split(data, args.split, config.RANDOM_STATE)

    calibs = {}
    for s in np.unique(subject[train_idx]):
        calibs[int(s)] = Calibration.fit(filtered[train_idx[subject[train_idx] == s]])
    global_calib = Calibration(np.median([c.scale for c in calibs.values()], axis=0))

    print("Extracting features...")
    F_tr, g_tr = features_for(train_idx, filtered, subject, calibs, global_calib)
    F_te, g_te = features_for(test_idx, filtered, subject, calibs, global_calib)
    y_tr, y_te = y[g_tr], y[g_te]
    print("Split by {}: train {} trials / {} windows, test {} trials / {} windows, {} features".format(
        args.split, len(train_idx), len(F_tr), len(test_idx), len(F_te), F_tr.shape[1]))

    rng = np.random.RandomState(config.RANDOM_STATE)
    cv_idx = subsample(len(F_tr), config.MAX_CV_SAMPLES, rng)
    fit_idx = subsample(len(F_tr), config.MAX_FIT_SAMPLES, rng)
    cv_groups = subject[g_tr] if args.split == "subject" else g_tr
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    models, report, fitted = build_models(), {"split": args.split}, {}
    print("\n{:5s}{:>14s}{:>10s}{:>10s}{:>12s}{:>12s}".format("model", "CV F1", "test acc", "test F1", "voted acc", "ms/window"))
    for name in args.models:
        model = models[name]
        cv_f1 = cross_val_score(model, F_tr[cv_idx], y_tr[cv_idx], groups=cv_groups[cv_idx], cv=cv,
                                scoring="f1_macro", n_jobs=-1)
        sel = np.arange(len(F_tr)) if name in ("lda", "rf") else fit_idx
        model.fit(F_tr[sel], y_tr[sel])
        if name == "rf":
            # Thread start-up dominates single-window prediction; real-time inference is one window at a time.
            model.set_params(n_jobs=1)

        proba = model.predict_proba(F_te)
        pred = model.classes_[np.argmax(proba, axis=1)]
        voted = []
        for i in test_idx:
            voted += [-1 if d is None else d for d in vote(proba[g_te == i], model.classes_)]
        t0 = time.perf_counter()
        for k in range(200):
            model.predict_proba(F_te[k : k + 1])
        ms = (time.perf_counter() - t0) / 200 * 1000

        cm = confusion_matrix(y_te, pred, labels=range(len(config.GESTURES)))
        report[name] = {
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "test_f1_macro": float(f1_score(y_te, pred, average="macro")),
            "test_voted_accuracy": float(np.mean(np.array(voted) == y_te)),
            "ms_per_window": ms,
            "confusion_matrix": cm.tolist(),
        }
        r = report[name]
        print("{:5s}{:>8.3f}±{:.3f}{:10.3f}{:10.3f}{:12.3f}{:12.3f}".format(
            name, r["cv_f1_mean"], r["cv_f1_std"], r["test_accuracy"], r["test_f1_macro"], r["test_voted_accuracy"], ms))
        fitted[name] = model

    if args.deploy == "best":
        args.deploy = max(args.models, key=lambda m: report[m]["cv_f1_mean"])
    deployed = fitted[args.deploy]
    deployed_cm = np.array(report[args.deploy]["confusion_matrix"])
    print("\nDeployed model: {} (highest CV F1 unless --deploy is given)".format(args.deploy.upper()))
    print("\nConfusion matrix ({}, rows = true, cols = predicted):".format(args.deploy.upper()))
    for label, row in zip(config.GESTURES, deployed_cm):
        print("  {:20s}".format(label) + "".join("{:7d}".format(v) for v in row))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay(deployed_cm / deployed_cm.sum(axis=1, keepdims=True),
                           display_labels=[g.split()[0] for g in config.GESTURES]).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("{} - split by {}".format(args.deploy.upper(), args.split))
    fig.savefig(config.CONFUSION_FIG, dpi=120)

    bundle = {
        "model": deployed,
        "model_name": args.deploy,
        "split": args.split,
        "gestures": config.GESTURES,
        "feature_names": config.FEATURE_NAMES,
        "calibrations": {s: c.to_list() for s, c in calibs.items()},
        "global_calibration": global_calib.to_list(),
        "test_trials": sorted({(int(data["mov"][i]), int(data["trial"][i])) for i in test_idx}),
    }
    joblib.dump(bundle, config.MODEL_PATH)
    report["_deployed"] = args.deploy
    report["_test_trials_example"] = bundle["test_trials"][:: max(1, len(test_idx) // 20)]
    with open(str(config.REPORT_PATH), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nSaved model -> {}\nSaved report -> {}\nSaved figure -> {}".format(
        config.MODEL_PATH, config.REPORT_PATH, config.CONFUSION_FIG))


if __name__ == "__main__":
    main()
