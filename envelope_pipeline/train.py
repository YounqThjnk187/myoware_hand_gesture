"""Train and compare classifiers on simulated KIT0012 envelopes, then save the model bundle.

Split: 20 % of the trials of every (subject, gesture) are held out for testing (within-session
evaluation, report E5). Calibration (rest baseline, MVC) uses only each subject's training trials.
"""

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from envelope_pipeline import config
from envelope_pipeline.dataset import load_envelope_dataset
from envelope_pipeline.preprocess import Calibration
from envelope_pipeline.recognizer import recording_features, vote


def build_models():
    seed = config.RANDOM_STATE
    return {
        "lda": LinearDiscriminantAnalysis(),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=seed)),
        "rf": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((32,), max_iter=300, early_stopping=True, random_state=seed)),
    }


def split_trials(y, subject, seed):
    strata = subject * len(config.GESTURES) + y
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=config.TEST_FRACTION, stratify=strata, random_state=seed)
    return np.sort(train), np.sort(test)


def fit_calibrations(E, y, subject, train_idx):
    calibs = {}
    for s in np.unique(subject):
        mine = train_idx[subject[train_idx] == s]
        rest = np.concatenate(E[mine[y[mine] == config.REST_LABEL]], axis=-1)
        active = np.concatenate(E[mine[y[mine] != config.REST_LABEL]], axis=-1)
        calibs[int(s)] = Calibration.fit(rest, active)
    return calibs


def windows_for(idx, data, calibs):
    feats = [recording_features(data["E"][i], calibs[int(data["subject"][i])]) for i in idx]
    counts = [len(f) for f in feats]
    F = np.concatenate(feats)
    y = np.repeat(data["y"][idx], counts)
    groups = np.repeat(idx, counts)
    return F, y, groups


def subsample(n_total, n_max, rng):
    return np.arange(n_total) if n_total <= n_max else np.sort(rng.choice(n_total, n_max, replace=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["lda", "knn", "rf", "mlp"], choices=list(build_models()),
                        help="svm is available but slow (probability calibration)")
    parser.add_argument("--deploy", default="lda", help="model saved for prediction (report: LDA is the main model)")
    parser.add_argument("--rebuild", action="store_true", help="re-read csv_data and re-simulate the envelopes")
    args = parser.parse_args(argv)
    if args.deploy not in args.models:
        parser.error("--deploy must be one of --models")

    data = load_envelope_dataset(rebuild=args.rebuild)
    y, subject = data["y"], data["subject"]
    print("Envelope dataset: {} trials, {} subjects, {} samples per trial at {} Hz".format(
        len(y), len(np.unique(subject)), data["E"].shape[-1], config.FS))

    train_idx, test_idx = split_trials(y, subject, config.RANDOM_STATE)
    calibs = fit_calibrations(data["E"], y, subject, train_idx)
    F_tr, y_tr, g_tr = windows_for(train_idx, data, calibs)
    F_te, y_te, g_te = windows_for(test_idx, data, calibs)
    print("Windows: train {}, test {} ({} features: {})".format(
        len(F_tr), len(F_te), F_tr.shape[1], ", ".join(config.FEATURE_NAMES)))

    rng = np.random.RandomState(config.RANDOM_STATE)
    cv_idx = subsample(len(F_tr), config.MAX_CV_SAMPLES, rng)
    fit_idx = subsample(len(F_tr), config.MAX_FIT_SAMPLES, rng)
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    models, report = build_models(), {}
    print("\n{:5s}{:>14s}{:>10s}{:>10s}{:>12s}{:>12s}".format("model", "CV F1", "test acc", "test F1", "voted acc", "ms/window"))
    for name in args.models:
        model = models[name]
        cv_f1 = cross_val_score(model, F_tr[cv_idx], y_tr[cv_idx], groups=g_tr[cv_idx], cv=cv, scoring="f1_macro", n_jobs=-1)
        sel = np.arange(len(F_tr)) if name in ("lda", "rf") else fit_idx
        model.fit(F_tr[sel], y_tr[sel])
        if name == "rf":
            # Thread start-up dominates single-window prediction; real-time inference is one window at a time.
            model.set_params(n_jobs=1)

        proba = model.predict_proba(F_te)
        pred = model.classes_[np.argmax(proba, axis=1)]
        voted, voted_true = [], []
        for i in test_idx:
            m = g_te == i
            d = vote(proba[m], model.classes_)
            voted += [-1 if v is None else v for v in d]
            voted_true += list(y_te[m])
        t0 = time.perf_counter()
        for k in range(200):
            model.predict_proba(F_te[k : k + 1])
        ms = (time.perf_counter() - t0) / 200 * 1000

        report[name] = {
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "test_f1_macro": float(f1_score(y_te, pred, average="macro")),
            "test_voted_accuracy": float(accuracy_score(voted_true, voted)),
            "ms_per_window": ms,
            "confusion_matrix": confusion_matrix(y_te, pred, labels=range(len(config.GESTURES))).tolist(),
        }
        r = report[name]
        print("{:5s}{:>8.3f}±{:.3f}{:10.3f}{:10.3f}{:12.3f}{:12.3f}".format(
            name, r["cv_f1_mean"], r["cv_f1_std"], r["test_accuracy"], r["test_f1_macro"], r["test_voted_accuracy"], ms))

        if name == args.deploy:
            deployed = model

    cm = np.array(report[args.deploy]["confusion_matrix"])
    print("\nConfusion matrix ({}, rows = true, cols = predicted):".format(args.deploy.upper()))
    for label, row in zip(config.GESTURES, cm):
        print("  {:20s}".format(label) + "".join("{:7d}".format(v) for v in row))

    global_calib = Calibration(
        np.median([c.baseline for c in calibs.values()], axis=0),
        np.median([c.mvc for c in calibs.values()], axis=0),
    )
    bundle = {
        "model": deployed,
        "model_name": args.deploy,
        "gestures": config.GESTURES,
        "fs": config.FS,
        "window": config.WINDOW,
        "step": config.STEP,
        "adc_scale": data["scale"],
        "calibrations": {s: c.to_dict() for s, c in calibs.items()},
        "global_calibration": global_calib.to_dict(),
        "test_trials": [(int(y[i]) + 1, int(data["trial"][i])) for i in test_idx],
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    report["_deployed"] = args.deploy
    report["_test_trials_example"] = bundle["test_trials"][:: max(1, len(test_idx) // 20)]
    with open(str(config.REPORT_PATH), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nSaved model -> {}\nSaved report -> {}".format(config.MODEL_PATH, config.REPORT_PATH))


if __name__ == "__main__":
    main()
