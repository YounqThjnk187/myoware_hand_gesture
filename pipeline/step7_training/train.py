"""Step 7: tune LDA, KNN, SVM and RF with 5-fold grouped CV, check learning curves, keep the best on validation."""

import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from pipeline import config

MODELS = {
    "lda": (
        LinearDiscriminantAnalysis(solver="lsqr"),
        {"shrinkage": [None, "auto"]},
    ),
    "knn": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    ),
    "svm": (
        SVC(kernel="rbf"),
        {"C": [1, 10, 100], "gamma": ["scale", 0.1, 1.0]},
    ),
    "rf": (
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        {"n_estimators": [50, 100], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 5]},
    ),
}


def subsample(X, y, g, max_n, seed):
    if not max_n or len(X) <= max_n:
        return X, y, g
    idx = np.random.RandomState(seed).choice(len(X), max_n, replace=False)
    return X[idx], y[idx], g[idx]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument(
        "--max-tune-samples", type=int, default=20000,
        help="windows used for grid search / learning curves (SVM and KNN scale badly); 0 = all",
    )
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args(argv)
    seed = config.RANDOM_STATE

    d = np.load(config.artifact("step6_dataset.npz"))
    X_train, y_train, g_train = d["X_train"], d["y_train"], d["rep_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    Xs, ys, gs = subsample(X_train, y_train, g_train, args.max_tune_samples, seed)
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    print("Train {} windows (tuning on {}), validation {}".format(len(X_train), len(Xs), len(X_val)))

    summary = {}
    fig, axes = plt.subplots(1, len(args.models), figsize=(4.5 * len(args.models), 4), squeeze=False, constrained_layout=True)
    for ax, name in zip(axes[0], args.models):
        estimator, grid = MODELS[name]
        print("\n=== {} ===".format(name.upper()))
        search = GridSearchCV(
            estimator, grid, scoring="f1_macro", cv=cv, n_jobs=args.jobs, return_train_score=True
        )
        search.fit(Xs, ys, groups=gs)
        best = search.best_index_
        cv_train = search.cv_results_["mean_train_score"][best]
        cv_val = search.cv_results_["mean_test_score"][best]
        cv_std = search.cv_results_["std_test_score"][best]
        print("Best params: {}".format(search.best_params_))
        print("5-fold CV macro-F1: {:.4f} +/- {:.4f} (train folds {:.4f}, gap {:.4f})".format(
            cv_val, cv_std, cv_train, cv_train - cv_val))

        model = clone(estimator).set_params(**search.best_params_)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        y_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, average="macro")
        print("Refit on full train in {:.1f} s -> validation accuracy {:.4f}, macro-F1 {:.4f}".format(
            fit_s, val_acc, val_f1))
        joblib.dump(model, config.model_path(name))

        sizes, tr_scores, cv_scores = learning_curve(
            clone(model), Xs, ys, groups=gs, cv=cv, scoring="f1_macro",
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=args.jobs,
        )
        for scores, label in ((tr_scores, "train"), (cv_scores, "CV")):
            mean, std = scores.mean(axis=1), scores.std(axis=1)
            ax.plot(sizes, mean, "o-", label=label)
            ax.fill_between(sizes, mean - std, mean + std, alpha=0.2)
        ax.set_title("{} learning curve".format(name.upper()))
        ax.set_xlabel("Training windows")
        ax.set_ylabel("Macro-F1")
        ax.grid(alpha=0.3)
        ax.legend()

        summary[name] = {
            "best_params": search.best_params_,
            "cv_f1_mean": float(cv_val),
            "cv_f1_std": float(cv_std),
            "cv_train_f1": float(cv_train),
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "fit_seconds": fit_s,
        }
    fig.savefig(config.figure("step7_learning_curves.png"), dpi=120)

    best_name = max(summary, key=lambda k: summary[k]["val_f1"])
    summary["_best_model"] = best_name
    with open(str(config.artifact("step7_summary.json")), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n{:6s}{:>10s}{:>10s}{:>10s}{:>10s}".format("model", "CV F1", "gap", "val acc", "val F1"))
    for name in args.models:
        s = summary[name]
        print("{:6s}{:10.4f}{:10.4f}{:10.4f}{:10.4f}".format(
            name, s["cv_f1_mean"], s["cv_train_f1"] - s["cv_f1_mean"], s["val_accuracy"], s["val_f1"]))
    print("Best on validation: {}. Summary -> {}".format(best_name, config.artifact("step7_summary.json")))

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
