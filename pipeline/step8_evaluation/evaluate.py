"""Step 8: test-set F1 / confusion matrices and streaming latency (filter + features + inference)."""

import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from pipeline import config
from pipeline.step2_filtering.filters import StreamingFilter, design_filters, group_delay_ms
from pipeline.step4_features.features import extract_features
from pipeline.step5_normalization.normalizer import MinMaxNormalizer

STAGES = ("filter", "buffer", "features", "normalize", "inference")


def measure_latency(model, normalizer, reps, sos, zc_threshold, warmup=5):
    """Replay raw repetitions in 100-sample hops, exactly as the device would see them."""
    times = {s: [] for s in STAGES}
    n = 0
    for x in reps:
        filt = StreamingFilter(sos, x.shape[0])
        buf = np.zeros((x.shape[0], config.WINDOW_SIZE))
        for start in range(0, x.shape[1] - config.WINDOW_STEP + 1, config.WINDOW_STEP):
            t0 = time.perf_counter()
            yf = filt.process(x[:, start : start + config.WINDOW_STEP])
            t1 = time.perf_counter()
            buf = np.concatenate([buf[:, config.WINDOW_STEP :], yf], axis=1)
            t2 = time.perf_counter()
            feats = extract_features(buf[None].astype(np.float32), threshold=zc_threshold)
            t3 = time.perf_counter()
            z = normalizer.transform(feats)
            t4 = time.perf_counter()
            model.predict(z)
            t5 = time.perf_counter()
            n += 1
            if n <= warmup:
                continue
            for s, dt in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                times[s].append(dt * 1000.0)
    times = {s: np.asarray(v) for s, v in times.items()}
    times["total"] = sum(times[s] for s in STAGES)
    return {s: {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())} for s, v in times.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, help="default: every model in artifacts/models")
    parser.add_argument("--latency-reps", type=int, default=20, help="test repetitions replayed for timing")
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args(argv)

    d = np.load(config.artifact("step6_dataset.npz"))
    X_test, y_test = d["X_test"], d["y_test"]
    normalizer = MinMaxNormalizer.load(config.artifact("normalizer_profile.json"))
    zc_threshold = float(np.load(config.artifact("step4_features.npz"))["zc_threshold"])
    names = args.models or sorted(p.stem for p in config.MODELS_DIR.glob("*.joblib"))
    if not names:
        raise SystemExit("No models found; run Step 7 first.")

    # Raw (cleaned, unfiltered) test repetitions for the streaming latency test.
    clean = np.load(config.artifact("step1_clean.npz"))
    test_reps = np.isin(clean["rep_uid"], np.unique(d["rep_test"]))
    reps = clean["X"][test_reps][: args.latency_reps].astype(np.float64)
    sos = design_filters()

    results = {"split_by": str(d["split_by"]), "filter_group_delay_ms": group_delay_ms(sos)}
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5), squeeze=False, constrained_layout=True)
    for ax, name in zip(axes[0], names):
        model = joblib.load(config.model_path(name))
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=range(len(config.GESTURES)))
        print("\n=== {} ===  test accuracy {:.4f}  macro-F1 {:.4f}".format(name.upper(), acc, f1))
        print(classification_report(y_test, y_pred, target_names=config.GESTURES, digits=4))

        ConfusionMatrixDisplay(
            cm / cm.sum(axis=1, keepdims=True), display_labels=[g.split()[0] for g in config.GESTURES]
        ).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title("{}  F1={:.3f}".format(name.upper(), f1))

        lat = measure_latency(model, normalizer, reps, sos, zc_threshold)
        print("Latency per decision (ms, Python on this PC):")
        for s in STAGES + ("total",):
            print("  {:10s} mean {:8.3f}  p95 {:8.3f}  max {:8.3f}".format(s, lat[s]["mean"], lat[s]["p95"], lat[s]["max"]))
        ok = lat["total"]["max"] <= config.LATENCY_BUDGET_MS
        print("  budget {:.0f} ms -> {}".format(config.LATENCY_BUDGET_MS, "PASS" if ok else "FAIL"))

        results[name] = {
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1),
            "confusion_matrix": cm.tolist(),
            "latency_ms": lat,
            "latency_within_budget": bool(ok),
        }
    fig.savefig(config.figure("step8_confusion_matrices.png"), dpi=120)

    with open(str(config.artifact("step8_results.json")), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nAlgorithmic latency: new decision every {} ms, window centre {} ms behind, filter group delay {:.1f} ms".format(
        1000 * config.WINDOW_STEP // config.FS, 1000 * config.WINDOW_SIZE // (2 * config.FS), results["filter_group_delay_ms"]))
    print("Saved {}".format(config.artifact("step8_results.json")))

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
