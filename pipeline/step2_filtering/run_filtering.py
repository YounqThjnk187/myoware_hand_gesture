"""Step 2: 50 Hz notch + 20-450 Hz Butterworth bandpass on the cleaned repetitions."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pipeline import config
from pipeline.step2_filtering.filters import apply_filters, design_filters, group_delay_ms


def band_power(f, psd, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(psd[m], f[m]))


def mean_psd(X, chunk=100):
    """Welch PSD averaged over repetitions and channels, computed in chunks to limit memory."""
    total = None
    for i in range(0, len(X), chunk):
        f, p = signal.welch(X[i : i + chunk], fs=config.FS, nperseg=512, axis=-1)
        s = p.sum(axis=(0, 1))
        total = s if total is None else total + s
    return f, total / (len(X) * X.shape[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args(argv)

    data = np.load(config.artifact("step1_clean.npz"))
    X = data["X"]

    sos = design_filters()
    _, poles, _ = signal.sos2zpk(sos)
    print("SOS cascade ({} sections): notch + Butterworth order {} bandpass {} Hz".format(
        len(sos), config.BUTTER_ORDER, config.BANDPASS))
    print(np.array2string(sos, precision=6, suppress_small=True))
    print("Max pole radius: {:.5f} (stable if < 1)".format(np.abs(poles).max()))
    print("Median group delay 50-150 Hz: {:.2f} ms".format(group_delay_ms(sos)))

    Y = apply_filters(X, sos)
    print("Filtered X {} (first {} samples dropped as filter transient)".format(Y.shape, config.TRANSIENT_SAMPLES))

    f, psd_raw = mean_psd(X[..., config.TRANSIENT_SAMPLES:])
    _, psd_filt = mean_psd(Y)
    for name, psd in (("raw", psd_raw), ("filtered", psd_filt)):
        total = band_power(f, psd, 0, config.FS / 2)
        print("{:9s} power <20 Hz: {:5.1f} %   48-52 Hz: {:5.2f} %   20-450 Hz: {:5.1f} %".format(
            name,
            100 * band_power(f, psd, 0, 20) / total,
            100 * band_power(f, psd, 48, 52) / total,
            100 * band_power(f, psd, 20, 450) / total,
        ))

    w, h = signal.sosfreqz(sos, worN=8192, fs=config.FS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-10)))
    axes[0].set_ylim(-80, 5)
    axes[0].set_title("Filter magnitude response")
    axes[0].set_ylabel("dB")
    axes[1].semilogy(f, psd_raw, label="raw (DC removed)")
    axes[1].semilogy(f, psd_filt, label="filtered")
    axes[1].set_title("Average PSD (Welch, all repetitions and channels)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(config.figure("step2_filter_response_psd.png"), dpi=120)

    t = np.arange(Y.shape[-1]) / config.FS
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.plot(t, X[0, 0, config.TRANSIENT_SAMPLES:], lw=0.5, label="raw")
    ax.plot(t, Y[0, 0], lw=0.5, label="filtered")
    ax.set_title("Repetition 0, Ch1")
    ax.set_xlabel("Time (s)")
    ax.legend()
    fig.savefig(config.figure("step2_before_after.png"), dpi=120)

    np.savez(config.artifact("step2_filtered.npz"), X=Y, y=data["y"], subject=data["subject"], rep_uid=data["rep_uid"])
    print("Saved {}".format(config.artifact("step2_filtered.npz")))

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
