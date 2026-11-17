"""Steps 2-5: filtering, amplitude calibration, windowing and feature extraction for raw sEMG."""

import numpy as np
from scipy import signal

from emg_pipeline import config


def design_filter():
    """50 Hz notch + Butterworth bandpass as one SOS cascade."""
    b, a = signal.iirnotch(config.NOTCH_HZ, config.NOTCH_Q, fs=config.FS)
    notch = signal.tf2sos(b, a)
    band = signal.butter(config.BUTTER_ORDER, config.BANDPASS_HZ, "bandpass", fs=config.FS, output="sos")
    return np.vstack([notch, band])


SOS = design_filter()


def filter_emg(x):
    """Causal filtering of (..., channels, time); the first TRANSIENT samples are dropped."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x[..., : config.TRANSIENT].mean(axis=-1, keepdims=True)
    return signal.sosfilt(SOS, x, axis=-1)[..., config.TRANSIENT :]


def sliding_windows(x, size=config.WINDOW, step=config.STEP):
    """(channels, time) -> (n_windows, channels, size)."""
    if x.shape[-1] < size:
        raise ValueError("recording shorter than one window ({} samples after filtering)".format(size))
    return np.lib.stride_tricks.sliding_window_view(x, size, axis=-1)[:, ::step].transpose(1, 0, 2)


def window_rms(x):
    """Per-window RMS (n_windows, channels) used for calibration."""
    return np.sqrt(np.mean(sliding_windows(x) ** 2, axis=-1))


class Calibration:
    """Per-channel amplitude reference so that features are comparable across people and sessions."""

    def __init__(self, scale):
        self.scale = np.asarray(scale, dtype=np.float64)

    @classmethod
    def fit(cls, filtered_recordings):
        """filtered_recordings: list of (channels, time) arrays covering rest and contractions."""
        rms = np.concatenate([window_rms(x) for x in filtered_recordings])
        return cls(np.percentile(rms, config.CALIB_PERCENTILE, axis=0))

    def apply(self, x):
        return x / self.scale[:, None]

    def to_list(self):
        return self.scale.tolist()


def _ar_coefficients(w, order):
    """Yule-Walker AR coefficients, (n, ch, size) -> (n, ch, order)."""
    w = w - w.mean(axis=-1, keepdims=True)
    n = w.shape[-1]
    r = np.stack([np.sum(w[..., : n - k] * w[..., k:], axis=-1) for k in range(order + 1)], axis=-1) / n
    r[..., 0] += 1e-12
    idx = np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])
    return np.linalg.solve(r[..., idx], r[..., 1 : order + 1, None])[..., 0]


def extract_features(w):
    """(n, channels, size) calibrated windows -> (n, channels * len(FEATURES))."""
    eps = 1e-8
    d = np.diff(w, axis=-1)
    mav = np.mean(np.abs(w), axis=-1)
    rms = np.sqrt(np.mean(w ** 2, axis=-1))
    wl = np.sum(np.abs(d), axis=-1)
    zc = np.sum((w[..., :-1] * w[..., 1:] < 0) & (np.abs(d) >= config.ZC_THRESHOLD), axis=-1)
    ssc = np.sum(d[..., :-1] * -d[..., 1:] >= config.SSC_THRESHOLD, axis=-1)

    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[-1]), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(w.shape[-1], 1.0 / config.FS)
    total = spec.sum(axis=-1) + eps
    mnf = (spec * freqs).sum(axis=-1) / total
    cum = np.cumsum(spec, axis=-1)
    mdf = freqs[np.argmax(cum >= 0.5 * cum[..., -1:], axis=-1)]

    ar = _ar_coefficients(w, config.AR_ORDER)
    feats = [np.log(mav + eps), np.log(rms + eps), np.log(wl + eps), zc, ssc, mnf / 100.0, mdf / 100.0]
    f = np.concatenate([np.stack(feats, axis=-1), ar], axis=-1)
    return f.reshape(len(w), -1)


def recording_features(raw, calib):
    """raw (channels, time) at FS -> (n_windows, n_features)."""
    return extract_features(sliding_windows(calib.apply(filter_emg(raw))))
