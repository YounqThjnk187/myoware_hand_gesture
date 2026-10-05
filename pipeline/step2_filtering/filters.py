"""Filter design and causal filtering shared by training, latency tests and C export."""

import numpy as np
from scipy import signal

from pipeline import config


def design_filters(
    fs=config.FS,
    notch_freq=config.NOTCH_FREQ,
    notch_q=config.NOTCH_Q,
    band=config.BANDPASS,
    order=config.BUTTER_ORDER,
):
    """Return one SOS cascade: 50 Hz notch followed by a Butterworth bandpass."""
    b, a = signal.iirnotch(notch_freq, notch_q, fs=fs)
    notch_sos = signal.tf2sos(b, a)
    band_sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    return np.vstack([notch_sos, band_sos])


def apply_filters(X, sos, discard=config.TRANSIENT_SAMPLES, chunk=200):
    """Causal (MCU-identical) filtering along the last axis; drops the start-up transient."""
    out = np.empty(X.shape[:-1] + (X.shape[-1] - discard,), dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i : i + chunk] = signal.sosfilt(sos, X[i : i + chunk], axis=-1)[..., discard:]
    return out


class StreamingFilter:
    """Keeps filter state between chunks, like the firmware does sample by sample."""

    def __init__(self, sos, n_channels=config.N_CHANNELS):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], n_channels, 2))

    def process(self, chunk):
        """chunk: (channels, n_samples)."""
        y, self.zi = signal.sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        return y


def group_delay_ms(sos, f_lo=50.0, f_hi=150.0, fs=config.FS):
    """Median group delay in the main EMG band."""
    w, h = signal.sosfreqz(sos, worN=8192, fs=fs)
    phase = np.unwrap(np.angle(h))
    gd_samples = -np.gradient(phase, 2 * np.pi * w / fs)
    band = (w >= f_lo) & (w <= f_hi)
    return float(np.median(gd_samples[band]) * 1000.0 / fs)
