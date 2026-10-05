"""Time-domain features: MAV, RMS, WL, ZC per channel (12 features for 3 channels)."""

import numpy as np

from pipeline import config


def mav(W):
    return np.mean(np.abs(W), axis=-1)


def rms(W):
    return np.sqrt(np.mean(np.square(W), axis=-1))


def wl(W):
    return np.sum(np.abs(np.diff(W, axis=-1)), axis=-1)


def zc(W, threshold=config.ZC_THRESHOLD):
    a, b = W[..., :-1], W[..., 1:]
    return np.sum((a * b < 0) & (np.abs(a - b) >= threshold), axis=-1)


def extract_features(W, threshold=config.ZC_THRESHOLD, chunk=10000):
    """(M, C, N) -> (M, 4*C) in config.FEATURE_NAMES order: [MAV_ch*, RMS_ch*, WL_ch*, ZC_ch*]."""
    out = np.empty((W.shape[0], len(config.FEATURES) * W.shape[1]), dtype=np.float32)
    for i in range(0, len(W), chunk):
        w = W[i : i + chunk]
        out[i : i + chunk] = np.concatenate([mav(w), rms(w), wl(w), zc(w, threshold)], axis=-1)
    return out
