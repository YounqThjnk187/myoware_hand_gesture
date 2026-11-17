"""Stages 5-6: sliding windows and the 12-D amplitude feature vector."""

import numpy as np

from envelope_pipeline import config


def sliding_windows(x, size=config.WINDOW, step=config.STEP):
    """(channels, time) -> (n_windows, channels, size)."""
    v = np.lib.stride_tricks.sliding_window_view(x, size, axis=-1)[:, ::step]
    return v.transpose(1, 0, 2)


def extract_features(w):
    """(n, channels, size) -> (n, 12): mean, max, std, slope (last - first) per channel."""
    f = np.stack([w.mean(-1), w.max(-1), w.std(-1), w[..., -1] - w[..., 0]], axis=-1)
    return f.reshape(len(w), -1)
