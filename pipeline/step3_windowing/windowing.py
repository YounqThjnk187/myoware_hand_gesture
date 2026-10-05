"""Sliding-window segmentation."""

import numpy as np

from pipeline import config


def sliding_windows(X, size=config.WINDOW_SIZE, step=config.WINDOW_STEP):
    """(N, C, T) -> (N, W, C, size) with W = (T - size) // step + 1."""
    if X.shape[-1] < size:
        raise ValueError("signal shorter than one window")
    view = np.lib.stride_tricks.sliding_window_view(X, size, axis=-1)[..., ::step, :]
    return np.ascontiguousarray(view.transpose(0, 2, 1, 3))


def segment_dataset(X, y, subject, rep_uid, size=config.WINDOW_SIZE, step=config.WINDOW_STEP):
    """Flatten windows of every repetition; each window inherits the labels of its repetition."""
    W = sliding_windows(X, size, step)
    n_reps, n_win = W.shape[:2]
    return (
        W.reshape(n_reps * n_win, X.shape[1], size),
        np.repeat(y, n_win),
        np.repeat(subject, n_win),
        np.repeat(rep_uid, n_win),
    )
