"""Stages 7-8 shared by training and prediction: features per recording and voted decisions."""

from collections import Counter, deque

import numpy as np

from envelope_pipeline import config
from envelope_pipeline.features import extract_features, sliding_windows
from envelope_pipeline.preprocess import preprocess


def recording_features(adc, calib):
    """adc (3, time) ADC counts at FS -> (n_windows, 12)."""
    if adc.shape[-1] < config.WINDOW:
        raise ValueError("recording shorter than one window ({} samples)".format(config.WINDOW))
    return extract_features(sliding_windows(preprocess(adc, calib)))


def vote(proba, classes, n=config.VOTE_N, min_proba=config.MIN_PROBA):
    """Per-window stable decisions: majority of the last n confident labels, else keep the previous one."""
    recent = deque(maxlen=n)
    stable, out = None, []
    for p in proba:
        k = int(np.argmax(p))
        recent.append(int(classes[k]) if p[k] >= min_proba else None)
        label, count = Counter(recent).most_common(1)[0]
        if label is not None and count > n // 2:
            stable = label
        out.append(stable)
    return out


def recognize(model, adc, calib):
    """Return per-window raw labels, probabilities, stable decisions and the overall gesture."""
    proba = model.predict_proba(recording_features(adc, calib))
    classes = model.classes_
    decisions = vote(proba, classes)
    counted = Counter(d for d in decisions if d is not None)
    if counted:
        gesture, n = counted.most_common(1)[0]
        share = n / len(decisions)
    else:
        gesture, share = None, 0.0
    return {
        "raw": classes[np.argmax(proba, axis=1)],
        "proba": proba,
        "decisions": decisions,
        "gesture": gesture,
        "share": share,
    }
