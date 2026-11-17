"""Step 7: turn per-window probabilities into stable decisions and one gesture per recording."""

from collections import Counter, deque

import numpy as np

from emg_pipeline import config
from emg_pipeline.processing import recording_features


def vote(proba, classes, n=config.VOTE_N, min_proba=config.MIN_PROBA):
    """Majority of the last n confident window labels; keep the previous decision when there is none."""
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


def recognize(model, raw, calib):
    """raw (3, time) at 1 kHz -> per-window probabilities, decisions and the overall gesture."""
    proba = model.predict_proba(recording_features(raw, calib))
    decisions = vote(proba, model.classes_)
    counted = Counter(d for d in decisions if d is not None)
    gesture, n = counted.most_common(1)[0] if counted else (None, 0)
    return {"proba": proba, "decisions": decisions, "gesture": gesture, "share": n / len(decisions)}
