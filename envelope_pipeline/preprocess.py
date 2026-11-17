"""Stage 4: 3-point median filter, rest-baseline subtraction and per-channel MVC normalization."""

import json

import numpy as np

from envelope_pipeline import config


class Calibration:
    def __init__(self, baseline, mvc):
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.mvc = np.maximum(np.asarray(mvc, dtype=np.float64), self.baseline + 1.0)

    @classmethod
    def fit(cls, rest, active):
        """rest, active: (channels, samples) ADC counts from the rest and contraction recordings."""
        return cls(np.median(rest, axis=1), np.percentile(active, config.MVC_PERCENTILE, axis=1))

    def normalize(self, x):
        """x: (..., channels, time) -> (x - baseline) / (MVC - baseline), clipped to [0, NORM_CLIP]."""
        b, m = self.baseline[:, None], self.mvc[:, None]
        return np.clip((x - b) / (m - b), 0.0, config.NORM_CLIP)

    def to_dict(self):
        return {"baseline": self.baseline.tolist(), "mvc": self.mvc.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["baseline"], d["mvc"])

    def save(self, path):
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(str(path), encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def median3(x):
    """Causal 3-point median along time, so offline output equals the streaming recognizer."""
    x = np.asarray(x, dtype=np.float64)
    p = np.concatenate([x[..., :1], x[..., :1], x], axis=-1)
    return np.median(np.stack([p[..., :-2], p[..., 1:-1], p[..., 2:]]), axis=0)


def preprocess(x, calib):
    return calib.normalize(median3(x))
