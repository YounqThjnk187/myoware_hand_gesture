"""Min-Max normalizer whose parameters are fitted on the training split only and saved as a profile."""

import json

import numpy as np


class MinMaxNormalizer:
    def __init__(self, feature_min=None, feature_max=None, feature_names=None):
        self.feature_min = None if feature_min is None else np.asarray(feature_min, dtype=np.float64)
        self.feature_max = None if feature_max is None else np.asarray(feature_max, dtype=np.float64)
        self.feature_names = list(feature_names) if feature_names is not None else None

    def fit(self, X):
        self.feature_min = X.min(axis=0).astype(np.float64)
        self.feature_max = X.max(axis=0).astype(np.float64)
        return self

    def transform(self, X, clip=True):
        span = self.feature_max - self.feature_min
        span = np.where(span > 0, span, 1.0)
        Z = (X - self.feature_min) / span
        # Unseen data can fall outside the training range; clipping keeps the model inputs in [0, 1].
        return (np.clip(Z, 0.0, 1.0) if clip else Z).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        profile = {
            "method": "min-max",
            "feature_names": self.feature_names,
            "min": self.feature_min.tolist(),
            "max": self.feature_max.tolist(),
        }
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(str(path), encoding="utf-8") as f:
            profile = json.load(f)
        return cls(profile["min"], profile["max"], profile.get("feature_names"))
