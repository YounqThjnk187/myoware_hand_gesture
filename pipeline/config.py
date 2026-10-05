"""Shared constants and paths for every pipeline step."""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "kaggle" / "DS1_EMG_SIGNALS_AVERAGE"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIRMWARE_DIR = ROOT / "pipeline" / "step9_deployment" / "firmware" / "emg_realtime"

# Dataset
FS = 1000
N_CHANNELS = 3
# Assumption: inside each MovK.mat the 400 repetitions are ordered subject by subject.
REPS_PER_SUBJECT = 20
GESTURES = ("Fist", "Thumb flexion", "Rest", "Finger extension", "Middle+ring flexion")
CHANNEL_NAMES = ("Ch1 anterior forearm", "Ch2 posterior forearm", "Ch3 anterior radius")

# Step 1 - cleaning
FLAT_BLOCK = 50
FLAT_EPS = 1e-6
CLIP_RATIO_LIMIT = 0.01
RMS_Z_LIMIT = 3.5

# Step 2 - filtering
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
BANDPASS = (20.0, 450.0)
BUTTER_ORDER = 4
TRANSIENT_SAMPLES = 300
BASELINE_CUTOFF_HZ = 1.0
BASELINE_ALPHA = 1.0 - math.exp(-2.0 * math.pi * BASELINE_CUTOFF_HZ / FS)

# Step 3 - windowing
WINDOW_SIZE = 200
WINDOW_STEP = 100

# Step 4 - features
FEATURES = ("MAV", "RMS", "WL", "ZC")
ZC_THRESHOLD = 0.01
FEATURE_NAMES = tuple(
    "{}_ch{}".format(f, c + 1) for f in FEATURES for c in range(N_CHANNELS)
)

# Step 6 - splitting
SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_STATE = 42

# Step 7 / 8
CV_FOLDS = 5
LATENCY_BUDGET_MS = 100.0


def artifact(name):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / name


def figure(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def model_path(name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / "{}.joblib".format(name)
