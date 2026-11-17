"""Shared paths and parameters of the raw-sEMG gesture pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "kaggle" / "DS1_EMG_SIGNALS_AVERAGE"
TDMS_CSV_DIR = ROOT / "csv_data" / "tdms"
TRIAL_INDEX = TDMS_CSV_DIR / "trial_index.csv"
OUT_DIR = ROOT / "artifacts" / "emg_pipeline"
DATASET_CACHE = OUT_DIR / "dataset_raw.npz"
MODEL_PATH = OUT_DIR / "gesture_model.joblib"
REPORT_PATH = OUT_DIR / "train_report.json"
CONFUSION_FIG = OUT_DIR / "confusion_matrix.png"

# Dataset
FS = 1000
N_CHANNELS = 3
N_SUBJECTS = 20
GESTURES = ("Fist", "Thumb flexion", "Rest", "Finger extension", "Middle+ring flexion")
CHANNEL_NAMES = ("Ch1 anterior forearm", "Ch2 posterior forearm", "Ch3 anterior radius")
# Fallback when trial_index.csv is missing: MovK trials are ordered subject by subject.
TRIALS_PER_SUBJECT = 20

# Filtering
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_HZ = (20.0, 450.0)
BUTTER_ORDER = 4
TRANSIENT = 300

# Calibration: per-channel amplitude reference = this percentile of window RMS
CALIB_PERCENTILE = 95.0

# Windowing: 200 ms window, 50 ms step
WINDOW = 200
STEP = 50

# Features (per channel); thresholds are in calibrated units
ZC_THRESHOLD = 0.02
SSC_THRESHOLD = 0.0005
AR_ORDER = 4
FEATURES = ("logMAV", "logRMS", "logWL", "ZC", "SSC", "MNF", "MDF") + tuple(
    "AR{}".format(k + 1) for k in range(AR_ORDER)
)
FEATURE_NAMES = tuple("{}_ch{}".format(f, c + 1) for c in range(N_CHANNELS) for f in FEATURES)

# Training / evaluation
TEST_FRACTION = 0.2
TEST_SUBJECTS = 4
CV_FOLDS = 5
RANDOM_STATE = 42
MAX_CV_SAMPLES = 10000
MAX_FIT_SAMPLES = 15000

# Post-processing
VOTE_N = 3
MIN_PROBA = 0.6
