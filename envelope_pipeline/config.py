"""Parameters of the KIT0012 envelope pipeline (report sections 4-5)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_DIR = ROOT / "csv_data"
OUT_DIR = ROOT / "artifacts" / "envelope"
SAMPLES_DIR = OUT_DIR / "samples"
DATASET_CACHE = OUT_DIR / "envelope_dataset.npz"
MODEL_PATH = OUT_DIR / "gesture_model.joblib"
REPORT_PATH = OUT_DIR / "train_report.json"

# Kaggle DS1 dataset (training source until real KIT0012 recordings exist)
RAW_FS = 1000
GESTURES = ("Fist", "Thumb flexion", "Rest", "Finger extension", "Middle+ring flexion")
REST_LABEL = 2
# Assumption: inside each MovK file the 400 trials are ordered subject by subject.
TRIALS_PER_SUBJECT = 20

# Stage 1 - sampling (Uno, 10-bit ADC)
FS = 200
N_CHANNELS = 3
ADC_MAX = 1023

# KIT0012 simulation: high-pass, full-wave rectify, low-pass envelope
SIM_HIGHPASS_HZ = 20.0
SIM_ENVELOPE_HZ = 5.0
SIM_TRANSIENT_S = 0.3
# Gain tuned so the strongest contractions peak around 800 counts (report 3.5: ADC 700-920).
SIM_ADC_PEAK = 800.0

# Stage 4 - preprocessing
MVC_PERCENTILE = 99.0
NORM_CLIP = 1.5

# Stage 5 - windowing: 200 ms window, 50 ms step
WINDOW = 40
STEP = 10

# Stage 6 - features
FEATURES = ("mean", "max", "std", "slope")
FEATURE_NAMES = tuple(
    "{}_ch{}".format(f, c + 1) for c in range(N_CHANNELS) for f in FEATURES
)

# Stage 7 - classification / evaluation
TEST_FRACTION = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
MAX_CV_SAMPLES = 10000
MAX_FIT_SAMPLES = 15000

# Stage 8 - post-processing
VOTE_N = 3
MIN_PROBA = 0.6
