from pathlib import Path

# =========================
# Project paths
# =========================
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_videos"
KEYPOINT_DIR = DATA_DIR / "keypoints"
WEIGHTS_DIR = BASE_DIR / "weights"
RESULTS_DIR = BASE_DIR / "results"

METADATA_PATH = KEYPOINT_DIR / "metadata.csv"
BEST_MODEL_PATH = WEIGHTS_DIR / "best_model.pth"

# =========================
# Data config
# =========================
SEQ_LEN = 30
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
INPUT_SIZE = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63

# =========================
# Training config
# =========================
BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 0  # Windows-safe default
RANDOM_SEED = 42

# =========================
# Runtime config
# =========================
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS_EXTRACT = 1
MAX_NUM_HANDS_DEMO = 2
SUPPORTED_VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def ensure_project_dirs() -> None:
    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
