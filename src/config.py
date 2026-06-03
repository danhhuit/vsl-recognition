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
#
SEQ_LEN = 30
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
INPUT_SIZE = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63

# =========================
# Training config
# =========================
BATCH_SIZE = 8           # Nhỏ hơn phù hợp với dataset nhỏ (~660 mẫu)
#no ron ẩn => kha nang ghi nho dac trung
HIDDEN_SIZE = 64         # Giảm model size để tránh overfitting với dataset nhỏ
#so lop LSTM xep chong len nhau
NUM_LAYERS = 2
#toc do hoc
LEARNING_RATE = 5e-4     # Giảm LR giúp training ổn định hơn
NUM_EPOCHS = 100
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15         # Tăng val/test để đánh giá đáng tin cậy hơn
TEST_RATIO = 0.15
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

# =========================
# Feature config for realtime app
# =========================
_BASE_INPUT = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63 keypoints gốc
USE_VELOCITY = False  # Giữ False nếu model đang train với INPUT_SIZE = 63

# =========================
# Realtime prediction config
# =========================
CONFIDENCE_THRESHOLD = 0.75      # Chỉ hiển thị dự đoán khi độ tin cậy >= 75%
MIN_HAND_FRAMES_RATIO = 0.60     # Ít nhất 60% frame trong buffer phải phát hiện được tay
APP_DEFAULT_LANGUAGE = "vi"      # "vi" hoặc "en"

