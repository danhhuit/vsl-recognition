import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

# =========================
# Cấu hình
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_videos"
KEYPOINT_DIR = BASE_DIR / "data" / "keypoints"
METADATA_PATH = KEYPOINT_DIR / "metadata.csv"

LABELS = ["bang_chu_cai", "cam_on", "chao", "ten_la", "toi"]
SEQ_LEN = 30

mp_hands = mp.solutions.hands


def ensure_dirs():
    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)


def extract_hand_keypoints(results):
    """
    Lấy 21 điểm khớp tay đầu tiên -> 63 giá trị (x, y, z).
    Nếu không nhận diện được tay thì trả về vector 0.
    """
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints, dtype=np.float32)

    return np.zeros(21 * 3, dtype=np.float32)


def sample_or_pad_sequence(sequence, target_len=30):
    """
    Chuẩn hóa chuỗi về đúng 30 frame.
    - Dài hơn: lấy mẫu đều
    - Ngắn hơn: thêm frame 0
    """
    if len(sequence) == 0:
        return np.zeros((target_len, 63), dtype=np.float32)

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) > target_len:
        indices = np.linspace(0, len(sequence) - 1, target_len).astype(int)
        sequence = sequence[indices]
    elif len(sequence) < target_len:
        pad = np.zeros((target_len - len(sequence), sequence.shape[1]), dtype=np.float32)
        sequence = np.vstack([sequence, pad])

    return sequence


def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARNING] Không mở được video: {video_path}")
        return np.zeros((SEQ_LEN, 63), dtype=np.float32)

    frames_keypoints = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            keypoints = extract_hand_keypoints(results)
            frames_keypoints.append(keypoints)

    cap.release()
    sequence = sample_or_pad_sequence(frames_keypoints, SEQ_LEN)
    return sequence


def main():
    ensure_dirs()
    rows = []

    for label_id, label_name in enumerate(LABELS):
        label_dir = RAW_DIR / label_name

        if not label_dir.exists():
            print(f"[WARNING] Không tìm thấy folder: {label_dir}")
            continue

        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(label_dir.glob(ext))

        if len(video_files) == 0:
            print(f"[WARNING] Folder không có video: {label_dir}")
            continue

        video_files = sorted(video_files)

        for idx, video_path in enumerate(video_files):
            print(f"[INFO] Đang xử lý: {label_name}/{video_path.name}")
            sequence = process_video(video_path)

            save_name = f"{label_name}_{idx:04d}.npy"
            save_path = KEYPOINT_DIR / save_name
            np.save(save_path, sequence)

            rows.append({
                "file_path": str(save_path),
                "label_name": label_name,
                "label_id": label_id,
                "video_name": video_path.name
            })

    if len(rows) == 0:
        print("[ERROR] Không có dữ liệu nào được trích xuất.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Đã lưu metadata: {METADATA_PATH}")
    print(df.head())
    print(f"[DONE] Tổng số mẫu: {len(df)}")


if __name__ == "__main__":
    main()
