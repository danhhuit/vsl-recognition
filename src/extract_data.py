# Script dùng MediaPipe trích xuất tọa độ từ video (Bản tinh gọn cho Train)
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import mediapipe.python.solutions.hands as mp_hands

# =========================
# Cấu hình đường dẫn và tham số
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_videos"
KEYPOINT_DIR = BASE_DIR / "data" / "keypoints"
METADATA_PATH = KEYPOINT_DIR / "metadata.csv"

# Chuẩn hóa về đúng 30 frame
SEQ_LEN = 30


def ensure_dirs():
    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)


def extract_hand_keypoints(results):
    """Trích xuất 63 tọa độ (x, y, z) tương đối (Lấy cổ tay làm gốc)."""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist = hand_landmarks.landmark[0]

        keypoints = []
        for lm in hand_landmarks.landmark:
            # Trừ tọa độ cổ tay cho tất cả 21 điểm
            keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(keypoints, dtype=np.float32)

    return np.zeros(21 * 3, dtype=np.float32)


def sample_or_pad_sequence(sequence, target_len=30):
    """Cắt bớt hoặc đệm frame rỗng để đảm bảo đủ 30 frame."""
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
    """Xử lý video. Báo lỗi và trả về None nếu không đọc được."""
    cap = cv2.VideoCapture(str(video_path))

    # 1. Kiểm tra xem OpenCV có mở được file không
    if not cap.isOpened():
        print(f"[ERROR] Không thể đọc được định dạng file này: {video_path.name}")
        return None

    frames_keypoints = []

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Bắt buộc 1 tay
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

    # 2. Kiểm tra xem file có chứa hình ảnh/frame nào không
    if len(frames_keypoints) == 0:
        print(f"[ERROR] Video rỗng hoặc độ dài bằng 0 giây: {video_path.name}")
        return None

    sequence = sample_or_pad_sequence(frames_keypoints, SEQ_LEN)
    return sequence


def main():
    ensure_dirs()
    rows = []

    if not RAW_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {RAW_DIR}")
        return

    # Tự động nhận diện nhãn từ thư mục con
    dynamic_labels = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])

    if len(dynamic_labels) == 0:
        print(f"[ERROR] Thư mục raw_videos đang trống.")
        return

    print(f"[INFO] Hệ thống tự động nhận diện các nhãn: {dynamic_labels}\n")

    for label_id, label_name in enumerate(dynamic_labels):
        label_dir = RAW_DIR / label_name
        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(label_dir.glob(ext))

        video_files = sorted(video_files)
        for idx, video_path in enumerate(video_files):
            sequence = process_video(video_path)

            # NẾU CÓ LỖI XẢY RA THÌ BÁO SKIP VÀ KHÔNG LƯU
            if sequence is None:
                print(f"[SKIP] ---> ĐÃ BỎ QUA file không hợp lệ: {label_name}/{video_path.name}\n")
                continue

            save_name = f"{label_name}_{idx:04d}.npy"
            save_path = KEYPOINT_DIR / save_name
            np.save(save_path, sequence)

            rows.append({
                "file_path": str(save_path),
                "label_name": label_name,
                "label_id": label_id,
                "video_name": video_path.name
            })
            print(f"[SUCCESS] Trích xuất thành công: {label_name}/{video_path.name}")

    if len(rows) == 0:
        print("\n[ERROR] Không có dữ liệu hợp lệ nào được trích xuất.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Đã lưu danh sách dữ liệu vào: {METADATA_PATH}")
    print(f"[DONE] TỔNG SỐ MẪU HỢP LỆ THỰC TẾ: {len(df)}")


if __name__ == "__main__":
    main()