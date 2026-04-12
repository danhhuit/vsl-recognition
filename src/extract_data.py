# Script dùng MediaPipe trích xuất tọa độ từ video
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

# =========================
# Cấu hình đường dẫn và tham số
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_videos"        # Thư mục chứa video gốc
KEYPOINT_DIR = BASE_DIR / "data" / "keypoints"    # Thư mục lưu tọa độ trích xuất
METADATA_PATH = KEYPOINT_DIR / "metadata.csv"     # File lưu danh sách quản lý dữ liệu

# Danh sách từ vựng và số lượng frame cố định cho mỗi chuỗi
LABELS = ["bang_chu_cai", "cam_on", "chao", "ten_la", "toi"]
SEQ_LEN = 30

# Khởi tạo module nhận diện tay của MediaPipe
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing


def ensure_dirs():
    """Tạo thư mục lưu trữ nếu chưa tồn tại"""
    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)


def extract_hand_keypoints(results):
    """
    Trích xuất tọa độ 21 điểm khớp tay (x, y, z)
    Trả về mảng phẳng 63 giá trị. Nếu không thấy tay thì trả về mảng 0.
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
    Đảm bảo mọi video đều trả về đúng 30 frame dữ liệu
    - Dài hơn 30: Cắt bớt frame đều nhau
    - Ngắn hơn 30: Chèn thêm các frame rỗng (0) vào cuối
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
    """Đọc từng frame video, dùng MediaPipe lấy tọa độ và trả về chuỗi sequence"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARNING] Không mở được video: {video_path}")
        return np.zeros((SEQ_LEN, 63), dtype=np.float32)

    frames_keypoints = []

    # Thiết lập tham số nhận diện cho MediaPipe
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

            # Tiền xử lý ảnh: Resize và chuyển sang RGB
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Lấy tọa độ khớp tay của frame hiện tại
            keypoints = extract_hand_keypoints(results)
            frames_keypoints.append(keypoints)

    cap.release()
    # Chuẩn hóa về độ dài 30 frame
    sequence = sample_or_pad_sequence(frames_keypoints, SEQ_LEN)
    return sequence


def main():
    ensure_dirs()
    rows = []

    # Duyệt qua từng thư mục nhãn (từ vựng)
    for label_id, label_name in enumerate(LABELS):
        label_dir = RAW_DIR / label_name

        if not label_dir.exists():
            print(f"[WARNING] Không tìm thấy folder: {label_dir}")
            continue

        # Tìm tất cả file video trong thư mục nhãn
        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(label_dir.glob(ext))

        if len(video_files) == 0:
            print(f"[WARNING] Folder không có video: {label_dir}")
            continue

        video_files = sorted(video_files)

        # Xử lý từng video một
        for idx, video_path in enumerate(video_files):
            print(f"[INFO] Đang xử lý: {label_name}/{video_path.name}")
            sequence = process_video(video_path)

            # Lưu tọa độ dưới dạng file Numpy (.npy)
            save_name = f"{label_name}_{idx:04d}.npy"
            save_path = KEYPOINT_DIR / save_name
            np.save(save_path, sequence)

            # Lưu thông tin vào danh sách để tạo file metadata CSV
            rows.append({
                "file_path": str(save_path),
                "label_name": label_name,
                "label_id": label_id,
                "video_name": video_path.name
            })

    # Kiểm tra nếu không có dữ liệu
    if len(rows) == 0:
        print("[ERROR] Không có dữ liệu nào được trích xuất.")
        return

    # Tạo DataFrame và xuất file CSV metadata
    df = pd.DataFrame(rows)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Đã lưu metadata: {METADATA_PATH}")
    print(df.head())
    print(f"[DONE] Tổng số mẫu: {len(df)}")


if __name__ == "__main__":
    main()