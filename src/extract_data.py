import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe.python.solutions.hands as mp_hands

from config import (
    RAW_DIR,
    KEYPOINT_DIR,
    METADATA_PATH,
    SEQ_LEN,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS_EXTRACT,
    SUPPORTED_VIDEO_EXTENSIONS,
    ensure_project_dirs,
)


def extract_hand_keypoints(results):
    """Trích xuất 63 tọa độ tương đối, lấy cổ tay làm gốc."""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist = hand_landmarks.landmark[0]

        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(keypoints, dtype=np.float32), True

    return np.zeros(63, dtype=np.float32), False


def sample_or_pad_sequence(sequence, target_len=SEQ_LEN):
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


def process_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở video: {video_path.name}")
        return None

    frames_keypoints = []
    num_frames_raw = 0
    num_detected_frames = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS_EXTRACT,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            num_frames_raw += 1
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            keypoints, detected = extract_hand_keypoints(results)
            if detected:
                num_detected_frames += 1
            frames_keypoints.append(keypoints)

    cap.release()

    if num_frames_raw == 0:
        print(f"[ERROR] Video rỗng hoặc không đọc được frame: {video_path.name}")
        return None

    sequence = sample_or_pad_sequence(frames_keypoints, SEQ_LEN)
    return {
        "sequence": sequence,
        "num_frames_raw": num_frames_raw,
        "num_detected_frames": num_detected_frames,
        "status": "ok" if num_detected_frames > 0 else "no_hand_detected",
    }


def collect_video_files(label_dir: Path):
    video_files = []
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        video_files.extend(label_dir.glob(ext))
    return sorted(video_files)


def main():
    ensure_project_dirs()
    rows = []

    if not RAW_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục raw_videos: {RAW_DIR}")
        return

    dynamic_labels = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    if not dynamic_labels:
        print("[ERROR] raw_videos đang trống hoặc chưa có thư mục nhãn.")
        return

    print(f"[INFO] Nhãn được phát hiện tự động: {dynamic_labels}\n")

    for label_id, label_name in enumerate(dynamic_labels):
        label_dir = RAW_DIR / label_name
        video_files = collect_video_files(label_dir)

        if not video_files:
            print(f"[WARNING] Không có video trong thư mục nhãn: {label_name}")
            continue

        for idx, video_path in enumerate(video_files):
            result = process_video(video_path)
            if result is None:
                print(f"[SKIP] Bỏ qua file lỗi: {label_name}/{video_path.name}\n")
                continue

            save_name = f"{label_name}_{idx:04d}.npy"
            save_path = KEYPOINT_DIR / save_name
            np.save(save_path, result["sequence"])

            rows.append(
                {
                    "file_path": str(save_path),
                    "label_name": label_name,
                    "label_id": label_id,
                    "video_name": video_path.name,
                    "num_frames_raw": result["num_frames_raw"],
                    "num_detected_frames": result["num_detected_frames"],
                    "status": result["status"],
                }
            )
            print(
                f"[DONE] {label_name}/{video_path.name} | "
                f"frames={result['num_frames_raw']} | "
                f"detected={result['num_detected_frames']} | "
                f"status={result['status']}"
            )

    if not rows:
        print("[ERROR] Không có dữ liệu hợp lệ nào được trích xuất.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Đã lưu metadata: {METADATA_PATH}")
    print(f"[DONE] Tổng số mẫu hợp lệ: {len(df)}")


if __name__ == "__main__":
    main()
