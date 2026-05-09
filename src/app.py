import cv2
import torch
import numpy as np
from collections import deque
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

from model import SignLanguageLSTM
from dataset import VSLDataset
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    SEQ_LEN,
    BEST_MODEL_PATH,
    METADATA_PATH,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS_EXTRACT,
)


# =====================================================================
# Load mô hình và label map
# =====================================================================
def load_model_for_inference(device):
    """Load best_model.pth và trả về model + label_map."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Không tìm thấy file weights: {BEST_MODEL_PATH}\n"
            "[HINT] Hãy huấn luyện mô hình trước (option 3)."
        )

    dataset = VSLDataset(METADATA_PATH)
    label_map = dataset.get_label_map()
    num_classes = dataset.num_classes

    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    ).to(device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[INFO] Đã load model từ : {BEST_MODEL_PATH}")
    print(f"[INFO] Số lớp           : {num_classes}")
    print(f"[INFO] Danh sách nhãn   : {list(label_map.values())}")

    return model, label_map


# =====================================================================
# Trích xuất keypoints từ 1 frame
# =====================================================================
def extract_keypoints_from_frame(results):
    """Trích xuất 63 tọa độ tương đối từ kết quả MediaPipe."""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist = hand_landmarks.landmark[0]

        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(keypoints, dtype=np.float32), True

    return np.zeros(INPUT_SIZE, dtype=np.float32), False


# =====================================================================
# Vẽ text có viền
# =====================================================================
def draw_text_with_outline(img, text, pos, font_scale, text_color, thickness):
    """Vẽ text có viền đen cho dễ đọc."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, font, font_scale, text_color, thickness)


# =====================================================================
# Vẽ giao diện HUD trên frame
# =====================================================================
def draw_hud(frame, prediction, confidence, buffer_len, hand_detected):
    """Vẽ thông tin dự đoán và trạng thái lên frame."""
    h, w, _ = frame.shape

    # --- Thanh trạng thái trên cùng ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Trạng thái buffer
    buffer_text = f"Buffer: {buffer_len}/{SEQ_LEN}"
    buffer_color = (0, 255, 0) if buffer_len >= SEQ_LEN else (0, 200, 255)
    draw_text_with_outline(frame, buffer_text, (10, 25), 0.6, buffer_color, 1)

    # Trạng thái tay
    hand_text = "Tay: DETECTED" if hand_detected else "Tay: NOT DETECTED"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    draw_text_with_outline(frame, hand_text, (10, 55), 0.6, hand_color, 1)

    # --- Kết quả dự đoán ở dưới ---
    if prediction is not None:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

        # Tên ký hiệu
        draw_text_with_outline(
            frame, f"Ky hieu: {prediction}", (10, h - 55), 1.0, (0, 255, 255), 2
        )

        # Confidence
        conf_text = f"Confidence: {confidence:.1f}%"
        conf_color = (0, 255, 0) if confidence > 70 else (0, 200, 255) if confidence > 40 else (0, 0, 255)
        draw_text_with_outline(frame, conf_text, (10, h - 20), 0.7, conf_color, 1)

    # Hướng dẫn
    draw_text_with_outline(frame, "Nhan 'Q' de thoat", (w - 220, 25), 0.5, (200, 200, 200), 1)


# =====================================================================
# Vòng lặp webcam chính
# =====================================================================
def run_inference(model, label_map, device):
    """Mở webcam, dùng sliding window 30 frames để dự đoán realtime."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam.")
        return

    # Sliding window buffer — lưu 30 frames gần nhất
    frame_buffer = deque(maxlen=SEQ_LEN)

    prediction = None
    confidence = 0.0

    print("[INFO] Webcam đang chạy. Nhấn 'Q' để thoát.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS_EXTRACT,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # --- Trích xuất keypoints ---
            keypoints, hand_detected = extract_keypoints_from_frame(results)
            frame_buffer.append(keypoints)

            # --- Vẽ hand landmarks ---
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # --- Dự đoán khi buffer đủ 30 frames ---
            if len(frame_buffer) == SEQ_LEN:
                sequence = np.array(list(frame_buffer), dtype=np.float32)
                tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities, dim=1)

                    predicted_label = predicted_idx.item()
                    confidence = max_prob.item() * 100
                    prediction = label_map.get(predicted_label, f"ID:{predicted_label}")

            # --- Vẽ HUD ---
            draw_hud(frame, prediction, confidence, len(frame_buffer), hand_detected)

            # --- Hiển thị ---
            cv2.imshow("Sign Language Recognition — Realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Đã đóng webcam.")


# =====================================================================
# Entry point
# =====================================================================
def main():
    print("\n" + "=" * 60)
    print("ỨNG DỤNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU — REALTIME")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị: {device}")

    model, label_map = load_model_for_inference(device)
    run_inference(model, label_map, device)


if __name__ == "__main__":
    main()
