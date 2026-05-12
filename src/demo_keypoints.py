import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS_DEMO


def draw_text_with_outline(img, text, pos, font_scale, text_color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, font, font_scale, text_color, thickness)


def run_full_demo():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS_DEMO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        print("[INFO] DEMO nhận diện tay đang chạy. Nhấn 'Q' để thoát.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for i, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    label = handedness.classification[0].label
                    if label == "Left":
                        hand_text = "Tay Trai"
                        hand_color = (255, 100, 100)
                    else:
                        hand_text = "Tay Phai"
                        hand_color = (100, 100, 255)

                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    norm_x = index_tip.x - wrist.x
                    norm_y = index_tip.y - wrist.y
                    norm_z = index_tip.z - wrist.z

                    h, w, _ = frame.shape
                    text_x = int(wrist.x * w)
                    text_y = int(wrist.y * h)

                    draw_text_with_outline(
                        img=frame,
                        text=hand_text,
                        pos=(text_x - 50, text_y + 30),
                        font_scale=0.7,
                        text_color=hand_color,
                        thickness=2,
                    )

                    y_offset = 25 + (i * 25)
                    xyz_text = f"{hand_text} - X: {norm_x:.2f} | Y: {norm_y:.2f} | Z: {norm_z:.2f}"
                    draw_text_with_outline(
                        img=frame,
                        text=xyz_text,
                        pos=(10, y_offset),
                        font_scale=0.6,
                        text_color=(255, 255, 255),
                        thickness=1,
                    )

            cv2.imshow("Demo: Nhan Dien Tay & Toa Do", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    run_full_demo()


if __name__ == "__main__":
    main()
