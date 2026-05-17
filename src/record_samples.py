"""
record_samples.py — Thu thập video mẫu từ webcam cho từng ký hiệu tay.

Cách dùng:
    python src/record_samples.py

Điều khiển:
    SPACE  → Bắt đầu / Dừng quay
    N      → Chuyển sang nhãn tiếp theo
    P      → Quay lại nhãn trước
    Q      → Thoát
    R      → Xem lại danh sách nhãn còn thiếu

Mỗi video được lưu tự động vào raw_videos/<label>/ khi bạn nhấn SPACE lần 2.
"""

import cv2
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, SUPPORTED_VIDEO_EXTENSIONS

# =====================================================================
# Cấu hình
# =====================================================================
TARGET_PER_CLASS = 20       # Số video mục tiêu cho mỗi class
RECORD_SECONDS   = 2.5      # Độ dài mỗi video (giây)
FPS              = 30       # FPS ghi video
RESOLUTION       = (640, 480)

# Màu sắc giao diện
COLOR_RED    = (50, 50, 220)
COLOR_GREEN  = (50, 200, 50)
COLOR_YELLOW = (30, 200, 220)
COLOR_WHITE  = (230, 230, 230)
COLOR_GRAY   = (120, 120, 120)
COLOR_BG     = (30, 30, 30)


# =====================================================================
# Đếm số video hiện có trong mỗi thư mục
# =====================================================================
def count_existing(label: str) -> int:
    label_dir = RAW_DIR / label
    if not label_dir.exists():
        return 0
    count = 0
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        count += len(list(label_dir.glob(ext)))
    return count


def get_next_filename(label: str) -> Path:
    label_dir = RAW_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = count_existing(label)
    return label_dir / f"{label}_{existing + 1:04d}.mp4"


# =====================================================================
# Vẽ UI lên frame
# =====================================================================
def draw_ui(frame, label, label_idx, total_labels, existing, target,
            recording, rec_elapsed, countdown, status_msg):
    H, W = frame.shape[:2]

    # Thanh trạng thái trên cùng
    cv2.rectangle(frame, (0, 0), (W, 70), COLOR_BG, -1)

    # Nhãn hiện tại
    pct = min(existing / target, 1.0)
    bar_w = int((W - 40) * pct)
    cv2.rectangle(frame, (20, 50), (W - 20, 62), (60, 60, 60), -1)
    bar_color = COLOR_GREEN if pct >= 1.0 else COLOR_YELLOW
    if bar_w > 0:
        cv2.rectangle(frame, (20, 50), (20 + bar_w, 62), bar_color, -1)

    label_text = f"[{label_idx + 1}/{total_labels}]  {label.upper()}"
    cv2.putText(frame, label_text, (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR_WHITE, 2)
    count_text = f"{existing}/{target} videos"
    cv2.putText(frame, count_text, (W - 160, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

    # Thanh dưới cùng
    cv2.rectangle(frame, (0, H - 80), (W, H), COLOR_BG, -1)

    if recording:
        # Thanh tiến trình quay
        prog = min(rec_elapsed / RECORD_SECONDS, 1.0)
        prog_w = int((W - 40) * prog)
        cv2.rectangle(frame, (20, H - 30), (W - 20, H - 18), (60, 60, 60), -1)
        if prog_w > 0:
            cv2.rectangle(frame, (20, H - 30), (20 + prog_w, H - 18), COLOR_RED, -1)

        secs_left = max(0, RECORD_SECONDS - rec_elapsed)
        rec_text = f"● ĐANG QUAY  {secs_left:.1f}s"
        cv2.putText(frame, rec_text, (20, H - 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_RED, 2)
        # Vòng tròn nhấp nháy
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (W - 30, H - 50), 10, COLOR_RED, -1)
    else:
        hint = "SPACE: Quay  |  N: Tiếp  |  P: Lùi  |  Q: Thoát"
        cv2.putText(frame, hint, (20, H - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1)
        if status_msg:
            cv2.putText(frame, status_msg, (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
        else:
            cv2.putText(frame, "Nhấn SPACE để bắt đầu quay", (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 1)

    # Countdown overlay trước khi quay
    if countdown > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cd_text = str(countdown)
        (tw, th), _ = cv2.getTextSize(cd_text, cv2.FONT_HERSHEY_DUPLEX, 6, 8)
        cv2.putText(frame, cd_text,
                    ((W - tw) // 2, (H + th) // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 6, COLOR_WHITE, 8)
        ready_text = f"Chuẩn bị ký hiệu:  {label.upper()}"
        (rw, _), _ = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, ready_text,
                    ((W - rw) // 2, H // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)

    return frame


# =====================================================================
# Main
# =====================================================================
def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {RAW_DIR}")
        print("        Hãy tạo raw_videos/ và các thư mục nhãn trước.")
        return

    # Lấy danh sách nhãn từ thư mục con
    labels = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    if not labels:
        print("[ERROR] Chưa có thư mục nhãn nào trong raw_videos/")
        return

    print(f"\n{'='*55}")
    print("  THU THẬP DỮ LIỆU VIDEO - VSL RECOGNITION")
    print(f"{'='*55}")
    print(f"  Mục tiêu     : {TARGET_PER_CLASS} video / class")
    print(f"  Độ dài video : {RECORD_SECONDS}s mỗi clip")
    print(f"  Tổng nhãn    : {len(labels)}")
    print(f"{'='*55}")

    # Hiện trạng
    counts = {lb: count_existing(lb) for lb in labels}
    done   = [lb for lb in labels if counts[lb] >= TARGET_PER_CLASS]
    todo   = [lb for lb in labels if counts[lb] < TARGET_PER_CLASS]
    print(f"\n  Đã đủ ({len(done)}): {', '.join(done) if done else 'chưa có'}")
    print(f"  Cần thêm ({len(todo)}): {', '.join(todo)}\n")

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)

    label_idx  = 0
    recording  = False
    writer     = None
    rec_start  = 0.0
    countdown  = 0
    cd_start   = 0.0
    status_msg = ""
    status_ts  = 0.0

    print("Webcam đã mở. Nhấn SPACE để bắt đầu quay, Q để thoát.")
    cv2.namedWindow("VSL - Thu thập dữ liệu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VSL - Thu thập dữ liệu", RESOLUTION[0], RESOLUTION[1])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không đọc được frame từ webcam.")
            break

        frame = cv2.flip(frame, 1)  # Mirror
        label   = labels[label_idx]
        existing = count_existing(label)

        # Xử lý countdown trước khi quay
        if countdown > 0:
            elapsed_cd = time.time() - cd_start
            remaining  = 3 - int(elapsed_cd)
            if remaining <= 0:
                countdown = 0
                # Bắt đầu quay thật sự
                recording = True
                rec_start = time.time()
                save_path = get_next_filename(label)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(save_path), fourcc, FPS, RESOLUTION
                )
                print(f"  [REC] Bắt đầu quay → {save_path.name}")
            else:
                countdown = remaining

        # Quay video
        if recording:
            writer.write(frame)
            rec_elapsed = time.time() - rec_start
            if rec_elapsed >= RECORD_SECONDS:
                # Tự động dừng
                recording = False
                writer.release()
                writer = None
                new_count = count_existing(label)
                status_msg = f"✓ Đã lưu video #{new_count} cho '{label}'"
                status_ts  = time.time()
                print(f"  [OK]  {label} → {new_count}/{TARGET_PER_CLASS} video")

                # Tự động sang nhãn kế nếu đủ rồi
                if new_count >= TARGET_PER_CLASS and label_idx < len(labels) - 1:
                    label_idx += 1
                    status_msg = f"'{label}' đủ {TARGET_PER_CLASS} video! Chuyển → '{labels[label_idx]}'"
                    print(f"\n  >> Đủ video cho '{label}', chuyển sang '{labels[label_idx]}'")
        else:
            rec_elapsed = 0.0

        # Ẩn status sau 3 giây
        if status_msg and time.time() - status_ts > 3:
            status_msg = ""

        # Vẽ UI
        display = draw_ui(
            frame.copy(), label, label_idx, len(labels),
            count_existing(label), TARGET_PER_CLASS,
            recording, rec_elapsed, countdown, status_msg,
        )
        cv2.imshow("VSL - Thu thập dữ liệu", display)

        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Đã thoát.")
            break

        elif key == ord(' '):  # SPACE
            if not recording and countdown == 0:
                # Bắt đầu đếm ngược 3 giây
                countdown = 3
                cd_start  = time.time()
                print(f"\n  >> Chuẩn bị quay '{label}'... (3s)")

        elif key == ord('n') or key == ord('N'):
            if not recording:
                label_idx = (label_idx + 1) % len(labels)
                status_msg = f"Chuyển sang: {labels[label_idx].upper()}"
                status_ts  = time.time()

        elif key == ord('p') or key == ord('P'):
            if not recording:
                label_idx = (label_idx - 1) % len(labels)
                status_msg = f"Chuyển sang: {labels[label_idx].upper()}"
                status_ts  = time.time()

        elif key == ord('r') or key == ord('R'):
            if not recording:
                print("\n  --- Trạng thái hiện tại ---")
                for lb in labels:
                    c   = count_existing(lb)
                    bar = "█" * c + "░" * max(0, TARGET_PER_CLASS - c)
                    tag = " ✓" if c >= TARGET_PER_CLASS else f" ({TARGET_PER_CLASS - c} còn thiếu)"
                    print(f"  {lb:10s} [{bar}] {c:2d}/{TARGET_PER_CLASS}{tag}")
                print()

    # Dọn dẹp
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    # Tổng kết
    print(f"\n{'='*55}")
    print("  TỔNG KẾT")
    print(f"{'='*55}")
    total = 0
    for lb in labels:
        c   = count_existing(lb)
        total += c
        tag = "✓ ĐỦ" if c >= TARGET_PER_CLASS else f"⚠ còn thiếu {TARGET_PER_CLASS - c}"
        print(f"  {lb:12s}: {c:3d} videos  [{tag}]")
    print(f"{'='*55}")
    print(f"  Tổng: {total} videos trên {len(labels)} nhãn")
    done2 = sum(1 for lb in labels if count_existing(lb) >= TARGET_PER_CLASS)
    print(f"  Nhãn đủ data: {done2}/{len(labels)}")

    if done2 == len(labels):
        print("\n  ✓ Đủ data! Bước tiếp theo:")
        print("    python src/extract_data.py")
        print("    python src/train.py")
    else:
        print(f"\n  ⚠ Còn {len(labels) - done2} nhãn chưa đủ. Chạy lại script để quay thêm.")


if __name__ == "__main__":
    main()