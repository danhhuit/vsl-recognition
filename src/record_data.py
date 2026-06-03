import cv2
import os
import time
from pathlib import Path
from config import RAW_DIR, ensure_project_dirs

def get_next_filename(label, ext=".mp4"):
    """Tự động tăng số thứ tự cho tên file video (VD: a_0000.mp4, a_0001.mp4)"""
    existing_files = list(RAW_DIR.glob(f"{label}_*{ext}"))
    if not existing_files:
        return RAW_DIR / f"{label}_0000{ext}"
    
    indices = []
    for f in existing_files:
        try:
            # Tên file có dạng label_0000.mp4
            idx = int(f.stem.split('_')[-1])
            indices.append(idx)
        except ValueError:
            pass
            
    next_idx = max(indices) + 1 if indices else 0
    return RAW_DIR / f"{label}_{next_idx:04d}{ext}"

def main():
    # Đảm bảo thư mục lưu video đã tồn tại
    ensure_project_dirs()
    
    print("\n" + "="*50)
    print("CHƯƠNG TRÌNH THU THẬP DỮ LIỆU VIDEO (WEBCAM)")
    print("="*50)
    
    label = input("Nhập tên nhãn động tác (ví dụ: a, b, cam_on, chao): ").strip()
    if not label:
        print("[ERROR] Tên nhãn không được để trống!")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam! Vui lòng kiểm tra lại thiết bị.")
        return

    # Lấy thông số video từ webcam
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 # Set cứng FPS lưu file (webcam thường là 30fps)
    
    # Định dạng video lưu trữ (MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    is_recording = False
    out = None
    video_path = None
    frames_recorded = 0
    
    print("\n" + "-"*40)
    print("👉 HƯỚNG DẪN SỬ DỤNG:")
    print("1. Đứng vào khung hình cho vừa vặn.")
    print("2. Nhấn phím 'r' để BẮT ĐẦU quay (xuất hiện chữ REC màu đỏ).")
    print("3. Thực hiện động tác.")
    print("4. Nhấn phím 'r' lần nữa để KẾT THÚC quay và lưu video.")
    print("5. Nhấn phím 'q' hoặc 'ESC' để THOÁT chương trình.")
    print("-" * 40 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không thể đọc khung hình từ webcam.")
            break
            
        # Lật ngược hình ảnh (mirror effect) để dễ nhìn khi tự quay
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        if is_recording:
            # Ghi frame gốc (đã lật) vào file
            out.write(frame)
            frames_recorded += 1
            
            # Vẽ UI trên frame hiển thị
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC: {frames_recorded} frames", (50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Trạng thái chờ
            cv2.putText(display_frame, f"Nhan: {label} | Nhan 'r' de quay", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Record Gesture Data", display_frame)
        
        # Đợi phím bấm
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            if not is_recording:
                # Bắt đầu quay
                video_path = get_next_filename(label)
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                is_recording = True
                frames_recorded = 0
                print(f"🔴 [BẮT ĐẦU QUAY] Đang ghi hình cho nhãn '{label}'...")
            else:
                # Kết thúc quay
                is_recording = False
                out.release()
                print(f"✅ [ĐÃ LƯU] Lưu thành công {frames_recorded} frames vào: {video_path.name}")
                
        elif key == ord('q') or key == 27: # 'q' hoặc ESC
            if is_recording:
                out.release()
                print(f"✅ [ĐÃ LƯU] Lưu thành công vào: {video_path.name}")
            print("\n[INFO] Đã thoát chương trình.")
            break

    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
