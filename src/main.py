"""
================================
MAIN MENU - Sign Language Recognition
================================

Hệ thống nhận diện ngôn ngữ ký hiệu Tiếng Việt (VSL - Vietnamese Sign Language)

Các tùy chọn:
  1. setup_model   - Khởi tạo & test kiến trúc mô hình
  2. extract_data  - Trích xuất keypoints từ video
  3. train         - Huấn luyện mô hình
  4. demo          - Demo visualize keypoints từ webcam
  5. app           - Ứng dụng inference real-time từ webcam
  6. exit          - Thoát chương trình

Quy trình khuyến nghị:
  Step 1: Chạy option 1 (setup_model) - kiểm tra mô hình có hoạt động
  Step 2: Chạy option 2 (extract_data) - trích xuất dữ liệu từ video
  Step 3: Chạy option 3 (train) - huấn luyện mô hình
  Step 4: Chạy option 4 (demo) hoặc option 5 (app) - kiểm tra kết quả
"""

import sys
import os
from pathlib import Path

# Thêm đường dẫn src vào path để import các module
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================
# HÀM HIỂN THỊ MENU
# ============================================================
def show_main_menu():
    """Hiển thị menu chính."""
    print("\n" + "="*75)
    print(" "*18 + "SIGN LANGUAGE RECOGNITION - MAIN MENU")
    print("="*75)
    print("\nCác tùy chọn khả dụng:")
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  [1] setup_model   → Khởi tạo & test kiến trúc mô hình LSTM        │")
    print("│  [2] extract_data  → Trích xuất keypoints từ video                 │")
    print("│  [3] train         → Huấn luyện mô hình                            │")
    print("│  [4] demo          → Demo visualize keypoints từ webcam            │")
    print("│  [5] app           → Ứng dụng inference real-time từ webcam        │")
    print("│  [6] exit          → Thoát chương trình                            │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print("="*75)
    choice = input("\nNhập lựa chọn (1-6): ").strip()
    return choice


# ============================================================
# OPTION 1: SETUP MODEL
# ============================================================
def setup_model_main():
    """OPTION 1: Khởi tạo & test mô hình."""
    print("\n" + "="*75)
    print(" "*18 + "[1] SETUP MODEL - KHỞI TẠO & TEST MÔ HÌNH LSTM")
    print("="*75)
    
    try:
        from model import (
            define_config, 
            setup_model, 
            setup_loss,
            setup_optimizer, 
            test_model
        )
        
        # Bước 1: Định nghĩa cấu hình
        print("\n[STEP 1/5] Định nghĩa cấu hình...")
        config = define_config(
            batch_size=16,
            sequence_length=30,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.001
        )
        
        # Bước 2: Khởi tạo mô hình
        print("\n[STEP 2/5] Khởi tạo mô hình...")
        model = setup_model(config)
        
        # Bước 3: Thiết lập Loss Function
        print("\n[STEP 3/5] Thiết lập Loss Function...")
        criterion = setup_loss()
        
        # Bước 4: Thiết lập Optimizer
        print("\n[STEP 4/5] Thiết lập Bộ Tối Ưu...")
        optimizer = setup_optimizer(model, config)
        
        # Bước 5: Test mô hình
        print("\n[STEP 5/5] Test mô hình...")
        test_model(model, criterion, optimizer, config)
        
        print("\n" + "="*75)
        print("✅ SETUP MODEL HOÀN THÀNH THÀNH CÔNG!")
        print("="*75 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}\n")


# ============================================================
# OPTION 2: EXTRACT DATA
# ============================================================
def extract_data_main():
    """OPTION 2: Trích xuất keypoints từ video."""
    print("\n" + "="*75)
    print(" "*18 + "[2] EXTRACT DATA - TRÍCH XUẤT KEYPOINTS TỪ VIDEO")
    print("="*75)
    
    try:
        import extract_data
        
        print("\nThông tin:")
        print("  • Trích xuất tọa độ keypoints từ các video")
        print("  • Sử dụng MediaPipe Hands Model")
        print("  • Lưu dữ liệu dưới dạng .npy files")
        print("  • Cập nhật metadata.csv")
        print("\nLưu ý: Đảm bảo thư mục data/raw_videos/ chứa các video\n")
        
        # Gọi hàm main trong extract_data.py
        if hasattr(extract_data, 'main'):
            extract_data.main()
        else:
            print("⚠️  Hàm main() không tìm thấy trong extract_data.py")
        
        print("\n" + "="*75)
        print("✅ EXTRACT DATA HOÀN THÀNH!")
        print("="*75 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}\n")


# ============================================================
# OPTION 3: TRAIN
# ============================================================
def train_main():
    """OPTION 3: Huấn luyện mô hình."""
    print("\n" + "="*75)
    print(" "*18 + "[3] TRAIN - HUẤN LUYỆN MÔ HÌNH")
    print("="*75)
    
    try:
        import train
        
        print("\nThông tin:")
        print("  • Huấn luyện mô hình LSTM trên dữ liệu keypoints")
        print("  • Chia dữ liệu: 80% training, 20% validation")
        print("  • Lưu mô hình tốt nhất vào weights/best_model.pth")
        print("\nLưu ý: Đảm bảo metadata.csv đã được tạo từ extract_data\n")
        
        # Gọi hàm main trong train.py
        if hasattr(train, 'main'):
            train.main()
        else:
            print("⚠️  Hàm main() không tìm thấy trong train.py")
        
        print("\n" + "="*75)
        print("✅ TRAIN HOÀN THÀNH!")
        print("="*75 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}\n")


# ============================================================
# OPTION 4: DEMO
# ============================================================
def demo_main():
    """OPTION 4: Demo - visualize keypoints từ webcam."""
    print("\n" + "="*75)
    print(" "*18 + "[4] DEMO - VISUALIZE KEYPOINTS TỪ WEBCAM")
    print("="*75)
    
    try:
        import demo_keypoints
        
        print("\nThông tin:")
        print("  • Sử dụng webcam để capture video real-time")
        print("  • Visualize tọa độ 21 khớp tay bằng MediaPipe Hands")
        print("  • Hiển thị tọa độ x, y, z của mỗi khớp")
        print("\nLưu ý: Đảm bảo webcam sẵn sàng")
        print("Nhấn 'Q' để thoát demo\n")
        
        # Gọi hàm main trong demo_keypoints.py
        if hasattr(demo_keypoints, 'run_full_demo'):
            demo_keypoints.run_full_demo()
        else:
            print("⚠️  Hàm run_full_demo() không tìm thấy trong demo_keypoints.py")
        
        print("\n" + "="*75)
        print("✅ DEMO HOÀN THÀNH!")
        print("="*75 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}\n")


# ============================================================
# OPTION 5: APP
# ============================================================
def app_main():
    """OPTION 5: Ứng dụng - inference real-time."""
    print("\n" + "="*75)
    print(" "*18 + "[5] APP - INFERENCE REAL-TIME TỪ WEBCAM")
    print("="*75)
    
    try:
        import app
        
        print("\nThông tin:")
        print("  • Sử dụng mô hình đã huấn luyện để nhận diện")
        print("  • Capture video từ webcam real-time")
        print("  • Hiển thị kết quả dự đoán và độ tin cậy")
        print("\nLưu ý: Đảm bảo mô hình đã được huấn luyện")
        print("Đảm bảo weights/best_model.pth tồn tại")
        print("Nhấn 'Q' để thoát ứng dụng\n")
        
        # Gọi hàm main trong app.py
        if hasattr(app, 'main'):
            app.main()
        else:
            print("⚠️  Hàm main() không tìm thấy trong app.py")
        
        print("\n" + "="*75)
        print("✅ APP HOÀN THÀNH!")
        print("="*75 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}\n")


# ============================================================
# HÀM MAIN - VÒNG LẶP MENU
# ============================================================
def main():
    """Hàm main - vòng lặp menu chính."""
    print("\n" + "="*75)
    print("╔" + "="*73 + "╗")
    print("║" + " "*73 + "║")
    print("║" + " "*15 + "WELCOME TO SIGN LANGUAGE RECOGNITION SYSTEM" + " "*15 + "║")
    print("║" + " "*73 + "║")
    print("╚" + "="*73 + "╝")
    print()
    print("Hệ thống nhận diện ngôn ngữ ký hiệu Tiếng Việt (VSL)")
    print("Vietnamese Sign Language Recognition System")
    print()
    
    while True:
        choice = show_main_menu()
        
        if choice == "1":
            setup_model_main()
        elif choice == "2":
            extract_data_main()
        elif choice == "3":
            train_main()
        elif choice == "4":
            demo_main()
        elif choice == "5":
            app_main()
        elif choice == "6":
            print("\n" + "="*75)
            print("╔" + "="*73 + "╗")
            print("║" + " "*30 + "GOODBYE! 👋" + " "*32 + "║")
            print("╚" + "="*73 + "╝\n")
            break
        else:
            print("\n⚠️  Lựa chọn không hợp lệ! Vui lòng chọn từ 1-6.\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*75)
        print("║" + " "*20 + "Chương trình bị dừng bởi người dùng!" + " "*20 + "║")
        print("="*75 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ LỖI KHÔNG NGỜ: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
