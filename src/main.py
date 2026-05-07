import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# =====================================================================
# Menu chính
# =====================================================================
def show_main_menu():
    """Hiển thị menu lựa chọn và trả về input."""
    print("\n" + "=" * 60)
    print("SIGN LANGUAGE RECOGNITION — MAIN MENU")
    print("=" * 60)
    print("  [1] setup_model   — Kiểm tra kiến trúc model")
    print("  [2] extract_data  — Trích xuất keypoints từ video")
    print("  [3] train         — Huấn luyện mô hình")
    print("  [4] evaluate      — Đánh giá mô hình (Metrics + Biểu đồ)")
    print("  [5] dataset_check — Kiểm tra dataset và dataloader")
    print("  [6] demo          — Demo keypoints từ webcam")
    print("  [7] app           — Ứng dụng inference realtime (CLI)")
    print("  [8] gui           — Giao diện ứng dụng (GUI)")
    print("  [9] exit          — Thoát")
    return input("\n  Nhập lựa chọn (1-9): ").strip()


# =====================================================================
# Các hàm dispatcher — mỗi hàm gọi đúng 1 module
# =====================================================================
def setup_model_main():
    """Kiểm tra kiến trúc model với dữ liệu giả."""
    from dataset import VSLDataset
    from model import SignLanguageLSTM
    from config import METADATA_PATH, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS
    import torch
    import torch.nn as nn

    print("\n[INFO] Kiểm tra model theo metadata thực tế...")
    dataset = VSLDataset(METADATA_PATH)
    num_classes = dataset.num_classes

    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    )

    dummy_input = torch.randn(4, 30, INPUT_SIZE)
    dummy_labels = torch.randint(0, num_classes, (4,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[INFO] Số lớp từ metadata : {num_classes}")
    print(f"[INFO] Output shape       : {tuple(output.shape)}")
    print(f"[DONE] Model hoạt động bình thường. Loss={loss.item():.4f}")


def extract_data_main():
    """Trích xuất keypoints từ video."""
    import extract_data
    extract_data.main()


def train_main():
    """Huấn luyện mô hình."""
    import train
    train.main()


def evaluate_main():
    """Đánh giá mô hình trên tập Test."""
    import evaluate
    evaluate.main()


def dataset_check_main():
    """Kiểm tra dataset và dataloader."""
    import dataset
    dataset.main()


def demo_main():
    """Demo keypoints từ webcam."""
    import demo_keypoints
    demo_keypoints.main()


def app_main():
    """Ứng dụng inference realtime (CLI)."""
    import app
    app.main()


def gui_main():
    """Giao diện ứng dụng (GUI)."""
    import gui
    gui.main()


def exit_main():
    """Thoát chương trình."""
    print("[DONE] Đã thoát chương trình.")
    sys.exit(0)


# =====================================================================
# Bảng dispatch — ánh xạ lựa chọn → hàm xử lý
# =====================================================================
ACTIONS = {
    "1": ("setup_model",   setup_model_main),
    "2": ("extract_data",  extract_data_main),
    "3": ("train",         train_main),
    "4": ("evaluate",      evaluate_main),
    "5": ("dataset_check", dataset_check_main),
    "6": ("demo",          demo_main),
    "7": ("app",           app_main),
    "8": ("gui",           gui_main),
    "9": ("exit",          exit_main),
}


# =====================================================================
# Entry point
# =====================================================================
def main():
    print("\n  WELCOME TO VIETNAMESE SIGN LANGUAGE RECOGNITION SYSTEM")

    while True:
        choice = show_main_menu()

        if choice not in ACTIONS:
            print("[WARNING] Lựa chọn không hợp lệ. Vui lòng chọn 1-9.")
            continue

        name, action = ACTIONS[choice]
        try:
            action()
        except Exception as e:
            print(f"[ERROR] Lỗi khi chạy '{name}': {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Chương trình đã dừng theo yêu cầu người dùng.")
        sys.exit(0)
