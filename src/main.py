import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def show_main_menu():
    print("\n" + "=" * 72)
    print("SIGN LANGUAGE RECOGNITION - MAIN MENU")
    print("=" * 72)
    print("[1] setup_model    - Kiểm tra kiến trúc model")
    print("[2] extract_data   - Trích xuất keypoints từ video")
    print("[3] train          - Huấn luyện mô hình")
    print("[4] dataset_check  - Kiểm tra dataset và dataloader")
    print("[5] demo           - Demo keypoints từ webcam")
    print("[6] app            - Ứng dụng inference realtime")
    print("[7] exit           - Thoát")
    return input("\nNhập lựa chọn (1-7): ").strip()


def setup_model_main():
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
    import extract_data
    extract_data.main()


def train_main():
    import train
    train.main()


def dataset_check_main():
    import dataset
    dataset.main()


def demo_main():
    import demo_keypoints
    demo_keypoints.main()


def app_main():
    import app
    app.main()


def main():
    print("\nWELCOME TO VIETNAMESE SIGN LANGUAGE RECOGNITION SYSTEM")

    while True:
        choice = show_main_menu()
        try:
            if choice == "1":
                setup_model_main()
            elif choice == "2":
                extract_data_main()
            elif choice == "3":
                train_main()
            elif choice == "4":
                dataset_check_main()
            elif choice == "5":
                demo_main()
            elif choice == "6":
                app_main()
            elif choice == "7":
                print("[DONE] Đã thoát chương trình.")
                break
            else:
                print("[WARNING] Lựa chọn không hợp lệ.")
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Chương trình đã dừng theo yêu cầu người dùng.")
        sys.exit(0)
