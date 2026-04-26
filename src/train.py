import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import SignLanguageLSTM
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    LEARNING_RATE
)

# Khai báo số lượng vòng lặp (Epochs)
NUM_EPOCHS = 50


def main():
    print("1. Đang tải DataLoader...")
    # Lấy DataLoader từ file dataset.py bạn đã có
    train_loader, val_loader, dataset = get_dataloaders()

    # Thiết lập thiết bị: Tự động nhận diện GPU nếu có, không thì chạy CPU (như ghi chú)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu huấn luyện trên thiết bị: {device}")

    print("\n2. Khởi tạo mô hình SignLanguageLSTM...")
    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=dataset.num_classes
    ).to(device)

    # Thiết lập hàm tính lỗi và thuật toán tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n3. Bắt đầu vòng lặp huấn luyện...\n")
    print("-" * 50)

    # =================================================================
    # Ý 1: VIẾT VÒNG LẶP HUẤN LUYỆN ĐI QUA NHIỀU EPOCHS
    # =================================================================
    for epoch in range(NUM_EPOCHS):
        # Chuyển mô hình sang chế độ huấn luyện (bật tính năng cập nhật trọng số)
        model.train()

        # Duyệt qua từng lô dữ liệu (batch) trong tập train
        for sequences, labels in train_loader:
            # Đưa dữ liệu vào cùng thiết bị với mô hình (CPU hoặc GPU)
            sequences, labels = sequences.to(device), labels.to(device)

            # =================================================================
            # Ý 2: THỰC HIỆN CHUẨN CÁC BƯỚC CỦA PYTORCH
            # =================================================================

            # Bước 1: zero_grad - Xóa bỏ các gradient (đạo hàm) cũ tích lũy từ batch trước
            optimizer.zero_grad()

            # Bước 2: forward - Đưa dữ liệu (sequences) qua mô hình để lấy dự đoán
            outputs = model(sequences)

            # Bước 3: Tính toán sai số (Loss) giữa dự đoán và nhãn thực tế
            loss = criterion(outputs, labels)

            # Bước 4: loss.backward() - Lan truyền ngược để tính toán đạo hàm cho các trọng số
            loss.backward()

            # Bước 5: optimizer.step() - Dựa vào đạo hàm vừa tính, cập nhật lại các trọng số
            optimizer.step()

        # In tạm kết quả của batch cuối cùng trong mỗi Epoch để theo dõi
        print(f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | Loss (batch cuối): {loss.item():.4f}")

    print("\n[DONE] Đã hoàn thành vòng lặp huấn luyện cơ bản.")


if __name__ == "__main__":
    main()