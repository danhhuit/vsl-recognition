import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import SignLanguageLSTM
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    BEST_MODEL_PATH,
    ensure_project_dirs,
)


# =====================================================================
# Hàm huấn luyện 1 epoch — tính Loss & Accuracy sau mỗi batch
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    """Huấn luyện 1 epoch, in Loss & Accuracy sau mỗi batch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)

    for batch_idx, (sequences, labels) in enumerate(loader, start=1):
        sequences, labels = sequences.to(device), labels.to(device)

        # --- Các bước chuẩn của PyTorch ---
        # Bước 1: Xóa gradient cũ
        optimizer.zero_grad()

        # Bước 2: Forward — đưa dữ liệu qua mô hình
        outputs = model(sequences)

        # Bước 3: Tính Loss
        loss = criterion(outputs, labels)

        # Bước 4: Backward — tính gradient
        loss.backward()

        # Bước 5: Cập nhật trọng số
        optimizer.step()

        # --- Tính toán metrics cho batch hiện tại ---
        batch_loss = loss.item()
        _, predicted = torch.max(outputs, dim=1)
        batch_correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)
        batch_acc = batch_correct / batch_size * 100

        # Tích lũy cho tổng epoch
        total_loss += batch_loss * batch_size
        total_correct += batch_correct
        total_samples += batch_size

        # In kết quả sau mỗi batch
        print(
            f"  Train | Batch [{batch_idx:02d}/{num_batches:02d}] | "
            f"Loss: {batch_loss:.4f} | Acc: {batch_acc:6.2f}%"
        )

    # Tính trung bình cho toàn epoch
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100

    print(f"  Train | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:6.2f}%")
    return avg_loss, avg_acc


# =====================================================================
# Hàm đánh giá trên tập Validation
# =====================================================================
def evaluate(model, loader, criterion, device):
    """Đánh giá mô hình trên tập Validation, trả về avg_loss và accuracy."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(loader, start=1):
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            batch_loss = loss.item()
            _, predicted = torch.max(outputs, dim=1)
            batch_correct = (predicted == labels).sum().item()
            batch_size = labels.size(0)
            batch_acc = batch_correct / batch_size * 100

            total_loss += batch_loss * batch_size
            total_correct += batch_correct
            total_samples += batch_size

            print(
                f"  Valid | Batch [{batch_idx:02d}/{num_batches:02d}] | "
                f"Loss: {batch_loss:.4f} | Acc: {batch_acc:6.2f}%"
            )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100

    print(f"  Valid | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:6.2f}%")
    return avg_loss, avg_acc


# =====================================================================
# Hàm lưu trọng số mô hình
# =====================================================================
def save_checkpoint(model, val_acc, epoch, path):
    """Lưu trọng số mô hình vào file .pth."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_accuracy": val_acc,
        },
        path,
    )
    print(f"  ★ Saved best model (val_acc: {val_acc:.2f}%) → {path}")


# =====================================================================
# Pipeline chính
# =====================================================================
def main():
    ensure_project_dirs()

    # --- 1. Tải DataLoader ---
    print("[INFO] Đang tải DataLoader...")
    train_loader, val_loader, _, dataset = get_dataloaders()

    # --- 2. Thiết lập thiết bị ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị huấn luyện: {device}")

    # --- 3. Khởi tạo mô hình ---
    print("[INFO] Khởi tạo mô hình SignLanguageLSTM...")
    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=dataset.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Vòng lặp huấn luyện ---
    print(f"[INFO] Bắt đầu huấn luyện {NUM_EPOCHS} epochs...\n")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]")

        # Huấn luyện 1 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )

        # Đánh giá trên tập Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Lưu model nếu Validation Accuracy tốt hơn
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, val_acc, epoch, BEST_MODEL_PATH)

        print("-" * 60)

    # --- 5. Tóm tắt kết quả ---
    print("\n" + "=" * 60)
    print("TÓM TẮT KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Best Epoch       : {best_epoch}")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved at   : {BEST_MODEL_PATH}")
    print("[DONE] Đã hoàn thành huấn luyện.")


if __name__ == "__main__":
    main()