"""
train.py — Huấn luyện mô hình LSTM nhận diện ngôn ngữ ký hiệu Việt Nam.

Xử lý dữ liệu mất cân bằng bằng:
  - Weighted CrossEntropyLoss (tự động tính trọng số theo tần suất lớp)
  - Data augmentation (thêm nhiễu, co giãn, dịch thời gian)
  - Oversampling các lớp thiểu số

Cách chạy:
    python src/train.py
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    RESULTS_DIR,
    NUM_WORKERS,
    RANDOM_SEED,
    ensure_project_dirs,
)
from dataset import get_dataloaders, VSLDataset
from model import SignLanguageLSTM


# =====================================================================
# Data Augmentation cho chuỗi keypoints
# =====================================================================
def augment_sequence(sequence):
    """Áp dụng augmentation ngẫu nhiên cho 1 chuỗi keypoints.

    Các phép biến đổi:
      - Thêm nhiễu Gaussian nhỏ
      - Co giãn biên độ ngẫu nhiên
      - Dịch chuyển thời gian (time shift)
    """
    seq = sequence.clone()

    # 1. Nhiễu Gaussian (50% xác suất)
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(seq) * 0.02
        seq = seq + noise

    # 2. Co giãn biên độ (50% xác suất)
    if torch.rand(1).item() > 0.5:
        scale = 0.9 + torch.rand(1).item() * 0.2  # [0.9, 1.1]
        seq = seq * scale

    # 3. Dịch thời gian — xáo trộn nhẹ thứ tự frame (30% xác suất)
    if torch.rand(1).item() > 0.7:
        shift = torch.randint(-2, 3, (1,)).item()
        if shift != 0:
            seq = torch.roll(seq, shifts=shift, dims=0)

    return seq


# =====================================================================
# Tính trọng số lớp cho dữ liệu mất cân bằng
# =====================================================================
def compute_class_weights(dataset):
    """Tính trọng số nghịch đảo tần suất cho từng lớp.

    Lớp có ít mẫu → trọng số lớn hơn → model tập trung học lớp đó nhiều hơn.
    """
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=dataset.dataset.num_classes
                               if hasattr(dataset, 'dataset') else max(labels) + 1)

    # Tránh chia cho 0
    class_counts = np.maximum(class_counts, 1)

    # Trọng số = tổng mẫu / (số lớp × số mẫu mỗi lớp)
    total = len(labels)
    num_classes = len(class_counts)
    weights = total / (num_classes * class_counts)

    return torch.FloatTensor(weights)


def compute_sample_weights(dataset, num_classes):
    """Tính trọng số cho từng mẫu để dùng với WeightedRandomSampler.

    Giúp oversampling các lớp thiểu số trong quá trình training.
    """
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return torch.DoubleTensor(sample_weights)


# =====================================================================
# Train 1 epoch
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, use_augment=True):
    """Chạy 1 epoch huấn luyện, trả về (loss trung bình, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        # Áp dụng augmentation nếu bật
        if use_augment:
            augmented = []
            for seq in sequences:
                augmented.append(augment_sequence(seq))
            sequences = torch.stack(augmented)

        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — giảm hiện tượng gradient bùng nổ
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


# =====================================================================
# Validate 1 epoch
# =====================================================================
@torch.no_grad()
def validate(model, loader, criterion, device):
    """Chạy validation (không augmentation), trả về (loss trung bình, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


# =====================================================================
# Pipeline huấn luyện chính
# =====================================================================
def train(num_epochs=NUM_EPOCHS, patience=15):
    """
    Pipeline huấn luyện đầy đủ:
      1. Tải dữ liệu và phân chia Train / Val / Test
      2. Tính trọng số lớp cho dữ liệu mất cân bằng
      3. Khởi tạo model, optimizer, scheduler
      4. Vòng lặp epoch: train (có augmentation) → validate → early stop
      5. Lưu best model và training history
    """
    ensure_project_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Tải dữ liệu ---
    print("\n" + "=" * 60)
    print("TẢI DỮ LIỆU")
    print("=" * 60)
    train_loader, val_loader, test_loader, dataset = get_dataloaders()
    num_classes = dataset.num_classes
    label_map = dataset.get_label_map()

    print(f"\n[INFO] Thiết bị         : {device}")
    print(f"[INFO] Số lớp           : {num_classes}")
    print(f"[INFO] Danh sách nhãn   : {list(label_map.values())}")

    # --- 2. Tính trọng số cho dữ liệu mất cân bằng ---
    print("\n[INFO] Tính trọng số lớp cho dữ liệu mất cân bằng...")
    class_weights = compute_class_weights(train_loader.dataset)
    class_weights = class_weights.to(device)
    print(f"[INFO] Trọng số lớp     : {class_weights.cpu().numpy().round(3)}")

    # Tạo WeightedRandomSampler cho oversampling
    sample_weights = compute_sample_weights(train_loader.dataset, num_classes)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=max(len(train_loader.dataset), num_classes * 10),
        replacement=True,
    )

    # Tạo lại train_loader với sampler (oversampling)
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --- 3. Khởi tạo model ---
    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Tổng tham số     : {total_params:,}")

    # --- 4. Vòng lặp huấn luyện ---
    print("\n" + "=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>11} {'Train Acc':>10} | "
          f"{'Val Loss':>9} {'Val Acc':>9} | {'LR':>10} | {'Thời gian':>9}")
    print("-" * 80)

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train (với augmentation)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_augment=True
        )

        # Validate (không augmentation)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Cập nhật learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Lưu lịch sử
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": current_lr,
            "time": round(epoch_time, 1),
        }
        history.append(entry)

        # In kết quả
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            marker = " << best"

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "num_classes": num_classes,
                "label_map": label_map,
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1

        print(
            f"{epoch:5d}  | "
            f"{train_loss:11.4f} {train_acc:9.2f}% | "
            f"{val_loss:9.4f} {val_acc:8.2f}% | "
            f"{current_lr:10.1e} | "
            f"{epoch_time:8.1f}s{marker}"
        )

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOP] Không cải thiện sau {patience} epoch liên tiếp.")
            break

    # --- 5. Lưu kết quả ---
    total_time = time.time() - start_time
    history_path = RESULTS_DIR / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Epoch tốt nhất    : {best_epoch}")
    print(f"  Val Accuracy      : {best_val_acc:.2f}%")
    print(f"  Tổng thời gian    : {total_time:.1f}s ({total_time / 60:.1f} phút)")
    print(f"  Model đã lưu tại  : {BEST_MODEL_PATH}")
    print(f"  Lịch sử lưu tại   : {history_path}")
    print("[DONE] Huấn luyện hoàn tất.")


# =====================================================================
# Entry point
# =====================================================================
def main():
    train()


if __name__ == "__main__":
    main()