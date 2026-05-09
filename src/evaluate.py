import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

from dataset import get_dataloaders
from model import SignLanguageLSTM
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    BEST_MODEL_PATH,
    RESULTS_DIR,
    ensure_project_dirs,
)


# =====================================================================
# Load mô hình đã huấn luyện
# =====================================================================
def load_best_model(dataset, device):
    """Load mô hình từ file best_model.pth."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Không tìm thấy file weights: {BEST_MODEL_PATH}\n"
            "[HINT] Hãy huấn luyện mô hình trước (option 3)."
        )

    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=dataset.num_classes,
    ).to(device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[INFO] Đã load model từ : {BEST_MODEL_PATH}")
    print(f"[INFO] Best epoch       : {checkpoint.get('epoch', '?')}")
    print(f"[INFO] Best val_acc     : {checkpoint.get('val_accuracy', '?'):.2f}%")

    return model


# =====================================================================
# Chạy dự đoán trên toàn bộ test set
# =====================================================================
def predict_all(model, loader, device):
    """Chạy inference trên toàn bộ loader, trả về y_true và y_pred."""
    all_true = []
    all_pred = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    return np.array(all_true), np.array(all_pred)


# =====================================================================
# In bảng Classification Report
# =====================================================================
def print_report(y_true, y_pred, label_names):
    """In Precision, Recall, F1-Score cho từng lớp."""
    all_labels = list(range(len(label_names)))
    report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=label_names,
        digits=4, zero_division=0
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Lưu ra file text
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    print(f"[INFO] Đã lưu report → {report_path}")
    return report


# =====================================================================
# Vẽ Confusion Matrix
# =====================================================================
def plot_confusion_matrix(y_true, y_pred, label_names):
    """Vẽ heatmap Confusion Matrix và lưu ảnh."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))

    fig, ax = plt.subplots(figsize=(max(8, len(label_names)), max(6, len(label_names) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Dự đoán (Predicted)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Thực tế (Actual)", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    save_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Đã lưu Confusion Matrix → {save_path}")


# =====================================================================
# Vẽ biểu đồ Precision / Recall / F1 cho từng lớp
# =====================================================================
def plot_per_class_metrics(y_true, y_pred, label_names):
    """Vẽ grouped bar chart cho Precision, Recall, F1-Score."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), zero_division=0
    )

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(label_names) * 1.5), 6))

    bars_p = ax.bar(x - width, precision, width, label="Precision", color="#4C72B0", edgecolor="white")
    bars_r = ax.bar(x, recall, width, label="Recall", color="#55A868", edgecolor="white")
    bars_f = ax.bar(x + width, f1, width, label="F1-Score", color="#C44E52", edgecolor="white")

    # Thêm giá trị lên đầu mỗi cột
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Nhãn (Label)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Điểm số", fontsize=12, fontweight="bold")
    ax.set_title("Precision / Recall / F1-Score theo từng lớp", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = RESULTS_DIR / "per_class_metrics.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Đã lưu Per-class Metrics → {save_path}")


# =====================================================================
# Vẽ biểu đồ tổng hợp Accuracy
# =====================================================================
def plot_overall_accuracy(y_true, y_pred, label_names):
    """Vẽ biểu đồ accuracy cho từng lớp (dạng horizontal bar)."""

    all_labels = list(range(len(label_names)))

    # BẮT BUỘC truyền labels để confusion matrix luôn có đủ số lớp
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Tính accuracy cho từng lớp
    per_class_acc = []
    for i in range(len(label_names)):
        total_samples_of_class = cm[i].sum()

        if total_samples_of_class > 0:
            acc = cm[i, i] / total_samples_of_class
        else:
            acc = 0.0

        per_class_acc.append(acc)

    if len(y_true) > 0:
        overall_acc = np.sum(y_true == y_pred) / len(y_true) * 100
    else:
        overall_acc = 0.0

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(label_names) * 0.6),
            max(5, len(label_names) * 0.5)
        )
    )

    colors = plt.cm.RdYlGn(np.array(per_class_acc))
    bars = ax.barh(
        label_names,
        per_class_acc,
        color=colors,
        edgecolor="white",
        height=0.6
    )

    # Thêm giá trị lên mỗi thanh
    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Accuracy theo từng lớp — Overall: {overall_acc:.2f}%",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.axvline(
        x=overall_acc / 100,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall: {overall_acc:.2f}%"
    )

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_path = RESULTS_DIR / "per_class_accuracy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Đã lưu Per-class Accuracy → {save_path}")


# =====================================================================
# Pipeline đánh giá đầy đủ
# =====================================================================
def main():

    ensure_project_dirs()

    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)

    # --- 1. Load dữ liệu ---
    print("\n[INFO] Đang tải DataLoader...")
    _, _, test_loader, dataset = get_dataloaders()

    # --- 2. Thiết lập thiết bị ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị: {device}")

    # --- 3. Load model ---
    print("[INFO] Đang load mô hình...")
    model = load_best_model(dataset, device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    best_epoch = checkpoint.get("epoch", "?")

    # --- 4. Dự đoán ---
    label_map = dataset.get_label_map()
    label_names = [label_map[i] for i in sorted(label_map.keys())]

    print("[INFO] Đang chạy dự đoán trên tập Test...")
    y_true, y_pred = predict_all(model, test_loader, device)

    overall_acc = np.sum(y_true == y_pred) / len(y_true) * 100
    print(f"[INFO] Overall Accuracy   : {overall_acc:.2f}%")

    # --- 5. In report & vẽ biểu đồ ---
    print_report(y_true, y_pred, label_names)
    plot_confusion_matrix(y_true, y_pred, label_names)
    plot_per_class_metrics(y_true, y_pred, label_names)
    plot_overall_accuracy(y_true, y_pred, label_names)

    # --- 6. Tạo dashboard HTML ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), zero_division=0
    )

    metrics = {
        "overall_accuracy": overall_acc,
        "label_names": label_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "num_test_samples": len(y_true),
        "best_epoch": best_epoch,
    }


    # --- 7. Tóm tắt ---
    print("\n" + "=" * 60)
    print("TÓM TẮT KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"  Overall Accuracy : {overall_acc:.2f}%")
    print(f"  Số lớp           : {len(label_names)}")
    print(f"  Số mẫu test      : {len(y_true)}")
    print(f"  Kết quả lưu tại  : {RESULTS_DIR}")
    print("[DONE] Đã hoàn thành đánh giá mô hình.")


if __name__ == "__main__":
    main()
