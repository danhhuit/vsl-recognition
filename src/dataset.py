# Class xây dựng PyTorch Dataset và DataLoader
# Dataset và DataLoader cho hệ thống Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt (VSL)
# Đọc dữ liệu từ file metadata.csv và các file .npy đã được trích xuất bởi extract_data.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

# =========================
# Cấu hình đường dẫn
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
KEYPOINT_DIR = BASE_DIR / "data" / "keypoints"
METADATA_PATH = KEYPOINT_DIR / "metadata.csv"

# Tham số huấn luyện mặc định
BATCH_SIZE = 16
NUM_WORKERS = 0          # Đặt 0 nếu chạy trên Windows để tránh lỗi multiprocessing
TRAIN_RATIO = 0.8        # 80% dữ liệu dùng để huấn luyện
VAL_RATIO = 0.2          # 20% dữ liệu dùng để kiểm thử


# =========================
# Class VSLDataset
# =========================
class VSLDataset(Dataset):
    """
    Dataset tùy chỉnh cho bài toán nhận diện ngôn ngữ ký hiệu Tiếng Việt.

    Mỗi mẫu dữ liệu gồm:
        - sequence: Tensor hình dạng (30, 63) — 30 frame, mỗi frame 63 tọa độ (21 khớp x 3)
        - label   : Tensor số nguyên đại diện cho lớp từ vựng tương ứng

    Cấu trúc metadata.csv (được tạo từ extract_data.py):
        file_path  | label_name | label_id | video_name
        -----------|------------|----------|------------
        ...npy     | cam_on     | 1        | video_001.mp4
    """

    def __init__(self, metadata_path, transform=None):
        """
        Khởi tạo Dataset.

        Args:
            metadata_path (str | Path): Đường dẫn đến file metadata.csv
            transform (callable, optional): Hàm biến đổi dữ liệu (tùy chọn mở rộng sau)
        """
        self.metadata_path = Path(metadata_path)
        self.transform = transform

        # Kiểm tra file metadata tồn tại
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Không tìm thấy file metadata: {self.metadata_path}\n"
                f"        Hãy chạy extract_data.py trước để tạo dữ liệu."
            )

        # Đọc toàn bộ danh sách mẫu từ CSV
        self.df = pd.read_csv(self.metadata_path, encoding="utf-8-sig")

        # Kiểm tra các cột bắt buộc
        required_cols = {"file_path", "label_name", "label_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"[ERROR] metadata.csv thiếu cột: {missing}")

        # Lọc bỏ những hàng có file .npy không tồn tại trên đĩa
        valid_mask = self.df["file_path"].apply(lambda p: Path(p).exists())
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"[WARNING] Bỏ qua {invalid_count} mẫu do file .npy không tìm thấy.")
        self.df = self.df[valid_mask].reset_index(drop=True)

        # Thống kê nhãn
        self.label_names = sorted(self.df["label_name"].unique().tolist())
        self.num_classes = len(self.label_names)

        print(f"[INFO] Tổng số mẫu hợp lệ : {len(self.df)}")
        print(f"[INFO] Số lớp từ vựng      : {self.num_classes} — {self.label_names}")

    def __len__(self):
        """Trả về tổng số mẫu trong dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu theo chỉ số idx.

        Returns:
            sequence (torch.FloatTensor): shape (30, 63)
            label    (torch.LongTensor) : số nguyên đại diện nhãn
        """
        row = self.df.iloc[idx]

        # Đọc chuỗi tọa độ từ file .npy
        sequence = np.load(row["file_path"])           # shape: (30, 63)
        sequence = sequence.astype(np.float32)

        # Chuyển sang Tensor PyTorch
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(int(row["label_id"]), dtype=torch.long)

        # Áp dụng transform nếu có (ví dụ: augmentation, normalization)
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)

        return sequence_tensor, label_tensor

    def get_label_map(self):
        """
        Trả về dictionary ánh xạ label_id -> label_name.
        Dùng để hiển thị kết quả dự đoán dưới dạng chữ.

        Returns:
            dict: {0: 'bang_chu_cai', 1: 'cam_on', ...}
        """
        label_map = self.df[["label_id", "label_name"]].drop_duplicates()
        return dict(zip(label_map["label_id"], label_map["label_name"]))


# =========================
# Hàm tạo DataLoader
# =========================
def get_dataloaders(
    metadata_path=METADATA_PATH,
    batch_size=BATCH_SIZE,
    train_ratio=TRAIN_RATIO,
    num_workers=NUM_WORKERS,
    seed=42,
):
    """
    Tạo DataLoader cho tập Train và Validation từ metadata.csv.

    Args:
        metadata_path (Path): Đường dẫn file metadata.csv
        batch_size    (int) : Số mẫu trong mỗi batch
        train_ratio   (float): Tỷ lệ dữ liệu dùng để huấn luyện (0.0 - 1.0)
        num_workers   (int) : Số luồng đọc dữ liệu song song
        seed          (int) : Hạt giống ngẫu nhiên để tái hiện kết quả

    Returns:
        train_loader (DataLoader): DataLoader cho tập huấn luyện
        val_loader   (DataLoader): DataLoader cho tập kiểm thử
        dataset      (VSLDataset): Dataset gốc (dùng để lấy label_map, num_classes)
    """
    # Khởi tạo dataset đầy đủ
    dataset = VSLDataset(metadata_path=metadata_path)

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = total - train_size

    # Chia ngẫu nhiên theo seed cố định để kết quả có thể tái hiện
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"[INFO] Tập huấn luyện : {train_size} mẫu")
    print(f"[INFO] Tập kiểm thử   : {val_size} mẫu")

    # Tạo DataLoader cho tập Train — shuffle để mô hình không học theo thứ tự
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),   # Tăng tốc nếu có GPU
        drop_last=True,                          # Bỏ batch cuối nếu không đủ kích thước
    )

    # Tạo DataLoader cho tập Validation — không shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, dataset


# =========================
# Kiểm thử nhanh (chạy trực tiếp file này)
# =========================
if __name__ == "__main__":
    print("=" * 55)
    print("  Kiểm thử Dataset và DataLoader — Dự án VSL")
    print("=" * 55)

    # Tạo DataLoader
    train_loader, val_loader, dataset = get_dataloaders()

    # Hiển thị ánh xạ nhãn
    label_map = dataset.get_label_map()
    print(f"\n[INFO] Ánh xạ nhãn: {label_map}")

    # Lấy thử 1 batch từ tập Train để kiểm tra shape
    print("\n--- Kiểm tra batch đầu tiên (Train) ---")
    for sequences, labels in train_loader:
        print(f"  sequences.shape : {sequences.shape}")   # Kỳ vọng: (batch, 30, 63)
        print(f"  labels.shape    : {labels.shape}")      # Kỳ vọng: (batch,)
        print(f"  labels mẫu      : {labels[:8].tolist()}")
        break

    print("\n[DONE] Dataset và DataLoader hoạt động bình thường.")
    # Alias để tương thích với train.py
    SignLanguageDataset = VSLDataset