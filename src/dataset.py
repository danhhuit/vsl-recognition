import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

from config import (
    METADATA_PATH,
    BATCH_SIZE,
    TRAIN_RATIO,
    TEST_RATIO,
    NUM_WORKERS,
    RANDOM_SEED,
    SEQ_LEN,
    INPUT_SIZE,
)


class VSLDataset(Dataset):
    """Dataset đọc metadata.csv và các file keypoints .npy."""

    REQUIRED_COLUMNS = {
        "file_path",
        "label_name",
        "label_id",
        "video_name",
        "num_frames_raw",
        "num_detected_frames",
        "status",
    }

    def __init__(self, metadata_path=METADATA_PATH, transform=None):
        self.metadata_path = Path(metadata_path)
        self.transform = transform

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Không tìm thấy metadata: {self.metadata_path}\n"
                "[HINT] Hãy chạy extract_data.py trước."
            )

        self.df = pd.read_csv(self.metadata_path, encoding="utf-8-sig")

        missing_cols = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"[ERROR] metadata.csv thiếu cột: {sorted(missing_cols)}")

        valid_rows = []
        skipped_missing_file = 0

        for _, row in self.df.iterrows():
            file_path = Path(row["file_path"])
            if not file_path.exists():
                skipped_missing_file += 1
                continue
            valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if skipped_missing_file:
            print(f"[WARNING] Bỏ qua {skipped_missing_file} mẫu do thiếu file .npy.")

        if self.df.empty:
            raise ValueError("[ERROR] Không còn mẫu hợp lệ nào sau khi kiểm tra metadata.")

        self.label_map = self._build_label_map()
        self.num_classes = len(self.label_map)

        print(f"[INFO] Tổng số mẫu hợp lệ  : {len(self.df)}")
        print(f"[INFO] Số lớp validation   : {self.num_classes}")
        print(f"[INFO] Danh sách nhãn      : {list(self.label_map.values())}")

    def _build_label_map(self):
        label_df = (
            self.df[["label_id", "label_name"]]
            .drop_duplicates()
            .sort_values("label_id")
        )
        return dict(zip(label_df["label_id"], label_df["label_name"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = np.load(row["file_path"]).astype(np.float32)

        expected_shape = (SEQ_LEN, INPUT_SIZE)
        if sequence.shape != expected_shape:
            raise ValueError(
                f"[ERROR] File keypoints sai shape: {row['file_path']}\n"
                f"        Expected: {expected_shape}, Actual: {sequence.shape}"
            )

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(int(row["label_id"]), dtype=torch.long)

        if self.transform is not None:
            sequence_tensor = self.transform(sequence_tensor)

        return sequence_tensor, label_tensor

    def get_label_map(self):
        return self.label_map


def get_dataloaders(
    metadata_path=METADATA_PATH,
    batch_size=BATCH_SIZE,
    train_ratio=TRAIN_RATIO,
    test_ratio=TEST_RATIO,
    num_workers=NUM_WORKERS,
    seed=RANDOM_SEED,
):
    """Chia dataset thành 3 phần: Train / Validation / Test."""
    dataset = VSLDataset(metadata_path=metadata_path)

    total = len(dataset)
    train_size = int(total * train_ratio)
    test_size = int(total * test_ratio)
    val_size = total - train_size - test_size

    if train_size == 0 or val_size == 0 or test_size == 0:
        raise ValueError(
            f"[ERROR] Không thể chia train/val/test với tổng mẫu = {total}.\n"
            f"        train={train_size}, val={val_size}, test={test_size}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"[INFO] Số mẫu train        : {train_size}")
    print(f"[INFO] Số mẫu validation   : {val_size}")
    print(f"[INFO] Số mẫu test         : {test_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, dataset


SignLanguageDataset = VSLDataset


def main():
    print("\n" + "=" * 60)
    print("KIỂM TRA DATASET & DATALOADER")
    print("=" * 60)

    train_loader, val_loader, test_loader, dataset = get_dataloaders()
    print(f"[INFO] Label map          : {dataset.get_label_map()}")

    for sequences, labels in train_loader:
        print(f"[INFO] Train batch shape  : {tuple(sequences.shape)}")
        print(f"[INFO] Label batch shape  : {tuple(labels.shape)}")
        break

    print("[DONE] Dataset và DataLoader hoạt động bình thường.")


if __name__ == "__main__":
    main()
