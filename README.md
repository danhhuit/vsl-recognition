# 🤟 Vietnamese Sign Language Recognition

Hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam sử dụng **LSTM + MediaPipe**.

## ⚡ Khởi chạy nhanh (Dành cho người mới)

**Chỉ cần 1 bước:**

1. Double-click file `run.bat` (Windows)
2. Chờ cài đặt tự động (lần đầu ~5-10 phút)
3. Trình duyệt sẽ tự mở giao diện web

> Không cần biết lập trình. Script sẽ tự tạo môi trường ảo, cài thư viện, và khởi chạy ứng dụng.

## 📁 Cấu trúc dự án

```
deep_learning/
├── run.bat                   # ⚡ Double-click để chạy
├── run.ps1                   # PowerShell alternative
├── data/
│   ├── raw_videos/           # Video gốc theo từng nhãn
│   └── keypoints/            # Keypoints đã trích xuất (.npy)
├── weights/
│   └── best_model.pth        # Model tốt nhất
├── results/                  # Kết quả đánh giá + training history
├── src/
│   ├── config.py             # Cấu hình chung
│   ├── model.py              # Kiến trúc LSTM
│   ├── dataset.py            # Dataset & DataLoader
│   ├── extract_data.py       # Trích xuất keypoints từ video
│   ├── train.py              # Huấn luyện mô hình
│   ├── evaluate.py           # Đánh giá mô hình
│   ├── record_samples.py     # Thu thập video mẫu từ webcam
│   └── app.py                # Giao diện Streamlit
└── requirements.txt
```

## 🚀 Cài đặt thủ công

```bash
# Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# Cài PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện khác
pip install -r requirements.txt
```

## 📖 Hướng dẫn sử dụng

### Bước 1: Thu thập video mẫu
```bash
python src/record_samples.py
```
Đặt video vào `data/raw_videos/<tên_nhãn>/` (≥ 20 video/nhãn).

### Bước 2: Trích xuất keypoints
```bash
python src/extract_data.py
```

### Bước 3: Huấn luyện
```bash
python src/train.py
```

### Bước 4: Đánh giá
```bash
python src/evaluate.py
```

### Bước 5: Mở giao diện
```bash
streamlit run src/app.py
```

> 💡 **Mẹo**: Bước 2-5 có thể chạy trực tiếp trên giao diện web (tab "Quy trình xử lý") mà không cần mở terminal.

## ⚙️ Cấu hình

Chỉnh sửa trong `src/config.py`:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| SEQ_LEN | 30 | Số frame mỗi chuỗi |
| HIDDEN_SIZE | 128 | Số unit ẩn LSTM |
| NUM_LAYERS | 2 | Số lớp LSTM |
| BATCH_SIZE | 16 | Kích thước batch |
| NUM_EPOCHS | 50 | Số epoch tối đa |
| LEARNING_RATE | 1e-3 | Tốc độ học |

## 🔧 Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|----------|
| Cài đặt bị đơ / hết RAM | `pip install --no-cache-dir -r requirements.txt` |
| mediapipe lỗi protobuf | `pip install mediapipe==0.10.9 protobuf==3.20.3` |
| Webcam không mở được | Kiểm tra quyền truy cập camera, đóng ứng dụng khác đang dùng camera |
| Model chưa có | Chạy extract → train → evaluate trước khi dùng realtime |
