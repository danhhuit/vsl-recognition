# 🤟 Vietnamese Sign Language Recognition
Live Demo: https://danhhuit-vsl-recognition-srcapp-danh-feature-vsl-cluhzc.streamlit.app/
Hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam sử dụng **LSTM + MediaPipe**.

## 📁 Cấu trúc dự án

```
deep_learning/
├── data/
│   ├── raw_videos/          # Video gốc theo từng nhãn
│   └── keypoints/           # Keypoints đã trích xuất (.npy)
├── weights/
│   └── best_model.pth       # Model tốt nhất
├── results/                  # Kết quả đánh giá + training history
├── src/
│   ├── config.py            # Cấu hình chung
│   ├── model.py             # Kiến trúc LSTM
│   ├── dataset.py           # Dataset & DataLoader
│   ├── extract_data.py      # Trích xuất keypoints từ video
│   ├── train.py             # Huấn luyện mô hình
│   ├── evaluate.py          # Đánh giá mô hình
│   └── app.py               # Giao diện Streamlit
└── requirements.txt
```

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
```

## 📖 Hướng dẫn sử dụng

### Bước 1: Trích xuất keypoints
```bash
python src/extract_data.py
```
Đặt video vào `data/raw_videos/<tên_nhãn>/` trước khi chạy.

### Bước 2: Huấn luyện
```bash
python src/train.py
```

### Bước 3: Đánh giá
```bash
python src/evaluate.py
```

### Bước 4: Mở giao diện
```bash
streamlit run src/app.py
```

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
