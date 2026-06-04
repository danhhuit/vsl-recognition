# 🤟 Vietnamese Sign Language (VSL) Recognition

Hệ thống nhận diện Ngôn ngữ Ký hiệu Việt Nam sử dụng **MediaPipe (trích xuất đặc trưng tay)** và **Mạng LSTM (nhận diện chuỗi thời gian)**. Dự án được thiết kế với giao diện web thân thiện qua Streamlit, giúp bạn dễ dàng thu thập dữ liệu, huấn luyện mô hình và dự đoán realtime mà không cần gõ lệnh.

---

## ⚡ Khởi chạy nhanh (Dành cho người mới)

Chỉ cần một thao tác đơn giản để cài đặt toàn bộ môi trường và mở ứng dụng:

1. Double-click vào file **`run.bat`** (trên Windows).
2. Chờ quá trình cài đặt tự động diễn ra (Lần đầu tiên có thể mất ~5-10 phút để tải PyTorch và các thư viện).
3. Trình duyệt sẽ tự động mở giao diện web tại địa chỉ `http://localhost:8501`.

> 💡 **Lưu ý:** Script sẽ tự động kiểm tra Python, tạo môi trường ảo `.venv`, cài đặt requirements và khởi chạy Streamlit. Lần chạy thứ 2 trở đi sẽ khởi động ngay lập tức chưa tới 5 giây.

---

## 📁 Cấu trúc thư mục

```text
deep_learning/
├── run.bat                   # ⚡ Khởi chạy nhanh bằng Command Prompt
├── run.ps1                   # ⚡ Khởi chạy nhanh bằng PowerShell
├── data/
│   ├── raw_videos/           # Thư mục chứa video gốc phân loại theo nhãn
│   └── keypoints/            # Thư mục chứa tọa độ bàn tay (.npy) đã trích xuất
├── weights/
│   └── best_model.pth        # File lưu trọng số mô hình tốt nhất sau khi train
├── results/                  # Biểu đồ, ma trận nhầm lẫn và file lịch sử train
├── src/
│   ├── app.py                # Giao diện web chính bằng Streamlit
│   ├── config.py             # File cấu hình (Hyperparameters, Paths)
│   ├── dataset.py            # Logic load data và chia tập Train/Val/Test
│   ├── evaluate.py           # Sinh báo cáo đánh giá và biểu đồ
│   ├── extract_data.py       # Quét raw_videos và dùng MediaPipe lưu ra .npy
│   ├── model.py              # Định nghĩa kiến trúc mạng Neural Network (LSTM)
│   ├── record_data.py        # Script phụ trợ để thu thập video từ Webcam
│   └── train.py              # Logic huấn luyện mô hình (Early stopping, LR scheduler)
└── requirements.txt          # Danh sách thư viện Python cần thiết
```

---

## 🚀 Cài đặt thủ công (Dành cho Developer)

Nếu bạn không muốn dùng `run.bat` hoặc đang dùng Linux/macOS, hãy chạy các lệnh sau trong Terminal:

```bash
# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên Windows CMD:
.venv\Scripts\activate.bat
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Cài đặt PyTorch (Bản CPU để nhẹ máy)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Cài đặt các thư viện còn lại
pip install -r requirements.txt

# 5. Khởi chạy ứng dụng
streamlit run src/app.py
```

---

## 📖 Quy trình 5 bước huấn luyện mô hình của riêng bạn

Toàn bộ quá trình từ dữ liệu thô đến mô hình nhận diện realtime:

### 1. Thu thập dữ liệu (Record)
- Mở terminal, chạy `python src/record_data.py`
- Nhập tên nhãn (vd: `chao`, `cam_on`) và làm theo hướng dẫn để ghi hình.
- Hoặc bạn có thể tự quay video bằng điện thoại và chép vào thư mục `data/raw_videos/<tên_nhãn>/` (Đề xuất tối thiểu 50 video/nhãn).

### 2. Trích xuất đặc trưng (Extract)
- Mở giao diện Web → Vào menu **Quy trình xử lý**.
- Bấm **Chạy Extract Data** để MediaPipe quét video và chuyển đổi bàn tay thành tọa độ số (`.npy`).

### 3. Huấn luyện mô hình (Train)
- Trên giao diện Web → **Quy trình xử lý** → Bấm **Chạy Train**.
- Theo dõi quá trình hội tụ của Loss/Accuracy. Mô hình tốt nhất sẽ tự lưu vào `weights/best_model.pth`.

### 4. Đánh giá (Evaluate)
- Bấm **Chạy Evaluate** để sinh ra các biểu đồ (Confusion Matrix, Precision/Recall) trong mục **Kết quả đánh giá**.

### 5. Sử dụng thực tế (Realtime Inference)
- Chuyển sang menu **Nhận diện Realtime**, bật camera và bắt đầu ra dấu!

---

## ⚙️ Cấu hình Model (src/config.py)

Bạn có thể tinh chỉnh các Hyperparameters trong file `config.py` để tối ưu kết quả:

| Biến số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `SEQ_LEN` | 30 | Số lượng frame (khung hình) đầu vào cho LSTM. |
| `HIDDEN_SIZE` | 128 | Số lượng nơ-ron ẩn trong mỗi lớp LSTM. Tăng nếu data nhiều. |
| `NUM_LAYERS` | 2 | Số lớp LSTM xếp chồng. |
| `BATCH_SIZE` | 16 | Cập nhật trọng số sau mỗi 16 mẫu. |
| `LEARNING_RATE` | 1e-3 | Tốc độ học của Optimizer Adam. |
| `NUM_EPOCHS` | 150 | Số vòng huấn luyện tối đa. (Sẽ tự dừng sớm nếu Model không tiến bộ). |

---

## 🔧 Xử lý sự cố thường gặp (Troubleshooting)

**1. Lỗi không mở được Webcam khi chạy Realtime**
- **Nguyên nhân:** Có phần mềm khác (Zoom, OBS, Zalo) đang dùng Camera, hoặc lỗi driver.
- **Cách sửa:** Đóng ứng dụng đó lại. Trên Windows, thử rút cáp Webcam cắm lại hoặc thay đổi cổng USB.

**2. Quá trình Extract hoặc Train báo lỗi "No model found"**
- **Nguyên nhân:** Bạn chưa chạy đủ các bước tuần tự.
- **Cách sửa:** Đảm bảo phải chạy **Extract** (để có file metadata.csv) -> **Train** (để có best_model.pth) -> thì mới chạy được **Realtime** hay **Evaluate**.

**3. Nhận diện sai tay Trái / Phải**
- **Cách sửa:** Vấn đề này đã được fix ở bản cập nhật mới nhất. Vui lòng đảm bảo `app.py` và `extract_data.py` xử lý trục X đồng nhất (không lật ngược tùy tiện).

**4. Lỗi UnicodeEncodeError (Charmap)**
- **Nguyên nhân:** Windows Console không hỗ trợ in ký tự tiếng Việt mặc định.
- **Cách sửa:** Script đã tích hợp hàm ép kiểu UTF-8 ở đầu file. Nếu bạn viết code mới, hãy chèn block `import sys, io` giống `train.py`.
