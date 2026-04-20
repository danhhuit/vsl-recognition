# Class định nghĩa kiến trúc mạng nơ-ron LSTM
import torch
import torch.nn as nn


class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Khởi tạo mô hình
        :param input_size: Số lượng features đầu vào của mỗi frame (ví dụ: 21 điểm x 3 tọa độ xyz = 63)
        :param hidden_size: Số lượng node ẩn trong mạng LSTM
        :param num_layers: Số lượng lớp LSTM xếp chồng lên nhau
        :param num_classes: Số lượng từ vựng đầu ra (output_size)
        """
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. Lớp LSTM xử lý chuỗi thời gian
        # batch_first=True ĐỂ ĐẢM BẢO tensor có dạng [batch_size, sequence_length, features] theo đúng ghi chú
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 2. Lớp Linear (Fully Connected) nối từ đầu ra LSTM để phân loại
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x có shape: [batch_size, sequence_length, features]

        # Khởi tạo trạng thái ẩn (hidden state) và trạng thái tế bào (cell state)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Đưa dữ liệu qua mạng LSTM
        out, _ = self.lstm(x, (h0, c0))

        # out có shape: [batch_size, sequence_length, hidden_size]
        # BƯỚC QUAN TRỌNG: Chỉ lấy output ở bước thời gian cuối cùng (khung hình cuối) để đưa vào lớp Linear
        out = out[:, -1, :]

        # Đưa qua lớp Linear để ra kết quả dự đoán
        out = self.fc(out)

        return out


# --- BỔ SUNG PHẦN TEST DUMMY DATA THEO YÊU CẦU ---
if __name__ == "__main__":
    # Các thông số đã chốt từ file dataset.py và extract_data.py
    BATCH_SIZE = 16  # Giả lập batch_size giống DataLoader
    SEQUENCE_LENGTH = 30  # SEQ_LEN từ extract_data.py
    INPUT_SIZE = 63  # 21 điểm x 3 tọa độ (x,y,z)
    HIDDEN_SIZE = 128  # Số node ẩn (có thể tùy chỉnh sau)
    NUM_LAYERS = 2  # Số lớp LSTM
    NUM_CLASSES = 5  # Dựa vào len(label_names) trong data hiện tại

    print("1. Đang khởi tạo mô hình SignLanguageLSTM...")
    model = SignLanguageLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                             num_layers=NUM_LAYERS, num_classes=NUM_CLASSES)

    print("\n2. Tạo Tensor giả (Dummy Data)...")
    # Yêu cầu tensor đầu vào phải có dạng: [batch_size, sequence_length, features]
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
    print(f"   -> Kích thước Input: {dummy_input.shape}")

    print("\n3. Chạy Forward Pass...")
    output = model(dummy_input)
    print(f"   -> Kích thước Output: {output.shape}")

    # Kiểm tra xem output có khớp kích thước (BATCH_SIZE, NUM_CLASSES) hay không
    if output.shape == (BATCH_SIZE, NUM_CLASSES):
        print("\n=> TEST THÀNH CÔNG! Không có lỗi về Shape Tensor. Mô hình đã sẵn sàng cho Training.")
    else:
        print("\n=> LỖI: Kích thước Output không như kỳ vọng!")