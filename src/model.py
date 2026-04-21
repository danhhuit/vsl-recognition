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


# ============================================================
# PHẦN 1: CÁC HÀM HELPER ĐỂ TÁCH LOGIC
# ============================================================

def define_config(batch_size=16, sequence_length=30, hidden_size=128, num_layers=2, learning_rate=0.001):
    """
    Định nghĩa toàn bộ cấu hình cho mô hình.
    
    Returns:
        dict: Chứa INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, SEQUENCE_LENGTH, 
              HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE, và LABEL_NAMES
    """
    config = {
        "INPUT_SIZE": 63,  # 21 khớp × 3 tọa độ
        "OUTPUT_SIZE": 5,  # 5 từ vựng
        "BATCH_SIZE": batch_size,
        "SEQUENCE_LENGTH": sequence_length,
        "HIDDEN_SIZE": hidden_size,
        "NUM_LAYERS": num_layers,
        "LEARNING_RATE": learning_rate,
        "LABEL_NAMES": ["bang_chu_cai", "cam_on", "chao", "ten_la", "toi"]
    }
    
    # Print cấu hình
    print("[CONFIG] Định nghĩa cấu hình mô hình:")
    print(f"  - INPUT_SIZE: {config['INPUT_SIZE']} (21 khớp × 3 tọa độ)")
    print(f"  - OUTPUT_SIZE: {config['OUTPUT_SIZE']} từ vựng")
    print(f"  - LABEL_NAMES: {config['LABEL_NAMES']}")
    print(f"  - BATCH_SIZE: {config['BATCH_SIZE']}")
    print(f"  - SEQUENCE_LENGTH: {config['SEQUENCE_LENGTH']}")
    print(f"  - HIDDEN_SIZE: {config['HIDDEN_SIZE']}")
    print(f"  - NUM_LAYERS: {config['NUM_LAYERS']}")
    print(f"  - LEARNING_RATE: {config['LEARNING_RATE']}")
    
    return config


def setup_model(config):
    """
    Khởi tạo mô hình SignLanguageLSTM.
    
    Args:
        config (dict): Cấu hình từ define_config()
        
    Returns:
        SignLanguageLSTM: Mô hình đã được khởi tạo
    """
    print("\n" + "="*60)
    print("BƯỚC 1: Khởi tạo mô hình SignLanguageLSTM...")
    print("="*60)
    
    model = SignLanguageLSTM(
        input_size=config["INPUT_SIZE"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        num_classes=config["OUTPUT_SIZE"]
    )
    
    print("✓ Mô hình đã được khởi tạo thành công!")
    print(f"  - Input Shape: [batch_size={config['BATCH_SIZE']}, "
          f"sequence={config['SEQUENCE_LENGTH']}, features={config['INPUT_SIZE']}]")
    print(f"  - Output Shape: [batch_size={config['BATCH_SIZE']}, num_classes={config['OUTPUT_SIZE']}]")
    print(f"  - Tổng tham số: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def setup_loss():
    """
    Thiết lập Loss Function (CrossEntropyLoss).
    
    Returns:
        nn.CrossEntropyLoss: Loss function để tính toán mất mát
    """
    print("\n" + "="*60)
    print("BƯỚC 2: Thiết lập Loss Function...")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    
    print("✓ CrossEntropyLoss đã được thiết lập!")
    print("  - Loại: Multi-class Classification Loss")
    print("  - Công dụng: So sánh dự đoán (logits) với nhãn thực tế")
    print("  - Input dự đoán: [batch_size, num_classes] (logits)")
    print("  - Input nhãn: [batch_size] (class indices)")
    
    return criterion


def setup_optimizer(model, config):
    """
    Thiết lập Bộ Tối Ưu (Adam Optimizer).
    
    Args:
        model (nn.Module): Mô hình cần tối ưu
        config (dict): Cấu hình từ define_config()
        
    Returns:
        torch.optim.Adam: Bộ tối ưu Adam
    """
    print("\n" + "="*60)
    print("BƯỚC 3: Thiết lập Bộ Tối Ưu (Optimizer)...")
    print("="*60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    
    print("✓ Bộ tối ưu Adam đã được thiết lập!")
    print(f"  - Learning Rate: {config['LEARNING_RATE']}")
    print(f"  - Số tham số cần huấn luyện: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Số tham số có gradient: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return optimizer


def test_model(model, criterion, optimizer, config):
    """
    Test toàn bộ quy trình: Forward Pass, Loss, Backward, Optimizer Step.
    
    Args:
        model (nn.Module): Mô hình cần test
        criterion: Loss function
        optimizer: Bộ tối ưu
        config (dict): Cấu hình từ define_config()
    """
    print("\n" + "="*60)
    print("BƯỚC 4: Test Forward Pass, Loss, Backward, Optimizer Step...")
    print("="*60)
    
    # Tạo dữ liệu giả
    print("\n1. Tạo Dummy Data...")
    dummy_input = torch.randn(config["BATCH_SIZE"], config["SEQUENCE_LENGTH"], config["INPUT_SIZE"])
    dummy_labels = torch.randint(0, config["OUTPUT_SIZE"], (config["BATCH_SIZE"],))
    print(f"   ✓ Input shape: {dummy_input.shape}")
    print(f"   ✓ Labels shape: {dummy_labels.shape}")
    
    # Forward Pass
    print("\n2. Chạy Forward Pass...")
    output = model(dummy_input)
    print(f"   ✓ Output shape: {output.shape}")
    
    # Tính Loss
    print("\n3. Tính Loss...")
    loss = criterion(output, dummy_labels)
    print(f"   ✓ Loss value: {loss.item():.4f}")
    
    # Backward Pass
    print("\n4. Chạy Backward Pass (tính gradient)...")
    optimizer.zero_grad()
    loss.backward()
    print("   ✓ Gradient đã được tính toán!")
    
    # Optimizer Step
    print("\n5. Chạy Optimizer Step (cập nhật tham số)...")
    optimizer.step()
    print("   ✓ Tham số mô hình đã được cập nhật!")
    
    # Summary
    print("\n" + "="*60)
    print("✓ TEST HOÀN THÀNH THÀNH CÔNG!")
    print("="*60)
    print("Mô hình, Loss Function, và Optimizer đã sẵn sàng cho huấn luyện!")


if __name__ == "__main__":
    # Test các hàm (có thể xóa khi sử dụng main.py)
    config = define_config()
    model = setup_model(config)
    criterion = setup_loss()
    optimizer = setup_optimizer(model, config)
    test_model(model, criterion, optimizer, config)