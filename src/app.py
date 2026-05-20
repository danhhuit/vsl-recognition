"""
app.py — Giao diện Streamlit cho hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam.

Cải tiến v2:
- Biểu đồ Plotly tương tác (zoom, hover, tooltip chi tiết).
- Giải thích biểu đồ rõ ràng ngay dưới mỗi chart.
- Tiến trình thực thi hiển thị % thực tế (parse stdout).
- UI mượt, loading spinner rõ ràng cho mọi thao tác.
- Evaluation: Confusion Matrix dạng heatmap Plotly tương tác.
- [FIXED] Giao diện pipeline: Cân bằng các nút bấm, tự động ẩn menu khi đang chạy script.
- [FIXED] Giao diện log: Cố định độ cao terminal log, chống tràn màn hình.

Yêu cầu bổ sung:
    pip install plotly

Chạy:
    streamlit run src/app.py
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import config as cfg
from dataset import VSLDataset
from model import SignLanguageLSTM

# ── Safe config reads ──────────────────────────────────────────────────────────
INPUT_SIZE            = cfg.INPUT_SIZE
HIDDEN_SIZE           = cfg.HIDDEN_SIZE
NUM_LAYERS            = cfg.NUM_LAYERS
SEQ_LEN               = cfg.SEQ_LEN
BATCH_SIZE            = cfg.BATCH_SIZE
LEARNING_RATE         = cfg.LEARNING_RATE
NUM_EPOCHS            = cfg.NUM_EPOCHS
BEST_MODEL_PATH       = cfg.BEST_MODEL_PATH
METADATA_PATH         = cfg.METADATA_PATH
RESULTS_DIR           = cfg.RESULTS_DIR
MIN_DETECTION_CONFIDENCE = cfg.MIN_DETECTION_CONFIDENCE
MIN_TRACKING_CONFIDENCE  = cfg.MIN_TRACKING_CONFIDENCE
MAX_NUM_HANDS_DEMO    = cfg.MAX_NUM_HANDS_DEMO
USE_VELOCITY          = getattr(cfg, "USE_VELOCITY", False)
_BASE_INPUT           = getattr(cfg, "_BASE_INPUT", cfg.NUM_HAND_LANDMARKS * cfg.COORDS_PER_LANDMARK)
CONFIDENCE_THRESHOLD  = getattr(cfg, "CONFIDENCE_THRESHOLD", 0.75)
MIN_HAND_FRAMES_RATIO = getattr(cfg, "MIN_HAND_FRAMES_RATIO", 0.60)

# ── Plotly ─────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#11111b",
    plot_bgcolor="#1e1e2e",
    font=dict(family="Inter, sans-serif", color="#cdd6f4", size=13), # Đã đổi font tại đây
    xaxis=dict(gridcolor="#313244", linecolor="#313244", zerolinecolor="#313244"),
    yaxis=dict(gridcolor="#313244", linecolor="#313244", zerolinecolor="#313244"),
    legend=dict(bgcolor="#1e1e2e", bordercolor="#313244", borderwidth=1),
    margin=dict(l=50, r=30, t=55, b=50),
    hoverlabel=dict(bgcolor="#1e1e2e", bordercolor="#89b4fa", font_color="#cdd6f4"),
)

st.set_page_config(
    page_title="VSL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# I18N
# ══════════════════════════════════════════════════════════════════════════════
TEXT: dict[str, dict[str, str]] = {
    "vi": {
        "app_title": "VSL Recognition",
        "app_subtitle": "Hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam sử dụng LSTM và MediaPipe",
        "language": "Ngôn ngữ",
        "navigation": "Điều hướng",
        "home": "Trang chủ",
        "pipeline": "Quy trình xử lý",
        "realtime": "Nhận diện thời gian thực",
        "training_results": "Kết quả huấn luyện",
        "evaluation": "Đánh giá mô hình",
        "history": "Lịch sử nhận diện",
        "device": "Thiết bị",
        "model_loaded": "Mô hình đã tải thành công",
        "model_not_loaded": "Chưa tải mô hình",
        "input_size": "Đầu vào",
        "seq_len": "Độ dài chuỗi",
        "hidden_size": "Ẩn LSTM",
        "num_layers": "Số lớp LSTM",
        "batch_size": "Batch size",
        "num_epochs": "Số epoch",
        "model_info": "Thông tin mô hình",
        "best_epoch": "Epoch tốt nhất",
        "val_accuracy": "Val Accuracy",
        "val_loss": "Val Loss",
        "num_classes": "Số lớp",
        "labels": "Danh sách nhãn nhận diện",
        "guide": "Hướng dẫn sử dụng",
        "loading_page": "Đang tải giao diện...",
        "run_on_ui": "Quy trình xử lý",
        "run_extract": "Trích xuất dữ liệu",
        "run_train": "Huấn luyện mô hình",
        "run_evaluate": "Đánh giá mô hình",
        "running_extract": "Đang trích xuất keypoints từ video...",
        "running_train": "Đang huấn luyện mô hình...",
        "running_evaluate": "Đang đánh giá mô hình...",
        "run_success": "Hoàn tất thành công.",
        "run_failed": "Thất bại. Xem log bên dưới.",
        "latest_log": "Nhật ký xử lý",
        "no_terminal_needed": "Bạn có thể chạy extract, train và evaluate tại đây mà không cần terminal.",
        "start": "Bắt đầu",
        "stop": "Dừng lại",
        "webcam_failed": "Không thể mở webcam.",
        "need_model": "Chưa tải được mô hình. Vui lòng huấn luyện trước.",
        "realtime_hint": "Nhấn Bắt đầu để mở webcam. Hệ thống chỉ hiện kết quả khi độ tin cậy đủ cao.",
        "buffer": "Bộ đệm khung hình",
        "hands_detected": "Số tay phát hiện",
        "no_hand": "Chưa phát hiện tay nào.",
        "make_hand_clear": "Đưa tay vào camera rõ hơn...",
        "low_confidence": "Độ tin cậy thấp",
        "collecting": "Đang thu thập frames...",
        "webcam_stopped": "Webcam đã dừng.",
        "action_log_title": "Các thao tác đã nhận diện",
        "action_log_empty": "Chưa có thao tác nào.",
        "left_hand": "Tay trái",
        "right_hand": "Tay phải",
        "index_finger": "Ngón trỏ",
        "no_history": "Chưa có lịch sử. Hãy dùng chức năng Nhận diện thời gian thực.",
        "total_records": "Tổng bản ghi",
        "unique_actions": "Số cử chỉ khác nhau",
        "avg_confidence": "Độ tin cậy trung bình",
        "max_confidence": "Độ tin cậy cao nhất",
        "time": "Thời gian",
        "action": "Cử chỉ",
        "confidence": "Độ tin cậy (%)",
        "hand_count": "Số tay",
        "hand_detail": "Chi tiết tay",
        "download_csv": "Xuất tệp CSV",
        "clear_history": "Xóa toàn bộ lịch sử",
        "no_train_data": "Chưa có dữ liệu huấn luyện.",
        "train_loss": "Train Loss cuối",
        "total_epoch": "Tổng epoch",
        "loss_chart": "Loss theo Epoch",
        "acc_chart": "Accuracy theo Epoch",
        "data_table": "Bảng dữ liệu",
        "classification_report": "Classification Report",
        "confusion_matrix": "Ma trận nhầm lẫn (Confusion Matrix)",
        "per_class_metrics": "Precision / Recall / F1-Score theo từng lớp",
        "per_class_accuracy": "Accuracy theo từng lớp",
        "no_eval_data": "Chưa có kết quả đánh giá.",
        "progress_extract_desc": "Đọc video trong data/raw_videos, trích xuất keypoints tay bằng MediaPipe và lưu metadata.csv.",
        "progress_train_desc": "Huấn luyện LSTM với augmentation, lưu best_model.pth và train_history.json.",
        "progress_eval_desc": "Đánh giá model trên tập test, tạo confusion matrix, classification report và biểu đồ metrics.",
        "important_note": "Mỗi thư mục trong raw_videos phải là một cử chỉ duy nhất. Nên có ≥ 20 video/nhãn để realtime ổn định.",
        "chart_loss_explain": "Biểu đồ này theo dõi sự thay đổi của hàm mất mát (Loss) qua từng epoch. Train Loss (đường liền) thể hiện mức độ model học trên tập huấn luyện — càng thấp càng tốt. Val Loss (đường đứt) phản ánh khả năng tổng quát hoá trên tập validation. Khoảng cách lớn giữa hai đường cho thấy hiện tượng overfitting (model học thuộc lòng dữ liệu train mà không tổng quát được).",
        "chart_acc_explain": "Accuracy là tỷ lệ dự đoán đúng (%). Train Acc cho biết model học tốt đến đâu. Val Acc là chỉ số quan trọng hơn vì phản ánh hiệu suất thực tế. Khi Val Acc bắt đầu giảm trong khi Train Acc vẫn tăng → dấu hiệu overfitting, lúc này Early Stopping sẽ dừng lại.",
        "chart_cm_explain": "Ma trận nhầm lẫn hiển thị số lần model dự đoán đúng/sai cho từng cặp nhãn. Đường chéo chính (góc trái-trên đến phải-dưới) là các dự đoán đúng. Ô nằm ngoài đường chéo là các lỗi nhầm lẫn giữa hai nhãn. Màu càng đậm = số lần xảy ra càng nhiều. Di chuột vào ô để xem số liệu chi tiết.",
        "chart_metrics_explain": "Precision: trong các lần model dự đoán là nhãn X, có bao nhiêu % thực sự đúng? Recall: trong tất cả các mẫu thực sự là nhãn X, model tìm được bao nhiêu %? F1-Score: trung bình điều hoà của Precision và Recall — chỉ số cân bằng và đáng tin cậy nhất khi dữ liệu mất cân bằng.",
        "chart_peracc_explain": "Accuracy riêng lẻ cho từng nhãn. Nhãn nào có cột ngắn hơn đường đỏ (overall accuracy) nghĩa là model đang nhận diện kém hơn mức trung bình — cần thu thập thêm dữ liệu hoặc điều chỉnh augmentation cho nhãn đó.",
        "chart_freq_explain": "Tần suất xuất hiện của từng cử chỉ trong phiên sử dụng hiện tại. Giúp bạn thấy nhãn nào được nhận diện nhiều nhất.",
        "chart_conftrend_explain": "Xu hướng độ tin cậy theo thời gian trong phiên. Đường ngang đỏ là ngưỡng tối thiểu (CONFIDENCE_THRESHOLD). Các điểm dưới ngưỡng bị loại bỏ, không hiển thị nhãn. Nếu đường thường xuyên dưới ngưỡng, hãy cải thiện điều kiện ánh sáng hoặc vị trí tay.",
    },
    "en": {
        "app_title": "VSL Recognition",
        "app_subtitle": "Vietnamese Sign Language recognition system using LSTM and MediaPipe",
        "language": "Language",
        "navigation": "Navigation",
        "home": "Home",
        "pipeline": "Processing Pipeline",
        "realtime": "Real-time Recognition",
        "training_results": "Training Results",
        "evaluation": "Model Evaluation",
        "history": "Recognition History",
        "device": "Device",
        "model_loaded": "Model loaded successfully",
        "model_not_loaded": "Model not loaded",
        "input_size": "Input size",
        "seq_len": "Sequence length",
        "hidden_size": "LSTM hidden size",
        "num_layers": "LSTM layers",
        "batch_size": "Batch size",
        "num_epochs": "Epochs",
        "model_info": "Model information",
        "best_epoch": "Best epoch",
        "val_accuracy": "Val Accuracy",
        "val_loss": "Val Loss",
        "num_classes": "Classes",
        "labels": "Recognizable labels",
        "guide": "User guide",
        "loading_page": "Loading interface...",
        "run_on_ui": "Processing Pipeline",
        "run_extract": "Extract data",
        "run_train": "Train model",
        "run_evaluate": "Evaluate model",
        "running_extract": "Extracting keypoints from videos...",
        "running_train": "Training model...",
        "running_evaluate": "Evaluating model...",
        "run_success": "Completed successfully.",
        "run_failed": "Task failed. See log below.",
        "latest_log": "Processing log",
        "no_terminal_needed": "Run extract, train, and evaluate here without opening a terminal.",
        "start": "Start",
        "stop": "Stop",
        "webcam_failed": "Cannot open webcam.",
        "need_model": "Model not available. Please train first.",
        "realtime_hint": "Click Start to open webcam. Results appear only when confidence is high enough.",
        "buffer": "Frame buffer",
        "hands_detected": "Detected hands",
        "no_hand": "No hand detected.",
        "make_hand_clear": "Move your hand clearly into view...",
        "low_confidence": "Low confidence",
        "collecting": "Collecting frames...",
        "webcam_stopped": "Webcam stopped.",
        "action_log_title": "Recognized actions",
        "action_log_empty": "No action recognized yet.",
        "left_hand": "Left hand",
        "right_hand": "Right hand",
        "index_finger": "Index finger",
        "no_history": "No history yet. Use Real-time Recognition first.",
        "total_records": "Total records",
        "unique_actions": "Unique actions",
        "avg_confidence": "Average confidence",
        "max_confidence": "Highest confidence",
        "time": "Time",
        "action": "Action",
        "confidence": "Confidence (%)",
        "hand_count": "Hands",
        "hand_detail": "Hand details",
        "download_csv": "Download CSV",
        "clear_history": "Clear all history",
        "no_train_data": "No training history yet.",
        "train_loss": "Final Train Loss",
        "total_epoch": "Total epochs",
        "loss_chart": "Loss by Epoch",
        "acc_chart": "Accuracy by Epoch",
        "data_table": "Data table",
        "classification_report": "Classification Report",
        "confusion_matrix": "Confusion Matrix",
        "per_class_metrics": "Precision / Recall / F1-Score by class",
        "per_class_accuracy": "Accuracy by class",
        "no_eval_data": "No evaluation results yet.",
        "progress_extract_desc": "Reads videos from data/raw_videos, extracts hand keypoints with MediaPipe and writes metadata.csv.",
        "progress_train_desc": "Trains LSTM with augmentation, saves best_model.pth and train_history.json.",
        "progress_eval_desc": "Evaluates model on the test set, generates confusion matrix, classification report and metric charts.",
        "important_note": "Each folder in raw_videos must represent exactly one gesture. At least 20 videos per label are recommended.",
        "chart_loss_explain": "This chart tracks how the loss function changes over training epochs. Train Loss (solid line) measures how well the model fits the training data — lower is better. Val Loss (dashed) measures generalisation on unseen data. A large and growing gap between the two lines signals overfitting: the model is memorising training samples instead of learning to generalise.",
        "chart_acc_explain": "Accuracy is the percentage of correct predictions. Train Acc shows how well the model learns the training set. Val Acc is the more important metric as it reflects real-world performance. When Val Acc stops improving while Train Acc keeps climbing → overfitting is occurring, and Early Stopping will halt training.",
        "chart_cm_explain": "The confusion matrix shows how many times each actual label was predicted as each other label. The main diagonal (top-left to bottom-right) contains correct predictions. Off-diagonal cells are mistakes — e.g. if row 'hello' has a value in column 'thank you', the model confused those two gestures. Darker colour = more occurrences. Hover over cells for exact counts.",
        "chart_metrics_explain": "Precision: of all times the model predicted label X, what % were actually X? Recall: of all real X samples, what % did the model find? F1-Score: the harmonic mean of Precision and Recall — the most balanced single metric, especially when data is imbalanced across classes.",
        "chart_peracc_explain": "Per-class accuracy for each label. Labels with bars shorter than the red dashed line (overall accuracy) are below average — consider collecting more training samples or tuning augmentation for those specific classes.",
        "chart_freq_explain": "Frequency of each recognised gesture in the current session. Helps you see which labels appear most often.",
        "chart_conftrend_explain": "Confidence trend over time in the current session. The red dashed line marks the minimum threshold (CONFIDENCE_THRESHOLD). Points below the threshold are suppressed and no label is displayed. If the line is frequently below threshold, improve lighting or hand positioning.",
    },
}

LABEL_TEXT = {
    "cam_on":  {"vi": "cảm ơn",  "en": "thank you"},
    "chao":    {"vi": "chào",    "en": "hello"},
    "ten_la":  {"vi": "tên là",  "en": "my name is"},
    "toi":     {"vi": "tôi",     "en": "I / me"},
}

PAGE_LABEL_KEY = {
    "home":             "home",
    "pipeline":         "pipeline",
    "realtime":         "realtime",
    "training_results": "training_results",
    "evaluation":       "evaluation",
    "history":          "history",
}


def lang() -> str:
    return st.session_state.get("lang", "vi")


def tr(key: str) -> str:
    return TEXT.get(lang(), TEXT["vi"]).get(key, key)


def display_label(label: Any) -> str:
    label = str(label)
    return LABEL_TEXT.get(label, {}).get(lang(), label)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton button, .stSelectbox, .stRadio, .stDataFrame, .stMetric, [data-testid="stCodeBlock"] pre {
  font-family: "Inter", "Segoe UI", Arial, Tahoma, sans-serif !important;
}

.main .block-container{padding-top:1rem;padding-bottom:2rem;max-width:1300px;}
#MainMenu{visibility:hidden}footer{visibility:hidden}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#11111b,#1e1e2e)}

/* Hero */
.hero{background:linear-gradient(135deg,#111827,#1e1e2e);border:1px solid #313244;border-radius:24px;
  padding:1.8rem 2rem;margin-bottom:1rem;box-shadow:0 18px 50px rgba(0,0,0,.25)}
.hero h1{font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#89b4fa,#b4befe,#cba6f7);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .3rem;letter-spacing:-1px;}
.hero p{color:#a6adc8;font-size:1rem;margin:0;}

/* Metric cards */
.metric-card{background:linear-gradient(135deg,#1e1e2e,#181825);border:1px solid #313244;border-radius:18px;
  padding:1rem;text-align:center;transition:all .2s;min-height:100px;display:flex;flex-direction:column;justify-content:center;}
.metric-card:hover{transform:translateY(-3px);border-color:#89b4fa;box-shadow:0 12px 28px rgba(137,180,250,.14)}
.metric-card .value{font-size:1.75rem;font-weight:900;
  background:linear-gradient(135deg,#89b4fa,#cba6f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;}
.metric-card .label{font-size:.75rem;color:#a6adc8;text-transform:uppercase;letter-spacing:.8px;margin-top:.4rem;}

/* Cards */
.card{background:linear-gradient(135deg,#1e1e2e,#181825);border:1px solid #313244;border-radius:18px;
  padding:1.1rem 1.2rem;margin:.5rem 0;box-shadow:0 8px 24px rgba(0,0,0,.16)}
.card-title{font-weight:800;color:#cdd6f4;font-size:1.05rem;margin-bottom:.3rem;}
.card-desc{color:#a6adc8;font-size:.9rem;line-height:1.55;}
.section-title{font-size:1.4rem;font-weight:900;color:#cdd6f4;margin:1rem 0 .55rem;}

/* Pipeline step specific card */
.step-card{
  background:linear-gradient(135deg,#1e1e2e,#181825);
  border:1px solid #313244;
  border-radius:18px;
  padding:1.1rem 1.2rem;
  margin:.5rem 0;
  box-shadow:0 8px 24px rgba(0,0,0,.16);
  min-height: 150px; /* Giữ 3 block text bằng nhau để các nút dưới cùng thẳng hàng */
  display: flex;
  flex-direction: column;
}

/* Prediction / waiting */
.pred-box{background:linear-gradient(135deg,#1e1e2e,#181825);border:2px solid #89b4fa;border-radius:18px;
  padding:1.3rem;text-align:center;margin:.5rem 0;box-shadow:0 10px 26px rgba(137,180,250,.13)}
.pred-box .sign{font-size:2rem;font-weight:900;color:#89b4fa;line-height:1.15;}
.pred-box .conf{font-size:1.15rem;font-weight:800;margin-top:.3rem;}
.waiting-box{background:#1e1e2e;border:2px dashed #45475a;border-radius:18px;padding:1.3rem;
  text-align:center;color:#a6adc8;line-height:1.55;}

/* Hand info */
.hand-info{background:#181825;border:1px solid #313244;border-radius:12px;padding:.8rem 1rem;margin:.3rem 0;font-size:.9rem;}
.hand-info .hand-label{font-weight:800;font-size:1rem;}

/* Action log */
.action-log{background:linear-gradient(135deg,#11111b,#181825);border:1px solid #45475a;border-radius:16px;
  padding:.9rem 1.1rem;margin-top:.7rem;color:#cdd6f4;font-size:1rem;line-height:1.65;box-shadow:0 8px 20px rgba(0,0,0,.14)}
.action-log .title{color:#89b4fa;font-weight:900;margin-bottom:.3rem;}
.action-log .content{color:#f9e2af;font-weight:800;word-break:break-word;}

/* Loader */
.load-card{background:linear-gradient(135deg,#1e1e2e,#11111b);border:1px solid #313244;border-radius:18px;
  padding:1rem 1.2rem;color:#cdd6f4;display:flex;align-items:center;gap:.8rem;margin:.5rem 0;}
.loader{width:22px;height:22px;border:3px solid #45475a;border-top-color:#89b4fa;
  border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}

/* Chart explain box */
.chart-explain{background:#181825;border-left:3px solid #89b4fa;border-radius:0 12px 12px 0;
  padding:.75rem 1rem;margin:.4rem 0 1rem;color:#a6adc8;font-size:.88rem;line-height:1.6;}
.chart-explain strong{color:#89b4fa;}

/* Pipeline step */
.step-header{display:flex;align-items:center;gap:.7rem;margin-bottom:.3rem;}
.step-num{background:linear-gradient(135deg,#89b4fa,#cba6f7);color:#11111b;border-radius:50%;
  width:28px;height:28px;display:flex;align-items:center;justify-content:center;
  font-weight:900;font-size:.88rem;flex-shrink:0;}
.step-title{font-weight:800;color:#cdd6f4;font-size:1rem;}

/* Progress bar custom */
.stProgress>div>div>div{background:linear-gradient(90deg,#89b4fa,#cba6f7)!important;border-radius:8px!important;}

/* Buttons */
.stButton button{border-radius:14px!important;font-weight:800!important;min-height:42px;}

/* GIỚI HẠN CHIỀU CAO CHO TERMINAL LOG (Chống tràn màn hình) */
[data-testid="stCodeBlock"] pre {
    max-height: 350px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Component helpers
# ══════════════════════════════════════════════════════════════════════════════
def loading_card(text: str) -> None:
    st.markdown(
        f'<div class="load-card"><div class="loader"></div><div>{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, col) -> None:
    col.markdown(
        f'<div class="metric-card"><div class="value">{html.escape(str(value))}</div>'
        f'<div class="label">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


def chart_explain(text: str) -> None:
    st.markdown(
        f'<div class="chart-explain">{text}</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    if not BEST_MODEL_PATH.exists():
        return None, None, "Chưa có model. Hãy huấn luyện trước."
    if not METADATA_PATH.exists():
        return None, None, "Chưa có metadata. Hãy chạy extract_data.py."
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            ds = VSLDataset(METADATA_PATH)
            lm = ds.get_label_map()
        finally:
            sys.stdout = old_stdout
        model = SignLanguageLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, ds.num_classes)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        ckpt = torch.load(BEST_MODEL_PATH, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, lm, None
    except Exception as e:
        return None, None, str(e)


def clear_model_cache() -> None:
    try:
        load_model.clear()
    except Exception:
        pass


def load_train_history():
    p = RESULTS_DIR / "train_history.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_info():
    if not BEST_MODEL_PATH.exists():
        return {}
    try:
        ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
        return {k: ckpt.get(k, "?") for k in
                ["epoch", "val_accuracy", "val_loss", "train_accuracy", "train_loss", "num_classes"]}
    except Exception:
        return {}


def build_sequence_with_velocity(raw_buffer: deque) -> np.ndarray:
    seq = np.array(list(raw_buffer), dtype=np.float32)
    velocity = np.zeros_like(seq)
    if len(seq) > 1:
        velocity[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, velocity], axis=1)


def style_ax(ax):
    """Fallback matplotlib styling (used only if plotly unavailable)."""
    import matplotlib.pyplot as plt
    ax.set_facecolor("#11111b")
    ax.tick_params(colors="#6c7086")
    ax.grid(alpha=0.2, color="#313244")
    for s in ax.spines.values():
        s.set_color("#313244")


# ══════════════════════════════════════════════════════════════════════════════
# Real-time progress runner
# ══════════════════════════════════════════════════════════════════════════════
def _parse_progress_extract(line: str, total_videos: int) -> int | None:
    """Returns 0-95 progress value by counting DONE lines."""
    if line.startswith("[DONE]") and "/" in line:
        return None  # signal: increment
    return None


def run_script(script_name: str, running_text: str, total_epochs: int = NUM_EPOCHS) -> bool:
    """
    Execute a Python script as a subprocess and stream its output to the UI.
    Progress bar uses real epoch/video counts parsed from stdout.
    """
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        st.error(f"Không tìm thấy file: {script_path}")
        return False

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # --- Phase indicators ---
    phase_ph   = st.empty()
    prog_bar   = st.progress(0)
    pct_ph     = st.empty()
    log_box    = st.empty()

    phase_ph.markdown(
        f'<div class="load-card"><div class="loader"></div><div><b>{html.escape(running_text)}</b></div></div>',
        unsafe_allow_html=True,
    )

    lines: list[str] = []
    tick = 3
    done_count = 0          # for extract progress
    epoch_count = 0         # for train progress

    # Epoch line pattern: "   5  | 0.1234  98.12% | ..."
    EPOCH_RE = re.compile(r"^\s*(\d+)\s+\|")
    DONE_RE  = re.compile(r"^\[DONE\]")
    SKIP_RE  = re.compile(r"^\[SKIP\]")
    EVAL_RE  = re.compile(r"^\[INFO\].*(?:Đang|Running|dự đoán|predict|load)", re.IGNORECASE)
    INFO_RE  = re.compile(r"^\[INFO\]")

    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None

        for line in process.stdout:
            clean = line.rstrip()
            if clean:
                lines.append(clean)

            # ── Parse real progress ──────────────────────────────────────────
            if script_name == "train.py":
                m = EPOCH_RE.match(clean)
                if m:
                    epoch_count = int(m.group(1))
                    tick = max(5, min(95, int(epoch_count / total_epochs * 93)))
                elif clean.strip().startswith("TẢI") or "Tải" in clean:
                    tick = 2
                elif "Tính trọng số" in clean or "class weight" in clean.lower():
                    tick = 6
                elif "BẮT ĐẦU" in clean or "TRAINING" in clean.upper():
                    tick = 8

            elif script_name == "extract_data.py":
                if DONE_RE.match(clean) or SKIP_RE.match(clean):
                    done_count += 1
                    tick = max(5, min(93, 5 + done_count * 3))
                elif "Nhãn" in clean or "label" in clean.lower():
                    tick = 5

            elif script_name == "evaluate.py":
                if "DataLoader" in clean:
                    tick = 10
                elif "load" in clean.lower() and "model" in clean.lower():
                    tick = 25
                elif "dự đoán" in clean.lower() or "predict" in clean.lower():
                    tick = 50
                elif "Confusion" in clean:
                    tick = 65
                elif "Per-class" in clean or "Metrics" in clean:
                    tick = 80
                elif "Accuracy" in clean and "lưu" in clean.lower():
                    tick = 90
                elif INFO_RE.match(clean):
                    tick = min(93, tick + 2)

            prog_bar.progress(tick)
            pct_ph.markdown(
                f'<div style="text-align:center;color:#89b4fa;font-weight:800;font-size:1.1rem;margin:.2rem 0">'
                f'{tick}%</div>',
                unsafe_allow_html=True,
            )
            # Giữ lại 120 dòng log mới nhất, css [data-testid="stCodeBlock"] sẽ khống chế chiều cao
            log_box.code("\n".join(lines[-120:]), language="text")

        return_code = process.wait()
        prog_bar.progress(100)
        pct_ph.markdown(
            '<div style="text-align:center;color:#a6e3a1;font-weight:800;font-size:1.1rem;margin:.2rem 0">100% ✓</div>',
            unsafe_allow_html=True,
        )
        phase_ph.empty()

        if return_code == 0:
            st.success(tr("run_success"))
            clear_model_cache()
            return True
        st.error(tr("run_failed"))
        return False

    except Exception as e:
        st.error(f"{tr('run_failed')}\n{e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline card renderer
# ══════════════════════════════════════════════════════════════════════════════
def render_pipeline_step(
    step_num: int,
    title_key: str,
    desc_key: str,
    button_key: str,
) -> bool:
    """Render a step card and a button, return True if clicked."""
    st.markdown(
        f"""<div class="step-card">
          <div class="step-header">
            <div class="step-num">{step_num}</div>
            <div class="step-title">{html.escape(tr(title_key))}</div>
          </div>
          <div class="card-desc">{html.escape(tr(desc_key))}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    return st.button(tr(title_key), key=button_key, type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Plotly chart builders
# ══════════════════════════════════════════════════════════════════════════════
def plotly_loss_chart(df: pd.DataFrame) -> go.Figure:
    best_idx = df["val_loss"].idxmin()
    best_ep  = int(df.loc[best_idx, "epoch"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["train_loss"],
        mode="lines", name="Train Loss",
        line=dict(color="#89b4fa", width=2.5),
        hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["val_loss"],
        mode="lines", name="Val Loss",
        line=dict(color="#cba6f7", width=2.5, dash="dash"),
        hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
    ))
    # Best epoch annotation
    fig.add_vline(
        x=best_ep, line_dash="dot", line_color="#a6e3a1", line_width=1.5,
        annotation_text=f"Best epoch {best_ep}",
        annotation_font_color="#a6e3a1",
        annotation_position="top right",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=tr("loss_chart"), font=dict(size=16, color="#cdd6f4")),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=380,
    )
    return fig


def plotly_acc_chart(df: pd.DataFrame) -> go.Figure:
    best_idx = df["val_acc"].idxmax()
    best_ep  = int(df.loc[best_idx, "epoch"])
    best_acc = float(df.loc[best_idx, "val_acc"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["train_acc"],
        mode="lines", name="Train Acc",
        line=dict(color="#89b4fa", width=2.5),
        hovertemplate="Epoch %{x}<br>Train Acc: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["val_acc"],
        mode="lines", name="Val Acc",
        line=dict(color="#a6e3a1", width=2.5, dash="dash"),
        hovertemplate="Epoch %{x}<br>Val Acc: %{y:.2f}%<extra></extra>",
    ))
    fig.add_vline(
        x=best_ep, line_dash="dot", line_color="#f9e2af", line_width=1.5,
        annotation_text=f"Best {best_acc:.1f}% @ epoch {best_ep}",
        annotation_font_color="#f9e2af",
        annotation_position="top right",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=tr("acc_chart"), font=dict(size=16, color="#cdd6f4")),
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        height=380,
    )
    return fig


def plotly_confusion_matrix(cm: np.ndarray, label_names: list[str]) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=label_names,
        y=label_names,
        colorscale=[[0, "#1e1e2e"], [0.5, "#3b5bdb"], [1.0, "#89b4fa"]],
        showscale=True,
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=13, color="#cdd6f4"),
        hovertemplate="Thực tế: %{y}<br>Dự đoán: %{x}<br>Số lần: %{z}<extra></extra>",
    ))
    n = len(label_names)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=tr("confusion_matrix"), font=dict(size=16, color="#cdd6f4")),
        xaxis=dict(title="Dự đoán (Predicted)", side="bottom", tickangle=-30, gridcolor="#313244"),
        yaxis=dict(title="Thực tế (Actual)", autorange="reversed", gridcolor="#313244"),
        height=max(350, n * 55 + 100),
    )
    return fig


def plotly_per_class_metrics(
    label_names: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    bar_kw = dict(barmode="group")
    for vals, name, color in [
        (precision, "Precision", "#89b4fa"),
        (recall,    "Recall",    "#a6e3a1"),
        (f1,        "F1-Score",  "#cba6f7"),
    ]:
        fig.add_trace(go.Bar(
            name=name,
            x=label_names,
            y=vals,
            marker_color=color,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate=f"{name}: %{{y:.4f}}<br>Nhãn: %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text=tr("per_class_metrics"), font=dict(size=16, color="#cdd6f4")),
        xaxis_title="Nhãn (Label)",
        yaxis_title="Điểm số",
        yaxis_range=[0, 1.18],
        height=420,
        bargap=0.18,
        bargroupgap=0.05,
    )
    return fig


def plotly_per_class_accuracy(label_names: list[str], per_class_acc: list[float], overall_acc: float) -> go.Figure:
    colors = [
        "#a6e3a1" if a >= overall_acc / 100 else "#f38ba8"
        for a in per_class_acc
    ]
    fig = go.Figure(go.Bar(
        x=per_class_acc,
        y=label_names,
        orientation="h",
        marker_color=colors,
        text=[f"{a:.1%}" for a in per_class_acc],
        textposition="outside",
        hovertemplate="Nhãn: %{y}<br>Accuracy: %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(
        x=overall_acc / 100, line_dash="dash",
        line_color="#f38ba8", line_width=2,
        annotation_text=f"Overall {overall_acc:.1f}%",
        annotation_font_color="#f38ba8",
        annotation_position="top right",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=tr("per_class_accuracy"), font=dict(size=16, color="#cdd6f4")),
        xaxis_title="Accuracy",
        xaxis_range=[0, 1.25],
        height=max(320, len(label_names) * 42 + 100),
    )
    return fig


def plotly_freq_chart(labels: list[str]) -> go.Figure:
    lc = pd.Series(labels).value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=lc.values,
        y=lc.index,
        orientation="h",
        marker_color="#89b4fa",
        text=lc.values,
        textposition="outside",
        hovertemplate="%{y}: %{x} lần<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=tr("unique_actions"), font=dict(size=15, color="#cdd6f4")),
        xaxis_title="Số lần",
        height=max(280, len(lc) * 44 + 100),
    )
    return fig


def plotly_conf_trend(confs: list[float]) -> go.Figure:
    threshold = CONFIDENCE_THRESHOLD * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=confs,
        mode="lines+markers",
        name="Confidence",
        line=dict(color="#89b4fa", width=2),
        marker=dict(size=5),
        hovertemplate="Lần %{x}<br>Độ tin cậy: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#f38ba8", line_width=1.8,
        annotation_text=f"Threshold {threshold:.0f}%",
        annotation_font_color="#f38ba8",
        annotation_position="bottom right",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=tr("avg_confidence"), font=dict(size=15, color="#cdd6f4")),
        xaxis_title="Lần nhận diện",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 105],
        height=320,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
if "lang" not in st.session_state:
    st.session_state.lang = getattr(cfg, "APP_DEFAULT_LANGUAGE", "vi")

with st.sidebar:
    st.markdown("### 🤟 VSL Recognition")
    st.caption("Vietnamese Sign Language")
    st.markdown("---")

    lang_choice = st.selectbox(
        tr("language"),
        options=["vi", "en"],
        index=0 if st.session_state.lang == "vi" else 1,
        format_func=lambda x: "Tiếng Việt" if x == "vi" else "English",
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    st.markdown("---")
    page = st.radio(
        tr("navigation"),
        list(PAGE_LABEL_KEY.keys()),
        format_func=lambda k: tr(PAGE_LABEL_KEY[k]),
    )

    st.markdown("---")
    dev = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"**{tr('device')}:** `{dev}`")

    with st.spinner(tr("loading_page")):
        model, label_map, err = load_model()
    if model:
        st.success(tr("model_loaded"))
    else:
        st.warning(err or tr("model_not_loaded"))

    st.caption(f"Python {sys.version.split()[0]} — PyTorch {torch.__version__}")
    st.caption(f"INPUT_SIZE = {INPUT_SIZE} ({'pos+vel' if USE_VELOCITY else 'position only'})")
    if not HAS_PLOTLY:
        st.warning("⚠️ `pip install plotly` để dùng biểu đồ tương tác.")

# Page transition flash
if st.session_state.get("_last_page") != page:
    ph = st.empty()
    with ph.container():
        loading_card(tr("loading_page"))
    time.sleep(0.15)
    ph.empty()
    st.session_state["_last_page"] = page


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown(
        f'<div class="hero"><h1>{tr("app_title")}</h1><p>{tr("app_subtitle")}</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    for c, (label, value) in zip(cols, [
        (tr("seq_len"), SEQ_LEN),
        (tr("input_size"), INPUT_SIZE),
        (tr("hidden_size"), HIDDEN_SIZE),
        (tr("num_layers"), NUM_LAYERS),
        (tr("batch_size"), BATCH_SIZE),
        (tr("num_epochs"), NUM_EPOCHS),
    ]):
        metric_card(label, str(value), c)

    section(tr("model_info"))
    ckpt = load_checkpoint_info()
    if ckpt:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("best_epoch"), ckpt.get("epoch", "?"))
        c2.metric(tr("val_accuracy"),
                  f"{ckpt.get('val_accuracy', 0):.1f}%"
                  if isinstance(ckpt.get("val_accuracy"), (int, float)) else "?")
        c3.metric(tr("val_loss"),
                  f"{ckpt.get('val_loss', 0):.4f}"
                  if isinstance(ckpt.get("val_loss"), (int, float)) else "?")
        c4.metric(tr("num_classes"), ckpt.get("num_classes", "?"))
    else:
        st.info(tr("model_not_loaded"))

    if label_map:
        section(tr("labels"))
        st.dataframe(
            pd.DataFrame([
                {"ID": k, tr("action"): display_label(v), "Raw label": v}
                for k, v in sorted(label_map.items())
            ]),
            hide_index=True,
            use_container_width=True,
        )

    section(tr("guide"))
    guide_df = pd.DataFrame([
        {"Step": 1, "Action": "raw_videos/<label>/", "Description": tr("important_note")},
        {"Step": 2, "Action": "extract_data.py",     "Description": tr("progress_extract_desc")},
        {"Step": 3, "Action": "train.py",             "Description": tr("progress_train_desc")},
        {"Step": 4, "Action": "evaluate.py",          "Description": tr("progress_eval_desc")},
        {"Step": 5, "Action": "app.py → Realtime",    "Description": tr("realtime_hint")},
    ])
    st.dataframe(guide_df, hide_index=True, use_container_width=True)
    st.info(tr("important_note"), icon="⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "pipeline":
    st.markdown(
        f'<div class="hero"><h1>{tr("run_on_ui")}</h1><p>{tr("no_terminal_needed")}</p></div>',
        unsafe_allow_html=True,
    )

    task_to_run = None
    ui_ph = st.empty()  # Container xử lý việc ẩn giao diện

    with ui_ph.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            if render_pipeline_step(1, "run_extract", "progress_extract_desc", "pipe_ext"):
                task_to_run = ("extract_data.py", "running_extract")
        with c2:
            if render_pipeline_step(2, "run_train", "progress_train_desc", "pipe_tr"):
                task_to_run = ("train.py", "running_train")
        with c3:
            if render_pipeline_step(3, "run_evaluate", "progress_eval_desc", "pipe_ev"):
                task_to_run = ("evaluate.py", "running_evaluate")

        st.markdown(
            f'<div class="card"><div class="card-desc">⚠️ {html.escape(tr("important_note"))}</div></div>',
            unsafe_allow_html=True,
        )

    # Nếu nút trong bất kỳ cột nào được bấm
    if task_to_run:
        ui_ph.empty()  # Xoá giao diện menu, chỉ giữ lại màn hình log
        script_name, running_key = task_to_run

        st.markdown(f"### {tr(running_key)}")
        run_script(script_name, tr(running_key))

        st.markdown("---")
        if st.button("Quay lại menu", type="primary", use_container_width=True):
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# REALTIME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "realtime":
    st.markdown(
        f'<div class="hero"><h1>{tr("realtime")}</h1><p>{tr("realtime_hint")}</p></div>',
        unsafe_allow_html=True,
    )
    if not model:
        st.error(tr("need_model"))
        st.stop()

    c1, c2 = st.columns([1, 4])
    start    = c1.button(tr("start"), type="primary", use_container_width=True)
    stop_ph  = c2.empty()

    col_v, col_i = st.columns([3.2, 1])
    vid_ph       = col_v.empty()
    log_box_ph   = col_v.empty()
    with col_i:
        pred_ph      = st.empty()
        hand_info_ph = st.empty()
        buf_ph       = st.empty()
        status_ph    = st.empty()

    if start:
        with st.spinner(tr("loading_page")):
            import mediapipe.python.solutions.drawing_utils as mp_drawing
            import mediapipe.python.solutions.hands as mp_hands

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error(tr("webcam_failed"))
            st.stop()

        raw_buf          = deque(maxlen=SEQ_LEN)
        hand_detected_buf = deque(maxlen=SEQ_LEN)
        dev_t  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stop   = stop_ph.button(tr("stop"), type="secondary")
        hands  = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS_DEMO,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        for key, default in [
            ("recognition_history", []),
            ("action_text_log", []),
            ("last_action_label", None),
            ("last_action_time", 0.0),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        LEFT_COLOR  = (248, 113, 113)
        RIGHT_COLOR = (56, 189, 248)

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h_f, w_f = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            kp           = np.zeros(_BASE_INPUT, dtype=np.float32)
            hand_detected = False
            hands_info    = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (hl, hd) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    raw_label     = hd.classification[0].label
                    hand_score    = hd.classification[0].score
                    hand_name_key = "left_hand" if raw_label == "Right" else "right_hand"
                    hand_name     = tr(hand_name_key)
                    hand_name_cv  = ("Left hand" if lang() == "en"
                                     else ("Tay trai" if hand_name_key == "left_hand" else "Tay phai"))
                    color = LEFT_COLOR if hand_name_key == "left_hand" else RIGHT_COLOR

                    wrist   = hl.landmark[0]
                    idx_tip = hl.landmark[8]
                    nx = round(idx_tip.x - wrist.x, 4)
                    ny = round(idx_tip.y - wrist.y, 4)
                    nz = round(idx_tip.z - wrist.z, 4)

                    hands_info.append({
                        "hand":        hand_name,
                        "score":       round(hand_score * 100, 1),
                        "wrist_x":     round(wrist.x, 3),
                        "wrist_y":     round(wrist.y, 3),
                        "index_tip_x": nx,
                        "index_tip_y": ny,
                        "index_tip_z": nz,
                    })

                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=color, thickness=2),
                    )
                    wx, wy = int(wrist.x * w_f), int(wrist.y * h_f)
                    cv2.putText(frame, hand_name_cv, (wx - 30, wy + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if i == 0:
                        vals = []
                        for lm in hl.landmark:
                            vals.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                        kp           = np.array(vals, dtype=np.float32)
                        hand_detected = True

            raw_buf.append(kp)
            hand_detected_buf.append(1 if hand_detected else 0)
            bl = len(raw_buf)

            buf_ph.metric(tr("buffer"),        f"{bl} / {SEQ_LEN}")
            status_ph.metric(tr("hands_detected"), f"{len(hands_info)}")

            if hands_info:
                hand_html = ""
                for hi in hands_info:
                    c = "#f87171" if hi["hand"] == tr("left_hand") else "#89b4fa"
                    hand_html += (
                        f'<div class="hand-info">'
                        f'<div class="hand-label" style="color:{c}">'
                        f'{html.escape(hi["hand"])} ({hi["score"]}%)</div>'
                        f'<div style="color:#a6adc8;font-size:.82rem;margin-top:4px">'
                        f'{tr("index_finger")}: X={hi["index_tip_x"]:+.3f} '
                        f'Y={hi["index_tip_y"]:+.3f} Z={hi["index_tip_z"]:+.3f}</div></div>'
                    )
                hand_info_ph.markdown(hand_html, unsafe_allow_html=True)
            else:
                hand_info_ph.markdown(
                    f'<div class="hand-info" style="color:#6c7086">{tr("no_hand")}</div>',
                    unsafe_allow_html=True,
                )

            if bl == SEQ_LEN:
                hand_ratio = sum(hand_detected_buf) / SEQ_LEN
                if hand_ratio < MIN_HAND_FRAMES_RATIO:
                    pred_ph.markdown(
                        f'<div class="waiting-box">{tr("make_hand_clear")}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    seq    = (build_sequence_with_velocity(raw_buf) if USE_VELOCITY
                              else np.array(list(raw_buf), dtype=np.float32))
                    tensor = torch.tensor(seq).unsqueeze(0).to(dev_t)
                    with torch.no_grad():
                        out   = model(tensor)
                        probs = torch.softmax(out, dim=1)
                        prob, idx = torch.max(probs, dim=1)
                        prediction_conf       = prob.item() * 100
                        raw_prediction_label  = label_map.get(idx.item(), f"ID:{idx.item()}")
                        prediction_label      = display_label(raw_prediction_label)

                    if prediction_conf < CONFIDENCE_THRESHOLD * 100:
                        pred_ph.markdown(
                            f'<div class="waiting-box">{tr("low_confidence")} '
                            f'({prediction_conf:.1f}%)<br>'
                            f'<small>Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%</small></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        cc = ("#a6e3a1" if prediction_conf > 80
                              else "#f9e2af" if prediction_conf > 65
                              else "#f38ba8")
                        pred_ph.markdown(
                            f'<div class="pred-box">'
                            f'<div class="sign">{html.escape(prediction_label)}</div>'
                            f'<div class="conf" style="color:{cc}">{prediction_conf:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        cv2.rectangle(frame, (0, h_f - 55), (w_f, h_f), (0, 0, 0), -1)
                        cv2.putText(
                            frame,
                            f"{raw_prediction_label}  ({prediction_conf:.1f}%)",
                            (15, h_f - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56, 189, 248), 2,
                        )

                        st.session_state.recognition_history.append({
                            "thoi_gian": datetime.now().strftime("%H:%M:%S"),
                            "cu_chi":    prediction_label,
                            "raw_label": raw_prediction_label,
                            "do_tin_cay": round(prediction_conf, 2),
                            "so_tay":    len(hands_info),
                            "chi_tiet_tay": hands_info.copy() if hands_info else [],
                        })

                        now_time   = time.time()
                        is_new_action = (
                            raw_prediction_label != st.session_state.last_action_label
                            or now_time - st.session_state.last_action_time > 1.2
                        )
                        if is_new_action:
                            st.session_state.action_text_log.append(prediction_label)
                            st.session_state.action_text_log = st.session_state.action_text_log[-20:]
                            st.session_state.last_action_label = raw_prediction_label
                            st.session_state.last_action_time  = now_time
            else:
                pred_ph.markdown(
                    f'<div class="waiting-box">{tr("collecting")}'
                    f'<br><small>{bl}/{SEQ_LEN}</small></div>',
                    unsafe_allow_html=True,
                )

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_ph.image(display_frame, channels="RGB", use_container_width=True)
            action_text = (
                " → ".join(st.session_state.get("action_text_log", []))
                or tr("action_log_empty")
            )
            log_box_ph.markdown(
                f'<div class="action-log">'
                f'<div class="title">{html.escape(tr("action_log_title"))}</div>'
                f'<div class="content">{html.escape(action_text)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.03)

        cap.release()
        hands.close()
        st.info(tr("webcam_stopped"))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "training_results":
    st.markdown(
        f'<div class="hero"><h1>{tr("training_results")}</h1><p>{tr("progress_train_desc")}</p></div>',
        unsafe_allow_html=True,
    )
    if st.button(tr("run_train"), key="train_from_results", type="primary", use_container_width=True):
        run_script("train.py", tr("running_train"))

    history = load_train_history()
    if history is None:
        st.warning(tr("no_train_data"))
        st.stop()

    df   = pd.DataFrame(history)
    best = df.loc[df["val_acc"].idxmax()]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(tr("total_epoch"), len(df))
    c2.metric(tr("best_epoch"),  int(best["epoch"]))
    c3.metric(tr("val_accuracy"), f"{best['val_acc']:.2f}%")
    c4.metric(tr("train_loss"),   f"{df.iloc[-1]['train_loss']:.4f}")
    c5.metric(tr("val_loss"),     f"{df.iloc[-1]['val_loss']:.4f}")

    tab1, tab2, tab3 = st.tabs([tr("loss_chart"), tr("acc_chart"), tr("data_table")])

    with tab1:
        if HAS_PLOTLY:
            st.plotly_chart(plotly_loss_chart(df), use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1e1e2e")
            style_ax(ax)
            ax.plot(df["epoch"], df["train_loss"], lw=2, label="Train Loss")
            ax.plot(df["epoch"], df["val_loss"],   lw=2, ls="--", label="Val Loss")
            ax.set_xlabel("Epoch", color="#cdd6f4")
            ax.set_ylabel("Loss",  color="#cdd6f4")
            ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")
            st.pyplot(fig); plt.close(fig)
        with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
            chart_explain(tr("chart_loss_explain"))

    with tab2:
        if HAS_PLOTLY:
            st.plotly_chart(plotly_acc_chart(df), use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1e1e2e")
            style_ax(ax)
            ax.plot(df["epoch"], df["train_acc"], lw=2, label="Train Acc")
            ax.plot(df["epoch"], df["val_acc"],   lw=2, ls="--", label="Val Acc")
            ax.set_xlabel("Epoch", color="#cdd6f4")
            ax.set_ylabel("Accuracy (%)", color="#cdd6f4")
            ax.set_ylim(0, 105)
            ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")
            st.pyplot(fig); plt.close(fig)
        with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
            chart_explain(tr("chart_acc_explain"))

    with tab3:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            tr("download_csv"),
            df.to_csv(index=False).encode("utf-8-sig"),
            "train_history.csv", "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "evaluation":
    st.markdown(
        f'<div class="hero"><h1>{tr("evaluation")}</h1><p>{tr("progress_eval_desc")}</p></div>',
        unsafe_allow_html=True,
    )
    if st.button(tr("run_evaluate"), key="eval_from_page", type="primary", use_container_width=True):
        run_script("evaluate.py", tr("running_evaluate"))

    has_any = False

    # ── Classification report ────────────────────────────────────────────────
    rp = RESULTS_DIR / "classification_report.txt"
    if rp.exists():
        has_any = True
        section(tr("classification_report"))
        with open(rp, encoding="utf-8") as f:
            report_text = f.read()
        st.code(report_text, language="text")
        with st.expander("💡 Đọc hiểu bảng này", expanded=False):
            chart_explain(tr("chart_metrics_explain"))

    # ── Confusion Matrix — interactive Plotly if data available ─────────────
    cm_img = RESULTS_DIR / "confusion_matrix.png"
    if cm_img.exists():
        has_any = True
        section(tr("confusion_matrix"))

        # Try to build interactive Plotly confusion matrix from saved .png data
        # We use the image as fallback but try sklearn results from numpy if available
        plotly_cm_built = False
        if HAS_PLOTLY:
            # Attempt to find saved predictions (not saved by default → fall back to image)
            try:
                # Check if evaluate.py saves numpy arrays alongside results
                y_true_path = RESULTS_DIR / "y_true.npy"
                y_pred_path = RESULTS_DIR / "y_pred.npy"
                lm_path     = RESULTS_DIR / "label_names.json"
                if y_true_path.exists() and y_pred_path.exists() and lm_path.exists():
                    from sklearn.metrics import confusion_matrix as sk_cm_fn
                    y_true = np.load(str(y_true_path))
                    y_pred = np.load(str(y_pred_path))
                    with open(lm_path, encoding="utf-8") as f:
                        lnames = json.load(f)
                    cm = sk_cm_fn(y_true, y_pred, labels=range(len(lnames)))
                    st.plotly_chart(plotly_confusion_matrix(cm, lnames), use_container_width=True)
                    plotly_cm_built = True
            except Exception:
                pass

        if not plotly_cm_built:
            st.image(str(cm_img), use_container_width=True)

        with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
            chart_explain(tr("chart_cm_explain"))

    # ── Per-class metrics ────────────────────────────────────────────────────
    metrics_img = RESULTS_DIR / "per_class_metrics.png"
    if metrics_img.exists():
        has_any = True
        section(tr("per_class_metrics"))

        plotly_metrics_built = False
        if HAS_PLOTLY:
            try:
                y_true_path = RESULTS_DIR / "y_true.npy"
                y_pred_path = RESULTS_DIR / "y_pred.npy"
                lm_path     = RESULTS_DIR / "label_names.json"
                if y_true_path.exists() and y_pred_path.exists() and lm_path.exists():
                    from sklearn.metrics import precision_recall_fscore_support
                    y_true = np.load(str(y_true_path))
                    y_pred = np.load(str(y_pred_path))
                    with open(lm_path, encoding="utf-8") as f:
                        lnames = json.load(f)
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred,
                        labels=range(len(lnames)),
                        zero_division=0,
                    )
                    st.plotly_chart(
                        plotly_per_class_metrics(lnames, prec, rec, f1),
                        use_container_width=True,
                    )
                    plotly_metrics_built = True
            except Exception:
                pass

        if not plotly_metrics_built:
            st.image(str(metrics_img), use_container_width=True)

        with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
            chart_explain(tr("chart_metrics_explain"))

    # ── Per-class accuracy ───────────────────────────────────────────────────
    acc_img = RESULTS_DIR / "per_class_accuracy.png"
    if acc_img.exists():
        has_any = True
        section(tr("per_class_accuracy"))

        plotly_acc_built = False
        if HAS_PLOTLY:
            try:
                y_true_path = RESULTS_DIR / "y_true.npy"
                y_pred_path = RESULTS_DIR / "y_pred.npy"
                lm_path     = RESULTS_DIR / "label_names.json"
                if y_true_path.exists() and y_pred_path.exists() and lm_path.exists():
                    from sklearn.metrics import confusion_matrix as sk_cm_fn
                    y_true = np.load(str(y_true_path))
                    y_pred = np.load(str(y_pred_path))
                    with open(lm_path, encoding="utf-8") as f:
                        lnames = json.load(f)
                    cm = sk_cm_fn(y_true, y_pred, labels=range(len(lnames)))
                    pca = []
                    for i in range(len(lnames)):
                        tot = cm[i].sum()
                        pca.append(cm[i, i] / tot if tot > 0 else 0.0)
                    ov = np.sum(y_true == y_pred) / len(y_true) * 100
                    st.plotly_chart(
                        plotly_per_class_accuracy(lnames, pca, ov),
                        use_container_width=True,
                    )
                    plotly_acc_built = True
            except Exception:
                pass

        if not plotly_acc_built:
            st.image(str(acc_img), use_container_width=True)

        with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
            chart_explain(tr("chart_peracc_explain"))

    if not has_any:
        st.warning(tr("no_eval_data"))

    # ── Tip: to enable full interactive charts, save arrays in evaluate.py ───
    if has_any and HAS_PLOTLY and not (RESULTS_DIR / "y_true.npy").exists():
        st.info(
            "💡 **Để mở khoá biểu đồ tương tác đầy đủ**, thêm các dòng sau vào cuối hàm `main()` "
            "trong `evaluate.py` (sau khi tính `y_true, y_pred`):\n\n"
            "```python\n"
            "import json\n"
            "np.save(RESULTS_DIR / 'y_true.npy', y_true)\n"
            "np.save(RESULTS_DIR / 'y_pred.npy', y_pred)\n"
            "with open(RESULTS_DIR / 'label_names.json', 'w', encoding='utf-8') as f:\n"
            "    json.dump(label_names, f, ensure_ascii=False)\n"
            "```"
        )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "history":
    st.markdown(
        f'<div class="hero"><h1>{tr("history")}</h1><p>{tr("action_log_title")}</p></div>',
        unsafe_allow_html=True,
    )
    if "recognition_history" not in st.session_state:
        st.session_state.recognition_history = []
    hist = st.session_state.recognition_history

    if not hist:
        st.info(tr("no_history"))
        st.stop()

    labels = [h["cu_chi"]    for h in hist]
    confs  = [h["do_tin_cay"] for h in hist]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("total_records"),  len(hist))
    c2.metric(tr("unique_actions"), len(set(labels)))
    c3.metric(tr("avg_confidence"), f"{np.mean(confs):.1f}%")
    c4.metric(tr("max_confidence"), f"{max(confs):.1f}%")

    # Table
    rows = []
    for i, h in enumerate(reversed(hist), 1):
        tay_str = "".join(
            f"{t['hand']} ({t['score']}%) " for t in h.get("chi_tiet_tay", [])
        )
        rows.append({
            "#":               i,
            tr("time"):        h["thoi_gian"],
            tr("action"):      h["cu_chi"],
            tr("confidence"):  h["do_tin_cay"],
            tr("hand_count"):  h.get("so_tay", 0),
            tr("hand_detail"): tay_str.strip() or "—",
        })
    hist_df = pd.DataFrame(rows)
    st.dataframe(hist_df, hide_index=True, use_container_width=True)

    if len(hist) > 1:
        tab1, tab2 = st.tabs(["Tần suất cử chỉ", "Xu hướng độ tin cậy"])
        with tab1:
            if HAS_PLOTLY:
                st.plotly_chart(plotly_freq_chart(labels), use_container_width=True)
            else:
                import matplotlib.pyplot as plt
                lc = pd.Series(labels).value_counts()
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor("#1e1e2e")
                style_ax(ax)
                ax.barh(lc.index, lc.values)
                st.pyplot(fig); plt.close(fig)
            with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
                chart_explain(tr("chart_freq_explain"))

        with tab2:
            if HAS_PLOTLY:
                st.plotly_chart(plotly_conf_trend(confs), use_container_width=True)
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor("#1e1e2e")
                style_ax(ax)
                ax.plot(range(len(confs)), confs, lw=1.5)
                ax.axhline(CONFIDENCE_THRESHOLD * 100, ls="--", alpha=.6)
                ax.set_ylim(0, 105)
                st.pyplot(fig); plt.close(fig)
            with st.expander("💡 Đọc hiểu biểu đồ này", expanded=False):
                chart_explain(tr("chart_conftrend_explain"))

    co1, co2 = st.columns(2)
    co1.download_button(
        tr("download_csv"),
        hist_df.to_csv(index=False).encode("utf-8-sig"),
        "lich_su_nhan_dien.csv", "text/csv",
        use_container_width=True,
    )
    if co2.button(tr("clear_history"), use_container_width=True):
        st.session_state.recognition_history = []
        st.session_state.action_text_log     = []
        st.rerun()