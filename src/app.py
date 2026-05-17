"""
app.py — Giao diện Streamlit cho hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam.

Tính năng:
- Hỗ trợ Tiếng Việt / English.
- Font Unicode ổn định cho giao diện và biểu đồ.
- Bố cục dashboard cân đối hơn.
- Hiệu ứng loading khi chuyển trang và khi chạy chức năng.
- Chạy extract_data.py, train.py, evaluate.py trực tiếp trên giao diện.
- Nhận diện realtime và lưu dòng thao tác đã nhận diện dưới camera.

Chạy:
    streamlit run src/app.py
"""
from __future__ import annotations

import html
import io
import json
import os
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

# =====================================================================
# Config an toàn — tránh lỗi khi config.py chưa khai báo biến mới
# =====================================================================
INPUT_SIZE = cfg.INPUT_SIZE
HIDDEN_SIZE = cfg.HIDDEN_SIZE
NUM_LAYERS = cfg.NUM_LAYERS
SEQ_LEN = cfg.SEQ_LEN
BATCH_SIZE = cfg.BATCH_SIZE
LEARNING_RATE = cfg.LEARNING_RATE
NUM_EPOCHS = cfg.NUM_EPOCHS
BEST_MODEL_PATH = cfg.BEST_MODEL_PATH
METADATA_PATH = cfg.METADATA_PATH
RESULTS_DIR = cfg.RESULTS_DIR
MIN_DETECTION_CONFIDENCE = cfg.MIN_DETECTION_CONFIDENCE
MIN_TRACKING_CONFIDENCE = cfg.MIN_TRACKING_CONFIDENCE
MAX_NUM_HANDS_DEMO = cfg.MAX_NUM_HANDS_DEMO
USE_VELOCITY = getattr(cfg, "USE_VELOCITY", False)
_BASE_INPUT = getattr(cfg, "_BASE_INPUT", cfg.NUM_HAND_LANDMARKS * cfg.COORDS_PER_LANDMARK)
CONFIDENCE_THRESHOLD = getattr(cfg, "CONFIDENCE_THRESHOLD", 0.75)
MIN_HAND_FRAMES_RATIO = getattr(cfg, "MIN_HAND_FRAMES_RATIO", 0.60)

# Matplotlib font Unicode cho Tiếng Việt
try:
    import matplotlib as mpl
    mpl.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Segoe UI", "Tahoma", "sans-serif"]
    mpl.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

st.set_page_config(page_title="VSL Recognition", page_icon="🤟", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# I18N
# =====================================================================
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
        "run_on_ui": "Chạy trực tiếp trên giao diện",
        "run_extract": "Trích xuất dữ liệu",
        "run_train": "Huấn luyện mô hình",
        "run_evaluate": "Đánh giá mô hình",
        "running_extract": "Đang trích xuất keypoints từ video...",
        "running_train": "Đang huấn luyện mô hình...",
        "running_evaluate": "Đang đánh giá mô hình...",
        "run_success": "Hoàn tất thành công.",
        "run_failed": "Chức năng chạy thất bại. Vui lòng xem log bên dưới.",
        "latest_log": "Nhật ký xử lý",
        "no_terminal_needed": "Bạn có thể chạy extract, train và evaluate tại đây mà không cần mở terminal trong IDE.",
        "start": "Bắt đầu",
        "stop": "Dừng lại",
        "webcam_failed": "Không thể mở webcam.",
        "need_model": "Chưa tải được mô hình. Vui lòng huấn luyện trước khi sử dụng.",
        "realtime_hint": "Nhấn Bắt đầu để mở webcam. Hệ thống chỉ hiện kết quả khi độ tin cậy đủ cao.",
        "buffer": "Bộ đệm khung hình",
        "hands_detected": "Số tay phát hiện",
        "no_hand": "Chưa phát hiện tay nào trong khung hình.",
        "make_hand_clear": "Đưa tay vào camera rõ hơn...",
        "low_confidence": "Độ tin cậy thấp",
        "collecting": "Đang thu thập frames...",
        "webcam_stopped": "Webcam đã dừng.",
        "action_log_title": "Các thao tác đã nhận diện",
        "action_log_empty": "Chưa có thao tác nào được nhận diện.",
        "left_hand": "Tay trái",
        "right_hand": "Tay phải",
        "index_finger": "Ngón trỏ",
        "no_history": "Chưa có dữ liệu lịch sử. Hãy mở trang Nhận diện thời gian thực và sử dụng webcam.",
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
        "no_train_data": "Chưa có dữ liệu huấn luyện. Bạn có thể chạy Train ngay trên giao diện.",
        "train_loss": "Train Loss cuối",
        "total_epoch": "Tổng epoch",
        "loss_chart": "Loss theo Epoch",
        "acc_chart": "Accuracy theo Epoch",
        "data_table": "Bảng dữ liệu",
        "classification_report": "Classification Report",
        "confusion_matrix": "Ma trận nhầm lẫn",
        "per_class_metrics": "Precision / Recall / F1-Score theo từng lớp",
        "per_class_accuracy": "Accuracy theo từng lớp",
        "no_eval_data": "Chưa có kết quả đánh giá. Bạn có thể chạy Evaluate ngay trên giao diện.",
        "progress_extract_desc": "Đọc video trong data/raw_videos và tạo data/keypoints/metadata.csv.",
        "progress_train_desc": "Huấn luyện LSTM và lưu weights/best_model.pth cùng results/train_history.json.",
        "progress_eval_desc": "Đánh giá model trên tập test và tạo confusion matrix, report, biểu đồ metrics.",
        "important_note": "Mỗi thư mục trong raw_videos phải đại diện cho một cử chỉ duy nhất. Nên có ít nhất 20 video cho mỗi nhãn để realtime ổn định.",
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
        "model_not_loaded": "Model is not loaded",
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
        "run_on_ui": "Run directly from UI",
        "run_extract": "Extract data",
        "run_train": "Train model",
        "run_evaluate": "Evaluate model",
        "running_extract": "Extracting keypoints from videos...",
        "running_train": "Training model...",
        "running_evaluate": "Evaluating model...",
        "run_success": "Completed successfully.",
        "run_failed": "The task failed. Please check the log below.",
        "latest_log": "Processing log",
        "no_terminal_needed": "You can run extract, train, and evaluate here without opening an IDE terminal.",
        "start": "Start",
        "stop": "Stop",
        "webcam_failed": "Cannot open webcam.",
        "need_model": "Model is not available. Please train the model first.",
        "realtime_hint": "Click Start to open webcam. The system displays a result only when confidence is high enough.",
        "buffer": "Frame buffer",
        "hands_detected": "Detected hands",
        "no_hand": "No hand detected in the frame.",
        "make_hand_clear": "Move your hand clearly into the camera...",
        "low_confidence": "Low confidence",
        "collecting": "Collecting frames...",
        "webcam_stopped": "Webcam stopped.",
        "action_log_title": "Recognized actions",
        "action_log_empty": "No action has been recognized yet.",
        "left_hand": "Left hand",
        "right_hand": "Right hand",
        "index_finger": "Index finger",
        "no_history": "No recognition history yet. Open Real-time Recognition and use the webcam.",
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
        "no_train_data": "No training history yet. You can run Train from the interface.",
        "train_loss": "Final Train Loss",
        "total_epoch": "Total epochs",
        "loss_chart": "Loss by Epoch",
        "acc_chart": "Accuracy by Epoch",
        "data_table": "Data table",
        "classification_report": "Classification Report",
        "confusion_matrix": "Confusion Matrix",
        "per_class_metrics": "Precision / Recall / F1-Score by class",
        "per_class_accuracy": "Accuracy by class",
        "no_eval_data": "No evaluation results yet. You can run Evaluate from the interface.",
        "progress_extract_desc": "Read videos in data/raw_videos and create data/keypoints/metadata.csv.",
        "progress_train_desc": "Train the LSTM model and save weights/best_model.pth plus results/train_history.json.",
        "progress_eval_desc": "Evaluate the model on the test set and generate confusion matrix, report, and metric charts.",
        "important_note": "Each folder in raw_videos must represent exactly one gesture. At least 20 videos per label are recommended for stable real-time performance.",
    },
}

LABEL_TEXT = {
    "cam_on": {"vi": "cảm ơn", "en": "thank you"},
    "chao": {"vi": "chào", "en": "hello"},
    "ten_la": {"vi": "tên là", "en": "my name is"},
    "toi": {"vi": "tôi", "en": "I / me"},
}

PAGE_LABEL_KEY = {
    "home": "home",
    "pipeline": "pipeline",
    "realtime": "realtime",
    "training_results": "training_results",
    "evaluation": "evaluation",
    "history": "history",
}


def lang() -> str:
    return st.session_state.get("lang", "vi")


def tr(key: str) -> str:
    return TEXT.get(lang(), TEXT["vi"]).get(key, key)


def display_label(label: Any) -> str:
    label = str(label)
    return LABEL_TEXT.get(label, {}).get(lang(), label)


# =====================================================================
# CSS
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stSelectbox, .stRadio, .stDataFrame, .stMetric{
    font-family:"Noto Sans","Segoe UI",Arial,Tahoma,sans-serif !important;
}
.main .block-container{padding-top:1.1rem; padding-bottom:2.5rem; max-width:1280px;}
#MainMenu{visibility:hidden} footer{visibility:hidden}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#11111b,#1e1e2e)}
.hero{background:linear-gradient(135deg,#111827,#1e1e2e); border:1px solid #313244; border-radius:24px; padding:2rem 2.2rem; margin-bottom:1rem; box-shadow:0 18px 50px rgba(0,0,0,.2)}
.hero h1{font-size:3rem; font-weight:900; background:linear-gradient(135deg,#89b4fa,#b4befe,#cba6f7); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 .35rem 0; letter-spacing:-1px;}
.hero p{color:#a6adc8; font-size:1.05rem; margin:0;}
.metric-card{background:linear-gradient(135deg,#1e1e2e,#181825);border:1px solid #313244;border-radius:18px;padding:1.05rem 1rem;text-align:center;transition:all .2s;min-height:108px;display:flex;flex-direction:column;justify-content:center;}
.metric-card:hover{transform:translateY(-3px); border-color:#89b4fa; box-shadow:0 12px 28px rgba(137,180,250,.12)}
.metric-card .value{font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#89b4fa,#cba6f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;}
.metric-card .label{font-size:.78rem;color:#a6adc8;text-transform:uppercase;letter-spacing:.8px;margin-top:.45rem;}
.card{background:linear-gradient(135deg,#1e1e2e,#181825);border:1px solid #313244;border-radius:18px;padding:1.1rem 1.2rem;margin:.55rem 0;box-shadow:0 10px 28px rgba(0,0,0,.16)}
.card-title{font-weight:800;color:#cdd6f4;font-size:1.05rem;margin-bottom:.35rem;}
.card-desc{color:#a6adc8;font-size:.92rem;line-height:1.55;}
.section-title{font-size:1.45rem;font-weight:900;color:#cdd6f4;margin:1rem 0 .6rem;}
.pred-box{background:linear-gradient(135deg,#1e1e2e,#181825);border:2px solid #89b4fa;border-radius:18px;padding:1.35rem;text-align:center;margin:.5rem 0;box-shadow:0 10px 26px rgba(137,180,250,.13)}
.pred-box .sign{font-size:2.05rem;font-weight:900;color:#89b4fa;line-height:1.15;}
.pred-box .conf{font-size:1.18rem;font-weight:800;margin-top:.35rem;}
.hand-info{background:#181825;border:1px solid #313244;border-radius:12px;padding:.85rem 1rem;margin:.35rem 0;font-size:.9rem;}
.hand-info .hand-label{font-weight:800;font-size:1rem;}
.waiting-box{background:#1e1e2e;border:2px dashed #45475a;border-radius:18px;padding:1.35rem;text-align:center;color:#a6adc8;line-height:1.55;}
.action-log{background:linear-gradient(135deg,#11111b,#181825);border:1px solid #45475a;border-radius:16px;padding:1rem 1.1rem;margin-top:.8rem;color:#cdd6f4;font-size:1rem;line-height:1.65;box-shadow:0 10px 22px rgba(0,0,0,.14)}
.action-log .title{color:#89b4fa;font-weight:900;margin-bottom:.35rem;}
.action-log .content{color:#f9e2af;font-weight:800;word-break:break-word;}
.load-card{background:linear-gradient(135deg,#1e1e2e,#11111b);border:1px solid #313244;border-radius:18px;padding:1.1rem 1.2rem;color:#cdd6f4;display:flex;align-items:center;gap:.8rem;margin:.5rem 0;}
.loader{width:22px;height:22px;border:3px solid #45475a;border-top-color:#89b4fa;border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.stButton button{border-radius:14px !important; font-weight:800 !important; min-height:42px;}
</style>
""",
    unsafe_allow_html=True,
)


def loading_card(text: str) -> None:
    st.markdown(f'<div class="load-card"><div class="loader"></div><div>{html.escape(text)}</div></div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, col) -> None:
    col.markdown(
        f'<div class="metric-card"><div class="value">{html.escape(str(value))}</div><div class="label">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# Helpers
# =====================================================================
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
        return {k: ckpt.get(k, "?") for k in ["epoch", "val_accuracy", "val_loss", "train_accuracy", "train_loss", "num_classes"]}
    except Exception:
        return {}


def build_sequence_with_velocity(raw_buffer: deque) -> np.ndarray:
    seq = np.array(list(raw_buffer), dtype=np.float32)
    velocity = np.zeros_like(seq)
    if len(seq) > 1:
        velocity[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, velocity], axis=1)


def run_script(script_name: str, running_text: str) -> bool:
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        st.error(f"Không tìm thấy file: {script_path}")
        return False

    log_box = st.empty()
    progress = st.progress(0)
    lines: list[str] = []
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    loading_card(running_text)
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
        tick = 5
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            if clean:
                lines.append(clean)
            tick = min(95, tick + 1)
            progress.progress(tick)
            log_box.code("\n".join(lines[-160:]), language="text")
        return_code = process.wait()
        progress.progress(100)
        if return_code == 0:
            st.success(tr("run_success"))
            clear_model_cache()
            return True
        st.error(tr("run_failed"))
        return False
    except Exception as e:
        st.error(f"{tr('run_failed')}\n{e}")
        return False


def render_pipeline_button(title_key: str, desc_key: str, script_name: str, running_key: str, button_key: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{html.escape(tr(title_key))}</div>
            <div class="card-desc">{html.escape(tr(desc_key))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(tr(title_key), key=button_key, type="primary", use_container_width=True):
        with st.spinner(tr(running_key)):
            run_script(script_name, tr(running_key))


def style_ax(ax):
    ax.set_facecolor("#11111b")
    ax.tick_params(colors="#6c7086")
    ax.grid(alpha=0.2, color="#313244")
    for s in ax.spines.values():
        s.set_color("#313244")


# =====================================================================
# Sidebar
# =====================================================================
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
    page = st.radio(tr("navigation"), list(PAGE_LABEL_KEY.keys()), format_func=lambda k: tr(PAGE_LABEL_KEY[k]))

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
    st.caption(f"INPUT_SIZE = {INPUT_SIZE} ({'position + velocity' if USE_VELOCITY else 'position only'})")

# Hiệu ứng loading khi chuyển trang
if st.session_state.get("_last_page") != page:
    placeholder = st.empty()
    with placeholder.container():
        loading_card(tr("loading_page"))
    time.sleep(0.18)
    placeholder.empty()
    st.session_state["_last_page"] = page

# =====================================================================
# HOME
# =====================================================================
if page == "home":
    st.markdown(f'<div class="hero"><h1>{tr("app_title")}</h1><p>{tr("app_subtitle")}</p></div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for c, (l, v) in zip(
        cols,
        [(tr("seq_len"), SEQ_LEN), (tr("input_size"), INPUT_SIZE), (tr("hidden_size"), HIDDEN_SIZE), (tr("num_layers"), NUM_LAYERS), (tr("batch_size"), BATCH_SIZE), (tr("num_epochs"), NUM_EPOCHS)],
    ):
        metric_card(l, str(v), c)

    st.markdown(f'<div class="section-title">{tr("model_info")}</div>', unsafe_allow_html=True)
    ckpt = load_checkpoint_info()
    if ckpt:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("best_epoch"), ckpt.get("epoch", "?"))
        c2.metric(tr("val_accuracy"), f"{ckpt.get('val_accuracy', 0):.1f}%" if isinstance(ckpt.get("val_accuracy", 0), (int, float)) else "?")
        c3.metric(tr("val_loss"), f"{ckpt.get('val_loss', 0):.4f}" if isinstance(ckpt.get("val_loss", 0), (int, float)) else "?")
        c4.metric(tr("num_classes"), ckpt.get("num_classes", "?"))
    else:
        st.info(tr("model_not_loaded"))

    if label_map:
        st.markdown(f'<div class="section-title">{tr("labels")}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"ID": k, tr("action"): display_label(v), "Raw label": v} for k, v in sorted(label_map.items())]), hide_index=True, width="stretch")

    st.markdown(f'<div class="section-title">{tr("guide")}</div>', unsafe_allow_html=True)
    guide_df = pd.DataFrame(
        [
            {"Step": 1, "Action": "data/raw_videos/<label>", "Description": tr("important_note")},
            {"Step": 2, "Action": "extract_data.py", "Description": tr("progress_extract_desc")},
            {"Step": 3, "Action": "train.py", "Description": tr("progress_train_desc")},
            {"Step": 4, "Action": "evaluate.py", "Description": tr("progress_eval_desc")},
            {"Step": 5, "Action": "app.py", "Description": tr("realtime_hint")},
        ]
    )
    st.dataframe(guide_df, hide_index=True, width="stretch")
    st.info(tr("important_note"), icon="⚠️")

# =====================================================================
# PIPELINE
# =====================================================================
elif page == "pipeline":
    st.markdown(f'<div class="hero"><h1>{tr("run_on_ui")}</h1><p>{tr("no_terminal_needed")}</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_pipeline_button("run_extract", "progress_extract_desc", "extract_data.py", "running_extract", "pipe_extract")
    with c2:
        render_pipeline_button("run_train", "progress_train_desc", "train.py", "running_train", "pipe_train")
    with c3:
        render_pipeline_button("run_evaluate", "progress_eval_desc", "evaluate.py", "running_evaluate", "pipe_eval")
    st.markdown(f'<div class="section-title">{tr("latest_log")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="card-desc">{html.escape(tr("important_note"))}</div></div>', unsafe_allow_html=True)

# =====================================================================
# REALTIME
# =====================================================================
elif page == "realtime":
    st.markdown(f'<div class="hero"><h1>{tr("realtime")}</h1><p>{tr("realtime_hint")}</p></div>', unsafe_allow_html=True)
    if not model:
        st.error(tr("need_model"))
        st.stop()

    c1, c2 = st.columns([1, 4])
    start = c1.button(tr("start"), type="primary", use_container_width=True)
    stop_ph = c2.empty()

    col_v, col_i = st.columns([3.2, 1])
    vid_ph = col_v.empty()
    log_box_ph = col_v.empty()
    with col_i:
        pred_ph = st.empty()
        hand_info_ph = st.empty()
        buf_ph = st.empty()
        status_ph = st.empty()

    if start:
        with st.spinner(tr("loading_page")):
            import mediapipe.python.solutions.drawing_utils as mp_drawing
            import mediapipe.python.solutions.hands as mp_hands

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error(tr("webcam_failed"))
            st.stop()

        raw_buf = deque(maxlen=SEQ_LEN)
        hand_detected_buf = deque(maxlen=SEQ_LEN)
        dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stop = stop_ph.button(tr("stop"), type="secondary")
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_NUM_HANDS_DEMO, min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

        if "recognition_history" not in st.session_state:
            st.session_state.recognition_history = []
        if "action_text_log" not in st.session_state:
            st.session_state.action_text_log = []
        if "last_action_label" not in st.session_state:
            st.session_state.last_action_label = None
        if "last_action_time" not in st.session_state:
            st.session_state.last_action_time = 0.0

        LEFT_COLOR = (248, 113, 113)
        RIGHT_COLOR = (56, 189, 248)

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h_f, w_f = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            kp = np.zeros(_BASE_INPUT, dtype=np.float32)
            hand_detected = False
            hands_info = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (hl, hd) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    raw_label = hd.classification[0].label
                    hand_score = hd.classification[0].score
                    hand_name_key = "left_hand" if raw_label == "Right" else "right_hand"
                    hand_name = tr(hand_name_key)
                    hand_name_cv = "Left hand" if lang() == "en" else ("Tay trai" if hand_name_key == "left_hand" else "Tay phai")
                    color = LEFT_COLOR if hand_name_key == "left_hand" else RIGHT_COLOR

                    wrist = hl.landmark[0]
                    idx_tip = hl.landmark[8]
                    nx = round(idx_tip.x - wrist.x, 4)
                    ny = round(idx_tip.y - wrist.y, 4)
                    nz = round(idx_tip.z - wrist.z, 4)

                    hands_info.append({"hand": hand_name, "score": round(hand_score * 100, 1), "wrist_x": round(wrist.x, 3), "wrist_y": round(wrist.y, 3), "index_tip_x": nx, "index_tip_y": ny, "index_tip_z": nz})

                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3), mp_drawing.DrawingSpec(color=color, thickness=2))
                    wx, wy = int(wrist.x * w_f), int(wrist.y * h_f)
                    cv2.putText(frame, hand_name_cv, (wx - 30, wy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if i == 0:
                        vals = []
                        for lm in hl.landmark:
                            vals.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                        kp = np.array(vals, dtype=np.float32)
                        hand_detected = True

            raw_buf.append(kp)
            hand_detected_buf.append(1 if hand_detected else 0)
            bl = len(raw_buf)

            buf_ph.metric(tr("buffer"), f"{bl} / {SEQ_LEN}")
            n_hands = len(hands_info)
            status_ph.metric(tr("hands_detected"), f"{n_hands}")

            if hands_info:
                hand_html = ""
                for hi in hands_info:
                    c = "#f87171" if hi["hand"] == tr("left_hand") else "#89b4fa"
                    hand_html += (
                        f'<div class="hand-info"><div class="hand-label" style="color:{c}">{html.escape(hi["hand"])} ({hi["score"]}%)</div>'
                        f'<div style="color:#a6adc8;font-size:.82rem;margin-top:4px">{tr("index_finger")}: X={hi["index_tip_x"]:+.3f}  Y={hi["index_tip_y"]:+.3f}  Z={hi["index_tip_z"]:+.3f}</div></div>'
                    )
                hand_info_ph.markdown(hand_html, unsafe_allow_html=True)
            else:
                hand_info_ph.markdown(f'<div class="hand-info" style="color:#6c7086">{tr("no_hand")}</div>', unsafe_allow_html=True)

            if bl == SEQ_LEN:
                hand_ratio = sum(hand_detected_buf) / SEQ_LEN
                if hand_ratio < MIN_HAND_FRAMES_RATIO:
                    pred_ph.markdown(f'<div class="waiting-box">{tr("make_hand_clear")}</div>', unsafe_allow_html=True)
                else:
                    seq = build_sequence_with_velocity(raw_buf) if USE_VELOCITY else np.array(list(raw_buf), dtype=np.float32)
                    tensor = torch.tensor(seq).unsqueeze(0).to(dev_t)
                    with torch.no_grad():
                        out = model(tensor)
                        probs = torch.softmax(out, dim=1)
                        prob, idx = torch.max(probs, dim=1)
                        prediction_conf = prob.item() * 100
                        raw_prediction_label = label_map.get(idx.item(), f"ID:{idx.item()}")
                        prediction_label = display_label(raw_prediction_label)

                    if prediction_conf < CONFIDENCE_THRESHOLD * 100:
                        pred_ph.markdown(f'<div class="waiting-box">{tr("low_confidence")} ({prediction_conf:.1f}%)<br><small>Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%</small></div>', unsafe_allow_html=True)
                    else:
                        cc = "#a6e3a1" if prediction_conf > 80 else "#f9e2af" if prediction_conf > 65 else "#f38ba8"
                        pred_ph.markdown(f'<div class="pred-box"><div class="sign">{html.escape(prediction_label)}</div><div class="conf" style="color:{cc}">{prediction_conf:.1f}%</div></div>', unsafe_allow_html=True)

                        cv2.rectangle(frame, (0, h_f - 55), (w_f, h_f), (0, 0, 0), -1)
                        cv2.putText(frame, f"{raw_prediction_label}  ({prediction_conf:.1f}%)", (15, h_f - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56, 189, 248), 2)

                        st.session_state.recognition_history.append({"thoi_gian": datetime.now().strftime("%H:%M:%S"), "cu_chi": prediction_label, "raw_label": raw_prediction_label, "do_tin_cay": round(prediction_conf, 2), "so_tay": n_hands, "chi_tiet_tay": hands_info.copy() if hands_info else []})

                        now_time = time.time()
                        is_new_action = raw_prediction_label != st.session_state.last_action_label or now_time - st.session_state.last_action_time > 1.2
                        if is_new_action:
                            st.session_state.action_text_log.append(prediction_label)
                            st.session_state.action_text_log = st.session_state.action_text_log[-20:]
                            st.session_state.last_action_label = raw_prediction_label
                            st.session_state.last_action_time = now_time
            else:
                pred_ph.markdown(f'<div class="waiting-box">{tr("collecting")}<br><small>{bl}/{SEQ_LEN}</small></div>', unsafe_allow_html=True)

            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_ph.image(display, channels="RGB", width="stretch")
            action_text = " → ".join(st.session_state.get("action_text_log", [])) or tr("action_log_empty")
            log_box_ph.markdown(f'<div class="action-log"><div class="title">{html.escape(tr("action_log_title"))}</div><div class="content">{html.escape(action_text)}</div></div>', unsafe_allow_html=True)
            time.sleep(0.03)

        cap.release()
        hands.close()
        st.info(tr("webcam_stopped"))

# =====================================================================
# TRAINING RESULTS
# =====================================================================
elif page == "training_results":
    st.markdown(f'<div class="hero"><h1>{tr("training_results")}</h1><p>{tr("progress_train_desc")}</p></div>', unsafe_allow_html=True)
    if st.button(tr("run_train"), key="train_from_results", type="primary", use_container_width=True):
        with st.spinner(tr("running_train")):
            run_script("train.py", tr("running_train"))

    history = load_train_history()
    if history is None:
        st.warning(tr("no_train_data"))
        st.stop()

    df = pd.DataFrame(history)
    best = df.loc[df["val_acc"].idxmax()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(tr("total_epoch"), len(df))
    c2.metric(tr("best_epoch"), int(best["epoch"]))
    c3.metric(tr("val_accuracy"), f"{best['val_acc']:.2f}%")
    c4.metric(tr("train_loss"), f"{df.iloc[-1]['train_loss']:.4f}")
    c5.metric(tr("val_loss"), f"{df.iloc[-1]['val_loss']:.4f}")

    import matplotlib.pyplot as plt
    tab1, tab2, tab3 = st.tabs([tr("loss_chart"), tr("acc_chart"), tr("data_table")])
    with tab1:
        with st.spinner(tr("loading_page")):
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1e1e2e")
            style_ax(ax)
            ax.plot(df["epoch"], df["train_loss"], lw=2, label="Train Loss")
            ax.plot(df["epoch"], df["val_loss"], lw=2, ls="--", label="Val Loss")
            ax.set_xlabel("Epoch", color="#cdd6f4")
            ax.set_ylabel("Loss", color="#cdd6f4")
            ax.set_title(tr("loss_chart"), color="#cdd6f4", fontsize=14, fontweight="bold")
            ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")
            st.pyplot(fig)
            plt.close(fig)
    with tab2:
        with st.spinner(tr("loading_page")):
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1e1e2e")
            style_ax(ax)
            ax.plot(df["epoch"], df["train_acc"], lw=2, label="Train Acc")
            ax.plot(df["epoch"], df["val_acc"], lw=2, ls="--", label="Val Acc")
            ax.set_xlabel("Epoch", color="#cdd6f4")
            ax.set_ylabel("Accuracy (%)", color="#cdd6f4")
            ax.set_title(tr("acc_chart"), color="#cdd6f4", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 105)
            ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")
            st.pyplot(fig)
            plt.close(fig)
    with tab3:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(tr("download_csv"), df.to_csv(index=False).encode("utf-8-sig"), "train_history.csv", "text/csv")

# =====================================================================
# EVALUATION
# =====================================================================
elif page == "evaluation":
    st.markdown(f'<div class="hero"><h1>{tr("evaluation")}</h1><p>{tr("progress_eval_desc")}</p></div>', unsafe_allow_html=True)
    if st.button(tr("run_evaluate"), key="eval_from_page", type="primary", use_container_width=True):
        with st.spinner(tr("running_evaluate")):
            run_script("evaluate.py", tr("running_evaluate"))

    has = False
    rp = RESULTS_DIR / "classification_report.txt"
    if rp.exists():
        has = True
        st.subheader(tr("classification_report"))
        with open(rp, encoding="utf-8") as f:
            st.code(f.read(), language="text")
    for title_key, fn in [("confusion_matrix", "confusion_matrix.png"), ("per_class_metrics", "per_class_metrics.png"), ("per_class_accuracy", "per_class_accuracy.png")]:
        p = RESULTS_DIR / fn
        if p.exists():
            has = True
            st.subheader(tr(title_key))
            st.image(str(p), width="stretch")
    if not has:
        st.warning(tr("no_eval_data"))

# =====================================================================
# HISTORY
# =====================================================================
elif page == "history":
    st.markdown(f'<div class="hero"><h1>{tr("history")}</h1><p>{tr("action_log_title")}</p></div>', unsafe_allow_html=True)
    if "recognition_history" not in st.session_state:
        st.session_state.recognition_history = []
    hist = st.session_state.recognition_history
    if not hist:
        st.info(tr("no_history"))
        st.stop()

    labels = [h["cu_chi"] for h in hist]
    confs = [h["do_tin_cay"] for h in hist]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("total_records"), len(hist))
    c2.metric(tr("unique_actions"), len(set(labels)))
    c3.metric(tr("avg_confidence"), f"{np.mean(confs):.1f}%")
    c4.metric(tr("max_confidence"), f"{max(confs):.1f}%")

    rows = []
    for i, h in enumerate(reversed(hist), 1):
        tay_str = ""
        for t in h.get("chi_tiet_tay", []):
            tay_str += f"{t['hand']} ({t['score']}%) "
        rows.append({"#": i, tr("time"): h["thoi_gian"], tr("action"): h["cu_chi"], tr("confidence"): h["do_tin_cay"], tr("hand_count"): h.get("so_tay", 0), tr("hand_detail"): tay_str.strip() or "—"})
    hist_df = pd.DataFrame(rows)
    st.dataframe(hist_df, hide_index=True, width="stretch")

    if len(hist) > 1:
        import matplotlib.pyplot as plt
        tab1, tab2 = st.tabs(["Frequency", tr("confidence")])
        with tab1:
            lc = pd.Series(labels).value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1e1e2e")
            style_ax(ax)
            ax.barh(lc.index, lc.values)
            ax.set_title(tr("unique_actions"), color="#cdd6f4", fontsize=13, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#1e1e2e")
            style_ax(ax)
            ax.plot(range(len(confs)), confs, lw=1.5)
            ax.axhline(CONFIDENCE_THRESHOLD * 100, ls="--", alpha=.6, label=f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
            ax.set_title(tr("avg_confidence"), color="#cdd6f4", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 105)
            ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")
            st.pyplot(fig)
            plt.close(fig)

    co1, co2 = st.columns(2)
    co1.download_button(tr("download_csv"), hist_df.to_csv(index=False).encode("utf-8-sig"), "lich_su_nhan_dien.csv", "text/csv")
    if co2.button(tr("clear_history"), use_container_width=True):
        st.session_state.recognition_history = []
        st.session_state.action_text_log = []
        st.rerun()
