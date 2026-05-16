"""
app.py — Giao diện Streamlit cho hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam.

Chạy: streamlit run src/app.py
"""
import streamlit as st
import json, sys, io, time
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from collections import deque
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS,
    BEST_MODEL_PATH, METADATA_PATH, RESULTS_DIR,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS_DEMO,
)
from model import SignLanguageLSTM
from dataset import VSLDataset

# =====================================================================
st.set_page_config(page_title="VSL Recognition", page_icon="🤟", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
.main .block-container{padding-top:1rem}
.metric-card{background:linear-gradient(135deg,#1e1e2e,#181825);border:1px solid #313244;border-radius:12px;padding:1.2rem;text-align:center;transition:transform .2s}
.metric-card:hover{transform:translateY(-2px)}
.metric-card .value{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#89b4fa,#cba6f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-card .label{font-size:.8rem;color:#6c7086;text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.hero{text-align:center;padding:2rem 0}
.hero h1{font-size:3.2rem;font-weight:900;background:linear-gradient(135deg,#89b4fa,#b4befe,#cba6f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0}
.hero p{color:#a6adc8;font-size:1.05rem;margin-top:.3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#11111b,#1e1e2e)}
#MainMenu{visibility:hidden}footer{visibility:hidden}
.pred-box{background:linear-gradient(135deg,#1e1e2e,#181825);border:2px solid #89b4fa;border-radius:16px;padding:1.5rem;text-align:center;margin:.5rem 0}
.pred-box .sign{font-size:2.2rem;font-weight:900;color:#89b4fa}
.pred-box .conf{font-size:1.3rem;font-weight:700}
.hand-info{background:#181825;border:1px solid #313244;border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;font-size:.9rem}
.hand-info .hand-label{font-weight:700;font-size:1rem}
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, col):
    col.markdown(f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

# =====================================================================
# Helpers
# =====================================================================
@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        return None, None, "Chưa có model. Hãy huấn luyện trước."
    if not METADATA_PATH.exists():
        return None, None, "Chưa có metadata. Hãy chạy extract_data.py."
    try:
        old = sys.stdout; sys.stdout = io.StringIO()
        try:
            ds = VSLDataset(METADATA_PATH); lm = ds.get_label_map()
        finally:
            sys.stdout = old
        model = SignLanguageLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, ds.num_classes)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        ckpt = torch.load(BEST_MODEL_PATH, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        return model, lm, None
    except Exception as e:
        return None, None, str(e)

def load_train_history():
    p = RESULTS_DIR / "train_history.json"
    if not p.exists(): return None
    with open(p, encoding="utf-8") as f: return json.load(f)

def load_checkpoint_info():
    if not BEST_MODEL_PATH.exists(): return {}
    try:
        ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
        return {k: ckpt.get(k, "?") for k in ["epoch","val_accuracy","val_loss","train_accuracy","train_loss","num_classes"]}
    except: return {}

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("### VSL Recognition")
    st.caption("Nhận diện ngôn ngữ ký hiệu Việt Nam")
    st.markdown("---")
    page = st.radio("Điều hướng", [
        "Trang chủ", "Nhận diện thời gian thực",
        "Kết quả huấn luyện", "Đánh giá mô hình", "Lịch sử nhận diện"
    ], label_visibility="collapsed")
    st.markdown("---")
    dev = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"**Thiết bị:** `{dev}`")
    model, label_map, err = load_model()
    if model: st.success("Mô hình đã tải thành công")
    else: st.warning(err or "Chưa tải mô hình")
    st.caption(f"Python {sys.version.split()[0]} — PyTorch {torch.__version__}")

# =====================================================================
# TRANG CHỦ
# =====================================================================
if page == "Trang chủ":
    st.markdown('<div class="hero"><h1>VSL Recognition</h1><p>Hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam sử dụng mô hình LSTM kết hợp MediaPipe</p></div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for c, (l, v) in zip(cols, [("Độ dài chuỗi", SEQ_LEN), ("Đầu vào", INPUT_SIZE), ("Ẩn LSTM", HIDDEN_SIZE), ("Số lớp LSTM", NUM_LAYERS), ("Batch Size", BATCH_SIZE), ("Số Epoch", NUM_EPOCHS)]):
        metric_card(l, str(v), c)
    st.markdown("---")
    ckpt = load_checkpoint_info()
    if ckpt:
        st.subheader("Thông tin mô hình")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Epoch tốt nhất", ckpt.get("epoch","?"))
        c2.metric("Val Accuracy", f"{ckpt.get('val_accuracy',0):.1f}%" if isinstance(ckpt.get('val_accuracy',0),(int,float)) else "?")
        c3.metric("Val Loss", f"{ckpt.get('val_loss',0):.4f}" if isinstance(ckpt.get('val_loss',0),(int,float)) else "?")
        c4.metric("Số lớp", ckpt.get("num_classes","?"))
    if label_map:
        st.subheader("Danh sách nhãn nhận diện")
        st.dataframe(pd.DataFrame([{"ID": k, "Tên nhãn": v} for k, v in sorted(label_map.items())]), hide_index=True, width="stretch")
    st.subheader("Hướng dẫn sử dụng")
    st.markdown("""
| Bước | Lệnh | Mô tả |
|------|-------|-------|
| 1 | `python src/extract_data.py` | Trích xuất keypoints từ video thô |
| 2 | `python src/train.py` | Huấn luyện mô hình LSTM |
| 3 | `python src/evaluate.py` | Đánh giá mô hình trên tập kiểm thử |
| 4 | `streamlit run src/app.py` | Khởi chạy giao diện Streamlit |
""")

# =====================================================================
# NHẬN DIỆN THỜI GIAN THỰC
# =====================================================================
elif page == "Nhận diện thời gian thực":
    st.header("Nhận diện thời gian thực")
    if not model:
        st.error("Chưa tải được mô hình. Vui lòng huấn luyện trước khi sử dụng."); st.stop()
    st.markdown("Nhấn **Bắt đầu** để mở webcam. Đưa tay vào camera — hệ thống sẽ nhận diện cả hai tay, phân biệt tay trái/tay phải và hiển thị kết quả dự đoán cử chỉ.")
    c1, c2 = st.columns([1, 4])
    start = c1.button("Bắt đầu", type="primary", use_container_width=True)
    stop_ph = c2.empty()
    col_v, col_i = st.columns([3, 1])
    vid_ph = col_v.empty()
    with col_i:
        pred_ph = st.empty(); hand_info_ph = st.empty(); buf_ph = st.empty(); status_ph = st.empty()

    if start:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể mở webcam."); st.stop()
        buf = deque(maxlen=SEQ_LEN)
        dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stop = stop_ph.button("Dừng lại", type="secondary")
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_NUM_HANDS_DEMO,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
        if "recognition_history" not in st.session_state:
            st.session_state.recognition_history = []

        LEFT_COLOR = (248, 113, 113)   # đỏ nhạt cho tay trái
        RIGHT_COLOR = (56, 189, 248)   # xanh dương cho tay phải

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h_f, w_f = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            kp = np.zeros(INPUT_SIZE, dtype=np.float32)
            hands_info = []  # thông tin từng tay

            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (hl, hd) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # MediaPipe trả về "Left"/"Right" dựa trên ảnh gốc (chưa flip)
                    # Sau cv2.flip, tay trái thực tế hiển thị bên trái màn hình
                    raw_label = hd.classification[0].label
                    hand_score = hd.classification[0].score
                    # Sau flip: Left trong MediaPipe = Tay phải thực tế
                    hand_name = "Tay trái" if raw_label == "Right" else "Tay phải"
                    color = LEFT_COLOR if hand_name == "Tay trái" else RIGHT_COLOR

                    wrist = hl.landmark[0]
                    idx_tip = hl.landmark[8]

                    # Tọa độ chuẩn hóa (lấy cổ tay làm gốc)
                    nx = round(idx_tip.x - wrist.x, 4)
                    ny = round(idx_tip.y - wrist.y, 4)
                    nz = round(idx_tip.z - wrist.z, 4)

                    hands_info.append({
                        "hand": hand_name,
                        "score": round(hand_score * 100, 1),
                        "wrist_x": round(wrist.x, 3), "wrist_y": round(wrist.y, 3),
                        "index_tip_x": nx, "index_tip_y": ny, "index_tip_z": nz,
                    })

                    # Vẽ landmarks với màu riêng cho mỗi tay
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=color, thickness=2))

                    # Vẽ nhãn tay lên frame
                    wx, wy = int(wrist.x * w_f), int(wrist.y * h_f)
                    cv2.putText(frame, hand_name, (wx - 30, wy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Lấy keypoints từ tay đầu tiên cho model
                    if i == 0:
                        vals = []
                        for lm in hl.landmark:
                            vals.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                        kp = np.array(vals, dtype=np.float32)

            buf.append(kp)
            bl = len(buf)

            # Cập nhật thông tin bên phải
            buf_ph.metric("Bộ đệm khung hình", f"{bl} / {SEQ_LEN}")
            n_hands = len(hands_info)
            status_ph.metric("Số tay phát hiện", f"{n_hands}")

            # Hiển thị chi tiết từng tay
            if hands_info:
                html = ""
                for hi in hands_info:
                    c = "#f87171" if hi["hand"] == "Tay trái" else "#89b4fa"
                    html += f'''<div class="hand-info">
                        <div class="hand-label" style="color:{c}">{hi["hand"]} ({hi["score"]}%)</div>
                        <div style="color:#a6adc8;font-size:.82rem;margin-top:4px">
                        Ngón trỏ: X={hi["index_tip_x"]:+.3f}  Y={hi["index_tip_y"]:+.3f}  Z={hi["index_tip_z"]:+.3f}
                        </div></div>'''
                hand_info_ph.markdown(html, unsafe_allow_html=True)
            else:
                hand_info_ph.markdown('<div class="hand-info" style="color:#6c7086">Chưa phát hiện tay nào trong khung hình.</div>', unsafe_allow_html=True)

            # Dự đoán
            prediction_label = None
            prediction_conf = 0.0
            if bl == SEQ_LEN:
                seq = np.array(list(buf), dtype=np.float32)
                tensor = torch.tensor(seq).unsqueeze(0).to(dev_t)
                with torch.no_grad():
                    out = model(tensor)
                    probs = torch.softmax(out, dim=1)
                    prob, idx = torch.max(probs, dim=1)
                    prediction_conf = prob.item() * 100
                    prediction_label = label_map.get(idx.item(), f"ID:{idx.item()}")

                cc = "#a6e3a1" if prediction_conf > 70 else "#f9e2af" if prediction_conf > 40 else "#f38ba8"
                pred_ph.markdown(f'''<div class="pred-box">
                    <div class="sign">{prediction_label}</div>
                    <div class="conf" style="color:{cc}">{prediction_conf:.1f}%</div>
                </div>''', unsafe_allow_html=True)

                # Vẽ dự đoán lên frame
                cv2.rectangle(frame, (0, h_f-55), (w_f, h_f), (0,0,0), -1)
                cv2.putText(frame, f"{prediction_label}  ({prediction_conf:.1f}%)",
                    (15, h_f-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56,189,248), 2)

                # Lưu lịch sử
                st.session_state.recognition_history.append({
                    "thoi_gian": datetime.now().strftime("%H:%M:%S"),
                    "cu_chi": prediction_label,
                    "do_tin_cay": round(prediction_conf, 2),
                    "so_tay": n_hands,
                    "chi_tiet_tay": hands_info.copy() if hands_info else [],
                })

            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_ph.image(display, channels="RGB", width="stretch")
            time.sleep(0.03)

        cap.release(); hands.close()
        st.info("Webcam đã dừng.")


# =====================================================================
# KẾT QUẢ HUẤN LUYỆN
# =====================================================================
elif page == "Kết quả huấn luyện":
    st.header("Kết quả huấn luyện")
    history = load_train_history()
    if history is None:
        st.warning("Chưa có dữ liệu huấn luyện. Hãy chạy lệnh sau trong terminal:")
        st.code("python src/train.py", language="bash")
        st.markdown("Sau khi hoàn tất, tệp `results/train_history.json` sẽ được tạo tự động. Quay lại trang này để xem kết quả.")
        st.stop()

    df = pd.DataFrame(history)
    best = df.loc[df["val_acc"].idxmax()]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Tổng epoch", len(df))
    c2.metric("Epoch tốt nhất", int(best["epoch"]))
    c3.metric("Val Accuracy cao nhất", f"{best['val_acc']:.2f}%")
    c4.metric("Train Loss cuối", f"{df.iloc[-1]['train_loss']:.4f}")
    c5.metric("Val Loss cuối", f"{df.iloc[-1]['val_loss']:.4f}")
    st.markdown("---")

    import matplotlib.pyplot as plt
    tab1, tab2, tab3 = st.tabs(["Loss theo Epoch", "Accuracy theo Epoch", "Bảng dữ liệu"])

    def style_ax(ax):
        ax.set_facecolor("#11111b"); ax.tick_params(colors="#6c7086"); ax.grid(alpha=0.2, color="#313244")
        for s in ax.spines.values(): s.set_color("#313244")

    with tab1:
        fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor("#1e1e2e"); style_ax(ax)
        ax.plot(df["epoch"], df["train_loss"], color="#89b4fa", lw=2, label="Train Loss")
        ax.plot(df["epoch"], df["val_loss"], color="#f38ba8", lw=2, ls="--", label="Val Loss")
        ax.fill_between(df["epoch"], df["train_loss"], alpha=0.1, color="#89b4fa")
        ax.set_xlabel("Epoch", color="#cdd6f4"); ax.set_ylabel("Loss", color="#cdd6f4")
        ax.set_title("Biểu đồ Loss theo Epoch", color="#cdd6f4", fontsize=14, fontweight="bold")
        ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4"); st.pyplot(fig); plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor("#1e1e2e"); style_ax(ax)
        ax.plot(df["epoch"], df["train_acc"], color="#a6e3a1", lw=2, label="Train Acc")
        ax.plot(df["epoch"], df["val_acc"], color="#f9e2af", lw=2, ls="--", label="Val Acc")
        ax.fill_between(df["epoch"], df["train_acc"], alpha=0.1, color="#a6e3a1")
        ax.set_xlabel("Epoch", color="#cdd6f4"); ax.set_ylabel("Accuracy (%)", color="#cdd6f4")
        ax.set_title("Biểu đồ Accuracy theo Epoch", color="#cdd6f4", fontsize=14, fontweight="bold")
        ax.set_ylim(0,105); ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4"); st.pyplot(fig); plt.close(fig)

    with tab3:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Tải xuống CSV", df.to_csv(index=False).encode("utf-8"), "train_history.csv", "text/csv")


# =====================================================================
# ĐÁNH GIÁ MÔ HÌNH
# =====================================================================
elif page == "Đánh giá mô hình":
    st.header("Đánh giá mô hình")
    has = False
    rp = RESULTS_DIR / "classification_report.txt"
    if rp.exists():
        has = True; st.subheader("Classification Report")
        with open(rp, encoding="utf-8") as f: st.code(f.read(), language="text")
    for title, fn in [("Ma trận nhầm lẫn (Confusion Matrix)","confusion_matrix.png"),
                      ("Precision / Recall / F1-Score theo từng lớp","per_class_metrics.png"),
                      ("Accuracy theo từng lớp","per_class_accuracy.png")]:
        p = RESULTS_DIR / fn
        if p.exists(): has = True; st.subheader(title); st.image(str(p), width="stretch")
    if not has:
        st.warning("Chưa có kết quả đánh giá. Hãy chạy lệnh sau:")
        st.code("python src/evaluate.py", language="bash")


# =====================================================================
# LỊCH SỬ NHẬN DIỆN
# =====================================================================
elif page == "Lịch sử nhận diện":
    st.header("Lịch sử nhận diện")
    if "recognition_history" not in st.session_state:
        st.session_state.recognition_history = []
    hist = st.session_state.recognition_history
    if not hist:
        st.info("Chưa có dữ liệu lịch sử. Hãy mở trang Nhận diện thời gian thực và sử dụng webcam.")
        st.stop()

    labels = [h["cu_chi"] for h in hist]
    confs = [h["do_tin_cay"] for h in hist]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tổng bản ghi", len(hist))
    c2.metric("Số cử chỉ khác nhau", len(set(labels)))
    c3.metric("Độ tin cậy trung bình", f"{np.mean(confs):.1f}%")
    c4.metric("Độ tin cậy cao nhất", f"{max(confs):.1f}%")
    st.markdown("---")

    # Bảng chính
    rows = []
    for i, h in enumerate(reversed(hist), 1):
        tay_str = ""
        for t in h.get("chi_tiet_tay", []):
            tay_str += f"{t['hand']} ({t['score']}%) "
        rows.append({
            "#": i,
            "Thời gian": h["thoi_gian"],
            "Cử chỉ": h["cu_chi"],
            "Độ tin cậy (%)": h["do_tin_cay"],
            "Số tay": h.get("so_tay", 0),
            "Chi tiết tay": tay_str.strip() or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Biểu đồ
    if len(hist) > 1:
        import matplotlib.pyplot as plt
        tab1, tab2 = st.tabs(["Tần suất cử chỉ", "Độ tin cậy theo thời gian"])
        with tab1:
            lc = pd.Series(labels).value_counts()
            fig, ax = plt.subplots(figsize=(8,4)); fig.patch.set_facecolor("#1e1e2e"); ax.set_facecolor("#11111b")
            ax.barh(lc.index, lc.values, color="#89b4fa"); ax.set_title("Tần suất nhận diện từng cử chỉ", color="#cdd6f4", fontsize=13)
            ax.tick_params(colors="#6c7086")
            for s in ax.spines.values(): s.set_color("#313244")
            st.pyplot(fig); plt.close(fig)
        with tab2:
            fig, ax = plt.subplots(figsize=(10,4)); fig.patch.set_facecolor("#1e1e2e"); ax.set_facecolor("#11111b")
            ax.plot(range(len(confs)), confs, color="#89b4fa", lw=1.5)
            ax.axhline(70, color="#a6e3a1", ls="--", alpha=.6, label="70%")
            ax.axhline(40, color="#f9e2af", ls="--", alpha=.6, label="40%")
            ax.fill_between(range(len(confs)), confs, alpha=.1, color="#89b4fa")
            ax.set_title("Biến thiên độ tin cậy theo thời gian", color="#cdd6f4", fontsize=13)
            ax.set_ylim(0,105); ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4"); ax.tick_params(colors="#6c7086")
            for s in ax.spines.values(): s.set_color("#313244")
            st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    co1, co2 = st.columns(2)
    co1.download_button("Xuất tệp CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), "lich_su_nhan_dien.csv", "text/csv")
    if co2.button("Xóa toàn bộ lịch sử"):
        st.session_state.recognition_history = []; st.rerun()
