"""
GUI Application — Vietnamese Sign Language Recognition
Giao diện nhận diện ngôn ngữ ký hiệu Việt Nam.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
from collections import deque
from datetime import datetime
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json
import os

from model import SignLanguageLSTM
from dataset import VSLDataset
from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN,
    BEST_MODEL_PATH, METADATA_PATH, RESULTS_DIR,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS_EXTRACT,
)

# =====================================================================
# Color Palette — Dark Cyberpunk Theme
# =====================================================================
BG       = "#080c14"
SURFACE  = "#0d1220"
CARD     = "#111827"
BORDER   = "#1e2d45"
TEXT     = "#e2e8f0"
DIM      = "#64748b"
ACCENT   = "#38bdf8"       # Cyan
ACCENT2  = "#818cf8"       # Indigo
GREEN    = "#34d399"
YELLOW   = "#fbbf24"
RED      = "#f87171"
PURPLE   = "#c084fc"
ORANGE   = "#fb923c"

FONT_TITLE  = ("Consolas", 14, "bold")
FONT_BODY   = ("Consolas", 11)
FONT_SMALL  = ("Consolas", 9)
FONT_BIG    = ("Consolas", 20, "bold")
FONT_HUGE   = ("Consolas", 32, "bold")

# =====================================================================
# Utility: Styled Widgets
# =====================================================================
def make_card(parent, **kwargs):
    f = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1,
                 highlightbackground=BORDER, **kwargs)
    return f

def make_label(parent, text, font=FONT_BODY, fg=TEXT, bg=CARD, **kwargs):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kwargs)

def make_btn(parent, text, cmd, color=ACCENT, fg=BG, width=None, **kwargs):
    b = tk.Button(
        parent, text=text, command=cmd,
        font=FONT_BODY, bg=color, fg=fg,
        activebackground=color, activeforeground=fg,
        relief="flat", cursor="hand2", bd=0,
        padx=14, pady=7,
    )
    if width:
        b.config(width=width)
    return b

def separator(parent, color=BORDER, height=1):
    return tk.Frame(parent, bg=color, height=height)

# =====================================================================
# History log path for persistence
# =====================================================================
HISTORY_FILE = RESULTS_DIR / "history_log.json"

def load_history():
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_history(history):
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# =====================================================================
# Sign gesture descriptions (customize as needed)
# =====================================================================
GESTURE_INFO = {
    # Add known gestures here; fallback = label name
}

def get_gesture_desc(label):
    return GESTURE_INFO.get(label, f"Ký hiệu: {label}")

# =====================================================================
# Main Application
# =====================================================================
class VSLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VSL Recognition System")
        self.geometry("1280x800")
        self.configure(bg=BG)
        self.minsize(1024, 680)

        # State
        self.history = load_history()
        self.cap = None
        self.model = None
        self.label_map = None
        self.hands_ctx = None
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.webcam_running = False
        self._photo_ref = None
        self._chart_refs = []
        self._current_page = None

        # Build layout
        self._build_layout()
        self.show_home()

    # ==============================================================
    # Layout skeleton: sidebar + content area
    # ==============================================================
    def _build_layout(self):
        # Top bar
        topbar = tk.Frame(self, bg=SURFACE, height=48)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="◈ VSL RECOGNITION", font=("Consolas", 14, "bold"),
                 bg=SURFACE, fg=ACCENT).pack(side="left", padx=20, pady=10)

        self.status_dot = tk.Label(topbar, text="●", font=("Consolas", 10),
                                   bg=SURFACE, fg=GREEN)
        self.status_dot.pack(side="right", padx=6)
        self.status_txt = tk.Label(topbar, text=f"Device: {self.device}",
                                   font=FONT_SMALL, bg=SURFACE, fg=DIM)
        self.status_txt.pack(side="right", padx=4)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = tk.Frame(body, bg=SURFACE, width=200)
        self.sidebar.pack(fill="y", side="left")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Content
        self.content = tk.Frame(body, bg=BG)
        self.content.pack(fill="both", expand=True, side="left")

    def _build_sidebar(self):
        tk.Label(self.sidebar, text="MENU", font=FONT_SMALL,
                 bg=SURFACE, fg=DIM).pack(pady=(20, 8), padx=16, anchor="w")

        self._nav_btns = {}
        nav_items = [
            ("home",    "⌂  Trang chủ",    self.show_home),
            ("webcam",  "◉  Nhận diện",     self.show_webcam),
            ("history", "≡  Lịch sử",       self.show_history),
            ("report",  "◈  Báo cáo",       self.show_report),
        ]
        for key, label, cmd in nav_items:
            btn = tk.Button(
                self.sidebar, text=label, font=FONT_BODY,
                bg=SURFACE, fg=TEXT, relief="flat", anchor="w",
                cursor="hand2", padx=18, pady=10,
                activebackground=CARD, activeforeground=ACCENT,
                command=lambda c=cmd, k=key: self._nav(k, c),
            )
            btn.pack(fill="x")
            self._nav_btns[key] = btn

        separator(self.sidebar, height=1).pack(fill="x", padx=16, pady=10)

        make_btn(self.sidebar, "✕  Thoát", self._quit_app,
                 color=RED, fg="white").pack(fill="x", padx=16, pady=4)

        # Model status box
        self.model_status_frame = make_card(self.sidebar)
        self.model_status_frame.pack(fill="x", padx=12, pady=(16, 8), side="bottom")
        tk.Label(self.model_status_frame, text="MODEL", font=FONT_SMALL,
                 bg=CARD, fg=DIM).pack(anchor="w", padx=10, pady=(8, 2))
        self.model_lbl = tk.Label(self.model_status_frame, text="● Not loaded",
                                  font=FONT_SMALL, bg=CARD, fg=YELLOW)
        self.model_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _nav(self, key, cmd):
        for k, b in self._nav_btns.items():
            b.config(bg=SURFACE, fg=TEXT)
        self._nav_btns[key].config(bg=CARD, fg=ACCENT)
        cmd()

    def _clear_content(self):
        self._stop_webcam()
        for w in self.content.winfo_children():
            w.destroy()

    def _update_model_status(self):
        if self.model:
            self.model_lbl.config(text="● Loaded ✓", fg=GREEN)
        else:
            self.model_lbl.config(text="● Not loaded", fg=YELLOW)

    # ==============================================================
    # Model loader
    # ==============================================================
    def _load_model(self):
        if self.model is not None:
            return True
        if not BEST_MODEL_PATH.exists():
            messagebox.showerror("Lỗi",
                "Không tìm thấy best_model.pth!\nHãy huấn luyện mô hình trước.")
            return False
        try:
            ds = VSLDataset(METADATA_PATH)
            self.label_map = ds.get_label_map()
            self.model = SignLanguageLSTM(
                INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, ds.num_classes,
            ).to(self.device)
            ckpt = torch.load(BEST_MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self._update_model_status()
            return True
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model:\n{e}")
            return False

    # ==============================================================
    # HOME PAGE
    # ==============================================================
    def show_home(self):
        self._clear_content()
        self._current_page = "home"

        frame = tk.Frame(self.content, bg=BG)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        tk.Label(frame, text="VSL", font=("Consolas", 72, "bold"),
                 bg=BG, fg=ACCENT).pack()
        tk.Label(frame, text="VIETNAMESE SIGN LANGUAGE RECOGNITION",
                 font=("Consolas", 13, "bold"), bg=BG, fg=TEXT).pack()
        tk.Label(frame, text="Hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam theo thời gian thực",
                 font=FONT_SMALL, bg=BG, fg=DIM).pack(pady=(4, 40))

        # Stats row
        stats_frame = tk.Frame(frame, bg=BG)
        stats_frame.pack(fill="x", pady=(0, 30))

        stats = [
            ("SEQ LEN", f"{SEQ_LEN}", "frames"),
            ("INPUT",   f"{INPUT_SIZE}", "keypoints"),
            ("HIDDEN",  f"{HIDDEN_SIZE}", "units"),
            ("LAYERS",  f"{NUM_LAYERS}", "LSTM"),
        ]
        for i, (title, val, unit) in enumerate(stats):
            card = make_card(stats_frame)
            card.grid(row=0, column=i, padx=8, pady=4, sticky="nsew")
            stats_frame.columnconfigure(i, weight=1)
            tk.Label(card, text=title, font=FONT_SMALL, bg=CARD, fg=DIM).pack(pady=(10, 2))
            tk.Label(card, text=val, font=("Consolas", 24, "bold"), bg=CARD, fg=ACCENT).pack()
            tk.Label(card, text=unit, font=FONT_SMALL, bg=CARD, fg=DIM).pack(pady=(0, 10))

        # Quick actions
        act_frame = tk.Frame(frame, bg=BG)
        act_frame.pack(pady=10)
        actions = [
            ("◉  Mở Webcam",    self.show_webcam,  ACCENT,  BG),
            ("≡  Lịch sử",      self.show_history, ACCENT2, "white"),
            ("◈  Báo cáo",      self.show_report,  GREEN,   BG),
        ]
        for text, cmd, color, fg in actions:
            make_btn(act_frame, text, cmd, color=color, fg=fg).pack(
                side="left", padx=8, ipady=4)

        # History summary
        if self.history:
            sep = separator(frame)
            sep.pack(fill="x", pady=20)
            tk.Label(frame, text=f"Phiên gần nhất: {self.history[-1]['label']} — {self.history[-1]['time']}",
                     font=FONT_SMALL, bg=BG, fg=DIM).pack()

    # ==============================================================
    # WEBCAM / REALTIME PAGE
    # ==============================================================
    def show_webcam(self):
        if not self._load_model():
            return
        self._clear_content()
        self._current_page = "webcam"
        self.frame_buffer.clear()

        root = tk.Frame(self.content, bg=BG)
        root.pack(fill="both", expand=True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=1)

        # ── Left: video feed ──────────────────────────────────────
        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.video_lbl = tk.Label(left, bg="#000000")
        self.video_lbl.grid(row=0, column=0, sticky="nsew")

        # ── Right: info panel ─────────────────────────────────────
        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        # -- Prediction card --
        pred_card = make_card(right)
        pred_card.pack(fill="x", pady=(0, 8))

        tk.Label(pred_card, text="KÝ HIỆU NHẬN DIỆN",
                 font=FONT_SMALL, bg=CARD, fg=DIM).pack(anchor="w", padx=12, pady=(10, 2))
        self.pred_lbl = tk.Label(pred_card, text="---",
                                 font=("Consolas", 22, "bold"), bg=CARD, fg=ACCENT)
        self.pred_lbl.pack(anchor="w", padx=12)
        self.gesture_desc_lbl = tk.Label(pred_card, text="Chờ nhận diện...",
                                         font=FONT_SMALL, bg=CARD, fg=DIM,
                                         wraplength=220, justify="left")
        self.gesture_desc_lbl.pack(anchor="w", padx=12, pady=(2, 10))

        # -- Confidence card --
        conf_card = make_card(right)
        conf_card.pack(fill="x", pady=(0, 8))

        tk.Label(conf_card, text="CONFIDENCE",
                 font=FONT_SMALL, bg=CARD, fg=DIM).pack(anchor="w", padx=12, pady=(10, 2))
        self.conf_lbl = tk.Label(conf_card, text="0.0%",
                                 font=("Consolas", 20, "bold"), bg=CARD, fg=GREEN)
        self.conf_lbl.pack(anchor="w", padx=12)

        # Confidence bar
        bar_bg = tk.Frame(conf_card, bg=BORDER, height=6)
        bar_bg.pack(fill="x", padx=12, pady=(4, 10))
        self.conf_bar = tk.Frame(bar_bg, bg=GREEN, height=6, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1, relwidth=0)

        # -- Buffer card --
        buf_card = make_card(right)
        buf_card.pack(fill="x", pady=(0, 8))

        tk.Label(buf_card, text="FRAME BUFFER",
                 font=FONT_SMALL, bg=CARD, fg=DIM).pack(anchor="w", padx=12, pady=(10, 2))
        self.buf_lbl = tk.Label(buf_card, text=f"0 / {SEQ_LEN}",
                                font=FONT_BODY, bg=CARD, fg=YELLOW)
        self.buf_lbl.pack(anchor="w", padx=12)
        buf_bar_bg = tk.Frame(buf_card, bg=BORDER, height=6)
        buf_bar_bg.pack(fill="x", padx=12, pady=(4, 10))
        self.buf_bar = tk.Frame(buf_bar_bg, bg=YELLOW, height=6)
        self.buf_bar.place(x=0, y=0, relheight=1, relwidth=0)

        # -- Hand detection status --
        hand_card = make_card(right)
        hand_card.pack(fill="x", pady=(0, 8))

        tk.Label(hand_card, text="TRẠNG THÁI",
                 font=FONT_SMALL, bg=CARD, fg=DIM).pack(anchor="w", padx=12, pady=(10, 2))
        self.hand_lbl = tk.Label(hand_card, text="● Chờ tay...",
                                 font=FONT_BODY, bg=CARD, fg=DIM)
        self.hand_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # -- Keypoint stats --
        kp_card = make_card(right)
        kp_card.pack(fill="x", pady=(0, 8))

        tk.Label(kp_card, text="KEYPOINT (INDEX TIP)",
                 font=FONT_SMALL, bg=CARD, fg=DIM).pack(anchor="w", padx=12, pady=(10, 2))
        self.kp_x = tk.Label(kp_card, text="X: ---", font=FONT_SMALL, bg=CARD, fg=ACCENT)
        self.kp_x.pack(anchor="w", padx=12)
        self.kp_y = tk.Label(kp_card, text="Y: ---", font=FONT_SMALL, bg=CARD, fg=ACCENT)
        self.kp_y.pack(anchor="w", padx=12)
        self.kp_z = tk.Label(kp_card, text="Z: ---", font=FONT_SMALL, bg=CARD, fg=ACCENT)
        self.kp_z.pack(anchor="w", padx=12, pady=(0, 10))

        # -- Model info card --
        info_card = make_card(right)
        info_card.pack(fill="x", pady=(0, 8))
        tk.Label(info_card, text="MODEL INFO",
                 font=FONT_SMALL, bg=CARD, fg=DIM).pack(anchor="w", padx=12, pady=(10, 2))
        classes = list(self.label_map.values()) if self.label_map else []
        tk.Label(info_card, text=f"Classes: {len(classes)}",
                 font=FONT_SMALL, bg=CARD, fg=TEXT).pack(anchor="w", padx=12)
        tk.Label(info_card, text=f"Device: {self.device}",
                 font=FONT_SMALL, bg=CARD, fg=TEXT).pack(anchor="w", padx=12, pady=(0, 10))

        # Start camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở webcam!")
            self.show_home()
            return

        self.hands_ctx = mp_hands.Hands(
            static_image_mode=False, max_num_hands=MAX_NUM_HANDS_EXTRACT,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.webcam_running = True
        self._update_webcam()

    def _update_webcam(self):
        if not self.webcam_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self._update_webcam)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_ctx.process(rgb)

        kp = np.zeros(INPUT_SIZE, dtype=np.float32)
        hand_detected = False
        norm_x = norm_y = norm_z = 0.0

        if results.multi_hand_landmarks:
            hand_detected = True
            hl = results.multi_hand_landmarks[0]
            wrist = hl.landmark[0]
            idx_tip = hl.landmark[8]
            norm_x = idx_tip.x - wrist.x
            norm_y = idx_tip.y - wrist.y
            norm_z = idx_tip.z - wrist.z
            vals = []
            for lm in hl.landmark:
                vals.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            kp = np.array(vals, dtype=np.float32)
            for hl2 in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl2, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(56, 189, 248), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(99, 102, 241), thickness=2))

        self.frame_buffer.append(kp)
        buf_len = len(self.frame_buffer)

        # Update keypoint display
        if hand_detected:
            self.kp_x.config(text=f"X: {norm_x:+.3f}")
            self.kp_y.config(text=f"Y: {norm_y:+.3f}")
            self.kp_z.config(text=f"Z: {norm_z:+.3f}")
            self.hand_lbl.config(text="● Đã phát hiện tay", fg=GREEN)
        else:
            self.hand_lbl.config(text="● Không có tay", fg=DIM)

        # Buffer bar
        buf_ratio = buf_len / SEQ_LEN
        buf_color = GREEN if buf_ratio >= 1.0 else YELLOW
        self.buf_lbl.config(text=f"{buf_len} / {SEQ_LEN}", fg=buf_color)
        self.buf_bar.place(relwidth=min(buf_ratio, 1.0))
        self.buf_bar.config(bg=buf_color)

        # Predict
        if buf_len == SEQ_LEN:
            seq = np.array(list(self.frame_buffer), dtype=np.float32)
            tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, dim=1)
                prob, idx = torch.max(probs, dim=1)
                conf = prob.item() * 100
                label = self.label_map.get(idx.item(), f"ID:{idx.item()}")

            self.pred_lbl.config(text=label)
            self.gesture_desc_lbl.config(text=get_gesture_desc(label))
            c_color = GREEN if conf > 70 else YELLOW if conf > 40 else RED
            self.conf_lbl.config(text=f"{conf:.1f}%", fg=c_color)
            self.conf_bar.place(relwidth=conf / 100)
            self.conf_bar.config(bg=c_color)

            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": label,
                "confidence": round(conf, 2),
                "conf_str": f"{conf:.1f}%",
                "hand_detected": hand_detected,
                "norm_x": round(float(norm_x), 4),
                "norm_y": round(float(norm_y), 4),
                "norm_z": round(float(norm_z), 4),
                "buffer_full": True,
            }
            self.history.append(entry)
            save_history(self.history)

        # Render frame to label
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            lw = self.video_lbl.winfo_width()
            lh = self.video_lbl.winfo_height()
            if lw > 50 and lh > 50:
                img = img.resize((lw, lh), Image.LANCZOS)
        except Exception:
            pass
        self._photo_ref = ImageTk.PhotoImage(img)
        self.video_lbl.config(image=self._photo_ref)
        self.after(30, self._update_webcam)

    def _stop_webcam(self):
        self.webcam_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if self.hands_ctx:
            self.hands_ctx.close()
            self.hands_ctx = None

    # ==============================================================
    # HISTORY PAGE
    # ==============================================================
    def show_history(self):
        self._clear_content()
        self._current_page = "history"

        root = tk.Frame(self.content, bg=BG)
        root.pack(fill="both", expand=True, padx=16, pady=12)

        # Header
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill="x", pady=(0, 10))
        tk.Label(hdr, text="LỊCH SỬ NHẬN DIỆN", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text=f"  {len(self.history)} bản ghi", font=FONT_SMALL, bg=BG, fg=DIM).pack(side="left", padx=6, pady=4)

        # Action buttons
        btn_row = tk.Frame(root, bg=BG)
        btn_row.pack(fill="x", pady=(0, 8))
        make_btn(btn_row, "↻  Làm mới", self.show_history, color=SURFACE, fg=TEXT).pack(side="left", padx=4)
        make_btn(btn_row, "✕  Xóa lịch sử", self._clear_history, color=RED, fg="white").pack(side="left", padx=4)
        make_btn(btn_row, "📊  Biểu đồ lịch sử", self._show_history_chart, color=ACCENT2, fg="white").pack(side="left", padx=4)

        if not self.history:
            tk.Label(root, text="Chưa có lịch sử.\nMở Webcam để bắt đầu nhận diện.",
                     font=FONT_BODY, bg=BG, fg=DIM, justify="center").pack(pady=60)
            return

        # Summary stats
        stats_frame = tk.Frame(root, bg=BG)
        stats_frame.pack(fill="x", pady=(0, 10))

        labels_seen = [h["label"] for h in self.history]
        unique_labels = list(set(labels_seen))
        high_conf = [h for h in self.history if h.get("confidence", 0) >= 70]
        avg_conf = np.mean([h.get("confidence", 0) for h in self.history])
        most_common = max(set(labels_seen), key=labels_seen.count) if labels_seen else "---"

        stat_data = [
            ("TỔNG BẢN GHI", str(len(self.history)), DIM),
            ("NHÃN DUY NHẤT", str(len(unique_labels)), ACCENT),
            ("CAO ≥ 70%", str(len(high_conf)), GREEN),
            ("CONF TB", f"{avg_conf:.1f}%", YELLOW),
            ("PHỔ BIẾN NHẤT", most_common, PURPLE),
        ]
        for i, (title, val, color) in enumerate(stat_data):
            card = make_card(stats_frame)
            card.grid(row=0, column=i, padx=5, sticky="nsew")
            stats_frame.columnconfigure(i, weight=1)
            tk.Label(card, text=title, font=FONT_SMALL, bg=CARD, fg=DIM).pack(pady=(8, 2), padx=8, anchor="w")
            tk.Label(card, text=val, font=("Consolas", 14, "bold"), bg=CARD, fg=color).pack(padx=8, anchor="w", pady=(0, 8))

        # Table
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("VSL.Treeview", background=CARD, foreground=TEXT,
                        fieldbackground=CARD, font=FONT_SMALL, rowheight=28)
        style.configure("VSL.Treeview.Heading", background=SURFACE, foreground=ACCENT,
                        font=("Consolas", 10, "bold"))
        style.map("VSL.Treeview", background=[("selected", ACCENT2)], foreground=[("selected", "white")])

        cols = ("stt", "time", "label", "confidence", "x", "y", "z")
        tree_frame = tk.Frame(root, bg=BG)
        tree_frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", style="VSL.Treeview")
        headers = {"stt": ("#", 45), "time": ("Thời gian", 155), "label": ("Ký hiệu", 160),
                   "confidence": ("Confidence", 100), "x": ("X", 80), "y": ("Y", 80), "z": ("Z", 80)}
        for col, (hdr_text, w) in headers.items():
            tree.heading(col, text=hdr_text)
            tree.column(col, width=w, anchor="center")

        for i, h in enumerate(reversed(self.history), 1):
            tag = "high" if h.get("confidence", 0) >= 70 else "mid" if h.get("confidence", 0) >= 40 else "low"
            tree.insert("", "end", values=(
                i, h.get("time", "---"), h.get("label", "---"),
                h.get("conf_str", f"{h.get('confidence', 0):.1f}%"),
                h.get("norm_x", "---"), h.get("norm_y", "---"), h.get("norm_z", "---"),
            ), tags=(tag,))

        tree.tag_configure("high", foreground=GREEN)
        tree.tag_configure("mid", foreground=YELLOW)
        tree.tag_configure("low", foreground=RED)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

    def _clear_history(self):
        if messagebox.askyesno("Xác nhận", "Xóa toàn bộ lịch sử?"):
            self.history.clear()
            save_history(self.history)
            self.show_history()

    def _show_history_chart(self):
        if not self.history:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu lịch sử.")
            return

        win = tk.Toplevel(self)
        win.title("Biểu đồ lịch sử nhận diện")
        win.geometry("900x550")
        win.configure(bg=BG)

        labels = [h["label"] for h in self.history]
        confs = [h.get("confidence", 0) for h in self.history]
        label_counts = {}
        for l in labels:
            label_counts[l] = label_counts.get(l, 0) + 1

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#111827")
        fig.subplots_adjust(wspace=0.35)

        # Bar: frequency
        ax1 = axes[0]
        ax1.set_facecolor("#111827")
        sorted_items = sorted(label_counts.items(), key=lambda x: -x[1])
        bar_labels = [x[0] for x in sorted_items]
        bar_vals = [x[1] for x in sorted_items]
        colors_bar = plt.cm.cool(np.linspace(0.2, 0.9, len(bar_labels)))
        ax1.barh(bar_labels, bar_vals, color=colors_bar, edgecolor="none")
        ax1.set_title("Tần suất nhận diện", color=TEXT, fontsize=11, pad=8)
        ax1.tick_params(colors=DIM)
        ax1.xaxis.label.set_color(DIM)
        for spine in ax1.spines.values():
            spine.set_color(BORDER)

        # Line: confidence over time
        ax2 = axes[1]
        ax2.set_facecolor("#111827")
        ax2.plot(range(len(confs)), confs, color=ACCENT, linewidth=1.5, alpha=0.9)
        ax2.axhline(70, color=GREEN, linestyle="--", linewidth=1, alpha=0.6, label="70%")
        ax2.axhline(40, color=YELLOW, linestyle="--", linewidth=1, alpha=0.6, label="40%")
        ax2.fill_between(range(len(confs)), confs, alpha=0.15, color=ACCENT)
        ax2.set_title("Confidence theo thời gian", color=TEXT, fontsize=11, pad=8)
        ax2.set_ylim(0, 105)
        ax2.tick_params(colors=DIM)
        ax2.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8)
        for spine in ax2.spines.values():
            spine.set_color(BORDER)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        plt.close(fig)

    # ==============================================================
    # REPORT PAGE
    # ==============================================================
    def show_report(self):
        self._clear_content()
        self._current_page = "report"

        root = tk.Frame(self.content, bg=BG)
        root.pack(fill="both", expand=True)

        # Tab bar
        tab_bar = tk.Frame(root, bg=SURFACE, height=42)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)

        self._report_tab_btns = {}
        tabs = [
            ("overview", "◈  Tổng quan"),
            ("charts",   "📊  Biểu đồ"),
            ("training", "📈  Training"),
        ]
        self._report_content = tk.Frame(root, bg=BG)
        self._report_content.pack(fill="both", expand=True)

        for key, label in tabs:
            btn = tk.Button(
                tab_bar, text=label, font=FONT_SMALL,
                bg=SURFACE, fg=DIM, relief="flat",
                cursor="hand2", padx=14, pady=10,
                activebackground=CARD, activeforeground=ACCENT,
                command=lambda k=key: self._switch_report_tab(k),
            )
            btn.pack(side="left")
            self._report_tab_btns[key] = btn

        self._switch_report_tab("overview")

    def _switch_report_tab(self, key):
        for k, b in self._report_tab_btns.items():
            b.config(bg=SURFACE, fg=DIM)
        self._report_tab_btns[key].config(bg=CARD, fg=ACCENT)

        for w in self._report_content.winfo_children():
            w.destroy()

        if key == "overview":
            self._render_overview()
        elif key == "charts":
            self._render_charts()
        elif key == "training":
            self._render_training()

    def _render_overview(self):
        frame = self._report_content

        # Scroll
        canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        pad = tk.Frame(inner, bg=BG)
        pad.pack(fill="both", padx=20, pady=16)

        tk.Label(pad, text="TỔNG QUAN MÔ HÌNH", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(anchor="w", pady=(0, 12))

        # Load checkpoint info
        ckpt_info = {}
        if BEST_MODEL_PATH.exists():
            try:
                ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
                ckpt_info = {
                    "best_epoch": ckpt.get("epoch", "?"),
                    "val_accuracy": ckpt.get("val_accuracy", 0),
                }
            except Exception:
                pass

        # Load classification report metrics if available
        report_path = RESULTS_DIR / "classification_report.txt"
        report_text = ""
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                report_text = f.read()

        # Metrics summary
        stats_row = tk.Frame(pad, bg=BG)
        stats_row.pack(fill="x", pady=(0, 16))
        summary_stats = [
            ("BEST EPOCH", str(ckpt_info.get("best_epoch", "?")), ACCENT),
            ("VAL ACC", f"{ckpt_info.get('val_accuracy', 0):.1f}%", GREEN),
            ("SEQ LEN", str(SEQ_LEN), ACCENT2),
            ("HIDDEN", str(HIDDEN_SIZE), YELLOW),
            ("LAYERS", str(NUM_LAYERS), ORANGE),
            ("INPUT", str(INPUT_SIZE), PURPLE),
        ]
        for i, (title, val, color) in enumerate(summary_stats):
            c = make_card(stats_row)
            c.grid(row=0, column=i, padx=5, sticky="nsew")
            stats_row.columnconfigure(i, weight=1)
            tk.Label(c, text=title, font=FONT_SMALL, bg=CARD, fg=DIM).pack(pady=(10,2), padx=10, anchor="w")
            tk.Label(c, text=val, font=("Consolas", 16, "bold"), bg=CARD, fg=color).pack(padx=10, anchor="w", pady=(0,10))

        # Classification report
        if report_text:
            tk.Label(pad, text="CLASSIFICATION REPORT", font=FONT_TITLE, bg=BG, fg=ACCENT2).pack(anchor="w", pady=(8, 4))
            report_box = tk.Text(pad, font=("Consolas", 10), bg=CARD, fg=TEXT,
                                 relief="flat", height=18, wrap="none",
                                 bd=0, padx=12, pady=8,
                                 highlightbackground=BORDER, highlightthickness=1)
            report_box.insert("1.0", report_text)
            report_box.config(state="disabled")
            report_box.pack(fill="x", pady=(0, 16))
        else:
            tk.Label(pad, text="Chưa có classification report.\nHãy chạy evaluate trước.",
                     font=FONT_BODY, bg=BG, fg=DIM).pack(pady=20)

    def _render_charts(self):
        frame = self._report_content
        canvas_outer = tk.Canvas(frame, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(frame, orient="vertical", command=canvas_outer.yview)
        inner = tk.Frame(canvas_outer, bg=BG)
        inner.bind("<Configure>", lambda e: canvas_outer.configure(scrollregion=canvas_outer.bbox("all")))
        canvas_outer.create_window((0, 0), window=inner, anchor="nw")
        canvas_outer.configure(yscrollcommand=sb.set)
        canvas_outer.bind_all("<MouseWheel>", lambda e: canvas_outer.yview_scroll(int(-1*(e.delta/120)), "units"))
        sb.pack(side="right", fill="y")
        canvas_outer.pack(fill="both", expand=True)

        pad = tk.Frame(inner, bg=BG)
        pad.pack(fill="both", padx=20, pady=16)

        tk.Label(pad, text="BIỂU ĐỒ ĐÁNH GIÁ", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(anchor="w", pady=(0, 12))

        self._chart_refs = []
        chart_files = [
            ("Confusion Matrix", "confusion_matrix.png"),
            ("Precision / Recall / F1-Score", "per_class_metrics.png"),
            ("Accuracy theo từng lớp", "per_class_accuracy.png"),
        ]
        found_any = False
        for title, fname in chart_files:
            path = RESULTS_DIR / fname
            if not path.exists():
                continue
            found_any = True
            tk.Label(pad, text=title, font=("Consolas", 12, "bold"), bg=BG, fg=TEXT).pack(anchor="w", pady=(12, 4))
            img = Image.open(path)
            max_w = 880
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._chart_refs.append(photo)
            tk.Label(pad, image=photo, bg=BG).pack(anchor="w", pady=(0, 8))

        if not found_any:
            tk.Label(pad, text="Chưa có biểu đồ.\nHãy chạy evaluate (option 4 trong CLI) trước.",
                     font=FONT_BODY, bg=BG, fg=DIM, justify="center").pack(pady=60)

    def _render_training(self):
        frame = self._report_content
        pad = tk.Frame(frame, bg=BG)
        pad.pack(fill="both", expand=True, padx=20, pady=16)

        tk.Label(pad, text="TRAINING METRICS", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(anchor="w", pady=(0, 12))

        # Try to load training history JSON if exists
        train_hist_path = RESULTS_DIR / "train_history.json"

        if not train_hist_path.exists():
            # Show placeholder / instructions
            info_card = make_card(pad)
            info_card.pack(fill="x", pady=8)
            tk.Label(info_card, text="Chưa có dữ liệu training.",
                     font=FONT_BODY, bg=CARD, fg=DIM).pack(padx=16, pady=(12, 4))
            tk.Label(info_card,
                     text="Để hiển thị biểu đồ Loss & Accuracy theo epoch, hãy thêm đoạn\n"
                          "code sau vào train.py sau mỗi epoch và lưu vào results/train_history.json:\n\n"
                          '  history_entry = {"epoch": epoch, "train_loss": ..., "train_acc": ...,\n'
                          '                   "val_loss": ..., "val_acc": ...}\n'
                          "  # append & json.dump to results/train_history.json",
                     font=("Consolas", 9), bg=CARD, fg=DIM, justify="left").pack(padx=16, pady=(4, 16))

            # Show demo placeholder chart
            self._render_demo_training_chart(pad)
            return

        with open(train_hist_path, encoding="utf-8") as f:
            hist = json.load(f)

        self._render_real_training_chart(pad, hist)

    def _render_demo_training_chart(self, parent):
        tk.Label(parent, text="DEMO — Biểu đồ mẫu (dữ liệu giả lập)",
                 font=FONT_SMALL, bg=BG, fg=YELLOW).pack(anchor="w", pady=(4, 4))

        epochs = list(range(1, 51))
        np.random.seed(42)
        train_loss = [1.8 * np.exp(-0.06 * e) + np.random.normal(0, 0.02) for e in epochs]
        val_loss   = [1.9 * np.exp(-0.055 * e) + np.random.normal(0, 0.03) for e in epochs]
        train_acc  = [100 * (1 - np.exp(-0.07 * e)) + np.random.normal(0, 0.5) for e in epochs]
        val_acc    = [100 * (1 - np.exp(-0.065 * e)) + np.random.normal(0, 0.8) for e in epochs]

        self._draw_training_charts(parent, epochs, train_loss, val_loss, train_acc, val_acc)

    def _render_real_training_chart(self, parent, hist):
        epochs     = [h["epoch"] for h in hist]
        train_loss = [h.get("train_loss", 0) for h in hist]
        val_loss   = [h.get("val_loss", 0) for h in hist]
        train_acc  = [h.get("train_acc", 0) for h in hist]
        val_acc    = [h.get("val_acc", 0) for h in hist]
        self._draw_training_charts(parent, epochs, train_loss, val_loss, train_acc, val_acc)

    def _draw_training_charts(self, parent, epochs, train_loss, val_loss, train_acc, val_acc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=CARD)
        fig.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.88, bottom=0.12)

        for ax in [ax1, ax2]:
            ax.set_facecolor(BG)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.tick_params(colors=DIM, labelsize=8)
            ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.5)

        # Loss
        ax1.plot(epochs, train_loss, color=ACCENT, linewidth=2, label="Train Loss")
        ax1.plot(epochs, val_loss, color=RED, linewidth=2, linestyle="--", label="Val Loss")
        ax1.fill_between(epochs, train_loss, alpha=0.1, color=ACCENT)
        ax1.set_title("Loss theo Epoch", color=TEXT, fontsize=12, pad=10)
        ax1.set_xlabel("Epoch", color=DIM, fontsize=9)
        ax1.set_ylabel("Loss", color=DIM, fontsize=9)
        ax1.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)

        # Accuracy
        ax2.plot(epochs, train_acc, color=GREEN, linewidth=2, label="Train Acc")
        ax2.plot(epochs, val_acc, color=YELLOW, linewidth=2, linestyle="--", label="Val Acc")
        ax2.fill_between(epochs, train_acc, alpha=0.1, color=GREEN)
        ax2.set_title("Accuracy theo Epoch", color=TEXT, fontsize=12, pad=10)
        ax2.set_xlabel("Epoch", color=DIM, fontsize=9)
        ax2.set_ylabel("Accuracy (%)", color=DIM, fontsize=9)
        ax2.set_ylim(0, 105)
        ax2.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", pady=(0, 12))
        plt.close(fig)

    # ==============================================================
    # Quit
    # ==============================================================
    def _quit_app(self):
        self._stop_webcam()
        save_history(self.history)
        self.destroy()


# =====================================================================
# Entry point
# =====================================================================
def main():
    app = VSLApp()
    app.mainloop()


if __name__ == "__main__":
    main()