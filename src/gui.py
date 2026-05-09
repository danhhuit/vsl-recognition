"""
GUI Application — Giao diện nhận diện ngôn ngữ ký hiệu Việt Nam.
Sử dụng tkinter, OpenCV, MediaPipe, PyTorch.
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

from model import SignLanguageLSTM
from dataset import VSLDataset
from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN,
    BEST_MODEL_PATH, METADATA_PATH, RESULTS_DIR,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS_EXTRACT,
)

# =====================================================================
# Theme colors
# =====================================================================
BG = "#0f0f1a"
SURFACE = "#131829"
CARD = "#1a1f35"
TEXT = "#e2e8f0"
DIM = "#8892b0"
ACCENT = "#6366f1"
GREEN = "#4ade80"
YELLOW = "#facc15"
RED = "#f87171"
BLUE = "#60a5fa"


# =====================================================================
# Main Application
# =====================================================================
class VSLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VSL Recognition")
        self.geometry("1024x700")
        self.configure(bg=BG)
        self.minsize(800, 600)

        # State
        self.history = []
        self.cap = None
        self.model = None
        self.label_map = None
        self.hands_ctx = None
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.webcam_running = False
        self._photo_ref = None
        self._chart_refs = []

        # Main container
        self.main_frame = tk.Frame(self, bg=BG)
        self.main_frame.pack(fill="both", expand=True)

        self.show_home()

    # ---------- Utilities ----------
    def _clear(self):
        self._stop_webcam()
        for w in self.main_frame.winfo_children():
            w.destroy()

    def _make_topbar(self, title, icon=""):
        top = tk.Frame(self.main_frame, bg=SURFACE, height=50)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Button(
            top, text="← Quay lại", font=("Segoe UI", 11),
            bg=SURFACE, fg=TEXT, relief="flat", cursor="hand2",
            command=self.show_home, activebackground=CARD, activeforeground=TEXT,
        ).pack(side="left", padx=10)
        tk.Label(
            top, text=f"{icon} {title}", font=("Segoe UI", 13, "bold"),
            bg=SURFACE, fg=TEXT,
        ).pack(side="left", padx=10)
        return top

    def _load_model(self):
        if self.model is not None:
            return True
        if not BEST_MODEL_PATH.exists():
            messagebox.showerror(
                "Lỗi", "Không tìm thấy best_model.pth!\nHãy huấn luyện mô hình trước."
            )
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
            return True
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model:\n{e}")
            return False

    # ======================== HOME ========================
    def show_home(self):
        self._clear()

        tk.Label(
            self.main_frame, text="🤟", font=("Segoe UI Emoji", 48), bg=BG, fg=TEXT,
        ).pack(pady=(50, 5))
        tk.Label(
            self.main_frame, text="Vietnamese Sign Language Recognition",
            font=("Segoe UI", 22, "bold"), bg=BG, fg=TEXT,
        ).pack()
        tk.Label(
            self.main_frame, text="Hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam",
            font=("Segoe UI", 11), bg=BG, fg=DIM,
        ).pack(pady=(4, 40))

        btn_frame = tk.Frame(self.main_frame, bg=BG)
        btn_frame.pack()

        btns = [
            ("📷  Mở Webcam", ACCENT, self.show_webcam, 0, 0),
            ("📊  Xem Báo cáo", "#2563eb", self.show_report, 0, 1),
            ("📜  Xem Lịch sử", "#0d9488", self.show_history, 1, 0),
            ("🚪  Thoát", "#dc2626", self._quit_app, 1, 1),
        ]
        for text, color, cmd, r, c in btns:
            tk.Button(
                btn_frame, text=text, font=("Segoe UI", 14, "bold"),
                bg=color, fg="white", activebackground=color, activeforeground="white",
                relief="flat", cursor="hand2", width=20, height=3, command=cmd,
            ).grid(row=r, column=c, padx=12, pady=12)

        status = "Loaded ✓" if self.model else "Not loaded"
        tk.Label(
            self.main_frame, text=f"Device: {self.device}  |  Model: {status}",
            font=("Segoe UI", 9), bg=BG, fg=DIM,
        ).pack(side="bottom", pady=10)

    # ======================== WEBCAM ========================
    def show_webcam(self):
        if not self._load_model():
            return
        self._clear()
        self.frame_buffer.clear()

        top = self._make_topbar("Nhận diện Realtime", "📷")
        self.status_lbl = tk.Label(top, text="● LIVE", font=("Segoe UI", 10), bg=SURFACE, fg=GREEN)
        self.status_lbl.pack(side="right", padx=10)

        self.video_lbl = tk.Label(self.main_frame, bg="#000000")
        self.video_lbl.pack(fill="both", expand=True, padx=10, pady=5)

        bot = tk.Frame(self.main_frame, bg=CARD, height=60)
        bot.pack(fill="x")
        bot.pack_propagate(False)

        self.pred_lbl = tk.Label(bot, text="Ký hiệu: ---", font=("Segoe UI", 16, "bold"), bg=CARD, fg=ACCENT)
        self.pred_lbl.pack(side="left", padx=20, pady=10)
        self.conf_lbl = tk.Label(bot, text="Confidence: ---", font=("Segoe UI", 12), bg=CARD, fg=DIM)
        self.conf_lbl.pack(side="left", padx=20)
        self.buf_lbl = tk.Label(bot, text=f"Buffer: 0/{SEQ_LEN}", font=("Segoe UI", 10), bg=CARD, fg=DIM)
        self.buf_lbl.pack(side="right", padx=20)

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
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_ctx.process(rgb)

        kp = np.zeros(INPUT_SIZE, dtype=np.float32)
        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            w = hl.landmark[0]
            vals = []
            for lm in hl.landmark:
                vals.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
            kp = np.array(vals, dtype=np.float32)
            for h in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        self.frame_buffer.append(kp)
        buf_len = len(self.frame_buffer)
        self.buf_lbl.config(text=f"Buffer: {buf_len}/{SEQ_LEN}", fg=GREEN if buf_len >= SEQ_LEN else YELLOW)

        if buf_len == SEQ_LEN:
            seq = np.array(list(self.frame_buffer), dtype=np.float32)
            tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, dim=1)
                prob, idx = torch.max(probs, dim=1)
                conf = prob.item() * 100
                label = self.label_map.get(idx.item(), f"ID:{idx.item()}")

            self.pred_lbl.config(text=f"Ký hiệu: {label}")
            c_color = GREEN if conf > 70 else YELLOW if conf > 40 else RED
            self.conf_lbl.config(text=f"Confidence: {conf:.1f}%", fg=c_color)

            self.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "label": label,
                "confidence": f"{conf:.1f}%",
            })

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            lw, lh = self.video_lbl.winfo_width(), self.video_lbl.winfo_height()
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

    # ======================== REPORT ========================
    def show_report(self):
        self._clear()
        self._make_topbar("Báo cáo đánh giá mô hình", "📊")

        canvas = tk.Canvas(self.main_frame, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        charts = [
            ("Confusion Matrix", "confusion_matrix.png"),
            ("Precision / Recall / F1-Score", "per_class_metrics.png"),
            ("Accuracy theo từng lớp", "per_class_accuracy.png"),
        ]

        self._chart_refs = []
        found = False
        for title, fname in charts:
            path = RESULTS_DIR / fname
            if not path.exists():
                continue
            found = True
            tk.Label(inner, text=title, font=("Segoe UI", 14, "bold"), bg=BG, fg=TEXT).pack(pady=(20, 5))
            img = Image.open(path)
            if img.width > 900:
                ratio = 900 / img.width
                img = img.resize((900, int(img.height * ratio)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._chart_refs.append(photo)
            tk.Label(inner, image=photo, bg=BG).pack(pady=(0, 10))

        if not found:
            tk.Label(
                inner, text="Chưa có kết quả đánh giá.\nHãy chạy Evaluate trước (option 4 trong CLI).",
                font=("Segoe UI", 14), bg=BG, fg=DIM, justify="center",
            ).pack(pady=100)

    # ======================== HISTORY ========================
    def show_history(self):
        self._clear()
        top = self._make_topbar("Lịch sử nhận diện", "📜")
        tk.Label(
            top, text=f"Tổng: {len(self.history)} mục",
            font=("Segoe UI", 10), bg=SURFACE, fg=DIM,
        ).pack(side="right", padx=10)

        if not self.history:
            tk.Label(
                self.main_frame, text="Chưa có lịch sử.\nHãy mở Webcam để bắt đầu nhận diện.",
                font=("Segoe UI", 14), bg=BG, fg=DIM, justify="center",
            ).pack(pady=100)
            return

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview", background=CARD, foreground=TEXT,
                        fieldbackground=CARD, font=("Segoe UI", 11), rowheight=32)
        style.configure("Dark.Treeview.Heading", background=SURFACE, foreground=TEXT,
                        font=("Segoe UI", 11, "bold"))
        style.map("Dark.Treeview", background=[("selected", ACCENT)])

        cols = ("stt", "time", "label", "confidence")
        tree = ttk.Treeview(self.main_frame, columns=cols, show="headings", style="Dark.Treeview")
        tree.heading("stt", text="#")
        tree.heading("time", text="Thời gian")
        tree.heading("label", text="Ký hiệu")
        tree.heading("confidence", text="Confidence")
        tree.column("stt", width=60, anchor="center")
        tree.column("time", width=120, anchor="center")
        tree.column("label", width=200, anchor="center")
        tree.column("confidence", width=120, anchor="center")

        for i, h in enumerate(reversed(self.history), 1):
            tree.insert("", "end", values=(i, h["time"], h["label"], h["confidence"]))

        vsb = ttk.Scrollbar(self.main_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------- Quit ----------
    def _quit_app(self):
        self._stop_webcam()
        self.destroy()


# =====================================================================
# Entry point
# =====================================================================
def main():
    app = VSLApp()
    app.mainloop()


if __name__ == "__main__":
    main()
