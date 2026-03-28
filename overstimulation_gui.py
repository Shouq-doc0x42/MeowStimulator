
import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import collections
import io
import sys
import urllib.request
import urllib.parse
import json

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk, ImageDraw

try:
    from plyer import notification
    PLYER_OK = True
except ImportError:
    PLYER_OK = False

try:
    import winsound
    SOUND_OK = True
except ImportError:
    SOUND_OK = False

# ──────────────────────────────────────────────────────────────
#  ICON — generated programmatically, no external file needed
# ──────────────────────────────────────────────────────────────

def _make_icon(size=64):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    d.ellipse([0, 0, size-1, size-1], fill="#1a1a2e")
    lw = max(2, size // 22)
    d.ellipse([lw*2, lw*2, size-lw*2-1, size-lw*2-1],
              outline="#5ec4a0", width=lw)
    r3 = size * 0.30
    d.ellipse([cx-r3, cy-r3, cx+r3, cy+r3], fill="#232340")
    lh, lw2 = size * 0.24, size * 0.12
    pts = []
    for t in range(0, 361, 4):
        rad = math.radians(t)
        pts.append((cx + lw2 * math.sin(rad),
                    (cy - lh*0.3) - lh*0.5*math.cos(rad)))
    d.polygon(pts, fill="#5ec4a0")
    d.line([(cx, cy+lh*0.22), (cx, cy+lh*0.55)],
           fill="#3a9e84", width=max(2, size//28))
    return img

# ──────────────────────────────────────────────────────────────
#  PALETTE
# ──────────────────────────────────────────────────────────────
P = {
    "bg":        "#1a1a2e",
    "surface":   "#232340",
    "surface2":  "#2c2c52",
    "calm":      "#5ec4a0",
    "watch":     "#e8c97a",
    "alert":     "#e07a8a",
    "text":      "#dcd9f0",
    "dim":       "#8a87a8",
    "accent":    "#9b8ec4",
    "btn":       "#3a3860",
    "btn_hover": "#4a4878",
    "black":     "#0e0e1e",
    "tg_blue":   "#2AABEE",
}

CALMING_ACTIVITIES = [
    ("🎯", "Fidget / stress ball",        "Squeeze it slowly, feel the texture"),
    ("🎧", "Noise-cancelling headphones", "Block it out, find your quiet"),
    ("🌬", "4-7-8 breathing",             "Inhale 4s · Hold 7s · Exhale 8s"),
    ("🚶", "Quiet room walk",             "5 minutes away from the screen"),
    ("💧", "Cold water",                  "Drink slowly, feel the temperature"),
    ("👐", "Shake your hands",            "Loose wrists, shake for 10 seconds"),
    ("🌀", "Free stim",                   "Rock, flap — whatever feels good"),
    ("😌", "Eyes closed rest",            "Do nothing for 60 seconds"),
]

# ──────────────────────────────────────────────────────────────
#  SCHEDULED REMINDERS  (rotate every fire)
# ──────────────────────────────────────────────────────────────
REMINDERS = [
    (
        "🌀", "Stim break",
        "Time to stim freely for a few minutes.",
        "Rock, flap, spin — whatever your body wants right now.\nNo rules. Just feel good. 🌀",
    ),
    (
        "👀", "Screen break",
        "Look away from the screen. Stretch your body.",
        "Focus on something far away for 20 seconds.\nRoll your shoulders, stretch your neck slowly. 🧘",
    ),
    (
        "🌬", "Breathing reset",
        "Pause for a 4-7-8 breathing cycle.",
        "Inhale for 4 seconds.\nHold for 7 seconds.\nExhale slowly for 8 seconds. 🌬",
    ),
    (
        "💧", "Water & snack check-in",
        "Have you had water recently? Maybe a snack?",
        "Drink some water slowly.\nEat something if you're hungry.\nYour body needs fuel. 💧",
    ),
]

DEFAULT_REMINDER_INTERVAL = 45 * 60   # 45 minutes in seconds

# ──────────────────────────────────────────────────────────────
#  MEDIAPIPE
# ──────────────────────────────────────────────────────────────
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

FACE_MESH = mp_face.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.5,
)
POSE = mp_pose.Pose(
    static_image_mode=False, model_complexity=0, smooth_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.5,
)

L_INNER_BROW, R_INNER_BROW = 105, 334
U_LIP,  L_LIP               = 13,  14
L_EYE_OUT, R_EYE_OUT        = 33,  263
L_EYE_TOP, L_EYE_BOT        = 159, 145
R_EYE_TOP, R_EYE_BOT        = 386, 374

# ──────────────────────────────────────────────────────────────
#  SIGNAL DETECTORS
# ──────────────────────────────────────────────────────────────

def face_stress_score(lm, iw, ih):
    def pt(i):
        p = lm[i]; return np.array([p.x*iw, p.y*ih])
    try:
        eye_w = np.linalg.norm(pt(L_EYE_OUT) - pt(R_EYE_OUT))
        if eye_w < 1: return 0.0
        brow_dist  = np.linalg.norm(pt(L_INNER_BROW) - pt(R_INNER_BROW)) / eye_w
        brow_score = max(0, min(100, (0.50 - brow_dist) / 0.25 * 100))
        lip_open   = np.linalg.norm(pt(U_LIP) - pt(L_LIP)) / eye_w
        lip_score  = max(0, min(100, (0.10 - lip_open) / 0.08 * 100))
        return brow_score * 0.6 + lip_score * 0.4
    except Exception:
        return 0.0


class BlinkTracker:
    def __init__(self):
        self.blinks      = collections.deque()
        self._was_closed = False

    def update(self, lm, iw, ih):
        try:
            def lpt(i): p = lm[i]; return np.array([p.x*iw, p.y*ih])
            eye_w  = np.linalg.norm(lpt(L_EYE_OUT) - lpt(R_EYE_OUT))
            l_ear  = np.linalg.norm(lpt(L_EYE_TOP) - lpt(L_EYE_BOT)) / max(eye_w, 1)
            r_ear  = np.linalg.norm(lpt(R_EYE_TOP) - lpt(R_EYE_BOT)) / max(eye_w, 1)
            closed = (l_ear + r_ear) / 2 < 0.018
            now    = time.time()
            if closed and not self._was_closed:
                self.blinks.append(now)
            self._was_closed = closed
            while self.blinks and now - self.blinks[0] > 6:
                self.blinks.popleft()
        except Exception:
            pass

    def score(self):
        c = len(self.blinks)
        return 0.0 if c <= 3 else min(100, (c-3)/7*100)


class MovementTracker:
    def __init__(self):
        self.pos_hist = collections.deque(maxlen=60)
        self.vel_hist = collections.deque(maxlen=60)
        self._last    = None
        self._last_t  = None

    def update(self, lm, iw, ih):
        try:
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            mx = (ls.x + rs.x) / 2 * iw
            my = (ls.y + rs.y) / 2 * ih
            now = time.time()
            if self._last:
                dt = max(now - self._last_t, 0.001)
                self.vel_hist.append(
                    math.hypot(mx-self._last[0], my-self._last[1]) / dt)
            self.pos_hist.append((mx, my))
            self._last = (mx, my); self._last_t = now
        except Exception:
            pass

    def score(self):
        if len(self.pos_hist) < 20: return 0.0
        pos  = np.array(self.pos_hist)
        rock = min(100, float(np.var(pos[:, 1])) / 80)
        flap = (min(100, float(np.sum(np.array(self.vel_hist) > 30)) / 15 * 100)
                if len(self.vel_hist) > 10 else 0.0)
        return rock*0.5 + flap*0.5


# ──────────────────────────────────────────────────────────────
#  MONITOR
# ──────────────────────────────────────────────────────────────

class Monitor:
    def __init__(self):
        self.history     = collections.deque(maxlen=90)
        self.last_alert  = 0
        self.alert_start = None
        self.score       = 0.0
        self.face_s = self.blink_s = self.move_s = 0.0

    def update(self, face_s, blink_s, move_s, weights):
        total = (face_s*weights[0] + blink_s*weights[1] + move_s*weights[2]) / 100
        if self.history:
            total = 0.8*self.history[-1] + 0.2*total
        self.history.append(total)
        self.score = total
        self.face_s = face_s; self.blink_s = blink_s; self.move_s = move_s
        return total

    def check_alert(self, threshold, hold_sec, cooldown_sec):
        now = time.time()
        if self.score >= threshold:
            if self.alert_start is None:
                self.alert_start = now
            elif (now - self.alert_start >= hold_sec and
                  now - self.last_alert  >= cooldown_sec):
                self.last_alert = now; self.alert_start = None
                return True
        else:
            self.alert_start = None
        return False

    def reset(self):
        self.history.clear(); self.score = 0.0; self.alert_start = None


# ──────────────────────────────────────────────────────────────
#  TELEGRAM BOT
# ──────────────────────────────────────────────────────────────

class TelegramBot:
    def __init__(self):
        self.token = ""; self.chat_id = ""; self.enabled = False

    def configure(self, token, chat_id):
        self.token   = token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = bool(self.token and self.chat_id)

    def send(self, message):
        if not self.enabled: return False
        try:
            url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id, "text": message, "parse_mode": "HTML"
            }).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read()).get("ok", False)
        except Exception as e:
            print(f"[Telegram] {e}"); return False

    def send_async(self, msg):
        threading.Thread(target=self.send, args=(msg,), daemon=True).start()

    def test(self):
        return self.send(
            "🌿 <b>MeowStimulator connected!</b>\n\n"
            "Your overstimulation alerts will appear here. Take care 💙")

TELEGRAM = TelegramBot()

# ──────────────────────────────────────────────────────────────
#  NOTIFICATIONS
# ──────────────────────────────────────────────────────────────

def play_calm_sound():
    if SOUND_OK:
        try: winsound.Beep(440,200); time.sleep(0.05); winsound.Beep(528,400)
        except Exception: pass

def send_desktop_notif(msg):
    if PLYER_OK:
        try:
            notification.notify(title="🌿 Time to regulate",
                                message=msg, app_name="MeowStimulator", timeout=12)
        except Exception: pass

def fire_all_alerts(activity):
    icon, title, desc = activity
    short = f"{icon} {title}: {desc}"
    threading.Thread(target=play_calm_sound,          daemon=True).start()
    threading.Thread(target=send_desktop_notif,
                     args=(short,),                   daemon=True).start()
    tg_msg = (
        f"🌿 <b>Time to regulate</b>\n\n"
        f"MeowStimulator detected overstimulation signs.\n\n"
        f"<b>Try this right now:</b>\n"
        f"{icon} <b>{title}</b>\n"
        f"<i>{desc}</i>\n\n"
        f"You've got this. Take your time 💙"
    )
    TELEGRAM.send_async(tg_msg)


def fire_reminder(reminder):
    icon, title, short_desc, long_desc = reminder
    desktop_msg = f"{icon} {title}: {short_desc}"
    tg_msg = (
        f"{icon} <b>{title}</b>\n\n"
        f"<i>{long_desc}</i>\n\n"
        f"— MeowStimulator reminder 🌿"
    )
    if PLYER_OK:
        try:
            notification.notify(
                title=f"{icon} Reminder — {title}",
                message=short_desc,
                app_name="MeowStimulator",
                timeout=15,
            )
        except Exception:
            pass
    TELEGRAM.send_async(tg_msg)


# ──────────────────────────────────────────────────────────────
#  APP
# ──────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌿 MeowStimulator")
        self.configure(bg=P["bg"])
        self.resizable(True, True)

        # Set window icon
        try:
            import tempfile, os
            icon_img = _make_icon(64)
            buf = io.BytesIO()
            icon_img.save(buf, format="ICO", sizes=[(64,64),(32,32),(16,16)])
            buf.seek(0)
            tmp = tempfile.NamedTemporaryFile(suffix=".ico", delete=False)
            tmp.write(buf.read()); tmp.close()
            self.iconbitmap(tmp.name)
            os.unlink(tmp.name)
        except Exception:
            pass

        self.monitor       = Monitor()
        self.blink_tracker = BlinkTracker()
        self.move_tracker  = MovementTracker()
        self.running       = False
        self.alert_visible = False
        self.cap           = None
        self._cam_thread   = None
        self._breath_angle = 0.0
        self._breath_dir   = 1
        self._current_frame = None
        self._pending_pil   = None
        self._frame_lock    = threading.Lock()

        self.var_threshold = tk.IntVar(value=65)
        self.var_hold      = tk.DoubleVar(value=4.0)
        self.var_cooldown  = tk.IntVar(value=90)
        self.var_w_face    = tk.IntVar(value=35)
        self.var_w_blink   = tk.IntVar(value=25)
        self.var_w_move    = tk.IntVar(value=40)
        self.var_tg_token  = tk.StringVar()
        self.var_tg_chat   = tk.StringVar()
        self._tg_status    = tk.StringVar(value="")

        # Reminder scheduler state
        self._reminder_idx      = 0
        self._reminder_interval = DEFAULT_REMINDER_INTERVAL
        self._reminder_enabled  = tk.BooleanVar(value=True)
        self._reminder_next     = time.time() + DEFAULT_REMINDER_INTERVAL
        self._reminder_paused   = False
        self.var_reminder_mins  = tk.IntVar(value=45)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._ui_loop()

    # ─── UI BUILD ────────────────────────────────────────────

    def _build_ui(self):
        self.geometry("1140x700")

        # Left column
        left = tk.Frame(self, bg=P["bg"], width=590)
        left.pack(side="left", fill="both", padx=(18,8), pady=18)
        left.pack_propagate(False)

        self._build_logo_bar(left)

        self.btn_start = self._btn(
            left, "▶  START MONITORING", self._toggle_camera,
            bg=P["calm"], fg=P["black"])
        self.btn_start.pack(fill="x", pady=(0,8))
        self._btn(left, "↺  RESET SCORE", self._reset_score,
                  bg=P["btn"], fg=P["dim"]).pack(fill="x", pady=(0,12))

        self.cam_label = tk.Label(
            left, bg=P["surface"],
            text="Press  ▶ START MONITORING  above",
            fg=P["dim"], font=("Courier New", 11))
        self.cam_label.pack(fill="both", expand=True)

        # Right column — scrollable
        right_outer = tk.Frame(self, bg=P["bg"])
        right_outer.pack(side="left", fill="both", expand=True, padx=(8,18), pady=18)

        # Canvas + scrollbar for scrollable right panel
        self._right_canvas = tk.Canvas(right_outer, bg=P["bg"],
                                       highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_outer, orient="vertical",
                                  command=self._right_canvas.yview)
        self._right_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._right_canvas.pack(side="left", fill="both", expand=True)

        # Inner frame that holds all cards
        right = tk.Frame(self._right_canvas, bg=P["bg"])
        self._right_canvas_window = self._right_canvas.create_window(
            (0, 0), window=right, anchor="nw")

        # Resize inner frame when canvas resizes
        def _on_canvas_resize(e):
            self._right_canvas.itemconfig(
                self._right_canvas_window, width=e.width)
        self._right_canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region when inner frame changes size
        right.bind("<Configure>", lambda e: self._right_canvas.configure(
            scrollregion=self._right_canvas.bbox("all")))

        # Mouse wheel scrolling
        def _on_mousewheel(e):
            self._right_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        self._right_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_score_card(self._card(right))
        self._build_signal_card(self._card(right))
        self._build_threshold_card(self._card(right))
        self._build_reminder_card(self._card(right))
        self._build_telegram_card(self._card(right))
        self._build_settings_card(self._card(right))
        self._build_alert_overlay()

    def _card(self, parent):
        f = tk.Frame(parent, bg=P["surface"],
                     highlightbackground=P["surface2"], highlightthickness=1)
        f.pack(fill="x", pady=(0,10))
        return f

    def _btn(self, parent, text, cmd, bg=None, fg=None):
        bg = bg or P["btn"]; fg = fg or P["text"]
        b  = tk.Label(parent, text=text, bg=bg, fg=fg,
                      font=("Courier New",11,"bold"), pady=11, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        orig = bg
        b.bind("<Enter>", lambda e: b.configure(bg=P["btn_hover"] if orig==P["btn"] else orig))
        b.bind("<Leave>", lambda e: b.configure(bg=orig))
        return b

    def _lbl(self, parent, text, size=9, color=None, bold=False, pady=0, padx=0):
        return tk.Label(parent, text=text, bg=P["surface"],
                        fg=color or P["dim"],
                        font=("Courier New", size, "bold" if bold else "normal"),
                        pady=pady, padx=padx)

    # Logo bar
    def _build_logo_bar(self, parent):
        frame = tk.Frame(parent, bg=P["surface"], height=54)
        frame.pack(fill="x", pady=(0,10))
        frame.pack_propagate(False)
        c = tk.Canvas(frame, bg=P["surface"], highlightthickness=0, height=54)
        c.pack(fill="x")
        # Leaf icon
        c.create_oval(12,7,46,47,  fill=P["surface2"], outline="")
        c.create_oval(18,13,40,41, fill="#1a1a2e",     outline="")
        c.create_oval(21,13,37,41, outline=P["calm"],  width=2)
        pts = []
        for t in range(0, 361, 10):
            r = math.radians(t)
            pts.extend([29+6*math.sin(r), 24-9*math.cos(r)])
        c.create_polygon(pts, fill=P["calm"], smooth=True)
        c.create_line(29,34,29,45, fill="#3a9e84", width=2, capstyle="round")
        c.create_text(56,19, text="MEOW STIMULATOR", anchor="w",
                      fill=P["text"], font=("Courier New",13,"bold"))
        c.create_text(56,37, text="overstimulation detector", anchor="w",
                      fill=P["dim"], font=("Courier New",9))

    # Score card
    def _build_score_card(self, parent):
        self._lbl(parent, "OVERSTIMULATION LEVEL", pady=8).pack()
        self.gauge_canvas = tk.Canvas(parent, width=220, height=150,
                                      bg=P["surface"], highlightthickness=0)
        self.gauge_canvas.pack()
        self.score_label = tk.Label(parent, text="—", bg=P["surface"],
                                    fg=P["calm"], font=("Courier New",36,"bold"))
        self.score_label.pack()
        self.status_label = tk.Label(parent, text="NOT RUNNING",
                                     bg=P["surface"], fg=P["dim"],
                                     font=("Courier New",11,"bold"), pady=6)
        self.status_label.pack()
        self._lbl(parent, "Alert hold progress").pack()
        self.hold_bar_bg = tk.Frame(parent, bg=P["surface2"], height=7)
        self.hold_bar_bg.pack(fill="x", padx=20, pady=(2,10))
        self.hold_bar_fill = tk.Frame(self.hold_bar_bg, bg=P["accent"], height=7)
        self.hold_bar_fill.place(x=0, y=0, width=0)

    # Signal bars
    def _build_signal_card(self, parent):
        self._lbl(parent, "SIGNALS", pady=6).pack(anchor="w", padx=14)
        self.bar_vars = {}
        for key, label in [("face","😤  Facial stress"),
                            ("blink","👁  Eye / blink rate"),
                            ("move","🔄  Repetitive movement")]:
            row = tk.Frame(parent, bg=P["surface"])
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=label, bg=P["surface"], fg=P["text"],
                     font=("Courier New",10), width=22, anchor="w").pack(side="left")
            bg_bar = tk.Frame(row, bg=P["surface2"], height=13, width=160)
            bg_bar.pack(side="left", padx=(4,8))
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=P["calm"], height=13)
            fill.place(x=0, y=0, relheight=1, width=0)
            lbl = tk.Label(row, text="0", bg=P["surface"], fg=P["calm"],
                           font=("Courier New",10,"bold"), width=4)
            lbl.pack(side="left")
            self.bar_vars[key] = (fill, lbl, bg_bar)
        tk.Frame(parent, bg=P["surface"], height=6).pack()

    # Threshold card — always visible
    def _build_threshold_card(self, parent):
        # Header row
        hdr = tk.Frame(parent, bg=P["surface"])
        hdr.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(hdr, text="🔔  ALERT THRESHOLD",
                 bg=P["surface"], fg=P["text"],
                 font=("Courier New", 10, "bold")).pack(side="left")

        # Big live percentage badge on the right
        self._thr_badge = tk.Label(hdr,
            text=f"{self.var_threshold.get()}%",
            bg=P["surface2"], fg=P["calm"],
            font=("Courier New", 14, "bold"),
            padx=10, pady=2)
        self._thr_badge.pack(side="right")

        # Description of current level
        self._thr_desc = tk.Label(parent,
            text=self._threshold_desc(self.var_threshold.get()),
            bg=P["surface"], fg=P["dim"],
            font=("Courier New", 9), anchor="w")
        self._thr_desc.pack(fill="x", padx=14, pady=(0, 6))

        # Big slider
        slider_frame = tk.Frame(parent, bg=P["surface"])
        slider_frame.pack(fill="x", padx=14, pady=(0, 4))

        tk.Label(slider_frame, text="30%", bg=P["surface"],
                 fg=P["dim"], font=("Courier New", 8)).pack(side="left")

        self._thr_slider = ttk.Scale(
            slider_frame,
            from_=30, to=90,
            variable=self.var_threshold,
            orient="horizontal",
            command=self._on_threshold_change,
        )
        self._thr_slider.pack(side="left", fill="x", expand=True, padx=6)

        tk.Label(slider_frame, text="90%", bg=P["surface"],
                 fg=P["dim"], font=("Courier New", 8)).pack(side="right")

        # Tick labels row
        tick_row = tk.Frame(parent, bg=P["surface"])
        tick_row.pack(fill="x", padx=14, pady=(0, 4))
        for pct, label, col in [
            ("30–45", "Very sensitive", P["alert"]),
            ("46–64", "Balanced",       P["watch"]),
            ("65–90", "Relaxed",        P["calm"]),
        ]:
            seg = tk.Frame(tick_row, bg=P["surface"])
            seg.pack(side="left", expand=True)
            tk.Frame(seg, bg=col, height=3).pack(fill="x", pady=(0, 2))
            tk.Label(seg, text=pct,   bg=P["surface"], fg=col,
                     font=("Courier New", 7)).pack()
            tk.Label(seg, text=label, bg=P["surface"], fg=P["dim"],
                     font=("Courier New", 7)).pack()

        tk.Frame(parent, bg=P["surface"], height=6).pack()

    def _threshold_desc(self, val):
        val = int(val)
        if val <= 45:
            return "Very sensitive — alerts early, even mild signs"
        elif val <= 64:
            return "Balanced — alerts on clear, sustained signals"
        else:
            return "Relaxed — only alerts during strong episodes"

    def _on_threshold_change(self, val):
        v = int(float(val))
        col = P["alert"] if v <= 45 else P["watch"] if v <= 64 else P["calm"]
        self._thr_badge.configure(text=f"{v}%", fg=col)
        self._thr_desc.configure(text=self._threshold_desc(v))

    # Reminder card
    def _build_reminder_card(self, parent):
        # Header
        hdr = tk.Frame(parent, bg=P["surface"])
        hdr.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(hdr, text="⏰  SCHEDULED REMINDERS",
                 bg=P["surface"], fg=P["text"],
                 font=("Courier New", 10, "bold")).pack(side="left")

        # Enabled toggle button
        self._rem_toggle_btn = tk.Label(
            hdr, text="ON", bg=P["calm"], fg=P["black"],
            font=("Courier New", 9, "bold"),
            padx=10, pady=2, cursor="hand2")
        self._rem_toggle_btn.pack(side="right")
        self._rem_toggle_btn.bind("<Button-1>", lambda e: self._toggle_reminders())

        # Countdown display
        countdown_row = tk.Frame(parent, bg=P["surface"])
        countdown_row.pack(fill="x", padx=14, pady=(4, 2))
        tk.Label(countdown_row, text="Next reminder in:",
                 bg=P["surface"], fg=P["dim"],
                 font=("Courier New", 9)).pack(side="left")
        self._rem_countdown = tk.Label(
            countdown_row, text="45:00",
            bg=P["surface"], fg=P["accent"],
            font=("Courier New", 14, "bold"))
        self._rem_countdown.pack(side="left", padx=(8, 0))

        # Countdown progress bar
        bar_bg = tk.Frame(parent, bg=P["surface2"], height=5)
        bar_bg.pack(fill="x", padx=14, pady=(2, 6))
        self._rem_bar = tk.Frame(bar_bg, bg=P["accent"], height=5)
        self._rem_bar.place(x=0, y=0, relwidth=1, relheight=1)

        # Interval slider row
        sl_row = tk.Frame(parent, bg=P["surface"])
        sl_row.pack(fill="x", padx=14, pady=(0, 4))
        tk.Label(sl_row, text="Every",
                 bg=P["surface"], fg=P["dim"],
                 font=("Courier New", 9)).pack(side="left")
        self._rem_mins_lbl = tk.Label(
            sl_row, text="45 min",
            bg=P["surface"], fg=P["accent"],
            font=("Courier New", 9, "bold"), width=7)
        self._rem_mins_lbl.pack(side="right")
        ttk.Scale(sl_row, from_=10, to=120,
                  variable=self.var_reminder_mins,
                  orient="horizontal",
                  command=self._on_reminder_interval_change
                  ).pack(side="left", fill="x", expand=True, padx=6)

        # Upcoming reminders preview
        tk.Label(parent, text="Reminder rotation:",
                 bg=P["surface"], fg=P["dim"],
                 font=("Courier New", 8)).pack(anchor="w", padx=14, pady=(2, 2))
        self._rem_preview_frames = []
        for i, (icon, title, short, _) in enumerate(REMINDERS):
            row = tk.Frame(parent, bg=P["surface"])
            row.pack(fill="x", padx=14, pady=1)
            dot = tk.Label(row, text="●", bg=P["surface"],
                           fg=P["dim"], font=("Courier New", 8))
            dot.pack(side="left", padx=(0, 4))
            lbl = tk.Label(row, text=f"{icon}  {title} — {short}",
                           bg=P["surface"], fg=P["dim"],
                           font=("Courier New", 8), anchor="w")
            lbl.pack(side="left", fill="x")
            self._rem_preview_frames.append((dot, lbl))

        # Send now button
        now_btn = tk.Label(parent, text="📨  SEND REMINDER NOW",
                           bg=P["btn"], fg=P["text"],
                           font=("Courier New", 9, "bold"),
                           pady=7, cursor="hand2")
        now_btn.pack(fill="x", padx=14, pady=(6, 10))
        now_btn.bind("<Button-1>", lambda e: self._fire_reminder_now())

    def _toggle_reminders(self):
        self._reminder_paused = not self._reminder_paused
        if self._reminder_paused:
            self._rem_toggle_btn.configure(text="OFF", bg=P["btn"], fg=P["dim"])
            self._rem_countdown.configure(fg=P["dim"])
        else:
            self._reminder_next = time.time() + self._reminder_interval
            self._rem_toggle_btn.configure(text="ON", bg=P["calm"], fg=P["black"])
            self._rem_countdown.configure(fg=P["accent"])

    def _on_reminder_interval_change(self, val):
        mins = int(float(val))
        self._reminder_interval = mins * 60
        self._reminder_next     = time.time() + self._reminder_interval
        self._rem_mins_lbl.configure(text=f"{mins} min")

    def _fire_reminder_now(self):
        reminder = REMINDERS[self._reminder_idx % len(REMINDERS)]
        threading.Thread(target=fire_reminder, args=(reminder,), daemon=True).start()
        self._reminder_idx  += 1
        self._reminder_next  = time.time() + self._reminder_interval
        self._update_reminder_preview()

    def _update_reminder_preview(self):
        for i, (dot, lbl) in enumerate(self._rem_preview_frames):
            is_next = (i == self._reminder_idx % len(REMINDERS))
            dot.configure(fg=P["accent"] if is_next else P["dim"])
            lbl.configure(fg=P["text"]   if is_next else P["dim"])

    def _update_reminder_countdown(self):
        if self._reminder_paused:
            self._rem_countdown.configure(text="paused")
            self._rem_bar.place(relwidth=0)
            return
        remaining = max(0, self._reminder_next - time.time())
        mins      = int(remaining // 60)
        secs      = int(remaining % 60)
        self._rem_countdown.configure(text=f"{mins:02d}:{secs:02d}")
        # Bar drains left→right as time passes
        progress = 1.0 - (remaining / max(self._reminder_interval, 1))
        self._rem_bar.place(x=0, y=0, relwidth=max(0.0, 1.0 - progress), relheight=1)
        # Fire when time is up
        if remaining <= 0:
            self._fire_reminder_now()

    # Telegram card
    def _build_telegram_card(self, parent):
        self._tg_open = False
        hdr = tk.Frame(parent, bg=P["surface"], cursor="hand2")
        hdr.pack(fill="x")
        self._tg_arrow = tk.Label(hdr, text="▶  TELEGRAM ALERTS",
                                  bg=P["surface"], fg=P["tg_blue"],
                                  font=("Courier New",10,"bold"),
                                  pady=7, padx=14, anchor="w")
        self._tg_arrow.pack(fill="x")
        for w in (hdr, self._tg_arrow):
            w.bind("<Button-1>", lambda e: self._toggle_section(
                "_tg_open", self._tg_body, self._tg_arrow,
                "▼  TELEGRAM ALERTS", "▶  TELEGRAM ALERTS"))

        self._tg_body = tk.Frame(parent, bg=P["surface"])

        # How-to text
        tk.Label(self._tg_body,
                 text="1. Message @BotFather → /newbot → copy token\n"
                      "2. Send any message to your new bot\n"
                      "3. Open: api.telegram.org/bot<TOKEN>/getUpdates\n"
                      "4. Copy your chat id  →  paste below",
                 bg=P["surface"], fg=P["dim"],
                 font=("Courier New",8), justify="left",
                 padx=14).pack(anchor="w", pady=(4,6))

        # Token row
        self._tg_entry("Bot Token :", self.var_tg_token, show="*")
        # Chat ID row
        self._tg_entry("Chat ID   :", self.var_tg_chat)

        btn_row = tk.Frame(self._tg_body, bg=P["surface"])
        btn_row.pack(fill="x", padx=14, pady=(6,4))

        save = tk.Label(btn_row, text="💾  SAVE",
                        bg=P["tg_blue"], fg=P["black"],
                        font=("Courier New",9,"bold"),
                        padx=14, pady=6, cursor="hand2")
        save.pack(side="left", padx=(0,6))
        save.bind("<Button-1>", lambda e: self._save_tg())

        test = tk.Label(btn_row, text="📨  SEND TEST",
                        bg=P["btn"], fg=P["text"],
                        font=("Courier New",9,"bold"),
                        padx=14, pady=6, cursor="hand2")
        test.pack(side="left")
        test.bind("<Button-1>", lambda e: self._test_tg())

        tk.Label(self._tg_body, textvariable=self._tg_status,
                 bg=P["surface"], fg=P["calm"],
                 font=("Courier New",8), pady=3).pack(anchor="w", padx=14)

        ind = tk.Frame(self._tg_body, bg=P["surface"])
        ind.pack(fill="x", padx=14, pady=(0,8))
        self._tg_dot = tk.Label(ind, text="●", bg=P["surface"],
                                fg=P["dim"], font=("Courier New",10))
        self._tg_dot.pack(side="left")
        tk.Label(ind, text=" Telegram alerts active",
                 bg=P["surface"], fg=P["dim"],
                 font=("Courier New",8)).pack(side="left")

    def _tg_entry(self, label, var, show=None):
        row = tk.Frame(self._tg_body, bg=P["surface"])
        row.pack(fill="x", padx=14, pady=2)
        tk.Label(row, text=label, bg=P["surface"], fg=P["text"],
                 font=("Courier New",9), width=12, anchor="w").pack(side="left")
        kw = dict(textvariable=var, bg=P["surface2"], fg=P["text"],
                  insertbackground=P["text"], relief="flat",
                  font=("Courier New",9), width=28)
        if show: kw["show"] = show
        tk.Entry(row, **kw).pack(side="left", padx=(4,0), ipady=4)

    def _save_tg(self):
        TELEGRAM.configure(self.var_tg_token.get(), self.var_tg_chat.get())
        if TELEGRAM.enabled:
            self._tg_status.set("✓ Saved! Press Send Test to verify.")
            self._tg_dot.configure(fg=P["calm"])
        else:
            self._tg_status.set("⚠ Fill in both Token and Chat ID.")
            self._tg_dot.configure(fg=P["watch"])

    def _test_tg(self):
        if not TELEGRAM.enabled:
            self._tg_status.set("⚠ Save your credentials first.")
            return
        self._tg_status.set("Sending test…")
        def _go():
            ok = TELEGRAM.test()
            self.after(0, lambda: self._tg_status.set(
                "✓ Check your Telegram!" if ok
                else "✗ Failed — check token & chat ID"))
        threading.Thread(target=_go, daemon=True).start()

    # Settings card
    def _build_settings_card(self, parent):
        self._sett_open = False
        hdr = tk.Frame(parent, bg=P["surface"], cursor="hand2")
        hdr.pack(fill="x")
        self._sett_arrow = tk.Label(hdr, text="▶  SETTINGS",
                                    bg=P["surface"], fg=P["accent"],
                                    font=("Courier New",10,"bold"),
                                    pady=7, padx=14, anchor="w")
        self._sett_arrow.pack(fill="x")
        for w in (hdr, self._sett_arrow):
            w.bind("<Button-1>", lambda e: self._toggle_section(
                "_sett_open", self._sett_body, self._sett_arrow,
                "▼  SETTINGS", "▶  SETTINGS"))

        self._sett_body = tk.Frame(parent, bg=P["surface"])

        def slider_row(lbl, var, lo, hi, fmt="{:.0f}"):
            row = tk.Frame(self._sett_body, bg=P["surface"])
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=lbl, bg=P["surface"], fg=P["text"],
                     font=("Courier New",9), width=22, anchor="w").pack(side="left")
            vl = tk.Label(row, text=fmt.format(var.get()),
                          bg=P["surface"], fg=P["accent"],
                          font=("Courier New",9,"bold"), width=5)
            vl.pack(side="right")
            ttk.Scale(row, from_=lo, to=hi, variable=var,
                      orient="horizontal", length=130,
                      command=lambda v, l=vl, f=fmt: l.configure(
                          text=f.format(float(v)))).pack(side="right", padx=4)

        slider_row("Alert threshold",  self.var_threshold, 30, 90)
        slider_row("Hold seconds",     self.var_hold,      1,  10, "{:.1f}")
        slider_row("Cooldown (sec)",   self.var_cooldown,  30, 300)
        tk.Frame(self._sett_body, bg=P["surface2"], height=1).pack(
            fill="x", padx=14, pady=4)
        self._lbl(self._sett_body,
                  "Signal weights (should sum ~100)").pack(anchor="w", padx=14)
        slider_row("Face stress weight",  self.var_w_face,  0, 100)
        slider_row("Eye/blink weight",    self.var_w_blink, 0, 100)
        slider_row("Movement weight",     self.var_w_move,  0, 100)
        tk.Frame(self._sett_body, bg=P["surface"], height=6).pack()

    def _toggle_section(self, flag_attr, body, arrow, open_txt, close_txt):
        is_open = not getattr(self, flag_attr)
        setattr(self, flag_attr, is_open)
        if is_open:
            body.pack(fill="x"); arrow.configure(text=open_txt)
        else:
            body.pack_forget(); arrow.configure(text=close_txt)

    # Alert overlay
    def _build_alert_overlay(self):
        self.overlay = tk.Toplevel(self)
        self.overlay.withdraw()
        self.overlay.overrideredirect(True)
        self.overlay.configure(bg=P["bg"])
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.97)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ow, oh = 520, 500
        self.overlay.geometry(f"{ow}x{oh}+{(sw-ow)//2}+{(sh-oh)//2}")

        self.breath_canvas = tk.Canvas(self.overlay, width=ow, height=110,
                                       bg=P["bg"], highlightthickness=0)
        self.breath_canvas.pack(pady=(20,0))

        tk.Label(self.overlay, text="🌿  TIME TO REGULATE",
                 bg=P["bg"], fg=P["calm"],
                 font=("Courier New",17,"bold")).pack(pady=(6,2))
        tk.Label(self.overlay,
                 text="Your body is showing overstimulation signals.\n"
                      "Here are some things that might help:",
                 bg=P["bg"], fg=P["dim"],
                 font=("Courier New",9), justify="center").pack(pady=(0,10))

        self._activity_idx = int(time.time()) % len(CALMING_ACTIVITIES)
        acts = tk.Frame(self.overlay, bg=P["bg"])
        acts.pack(fill="x", padx=22)
        for i in range(3):
            idx = (self._activity_idx + i) % len(CALMING_ACTIVITIES)
            icon, title, desc = CALMING_ACTIVITIES[idx]
            card = tk.Frame(acts, bg=P["surface"], pady=7, padx=10)
            card.pack(fill="x", pady=3)
            tk.Label(card, text=f"{icon}  {title}", bg=P["surface"],
                     fg=P["text"], font=("Courier New",10,"bold"),
                     anchor="w").pack(fill="x")
            tk.Label(card, text=desc, bg=P["surface"], fg=P["dim"],
                     font=("Courier New",8), anchor="w").pack(fill="x")

        dismiss = tk.Label(self.overlay, text="✓  I'M OKAY, DISMISS",
                           bg=P["calm"], fg=P["black"],
                           font=("Courier New",11,"bold"),
                           pady=11, cursor="hand2")
        dismiss.pack(fill="x", padx=22, pady=14)
        dismiss.bind("<Button-1>", lambda e: self._dismiss_alert())

    def _show_alert(self):
        if self.alert_visible: return
        self.alert_visible = True
        self.overlay.deiconify(); self.overlay.lift()

    def _dismiss_alert(self):
        self.alert_visible = False
        self.overlay.withdraw()

    # Gauge
    def _draw_gauge(self, score):
        c = self.gauge_canvas; c.delete("all")
        cx, cy, r = 110, 76, 57
        c.create_arc(cx-r,cy-r,cx+r,cy+r,
                     start=210,extent=-240,outline=P["surface2"],width=8,style="arc")
        if score > 0:
            col = P["calm"] if score<40 else P["watch"] if score<65 else P["alert"]
            c.create_arc(cx-r,cy-r,cx+r,cy+r,
                         start=210,extent=-240*score/100,
                         outline=col,width=8,style="arc")
        self._breath_angle += 0.025 * self._breath_dir
        if self._breath_angle > 1: self._breath_dir = -1
        if self._breath_angle < 0: self._breath_dir =  1
        pr  = int(5 + 6*self._breath_angle)
        col = P["calm"] if score<40 else P["watch"] if score<65 else P["alert"]
        c.create_oval(cx-pr,cy-pr,cx+pr,cy+pr,fill=col,outline="")

    def _draw_breathing_anim(self):
        c = self.breath_canvas; c.delete("all")
        val = (math.sin(time.time()*0.7)+1)/2
        r   = int(18+32*val); cx,cy = 260,56
        for dr in range(5,0,-1):
            gr=r+dr*5; c.create_oval(cx-gr,cy-gr,cx+gr,cy+gr,
                                     outline=P["calm"],width=1)
        c.create_oval(cx-r,cy-r,cx+r,cy+r,fill=P["calm"],outline="")
        c.create_text(cx,cy+r+22,
                      text="Inhale slowly..." if val>0.5 else "Exhale slowly...",
                      fill=P["dim"],font=("Courier New",10))

    # Camera thread
    def _camera_worker(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.running = False; return
        while self.running:
            ok, frame = self.cap.read()
            if not ok: continue
            frame = cv2.flip(frame,1)
            h,w   = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fr    = FACE_MESH.process(rgb)
            pr    = POSE.process(rgb)
            fs=bs=ms=0.0
            if fr.multi_face_landmarks:
                fl = fr.multi_face_landmarks[0].landmark
                fs = face_stress_score(fl,w,h)
                self.blink_tracker.update(fl,w,h)
                bs = self.blink_tracker.score()
                mp_draw.draw_landmarks(frame,fr.multi_face_landmarks[0],
                    mp_face.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(60,55,90),thickness=1,circle_radius=1),
                    mp_draw.DrawingSpec(color=(50,48,80),thickness=1))
            if pr.pose_landmarks:
                self.move_tracker.update(pr.pose_landmarks.landmark,w,h)
                ms = self.move_tracker.score()
                mp_draw.draw_landmarks(frame,pr.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(100,90,160),thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(70,65,120),thickness=2))
            weights = [self.var_w_face.get(),
                       self.var_w_blink.get(),
                       self.var_w_move.get()]
            self.monitor.update(fs,bs,ms,weights)
            if self.monitor.check_alert(self.var_threshold.get(),
                                        self.var_hold.get(),
                                        self.var_cooldown.get()):
                act = CALMING_ACTIVITIES[int(time.time())%len(CALMING_ACTIVITIES)]
                threading.Thread(target=fire_all_alerts,args=(act,),daemon=True).start()
                self.after(0, self._show_alert)
            display = cv2.resize(frame,(540,360))
            pil     = Image.fromarray(cv2.cvtColor(display,cv2.COLOR_BGR2RGB))
            with self._frame_lock:
                self._pending_pil = pil
        if self.cap: self.cap.release()

    # UI loop
    def _ui_loop(self):
        score  = self.monitor.score
        fs,bs,ms = self.monitor.face_s, self.monitor.blink_s, self.monitor.move_s

        with self._frame_lock:
            pil = self._pending_pil; self._pending_pil = None
        if pil is not None:
            self._current_frame = ImageTk.PhotoImage(pil)
        if self._current_frame:
            self.cam_label.configure(image=self._current_frame,text="")
            self.cam_label.image = self._current_frame

        self._draw_gauge(score)
        if self.running:
            col = P["calm"] if score<40 else P["watch"] if score<65 else P["alert"]
            self.score_label.configure(text=str(int(score)),fg=col)
            self.status_label.configure(
                text=("CALM  🌿" if score<40 else
                      "WATCH  🟡" if score<65 else "OVERSTIMULATED  🔴"),fg=col)
        else:
            self.score_label.configure(text="—",fg=P["dim"])
            self.status_label.configure(text="NOT RUNNING",fg=P["dim"])

        now=time.time(); hw=self.hold_bar_bg.winfo_width()
        if self.monitor.alert_start and hw>0:
            prog=min((now-self.monitor.alert_start)/max(self.var_hold.get(),0.1),1.0)
            self.hold_bar_fill.place(x=0,y=0,width=int(hw*prog),height=7)
        else:
            self.hold_bar_fill.place(x=0,y=0,width=0,height=7)

        for key,val in [("face",fs),("blink",bs),("move",ms)]:
            fill,lbl,bg=self.bar_vars[key]
            bw=bg.winfo_width()
            fw=int(bw*val/100) if bw>0 else 0
            col=P["calm"] if val<40 else P["watch"] if val<65 else P["alert"]
            fill.place(x=0,y=0,width=fw,relheight=1); fill.configure(bg=col)
            lbl.configure(text=str(int(val)),fg=col)

        if hasattr(self,"_tg_dot"):
            self._tg_dot.configure(fg=P["calm"] if TELEGRAM.enabled else P["dim"])
        if hasattr(self, "_rem_countdown"):
            self._update_reminder_countdown()
            self._update_reminder_preview()
        if self.alert_visible:
            self._draw_breathing_anim()

        self.after(50, self._ui_loop)

    # Controls
    def _toggle_camera(self):
        if self.running:
            self.running=False
            self.btn_start.configure(text="▶  START MONITORING",bg=P["calm"])
            self.cam_label.configure(image="",
                text="Press  ▶ START MONITORING  above")
            self._current_frame=None
        else:
            self.running=True
            self.btn_start.configure(text="⏹  STOP MONITORING",bg=P["alert"])
            self._cam_thread=threading.Thread(target=self._camera_worker,daemon=True)
            self._cam_thread.start()

    def _reset_score(self):
        self.monitor.reset(); self._dismiss_alert()

    def _on_close(self):
        self.running=False; time.sleep(0.15)
        FACE_MESH.close(); POSE.close(); self.destroy()


if __name__ == "__main__":
    App().mainloop()
