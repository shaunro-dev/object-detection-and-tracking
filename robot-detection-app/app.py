"""
AI-Based Robot Remote Detection and Retrieval System
Unified Flask + YOLO Backend Server (High-Performance Tracking Edition)
------------------------------------------------------------------------
Run:  python app.py
Opens browser automatically at http://localhost:5001
"""

import os
import sys
import time
import json
import random
import shutil
import sqlite3
import hashlib
import logging
import platform
import threading
import subprocess
import webbrowser
from collections import deque

import cv2
import torch
import numpy as np
from ultralytics import YOLO, YOLOWorld
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

# Disable Flask/Werkzeug request logging to keep console clean for terminal inputs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ─────────────────────────────────────────────
#  FLASK + SOCKETIO SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='static')
app.config['SECRET_KEY'] = 'robot_secret_key'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False
)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
CUSTOM_MODELS_DIR = os.path.join(BASE_DIR, "custom_models")
DB_PATH = os.path.join(BASE_DIR, "robot_data.db")


# ─────────────────────────────────────────────
#  SQLITE DATABASE
# ─────────────────────────────────────────────
def init_db():
    """Create tables on first run."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            label     TEXT    NOT NULL,
            confidence REAL   NOT NULL,
            position  TEXT,
            direction TEXT,
            speed     REAL,
            mode      TEXT,
            bbox_json TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            command   TEXT NOT NULL,
            target    TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS session_reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at  TEXT NOT NULL,
            ended_at    TEXT,
            detections  INTEGER DEFAULT 0,
            top_label   TEXT,
            notes       TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("[DB] SQLite database ready:", DB_PATH)


def db_save_detection(label, confidence, position, direction, speed, mode, bbox=None):
    """Persist a detection event to SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO detections (timestamp,label,confidence,position,direction,speed,mode,bbox_json) VALUES (?,?,?,?,?,?,?,?)",
            (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                label,
                confidence,
                position,
                direction,
                speed,
                mode,
                json.dumps(bbox) if bbox else None
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Save detection error: {e}")


def db_save_command(command, target=None):
    """Persist a control command to SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO commands (timestamp,command,target) VALUES (?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), command, target)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Save command error: {e}")


def db_get_detections(limit=100):
    """Return recent detections as list of dicts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Get detections error: {e}")
        return []


def db_get_commands(limit=50):
    """Return recent commands as list of dicts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM commands ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Get commands error: {e}")
        return []


# ─────────────────────────────────────────────
#  COCO CLASSES
# ─────────────────────────────────────────────
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
current_target = None
target_lock = threading.Lock()

latest_preds = None
latest_preds_lock = threading.Lock()

latest_frame = None
latest_frame_lock = threading.Lock()

app_state = "TRACKING"
state_lock = threading.Lock()

custom_class_name = "mobile"
custom_epochs = 20
use_custom_model = False
custom_model = None

cam_global = None
world_model_global = None

# Live detection summary for the dashboard
detection_summary = {
    "detected": False,
    "label": "--",
    "confidence": 0.0,
    "position": "--",
    "direction": "---",
    "speed": 0.0,
    "fps": 0.0,
    "mode": "ALL",
    "target": "all",
    "objects_count": 0,
    "status": "SEARCHING...",
    "custom_models": [],
    "app_state": "TRACKING"
}
summary_lock = threading.Lock()

# ─────────────────────────────────────────────
#  TTS WORKER
# ─────────────────────────────────────────────
latest_speech_text = None
speech_lock = threading.Lock()
last_announced_location = None
last_announced_time = 0
last_announced_target = None


def speech_worker():
    global latest_speech_text
    system_os = platform.system().lower()
    while True:
        text_to_say = None
        with speech_lock:
            if latest_speech_text is not None:
                text_to_say = latest_speech_text
                latest_speech_text = None
        if text_to_say:
            try:
                if "darwin" in system_os:
                    subprocess.run(["say", text_to_say], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif "windows" in system_os:
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text_to_say}')"
                    subprocess.run(["powershell", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["espeak", text_to_say], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        time.sleep(0.1)


def reset_speech_tracking():
    global last_announced_location, last_announced_target
    last_announced_location = None
    last_announced_target = None


def queue_speech(target, cx, cy, w, h):
    global latest_speech_text, last_announced_location, last_announced_time, last_announced_target
    current_time = time.time()
    x_part = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "middle")
    y_part = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")

    if x_part == "middle" and y_part == "middle":
        location = "center screen"
    elif y_part == "middle":
        location = f"middle {x_part}"
    elif x_part == "middle":
        location = f"{y_part} middle"
    else:
        location = f"{y_part} {x_part}"

    if (
        target != last_announced_target
        or location != last_announced_location
        or (current_time - last_announced_time > 7.0)
    ):
        last_announced_location = location
        last_announced_target = target
        last_announced_time = current_time
        with speech_lock:
            latest_speech_text = f"{target} on the {location}"


# ─────────────────────────────────────────────
#  HARDWARE DEVICE SELECTION
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    DETECTOR_DEVICE = 'cuda'
    WORLD_DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DETECTOR_DEVICE = 'mps'
    WORLD_DEVICE = 'cpu'
else:
    DETECTOR_DEVICE = 'cpu'
    WORLD_DEVICE = 'cpu'

print(f"\n[INFO] Primary Detector Device: {DETECTOR_DEVICE}")
print(f"[INFO] YOLO-World Device: {WORLD_DEVICE}")


# ─────────────────────────────────────────────
#  DIRECTION & POSITION TRACKER
# ─────────────────────────────────────────────
class DirectionTracker:
    def __init__(self, history=10, threshold=6):
        self.positions = deque(maxlen=history)
        self.threshold = threshold
        self.last_dir = "STATIONARY"
        self.last_spd = 0.0

    def update(self, cx, cy):
        self.positions.append((cx, cy, time.time()))

    def reset(self):
        self.positions.clear()
        self.last_dir = "STATIONARY"
        self.last_spd = 0.0

    def get(self):
        if len(self.positions) < 3:
            return "TRACKING...", 0.0
        x0, y0, t0 = self.positions[0]
        xn, yn, tn = self.positions[-1]
        dt = max(tn - t0, 0.001)
        dx, dy = xn - x0, yn - y0
        speed = round(np.hypot(dx, dy) / dt, 1)

        mh = abs(dx) > self.threshold
        mv = abs(dy) > self.threshold

        if not mh and not mv:
            direction = "STATIONARY"
        else:
            h = ("RIGHT" if dx > 0 else "LEFT") if mh else ""
            v = ("DOWN" if dy > 0 else "UP") if mv else ""
            direction = f"{v}-{h}" if h and v else h or v

        self.last_dir = direction
        self.last_spd = speed
        return direction, speed

    def draw_arrow(self, frame, cx, cy):
        d, _ = self.get()
        vectors = {
            "LEFT": (-50, 0), "RIGHT": (50, 0),
            "UP": (0, -50), "DOWN": (0, 50),
            "UP-LEFT": (-35, -35), "UP-RIGHT": (35, -35),
            "DOWN-LEFT": (-35, 35), "DOWN-RIGHT": (35, 35)
        }
        if d in vectors:
            dx2, dy2 = vectors[d]
            ex, ey = cx + dx2, cy + dy2
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.35)


# ─────────────────────────────────────────────
#  THREADED CAMERA STREAM (Pre-flipped at source)
# ─────────────────────────────────────────────
class CameraStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror once at camera capture so all threads have identical coordinates
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.ret, self.frame = ret, frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()


# ─────────────────────────────────────────────
#  UI DRAWING HELPERS
# ─────────────────────────────────────────────
def get_box_color(label):
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return (max(h & 0xFF, 70), max((h >> 8) & 0xFF, 70), max((h >> 16) & 0xFF, 70))


def draw_tracked_object(frame, box, class_name, score, track_id, w, h):
    """Renders bounding box with Track ID + Class Name + Confidence."""
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    color = get_box_color(class_name)

    id_label = f"ID:#{track_id} " if track_id is not None else ""
    lbl = f"{id_label}{class_name.upper()} {int(score * 100)}%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    ly = max(y1 - 5, th + 8)
    cv2.rectangle(frame, (x1, ly - th - 5), (x1 + tw + 6, ly + 2), color, -1)
    tc = (0, 0, 0) if sum(color) > 360 else (255, 255, 255)
    cv2.putText(frame, lbl, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.44, tc, 1, cv2.LINE_AA)
    return cx, cy


def draw_hud(frame, mode, target, status, direction, speed, obj_count, frame_cx, fps):
    h2, w2 = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w2, 60), (12, 12, 12), -1)
    cv2.line(frame, (frame_cx, 60), (frame_cx, h2), (50, 50, 50), 1, cv2.LINE_AA)

    fc = (0, 255, 0) if fps > 22 else (0, 165, 255) if fps > 14 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w2 - 85, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, fc, 1, cv2.LINE_AA)

    if mode == "ALL":
        cv2.putText(frame, "MODE: MULTI-OBJECT TRACKING (YOLO11 + ByteTrack)", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"OBJECTS ACTIVE: {obj_count}", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        sc = (0, 255, 0) if "CENTER" in status else (0, 165, 255)
        if "LOST" in status or "SEARCH" in status:
            sc = (80, 80, 255)
        cv2.putText(frame, f"MODE: TARGET LOCKED ('{target.upper()}')", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"STATUS: {status} | DIR: {direction} | SPD: {speed:.0f} px/s", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sc, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h2 - 24), (w2, h2), (12, 12, 12), -1)
    cv2.putText(frame, "AI Robot Tracking & Retrieval System | Live Telemetry Active", (12, h2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  INFERENCE THREAD (Optimized GPU Tracking)
# ─────────────────────────────────────────────
def inference_thread(world_model, cam):
    global latest_preds, current_target, app_state, custom_class_name, use_custom_model, custom_model

    print("[INFO] Initializing YOLO11 with ByteTrack native tracker...")
    yolo11_model = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))
    current_vocab = None

    while True:
        with state_lock:
            state = app_state
        if state != "TRACKING":
            time.sleep(0.08)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        with target_lock:
            active_tgt = current_target

        boxes = np.empty((0, 4))
        scores = np.empty((0,))
        labels_names = []
        track_ids = []

        is_custom_ready = use_custom_model and custom_model is not None

        try:
            # Mode 1: Custom Trained Model
            if is_custom_ready and custom_model is not None and (active_tgt == "custom" or active_tgt == custom_class_name):
                results = custom_model.track(
                    frame, imgsz=480, conf=0.55, persist=True,
                    tracker="bytetrack.yaml", verbose=False, device=DETECTOR_DEVICE
                )
                if len(results) > 0 and len(results[0].boxes) > 0:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    labels_names = [custom_class_name] * len(boxes)
                    track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes)

            # Mode 2: Multi-Object Tracking on All Objects (Fast YOLO11)
            elif active_tgt is None or active_tgt == "all":
                results = yolo11_model.track(
                    frame, imgsz=480, conf=0.58, persist=True,
                    tracker="bytetrack.yaml", verbose=False, device=DETECTOR_DEVICE
                )
                if len(results) > 0 and len(results[0].boxes) > 0:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    labels_idx = r.boxes.cls.cpu().numpy().astype(int)
                    labels_names = [r.names[idx] for idx in labels_idx]
                    track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes)

            # Mode 3: Open-Vocabulary Tracking (YOLO-World)
            else:
                if current_vocab != active_tgt:
                    world_model.set_classes([active_tgt])
                    current_vocab = active_tgt
                results = world_model.predict(
                    frame, imgsz=480, conf=0.30, verbose=False, device=WORLD_DEVICE
                )
                if len(results) > 0 and len(results[0].boxes) > 0:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    labels_idx = r.boxes.cls.cpu().numpy().astype(int)
                    labels_names = [r.names[idx] for idx in labels_idx]
                    track_ids = [None] * len(boxes)

        except Exception as e:
            print(f"\n[ERROR] Multi-object tracking inference error: {e}")

        with latest_preds_lock:
            latest_preds = {
                'boxes': boxes,
                'labels': labels_names,
                'scores': scores,
                'track_ids': track_ids
            }

        time.sleep(0.005)


# ─────────────────────────────────────────────
#  FRAME BUILDER & DISPLAY THREAD
# ─────────────────────────────────────────────
def detector_thread(cam):
    global latest_frame, app_state, custom_class_name, use_custom_model, custom_model, detection_summary

    dir_tracker = DirectionTracker(history=10, threshold=5)
    fps = 0.0
    fps_t = time.time()
    fps_c = 0
    direction = "---"
    speed = 0.0
    last_emit = 0.0

    ret, test = cam.read()
    frame_w = test.shape[1] if (ret and test is not None) else 1280
    frame_h = test.shape[0] if (ret and test is not None) else 720
    frame_cx = frame_w // 2

    while True:
        with state_lock:
            current_state = app_state

        if current_state != "TRACKING":
            time.sleep(0.05)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

        with latest_preds_lock:
            preds = latest_preds

        if preds is None:
            time.sleep(0.02)
            continue

        boxes = preds['boxes']
        labels = preds['labels']
        scores = preds['scores']
        track_ids = preds.get('track_ids', [None] * len(boxes))

        with target_lock:
            active_target = current_target

        detected_flag = False
        det_label = "--"
        det_conf = 0.0
        det_position = "--"
        det_status = "SEARCHING..."
        obj_count = 0

        # ── ALL OBJECTS TRACKING MODE ──
        if active_target is None or active_target == "all":
            dir_tracker.reset()
            direction = "---"
            speed = 0.0
            mode_str = "ALL"

            for box, cn, score, tid in zip(boxes, labels, scores, track_ids):
                if score < 0.55:
                    continue
                cx, cy = draw_tracked_object(frame, box, cn, score, tid, w, h)
                obj_count += 1
                if not detected_flag:
                    detected_flag = True
                    det_label = f"#{tid} {cn}" if tid else cn
                    det_conf = float(score)
                    det_position = "Left" if cx < w / 3 else ("Right" if cx > 2 * w / 3 else "Center")

            draw_hud(frame, "ALL", "", "", direction, speed, obj_count, frame_cx, fps)

        # ── TARGET FOCUS MODE (Custom or Open-Vocab) ──
        else:
            mode_str = "FOCUS"
            target_found = False

            # Subtle dark overlay to focus on target
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            for box, cn, score, tid in zip(boxes, labels, scores, track_ids):
                match_condition = (
                    (cn.lower() == custom_class_name.lower()) if use_custom_model else (active_target in cn.lower())
                )
                if match_condition:
                    target_found = True
                    cx, cy = draw_tracked_object(frame, box, cn, score, tid, w, h)
                    obj_count += 1

                    dir_tracker.update(cx, cy)
                    direction, speed = dir_tracker.get()
                    dir_tracker.draw_arrow(frame, cx, cy)

                    DEADZONE = 50
                    if cx < (frame_cx - DEADZONE):
                        cmd = "MOVE LEFT"
                    elif cx > (frame_cx + DEADZONE):
                        cmd = "MOVE RIGHT"
                    else:
                        cmd = "STAY CENTER"

                    det_status = cmd
                    detected_flag = True
                    det_label = cn
                    det_conf = float(score)
                    det_position = "Left" if cx < w / 3 else ("Right" if cx > 2 * w / 3 else "Center")

                    queue_speech(active_target, cx, cy, w, h)
                    break

            if not target_found:
                dir_tracker.reset()
                det_status = "TARGET LOST"
                direction = "---"
                speed = 0.0
                cv2.putText(
                    frame, f"'{active_target.upper()}' NOT IN SIGHT",
                    (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA
                )
                reset_speech_tracking()

            draw_hud(frame, "TARGET", active_target, det_status, direction, speed, obj_count, frame_cx, fps)

        # FPS Calculation
        fps_c += 1
        el = time.time() - fps_t
        if el >= 1.0:
            fps = fps_c / el
            fps_t = time.time()
            fps_c = 0

        available_customs = [
            f[:-3] for f in os.listdir(CUSTOM_MODELS_DIR) if f.endswith(".pt") and f != "best.pt"
        ] if os.path.exists(CUSTOM_MODELS_DIR) else []

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with latest_frame_lock:
            latest_frame = jpeg.tobytes()

        with summary_lock:
            detection_summary.update({
                "detected": detected_flag,
                "label": det_label,
                "confidence": round(det_conf * 100, 1),
                "position": det_position,
                "direction": direction,
                "speed": round(speed, 1),
                "fps": round(fps, 1),
                "mode": mode_str,
                "target": active_target or "all",
                "objects_count": obj_count,
                "status": det_status,
                "custom_models": available_customs,
                "app_state": app_state
            })

        # SocketIO telemetry broadcast (~15Hz)
        now = time.time()
        if now - last_emit > 0.06:
            last_emit = now
            with summary_lock:
                data = dict(detection_summary)
            socketio.emit('detection_update', data)

        # Persist detection snapshots to SQLite periodically
        if detected_flag and int(time.time()) % 5 == 0:
            db_save_detection(
                label=det_label,
                confidence=round(det_conf * 100, 1),
                position=det_position,
                direction=direction,
                speed=round(speed, 1),
                mode=mode_str
            )


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
def load_models():
    global use_custom_model, custom_class_name, custom_model

    print("\n" + "=" * 60)
    print("  ROBOT VISION  |  MULTI-OBJECT TRACKING SERVER")
    print("=" * 60)

    os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)

    # Check for existing custom weights
    if os.path.exists(CUSTOM_MODEL_PATH):
        try:
            temp_model = YOLO(CUSTOM_MODEL_PATH)
            if hasattr(temp_model, 'names') and len(temp_model.names) > 0:
                custom_class_name = temp_model.names[0]
                custom_model = temp_model
                use_custom_model = True
                print(f"[INFO] Auto-loaded custom model: '{custom_class_name.upper()}'")
        except Exception as e:
            print(f"[WARNING] Could not load custom best.pt: {e}")

    print("[INFO] Loading YOLO-World for open-vocabulary querying...")
    world_model = YOLOWorld(os.path.join(BASE_DIR, "yolov8m-world.pt"))
    print("[INFO] Models initialized successfully.\n")
    return world_model, custom_model


def emit_status():
    with summary_lock:
        data = dict(detection_summary)
    socketio.emit('detection_update', data)


def set_app_state(new_state, status_text=None):
    global app_state
    with state_lock:
        app_state = new_state
    with summary_lock:
        detection_summary["app_state"] = new_state
        if status_text is not None:
            detection_summary["status"] = status_text
    emit_status()


# ─────────────────────────────────────────────
#  GUI DATASET & TRAINING MANAGER
# ─────────────────────────────────────────────
def gui_state_loop(cam, world_model):
    global app_state, custom_class_name, custom_epochs, custom_model, use_custom_model

    while True:
        with state_lock:
            current_state = app_state

        if current_state == "TRACKING":
            time.sleep(0.1)

        elif current_state == "COLLECT_REQUESTED":
            save_dir = os.path.join(BASE_DIR, "dataset", custom_class_name)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            set_app_state("COLLECTING", f"COLLECTING: '{custom_class_name}' (Press S / Q)")

            count = 0
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                disp = frame.copy()
                cv2.putText(disp, f"COLLECTING: {custom_class_name.upper()} | Saved: {count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(disp, "Press 'S' to Save | 'Q' when Done", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Capture Dataset", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    filename = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
                    cv2.imwrite(filename, frame)
                    count += 1
                    print(f"  [SAVED] {filename} (Total: {count})")
                elif key == ord('q') or key == ord('Q'):
                    break

            cv2.destroyWindow("Capture Dataset")
            cv2.waitKey(1)
            set_app_state("LABELING", f"LABELING: '{custom_class_name}'")

            # Automated setup for dataset folders
            yolo_dir = os.path.join(BASE_DIR, "yolo_dataset")
            if os.path.exists(yolo_dir):
                shutil.rmtree(yolo_dir)
            for split in ['train', 'val']:
                os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

            images = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            random.shuffle(images)
            split_idx = max(1, int(len(images) * 0.8))
            splits = {'train': images[:split_idx], 'val': images[split_idx:]}

            for split_name, img_list in splits.items():
                for idx, img_name in enumerate(img_list):
                    img_path = os.path.join(save_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h_img, w_img = img.shape[:2]
                    title = f"Label {split_name.upper()} ({idx + 1}/{len(img_list)}) | Space=Lock | Esc=Next"
                    rois = cv2.selectROIs(title, img, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow(title)
                    cv2.waitKey(1)

                    shutil.copy(img_path, os.path.join(yolo_dir, "images", split_name, img_name))
                    txt_path = os.path.join(yolo_dir, "labels", split_name, img_name.rsplit('.', 1)[0] + ".txt")
                    with open(txt_path, "w") as f:
                        for roi in rois:
                            rx, ry, rw, rh = roi
                            if rw > 0 and rh > 0:
                                cx = (rx + (rw / 2)) / w_img
                                cy = (ry + (rh / 2)) / h_img
                                f.write(f"0 {cx:.6f} {cy:.6f} {rw / w_img:.6f} {rh / h_img:.6f}\n")

            # Save data.yaml
            yaml_content = f"path: {os.path.abspath(yolo_dir)}\ntrain: images/train\nval: images/val\n\nnames:\n  0: {custom_class_name}\n"
            with open(os.path.join(BASE_DIR, "data.yaml"), "w") as f:
                f.write(yaml_content)

            # Training execution
            set_app_state("TRAINING", f"TRAINING: '{custom_class_name}' ({custom_epochs} ep)")
            base_model = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))
            base_model.train(
                data=os.path.join(BASE_DIR, "data.yaml"),
                epochs=custom_epochs,
                imgsz=480,
                batch=16,
                device=DETECTOR_DEVICE,
                name=f"custom_{custom_class_name}",
                exist_ok=True,
                verbose=False
            )

            trained_weights = os.path.join(BASE_DIR, "runs", "detect", f"custom_{custom_class_name}", "weights", "best.pt")
            if os.path.exists(trained_weights):
                shutil.copy(trained_weights, CUSTOM_MODEL_PATH)
                dest_stored = os.path.join(CUSTOM_MODELS_DIR, f"{custom_class_name}.pt")
                shutil.copy(trained_weights, dest_stored)
                custom_model = YOLO(CUSTOM_MODEL_PATH)
                use_custom_model = True
                print(f"\n[SUCCESS] Custom model deployed: {dest_stored}\n")

            set_app_state("TRACKING", "TRACKING")

        else:
            time.sleep(0.1)


# ─────────────────────────────────────────────
#  INTERACTIVE COMMAND LINE
# ─────────────────────────────────────────────
def input_listener():
    global current_target, custom_class_name, custom_epochs, use_custom_model, app_state
    print("\n" + "=" * 60)
    print("  INTERACTIVE TERMINAL")
    print("  Type 'all'           -> Multi-object tracking across everything")
    print("  Type [object]        -> Lock onto a specific object (e.g. bottle)")
    print("  Type 'collect'       -> Train an on-spot custom object")
    print("  Type 'list'          -> Print detectable classes")
    print("=" * 60 + "\n")

    while True:
        try:
            cmd = input("  >> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            if cam_global:
                cam_global.release()
            cv2.destroyAllWindows()
            os._exit(0)

        if not cmd:
            continue

        if cmd == "all":
            with target_lock:
                current_target = None
            use_custom_model = False
            print("  [OK] Tracking all objects with persistent IDs.\n")

        elif cmd == "collect":
            name = input("  Enter target class name (e.g., pen, remote): ").strip().lower()
            if name:
                with state_lock:
                    custom_class_name = name
                    custom_epochs = 20
                    app_state = "COLLECT_REQUESTED"
                print(f"  [OK] Starting dataset collection for '{name}'...\n")

        elif cmd == "list":
            print("\nStandard Classes:")
            for i, cn in enumerate(sorted(COCO_CLASSES)):
                print(f"  {cn:<16}", end="")
                if (i + 1) % 4 == 0:
                    print()
            print("\n")

        else:
            with target_lock:
                current_target = cmd
            print(f"  [OK] Locking onto target: '{cmd.upper()}'\n")


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/style.css')
def serve_css():
    return send_from_directory('static', 'style.css')


@app.route('/script.js')
def serve_js():
    return send_from_directory('static', 'script.js')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with latest_frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.04)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with summary_lock:
        return jsonify(dict(detection_summary))


@app.route('/api/detection/history')
def api_detection_history():
    limit = int(request.args.get('limit', 100))
    return jsonify(db_get_detections(limit))


@app.route('/api/commands/history')
def api_commands_history():
    limit = int(request.args.get('limit', 50))
    return jsonify(db_get_commands(limit))


@app.route('/api/target', methods=['POST'])
def api_set_target():
    global current_target, use_custom_model
    data = request.get_json() or {}
    target = data.get('target', 'all').strip().lower()
    if target in ['all', '']:
        with target_lock:
            current_target = None
        use_custom_model = False
        return jsonify({"ok": True, "target": "all", "mode": "ALL"})
    else:
        with target_lock:
            current_target = target
        return jsonify({"ok": True, "target": target, "mode": "FOCUS"})


# ─────────────────────────────────────────────
#  SOCKET.IO HANDLERS
# ─────────────────────────────────────────────
active_clients = set()
active_clients_lock = threading.Lock()


@socketio.on('connect')
def on_connect():
    with active_clients_lock:
        active_clients.add(request.sid)
    with summary_lock:
        emit('detection_update', dict(detection_summary))


@socketio.on('disconnect')
def on_disconnect():
    with active_clients_lock:
        if request.sid in active_clients:
            active_clients.remove(request.sid)


@socketio.on('set_target')
def on_set_target(data):
    global current_target, use_custom_model
    target = (data.get('target') or 'all').strip().lower()
    if target in ['all', '']:
        with target_lock:
            current_target = None
        use_custom_model = False
        emit('target_changed', {"target": "all", "mode": "ALL"})
    else:
        with target_lock:
            current_target = target
        emit('target_changed', {"target": target, "mode": "FOCUS"})


@socketio.on('manual_command')
def on_manual_command(data):
    cmd = (data.get('command') or '').strip().upper()
    if cmd:
        db_save_command(cmd, target="manual")
        with summary_lock:
            detection_summary["status"] = f"MANUAL: {cmd}"
        socketio.emit('detection_update', dict(detection_summary))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AI Robot Detection & Retrieval System")
    print("  Dashboard -> http://localhost:5001")
    print("=" * 60 + "\n")

    init_db()
    world_model, custom_model = load_models()
    world_model_global = world_model

    cam = CameraStream(src=0, width=1280, height=720)
    cam_global = cam
    time.sleep(0.8)

    threading.Thread(target=inference_thread, args=(world_model, cam), daemon=True).start()
    threading.Thread(target=detector_thread, args=(cam,), daemon=True).start()
    threading.Thread(target=speech_worker, daemon=True).start()

    flask_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=5001, debug=False, use_reloader=False, allow_unsafe_werkzeug=True),
        daemon=True
    )
    flask_thread.start()

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:5001")

    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=input_listener, daemon=True).start()

    try:
        gui_state_loop(cam, world_model)
    except KeyboardInterrupt:
        if cam_global:
            cam_global.release()
        cv2.destroyAllWindows()
        os._exit(0)