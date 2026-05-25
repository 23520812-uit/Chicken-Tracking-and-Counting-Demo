"""
Chicken Tracking & Counting Demo – Streamlit Application
Refinement modules: BGD, CL‑ID, TVM
Simple UI: select module, upload video -> automatic tracking with side‑by‑side view.
No frame limit – processes the entire video.
Optimised for Streamlit Cloud (CPU, lightweight packages).
"""
import subprocess
import sys

# ── Uninstall opencv‑python (comes with ultralytics) and install headless ──
def _ensure_headless():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"],
            check=False, capture_output=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"],
            check=True
        )
    except Exception:
        # fallback: use whatever is already installed
        pass

_ensure_headless()

import streamlit as st
import cv2
import numpy as np
import tempfile
import sys
from pathlib import Path
from collections import defaultdict
import torch
import pandas as pd

# Monkey-patch for pkg_resources
import importlib.resources
import importlib.metadata as importlib_metadata

class FakePkgResources:
    @staticmethod
    def resource_filename(package, resource):
        return str(importlib.resources.files(package) / resource)
    @staticmethod
    def get_distribution(name):
        return importlib_metadata.distribution(name)

sys.modules.setdefault('pkg_resources', FakePkgResources)

from ultralytics import YOLO
from boxmot.trackers import ByteTrack as BYTETracker

# ============================================================
# CONFIG
# ============================================================
APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = APP_DIR / "weights" / "best.pt"
TEMP_DIR = Path(tempfile.gettempdir()) / "chicken_tracking_demo"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

MAX_FRAMES = None          # process entire video, no limit
DETECT_CONF = 0.12
DETECT_IOU  = 0.50
IMGSZ = 640

# ============================================================
# REFINEMENT MODULES (latest versions with visualisation)
# ============================================================

class TwoWaySmartLineBGD:
    """
    Two‑way BGD with birth mask and death at zone boundaries.
    - Birth: new tracks only accepted if first position is outside the growth zone.
    - Death: track removed when centre leaves the growth zone on the opposite side.
    - Direction is locked based on first position.
    """
    def __init__(self, base_tracker, line_x, growth_offset=70, birth_conf=0.35):
        self.tracker = base_tracker
        self.line_x = line_x
        self.growth_offset = growth_offset
        self.birth_conf = birth_conf
        self.known_tracks = set()
        self.track_history = defaultdict(list)
        self.locked_direction = {}
        self.pending_x = {}

    def update(self, detections_xyxy, scores, frame=None):
        if frame is None:
            return None
        img_w = frame.shape[1]
        left_bound  = self.line_x - self.growth_offset
        right_bound = self.line_x + self.growth_offset

        dets = np.column_stack([detections_xyxy, scores, np.zeros(len(scores), dtype=np.int32)])
        outputs = self.tracker.update(dets, frame)

        valid = []
        if outputs is not None and len(outputs) > 0:
            for out in outputs:
                x1, y1, x2, y2, tid, conf, cls, *rest = out
                cx = (x1 + x2) / 2.0
                self.track_history[tid].append(cx)
                if len(self.track_history[tid]) > 10:
                    self.track_history[tid] = self.track_history[tid][-10:]

                if tid not in self.known_tracks:
                    if left_bound <= cx <= right_bound:
                        continue
                    if cx < left_bound:
                        self.locked_direction[tid] = 'left'
                    else:
                        self.locked_direction[tid] = 'right'
                    self.known_tracks.add(tid)

                direction = self.locked_direction.get(tid, 'left')
                if direction == 'left' and cx > right_bound:
                    continue
                if direction == 'right' and cx < left_bound:
                    continue

                valid.append(out)

        if outputs is not None:
            for out in outputs:
                self.known_tracks.add(out[4])

        return valid if len(valid) > 0 else None

    def register_death(self, display_id, cx, cy):
        self.pending_x[display_id] = (cx, cy, 3)

    def draw_pending_x(self, frame):
        expired = []
        for disp_id, (cx, cy, remaining) in self.pending_x.items():
            cv2.circle(frame, (int(cx), int(cy)), 15, (0,0,255), 3)
            cv2.putText(frame, "X", (int(cx)-10, int(cy)+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if remaining <= 1:
                expired.append(disp_id)
            else:
                self.pending_x[disp_id] = (cx, cy, remaining-1)
        for disp_id in expired:
            del self.pending_x[disp_id]


class TVMFilter:
    """Trajectory Validation Module – keeps only smooth tracks."""
    def __init__(self, max_sigma=30.0):
        self.max_sigma = max_sigma
        self.history = defaultdict(list)

    def filter(self, tracked_boxes):
        valid = []
        for x1, y1, x2, y2, tid in tracked_boxes:
            tid = int(tid)
            cx = (x1+x2)/2; cy = (y1+y2)/2
            self.history[tid].append((cx, cy))
            if len(self.history[tid]) > 10:
                self.history[tid] = self.history[tid][-10:]
            if len(self.history[tid]) >= 5:
                pts = np.array(self.history[tid])
                dx = np.diff(pts[:,0]); dy = np.diff(pts[:,1])
                if np.std(dx) < self.max_sigma and np.std(dy) < self.max_sigma:
                    valid.append((x1, y1, x2, y2, tid))
        return valid


def count_crossings_filtered(pred_df, line_x, min_length=5):
    """CL‑ID: count only tracks that live at least min_length frames."""
    if pred_df is None or pred_df.empty:
        return 0
    df = pred_df.copy()
    df['cx'] = df['x'] + df['w']/2.0
    track_lengths = df.groupby('id').size()
    valid_ids = set(track_lengths[track_lengths >= min_length].index)
    df = df[df['id'].isin(valid_ids)]
    if df.empty:
        return 0
    counted = set()
    for tid, grp in df.groupby('id'):
        grp = grp.sort_values('frame')
        cxs = grp['cx'].values
        sides = cxs >= line_x
        if np.any(sides[1:] != sides[:-1]):
            counted.add(tid)
    return len(counted)


# ============================================================
# HELPERS
# ============================================================
COLOR_CACHE = {}
def get_color(track_id):
    if track_id not in COLOR_CACHE:
        rng = np.random.default_rng(track_id + 12345)
        COLOR_CACHE[track_id] = tuple(int(x) for x in rng.integers(50, 255, size=3))
    return COLOR_CACHE[track_id]

def draw_tracks(frame, tracks_xyxy_id, trajectories, unique_ids):
    for x1, y1, x2, y2, tid in tracks_xyxy_id:
        tid = int(tid)
        unique_ids.add(tid)
        color = get_color(tid)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID {tid}", (int(x1), max(0, int(y1)-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
        trajectories.setdefault(tid, []).append((cx, cy))
        if len(trajectories[tid]) > 30:
            trajectories[tid] = trajectories[tid][-30:]
        for i in range(1, len(trajectories[tid])):
            cv2.line(frame, trajectories[tid][i-1], trajectories[tid][i], color, 2)
    return frame

@st.cache_resource
def load_detector():
    if not WEIGHTS_PATH.exists():
        st.error(f"Weights not found at {WEIGHTS_PATH}")
        return None
    return YOLO(str(WEIGHTS_PATH))

def process_video(video_path, module_name, progress_bar):
    detector = load_detector()
    if detector is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("Cannot open video file.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_budget = total_frames if MAX_FRAMES is None else min(total_frames, MAX_FRAMES)

    line_x = w // 2

    # Base ByteTrack (CPU‑friendly)
    base_tracker = BYTETracker(
        track_thresh=0.35, track_buffer=120, match_thresh=0.9, frame_rate=int(fps)
    )

    # Apply module
    if module_name == "BGD":
        tracker = TwoWaySmartLineBGD(base_tracker, line_x=line_x, growth_offset=70, birth_conf=0.35)
    else:
        tracker = base_tracker

    tvm = TVMFilter(max_sigma=30.0) if module_name == "TVM" else None

    # Prediction storage for CL‑ID
    pred_rows = []

    out_path = TEMP_DIR / f"{module_name}_{Path(video_path).stem}.mp4"
    codec_list = [
        ('VP80', '.webm'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
        ('mp4v', '.mp4')
    ]
    writer = None
    for codec, ext in codec_list:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_path = TEMP_DIR / f"{module_name}_{Path(video_path).stem}{ext}"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            break

    if not writer or not writer.isOpened():
        cap.release()
        st.error("Cannot create output video writer.")
        return None

    unique_ids, trajectories = set(), {}
    cross_states = {}
    counted_ids = set()
    current_count = 0
    id_map = {}
    next_disp_id = 1

    # BGD zone boundaries (for visual lines)
    growth_offset = 70
    left_bound  = line_x - growth_offset
    right_bound = line_x + growth_offset

    for frame_idx in range(frame_budget):
        ok, frame = cap.read()
        if not ok:
            break

        # Detection (CPU, no half precision)
        results = detector.predict(frame, verbose=False, conf=DETECT_CONF,
                                   iou=DETECT_IOU, imgsz=IMGSZ, device='cpu',
                                   half=False, stream=False)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            boxes = np.empty((0,4), dtype=float)
            confs = np.empty((0,), dtype=float)
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(float)
            confs = results[0].boxes.conf.cpu().numpy().astype(float)

        # NMS
        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            confs_t = torch.tensor(confs, dtype=torch.float32)
            keep = torch.ops.torchvision.nms(boxes_t, confs_t, 0.4)
            boxes = boxes[keep.numpy()]
            confs = confs[keep.numpy()]

        # Tracker update
        if module_name == "BGD":
            outputs = tracker.update(boxes, confs, frame)
        else:
            dets = np.column_stack([boxes, confs, np.zeros((len(boxes),), dtype=np.int32)]) if len(boxes) else np.empty((0,6))
            outputs = tracker.update(dets, frame)

        tracked_boxes = []
        if outputs is not None and len(outputs) > 0:
            for out in outputs:
                x1, y1, x2, y2, tid, conf, cls, *rest = out
                tid = int(tid)
                if tid not in id_map:
                    id_map[tid] = next_disp_id
                    next_disp_id += 1
                disp_id = id_map[tid]
                tracked_boxes.append((x1, y1, x2, y2, disp_id))

                if module_name == "CL‑ID":
                    pred_rows.append([frame_idx+1, disp_id, x1, y1, x2-x1, y2-y1, conf])

                if module_name != "CL‑ID":
                    cx = (x1 + x2) / 2.0
                    curr_side = cx > line_x
                    if disp_id not in cross_states:
                        cross_states[disp_id] = curr_side
                    else:
                        prev_side = cross_states[disp_id]
                        if prev_side != curr_side and disp_id not in counted_ids:
                            counted_ids.add(disp_id)
                            current_count = len(counted_ids)
                        cross_states[disp_id] = curr_side

        # TVM filtering
        all_tracked = tracked_boxes  
        if tvm is not None:
            validated = tvm.filter(all_tracked)
            accepted = len(validated)
            rejected = len(all_tracked) - accepted

            for x1, y1, x2, y2, disp_id in validated:
                cx = (x1 + x2) / 2.0
                curr_side = cx > line_x
                if disp_id not in cross_states:
                    cross_states[disp_id] = curr_side
                else:
                    prev_side = cross_states[disp_id]
                    if prev_side != curr_side and disp_id not in counted_ids:
                        counted_ids.add(disp_id)
                        current_count = len(counted_ids)
                    cross_states[disp_id] = curr_side

            tracked_boxes = validated      
        else:
            accepted = rejected = 0

        # ---- Draw visual elements ----
        frame = draw_tracks(frame, tracked_boxes, trajectories, unique_ids)

        # BGD visual: three lines
        if module_name == "BGD":
            cv2.line(frame, (left_bound, 0), (left_bound, h), (0, 0, 0), 3)
            cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 3)
            cv2.line(frame, (right_bound, 0), (right_bound, h), (0, 0, 0), 3)
            cv2.putText(frame, "BIRTH / DEATH", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            cv2.putText(frame, "COUNT", (line_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            cv2.putText(frame, "BIRTH / DEATH", (right_bound + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            cv2.putText(frame, "GROWTH", (left_bound + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # CL‑ID visual: short track counter + thickness
        if module_name == "CL‑ID" and pred_rows:
            temp_df = pd.DataFrame(pred_rows, columns=["frame","id","x","y","w","h","conf"])
            track_lens = temp_df.groupby("id").size()
            all_ids = set(track_lens.index)
            valid_ids = set(track_lens[track_lens >= 5].index)
            ignored_short = len(all_ids - valid_ids)
            cv2.putText(frame, f"Short tracks ignored: {ignored_short}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            # Re-draw with thickness
            for x1, y1, x2, y2, disp_id in tracked_boxes:
                length = track_lens.get(disp_id, 0)
                color = get_color(disp_id)
                thick = 2 if length >= 5 else 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)

        # TVM visual: accepted / rejected counter + colour
        if tvm is not None:
            cv2.putText(frame, f"Accepted: {accepted} | Rejected: {rejected}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            # Draw original all_tracked with colour
            for x1, y1, x2, y2, disp_id in all_tracked:
                is_valid = disp_id in {v[4] for v in validated}
                color = (0,255,0) if is_valid else (0,0,255)
                thick = 2 if is_valid else 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)

        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 3)
        if module_name == "CL‑ID":
            if pred_rows:
                temp_df = pd.DataFrame(pred_rows, columns=["frame","id","x","y","w","h","conf"])
                current_count = count_crossings_filtered(temp_df, line_x, min_length=5)
            else:
                current_count = 0
        cv2.putText(frame, f"{module_name} | Cross Count: {current_count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(frame)
        progress_bar.progress((frame_idx + 1) / frame_budget)

    cap.release()
    writer.release()

    # CL‑ID final count
    if module_name == "CL‑ID":
        pred_df = pd.DataFrame(pred_rows, columns=["frame","id","x","y","w","h","conf"])
        current_count = count_crossings_filtered(pred_df, line_x, min_length=5)

    if not out_path.exists() or out_path.stat().st_size == 0:
        st.error("Output video is empty. Tracking may have failed.")
        return None

    return out_path, current_count


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Chicken Tracking & Counting Demo", layout="wide")
st.title("Chicken Tracking & Counting Demo")
st.markdown("Select a refinement module and upload a video to see tracking results.")

module_choice = st.selectbox("Refinement Module", ["BGD", "CL‑ID", "TVM"], index=0)
uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    temp_video = TEMP_DIR / f"input_{uploaded_file.name}"
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Video")
        with open(temp_video, "rb") as f:
            st.video(f.read())

    progress_bar = st.progress(0, text="Processing...")
    result = process_video(temp_video, module_choice, progress_bar)
    progress_bar.empty()

    if result is not None:
        output_path, final_count = result
        if output_path.exists():
            with col2:
                st.subheader(f"Tracked Video ({module_choice})")
                with open(output_path, "rb") as f:
                    st.video(f.read())
            st.metric("Chickens Crossed Line", final_count)
        else:
            with col2:
                st.error("Tracking failed. Output video not found.")
    else:
        with col2:
            st.error("Tracking failed. Please check that the weights file exists and the video is valid.")