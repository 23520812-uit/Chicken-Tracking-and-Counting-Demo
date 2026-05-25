"""
Chicken Tracking Demo – Streamlit Application
Refinement modules: BGD, CL‑ID, TVM
Simple UI: select module, upload video -> automatic tracking with side-by-side view.
"""
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

MAX_FRAMES = 600
DETECT_CONF = 0.12
DETECT_IOU  = 0.50
IMGSZ = 640

# ============================================================
# REFINEMENT MODULES
# ============================================================

class DirectionalBGD:
    """Bidirectional Birth‑Growth‑Death tracker wrapper."""
    def __init__(self, base_tracker, birth_conf=0.35, death_margin=70):
        self.tracker = base_tracker
        self.birth_conf = birth_conf
        self.death_margin = death_margin
        self.img_w = None
        self.track_history = defaultdict(list)
        self.locked_direction = {}

    def update(self, detections_xyxy, scores, frame=None):
        if frame is not None and self.img_w is None:
            self.img_w = frame.shape[1]

        N = len(detections_xyxy)
        birth_mask = scores >= self.birth_conf
        modified_scores = scores.copy()
        modified_scores[~birth_mask] = 0.0

        dets = np.column_stack([detections_xyxy, modified_scores, np.zeros(N, dtype=np.int32)])
        outputs = self.tracker.update(dets, frame)

        if outputs is not None and len(outputs) > 0:
            valid = []
            for out in outputs:
                x1, y1, x2, y2, tid, conf, cls, *rest = out
                cx = (x1+x2)/2
                self.track_history[tid].append(cx)
                if len(self.track_history[tid]) > 10:
                    self.track_history[tid] = self.track_history[tid][-10:]

                direction = self._get_direction(tid)
                if direction == 'left' and cx > self.img_w - self.death_margin:
                    continue
                if direction == 'right' and cx < self.death_margin:
                    continue
                valid.append(out)
            return valid
        return outputs

    def _get_direction(self, tid):
        if tid in self.locked_direction:
            return self.locked_direction[tid]
        hist = self.track_history[tid]
        if len(hist) < 2:
            return 'left' if hist[0] < self.img_w/2 else 'right'
        if len(hist) >= 15:
            moves = np.diff(hist[-10:])
            rightward = np.sum(moves > 0)
            leftward = len(moves) - rightward
            if rightward > leftward:
                self.locked_direction[tid] = 'left'
            else:
                self.locked_direction[tid] = 'right'
            return self.locked_direction[tid]
        first, last = hist[0], hist[-1]
        return 'left' if last > first else 'right'


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
    """CL‑ID: count only tracks that live long enough."""
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
    frame_budget = min(total_frames, MAX_FRAMES) if total_frames > 0 else MAX_FRAMES

    line_x = w // 2

    # Base ByteTrack tracker (same for all modules)
    base_tracker = BYTETracker(
        track_thresh=0.35, track_buffer=120, match_thresh=0.9, frame_rate=int(fps)
    )

    # Apply the selected refinement module
    if module_name == "BGD":
        tracker = DirectionalBGD(base_tracker, birth_conf=0.35, death_margin=70)
    else:
        tracker = base_tracker  # for CL‑ID and TVM we use vanilla ByteTrack + post‑processing

    # For TVM we need a filter instance
    tvm = TVMFilter(max_sigma=30.0) if module_name == "TVM" else None

    # Store predictions for CL‑ID (which does post‑processing)
    pred_rows = []

    # Video writer (try browser-friendly codecs)
    out_path = TEMP_DIR / f"{module_name}_{Path(video_path).stem}.mp4"
    writer = None
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
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

    for frame_idx in range(frame_budget):
        ok, frame = cap.read()
        if not ok:
            break

        # Detection
        results = detector.predict(frame, verbose=False, conf=DETECT_CONF,
                                   iou=DETECT_IOU, imgsz=IMGSZ, device=0,
                                   half=torch.cuda.is_available(), stream=False)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            boxes = np.empty((0,4), dtype=float)
            confs = np.empty((0,), dtype=float)
        else:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy().astype(float)
            confs = results[0].boxes.conf.detach().cpu().numpy().astype(float)

        # NMS
        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            confs_t = torch.tensor(confs, dtype=torch.float32)
            keep = torch.ops.torchvision.nms(boxes_t, confs_t, 0.4)
            boxes = boxes[keep.numpy()]
            confs = confs[keep.numpy()]

        # Run tracker (with or without BGD wrapper)
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
                # Map internal ID to display ID
                if tid not in id_map:
                    id_map[tid] = next_disp_id
                    next_disp_id += 1
                disp_id = id_map[tid]
                tracked_boxes.append((x1, y1, x2, y2, disp_id))

                # Store for CL‑ID
                if module_name == "CL‑ID":
                    pred_rows.append([frame_idx+1, disp_id, x1, y1, x2-x1, y2-y1, conf])

                # Bidirectional counting (live, for BGD and TVM)
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

        # TVM filtering (post‑tracker)
        if tvm is not None:
            tracked_boxes = tvm.filter(tracked_boxes)
            # Update counting after filtering
            current_count = 0
            counted_ids.clear()
            cross_states.clear()
            for x1, y1, x2, y2, disp_id in tracked_boxes:
                cx = (x1+x2)/2.0
                curr_side = cx > line_x
                if disp_id not in cross_states:
                    cross_states[disp_id] = curr_side
                else:
                    prev_side = cross_states[disp_id]
                    if prev_side != curr_side and disp_id not in counted_ids:
                        counted_ids.add(disp_id)
                        current_count = len(counted_ids)
                    cross_states[disp_id] = curr_side

        frame = draw_tracks(frame, tracked_boxes, trajectories, unique_ids)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 3)
        cv2.putText(frame, f"{module_name} | Cross Count: {current_count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(frame)
        progress_bar.progress((frame_idx + 1) / frame_budget)

    cap.release()
    writer.release()

    # CL‑ID final count (from stored predictions)
    if module_name == "CL‑ID":
        pred_df = pd.DataFrame(pred_rows, columns=["frame","id","x","y","w","h","conf"])
        current_count = count_crossings_filtered(pred_df, line_x, min_length=5)

    # Overwrite the count on the video? Not needed here, just return the final count
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
    # Save input video to temp
    temp_video = TEMP_DIR / f"input_{uploaded_file.name}"
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.read())

    # Show original video on the left
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Video")
        with open(temp_video, "rb") as f:
            st.video(f.read())

    # Process automatically
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