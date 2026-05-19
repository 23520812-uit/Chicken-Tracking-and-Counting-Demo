"""
Chicken Tracking Demo – Streamlit Application
Simple UI: select tracker, upload video -> automatic tracking with side-by-side view.
No download – results are streamed directly.
"""
import os
# Lệnh này sẽ chạy ngầm để xóa bản OpenCV lỗi do Ultralytics tự tải về, 
# ép hệ thống phải dùng bản opencv-python-headless ở trong requirements.txt
os.system("pip uninstall -y opencv-python opencv-contrib-python")

import streamlit as st
import cv2
import numpy as np
import tempfile
import sys
from pathlib import Path
from collections import defaultdict
import torch

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
from boxmot.trackers import OcSort as OCSORT
from boxmot.trackers import BotSort as BoTSORT

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

def process_video(video_path, tracker_name, progress_bar):
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

    # ---------- Tuned tracker configurations ----------
    if tracker_name == "ByteTrack":
        tracker = BYTETracker(
            track_thresh=0.35,
            track_buffer=120,
            match_thresh=0.9,
            frame_rate=int(fps)
        )
    elif tracker_name == "OC-SORT":
        tracker = OCSORT(
            det_thresh=0.35,
            iou_threshold=0.3,
            max_age=50,
            min_hits=3,
            max_obs=200          # avoids warning
        )
    else:  # BoT-SORT
        tracker = BoTSORT(
            track_high_thresh=0.35,
            track_low_thresh=0.1,
            new_track_thresh=0.5,
            match_thresh=0.7,
            track_buffer=60,
            min_hits=3,
            max_age=30,
            with_reid=False,
            frame_rate=int(fps)
        )

    # Video writer (try browser-friendly codecs)
    out_path = TEMP_DIR / f"{tracker_name}_{Path(video_path).stem}.mp4"
    writer = None
    for codec in ['VP80', 'avc1', 'mp4v', 'XVID', 'MJPG']:
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

    for frame_idx in range(frame_budget):
        ok, frame = cap.read()
        if not ok:
            break
        
        DEVICE = "0" if torch.cuda.is_available() else "cpu"
        
        # Detection
        results = detector.predict(frame, verbose=False, conf=DETECT_CONF,
                                   iou=DETECT_IOU, imgsz=IMGSZ, device=DEVICE,
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

        if len(boxes) > 0:
            dets = np.column_stack([boxes, confs, np.zeros((len(boxes),), dtype=np.int32)])
        else:
            dets = np.empty((0, 6))

        outputs = tracker.update(dets, frame)
        tracked_boxes = []
        if outputs is not None and len(outputs) > 0:
            for out in outputs:
                x1, y1, x2, y2, tid, conf, cls, *rest = out
                tid = int(tid)
                tracked_boxes.append((x1, y1, x2, y2, tid))

                # Bidirectional counting
                cx = (x1 + x2) / 2.0
                curr_side = cx > line_x
                if tid not in cross_states:
                    cross_states[tid] = curr_side
                else:
                    prev_side = cross_states[tid]
                    if prev_side != curr_side and tid not in counted_ids:
                        counted_ids.add(tid)
                        current_count = len(counted_ids)
                    cross_states[tid] = curr_side

        frame = draw_tracks(frame, tracked_boxes, trajectories, unique_ids)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 3)
        cv2.putText(frame, f"{tracker_name} | Cross Count: {current_count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(frame)
        progress_bar.progress((frame_idx + 1) / frame_budget)

    cap.release()
    writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        st.error("Output video is empty. Tracking may have failed.")
        return None

    return out_path


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Chicken Tracking Demo", layout="wide")
st.title("Chicken Tracking Demo")
st.markdown("Select a tracker and upload a video to see tracking results.")

tracker_choice = st.selectbox("Tracker", ["ByteTrack", "OC-SORT", "BoT-SORT"], index=0)
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
    output_path = process_video(temp_video, tracker_choice, progress_bar)
    progress_bar.empty()

    if output_path and output_path.exists():
        with col2:
            st.subheader(f"Tracked Video ({tracker_choice})")
            with open(output_path, "rb") as f:
                st.video(f.read())
    else:
        with col2:
            st.error("Tracking failed. Please check that the weights file exists and the video is valid.")