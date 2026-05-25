# Chicken Tracking & Counting Demo

A Streamlit application for demonstrating chicken counting using YOLOv8n and three tracking refinement modules: BGD, CL‑ID, and TVM.

## Features

- Upload your own video (MP4, AVI, MOV, MKV)
- Choose a refinement module: **BGD** (Birth‑Growth‑Death), **CL‑ID** (Counting Line with ID Filtering), or **TVM** (Trajectory Validation Module)
- Side‑by‑side display of the original and tracked video
- Live chicken count overlaid on the processed video
- No download required – results are streamed directly in the browser

## Refinement Modules

| Module | Mechanism | Purpose |
|--------|-----------|---------|
| **BGD** | Spatial birth‑death zones | Prevents track creation in unwanted areas and removes tracks when they reach the opposite side, reducing ID switches. |
| **CL‑ID** | Temporal filter (min track length) | Counts only tracks that persist for at least 5 frames, suppressing noise and short‑lived false tracks. |
| **TVM** | Motion consistency check (σ ≤ 30) | Keeps only tracks with smooth trajectories, eliminating erratic ghost tracks. |

All modules use **ByteTrack** as the underlying tracker with the same tuned configuration (`track_thresh=0.35`, `track_buffer=120`, `match_thresh=0.9`).

## Installation

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
