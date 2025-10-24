# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import glob

# ===== Cấu hình =====
VIDEO_PATH = "/home/B2205945-CT255/ProJect_yolo/data/raw/dataset/test/Trikasana/video 12.mp4"
MODEL_PATH = "yolov8n-pose.pt"  # hoặc yolov8m-pose.pt
OUT_DIR = Path("/home/B2205945-CT255/ProJect_yolo/data/test_csv")
OUT_CSV_DIR = OUT_DIR / "csv"

# ===== Load model =====
model = YOLO(MODEL_PATH)

video_list = glob.glob(VIDEO_PATH, recursive=True)
assert video_list, f"Không tìm thấy video: {VIDEO_PATH}"

kpt_names = [
    "nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
    "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
    "l_knee","r_knee","l_ankle","r_ankle"
]
cols_xy = [f"{n}_{c}" for n in kpt_names for c in ["x","y"]]
columns = ["frame","time_s"] + cols_xy

NORMALIZE_01 = False  # giữ nguyên pixel

for vp in video_list:
    cap = cv2.VideoCapture(str(vp))
    assert cap.isOpened(), f"Không mở được video: {vp}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    path = Path(vp)
    # Giữ nguyên cấu trúc gốc (train/.../video.mp4 → csv/train/.../video.csv)
    rel_path = path.relative_to("/home/B2205945-CT255/ProJect_yolo/data/raw/dataset")
    CSV_PATH = OUT_CSV_DIR / rel_path.with_suffix(".csv")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)  # tạo thư mục trước

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_sec = frame_idx / fps

        results = model.predict(frame, verbose=False)
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
            best_i = int(res.boxes.conf.argmax().item())
            xy = res.keypoints.xy[best_i].cpu().numpy()  # (17,2)

            if NORMALIZE_01 and width > 0 and height > 0:
                xy_norm = xy.copy()
                xy_norm[:, 0] = xy[:, 0] / width
                xy_norm[:, 1] = xy[:, 1] / height
                flat_xy = xy_norm.flatten().tolist()
            else:
                flat_xy = xy.flatten().tolist()

            row = [frame_idx, t_sec] + flat_xy
            rows.append(row)
        else:
            row = [frame_idx, t_sec] + [np.nan] * len(cols_xy)
            rows.append(row)

    cap.release()

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"✅ Saved CSV to {CSV_PATH}")
