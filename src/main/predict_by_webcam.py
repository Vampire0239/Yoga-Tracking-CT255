# realtime_pose_delay.py
# -*- coding: utf-8 -*-
"""
Realtime yoga pose (webcam/video) với delay smoothing:
- YOLO pose -> keypoints (x,y) theo COCO-17 (pixel, KHÔNG normalize 0-1)
- Làm sạch giống train: interpolate -> bfill/ffill
  (Tránh rolling center vì không causal; thay bằng centered average qua DELAY)
- Chuẩn hoá bằng StandardScaler (scaler.pkl)
- Dự đoán bằng best_model.keras (Conv1D+BiGRU)
- Gán xác suất của mỗi cửa sổ vào frame "giữa" cửa sổ;
  hiển thị nhãn trễ DELAY khung bằng trung bình centered.
- Không lưu file; chỉ hiển thị cửa sổ. Bấm 'q' để thoát.
"""
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from collections import deque
from joblib import load
from ultralytics import YOLO
import tensorflow as tf

# ========== ARGS ==========
p = argparse.ArgumentParser()
src = p.add_mutually_exclusive_group(required=True)
src.add_argument("--cam", type=int, help="Chỉ số webcam (0,1,...)")
src.add_argument("--video", type=str, help="Đường dẫn file video để thử như webcam")
p.add_argument("--out", type=str, required=True, help="Thư mục chứa: best_model.keras, scaler.pkl, classes.txt, feature_cols.txt")
p.add_argument("--yolo", type=str, default="yolov8n-pose.pt", help="Đường dẫn YOLO pose model")
p.add_argument("--imgsz", type=int, default=384, help="YOLO imgsz (nhỏ = nhanh hơn)")
p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
p.add_argument("--seq_len", type=int, default=30, help="Độ dài cửa sổ trượt (SEQ_LEN)")
p.add_argument("--delay", type=int, default=10, help="Độ trễ (số khung); hiển thị nhãn cho frame_id-DELAY")
p.add_argument("--yolo_stride", type=int, default=1, help="Chạy YOLO mỗi N khung (1= mỗi khung; 2 = mỗi 2 khung)")
p.add_argument("--show_fps", action="store_true", help="Hiển thị FPS")
p.add_argument("--mirror", action="store_true", help="Lật gương khi dùng webcam")
p.add_argument("--backend", type=str, default="auto", choices=["auto","dshow","msmf","v4l2"], help="Backend mở webcam (Windows: dshow/msmf)")
args = p.parse_args()

# ========== LOAD CLASSIFIER & ARTIFACTS ==========
OUT = args.out
MODEL_PATH    = os.path.join(OUT, "best_model.keras")
SCALER_PATH   = os.path.join(OUT, "scaler.pkl")
CLASSES_PATH  = os.path.join(OUT, "classes.txt")
FEATCOLS_PATH = os.path.join(OUT, "feature_cols.txt")

print("Loading classifier:", MODEL_PATH)
clf = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = load(SCALER_PATH)
classes = [l.strip() for l in open(CLASSES_PATH, encoding="utf-8")]
feat_cols = [l.strip() for l in open(FEATCOLS_PATH, encoding="utf-8")]
print(f"✅ {len(classes)} lớp | {len(feat_cols)} feature cols")

# ========== LOAD YOLO ==========
print("Loading YOLO pose:", args.yolo)
yolo = YOLO(args.yolo)

# ========== COCO-17 ==========
kpt_names = [
    "nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
    "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
    "l_knee","r_knee","l_ankle","r_ankle"
]
pairs = [
    ("nose","l_eye"),("nose","r_eye"),
    ("l_eye","l_ear"),("r_eye","r_ear"),
    ("l_shoulder","r_shoulder"),
    ("l_shoulder","l_elbow"),("r_shoulder","r_elbow"),
    ("l_elbow","l_wrist"),("r_elbow","r_wrist"),
    ("l_shoulder","l_hip"),("r_shoulder","r_hip"),
    ("l_hip","r_hip"),
    ("l_hip","l_knee"),("r_hip","r_knee"),
    ("l_knee","l_ankle"),("r_knee","r_ankle")
]

def color_for_label(label: str):
    import hashlib
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
    return (h % 255, (h//255) % 255, (h//65025) % 255)

# ========== CAMERA/VIDEO OPEN HELPERS ==========
def open_camera(idx: int, backend: str):
    if backend == "dshow":
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    elif backend == "msmf":
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    elif backend == "v4l2":
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    else:  # auto: thử DSHOW -> MSMF -> default
        for be in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                # ép MJPG + 640x480 để ổn định
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # warmup
                for _ in range(5):
                    cap.read()
                return cap
            cap.release()
        return cv2.VideoCapture(idx)  # fallback
    # cấu hình chung
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            cap.read()
    return cap

if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = open_camera(args.cam, args.backend)

assert cap.isOpened(), "❌ Không mở được nguồn hình ảnh"
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
print("Opened:", True, "| W,H:", W, H, "| FPS(reported):", cap.get(cv2.CAP_PROP_FPS))

# ========== FEATURE ROW BUILDER ==========
def make_feature_row_from_xy(xy17):
    """
    xy17: ndarray shape (17,2) pixel coords (KHÔNG normalize 0-1)
    Trả về dict {feat_col: value} theo đúng thứ tự feature_cols.
    """
    row_base = {}
    for i, name in enumerate(kpt_names):
        if xy17 is None or np.any(np.isnan(xy17[i])):
            x, y = np.nan, np.nan
        else:
            x, y = float(xy17[i,0]), float(xy17[i,1])
        row_base[f"{name}_x"] = x
        row_base[f"{name}_y"] = y
    return {c: row_base.get(c, np.nan) for c in feat_cols}

# ========== CLEAN LIKE TRAIN (causal-friendly) ==========
def clean_like_train(df: pd.DataFrame, feat_cols):
    # 1) interpolate 2 chiều
    df[feat_cols] = df[feat_cols].interpolate(limit_direction="both")
    # 2) ffill/bfill một phía
    df[feat_cols] = df[feat_cols].fillna(method="bfill").fillna(method="ffill")
    # 3) nếu còn NaN trường hợp cực đoan: fill median cột
    if df[feat_cols].isna().sum().sum() > 0:
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    return df

# ========== PREDICT WINDOW ==========
def predict_window(buf_df: pd.DataFrame):
    df_clean = clean_like_train(buf_df.copy(), feat_cols)
    X = scaler.transform(df_clean[feat_cols].values.astype(np.float32))  # (SEQ_LEN, n_feats)
    X = X[np.newaxis, ...]                                              # (1, SEQ_LEN, n_feats)
    probs = clf.predict(X, verbose=0)[0]                                # (n_classes,)
    return probs

# ========== MAIN LOOP WITH DELAY SMOOTHING ==========
SEQ_LEN = args.seq_len
DELAY   = args.delay
yolo_stride = max(1, args.yolo_stride)

buf = deque(maxlen=SEQ_LEN)              # chứa dict feature theo frame (giữ mới nhất)
probs_buf = []                           # list xác suất theo frame (None hoặc np.array(n_classes,))
last_label, last_conf = "...", 0.0
last_xy17 = None
t0, frames, frame_id = time.time(), 0, 0

def draw_skeleton(frame, xy17):
    if xy17 is None: return
    # edges
    for (a,b) in pairs:
        ia, ib = kpt_names.index(a), kpt_names.index(b)
        xa, ya = xy17[ia]; xb, yb = xy17[ib]
        cv2.line(frame, (int(xa),int(ya)), (int(xb),int(yb)), (0,255,0), 2)
    # nodes
    for i in range(17):
        x, y = int(xy17[i,0]), int(xy17[i,1])
        cv2.circle(frame, (x,y), 3, (0,255,255), -1)

while True:
    ok, frame = cap.read()
    if not ok:
        # thử đọc lại vài lần thay vì thoát ngay
        retry = False
        for _ in range(3):
            ok, frame = cap.read()
            if ok:
                retry = True
                break
        if not retry:
            break

    if args.mirror and args.cam is not None:
        frame = cv2.flip(frame, 1)

    # đảm bảo probs_buf có ô trống cho frame hiện tại
    probs_buf.append(None)

    # ---- YOLO mỗi yolo_stride khung ----
    run_yolo = (frame_id % yolo_stride == 0)
    xy17 = None
    if run_yolo:
        res = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if res.boxes is not None and len(res.boxes)>0 and res.keypoints is not None:
            best_i = int(res.boxes.conf.argmax().item())
            xy = res.keypoints.xy[best_i].cpu().numpy()  # (17,2) pixel
            xy17 = xy
            last_xy17 = xy17
        else:
            xy17 = last_xy17
    else:
        xy17 = last_xy17

    # ---- push vào buffer (dù có/không có kp) ----
    row = make_feature_row_from_xy(xy17)
    buf.append(row)

    # ---- khi đủ SEQ_LEN: predict và gán prob vào frame giữa cửa sổ ----
    if len(buf) == SEQ_LEN:
        win_df = pd.DataFrame(list(buf), columns=feat_cols)
        probs = predict_window(win_df)  # (n_classes,)

        # center frame index của cửa sổ hiện tại
        center_offset = SEQ_LEN // 2
        center_idx = frame_id - (SEQ_LEN - 1) + center_offset
        if 0 <= center_idx < len(probs_buf):
            if probs_buf[center_idx] is None:
                probs_buf[center_idx] = probs.astype(np.float32)
            else:
                # trộn nhẹ nếu đã có (tuỳ chọn)
                probs_buf[center_idx] = 0.5*probs_buf[center_idx] + 0.5*probs.astype(np.float32)

    # ---- chốt nhãn với độ trễ DELAY (centered averaging) ----
    d = frame_id - DELAY
    if d >= 0:
        L = max(0, d - DELAY)
        R = min(len(probs_buf) - 1, d + DELAY)
        chunk = [p for p in probs_buf[L:R+1] if p is not None]
        if chunk:
            avg_prob = np.mean(chunk, axis=0)
            cls_id = int(np.argmax(avg_prob))
            last_label = classes[cls_id]
            last_conf  = float(avg_prob[cls_id])

    # ---- vẽ skeleton + overlay ----
    draw_skeleton(frame, xy17)
    color = color_for_label(last_label)
    text  = f"{last_label}  ({last_conf*100:.1f}%)" if last_conf>0 else last_label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x0, y0 = 10, 30
    cv2.rectangle(frame, (x0-5, y0-th-6), (x0+tw+5, y0+6), (0,0,0), -1)
    cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # FPS (tuỳ chọn)
    frames += 1
    if args.show_fps:
        dt = time.time() - t0
        if dt > 0:
            fps = frames / dt
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y0+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

    cv2.imshow("Yoga Pose - Realtime (DELAY smoothing)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
print("✅ Realtime stopped (no files saved).")
