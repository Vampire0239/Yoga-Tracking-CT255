# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from collections import Counter
from joblib import load
import tensorflow as tf



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True,
                    help="Path to input video file")
parser.add_argument("--out", type=str, default=64, help="out")
args = parser.parse_args()


# ===== Cấu hình =====
VIDEO_PATH = args.video_path
MODEL_PATH = "yolov8n-pose.pt"  # hoặc yolov8m-pose.pt


# ===== Load model =====
model = YOLO(MODEL_PATH)

kpt_names = [
    "nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
    "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
    "l_knee","r_knee","l_ankle","r_ankle"
]
cols_xy = [f"{n}_{c}" for n in kpt_names for c in ["x","y"]]
columns = ["frame","time_s"] + cols_xy

cap = cv2.VideoCapture(str(VIDEO_PATH))
assert cap.isOpened(), f"Không mở được video: {VIDEO_PATH}"
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)


rows = []
frame_idx = 0
NORMALIZE_01 = False  # giữ nguyên pixel

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


OUT_DIR     = args.out
MODEL_PATH  = os.path.join(OUT_DIR, "best_model.keras")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
CLASSES_PATH = os.path.join(OUT_DIR, "classes.txt")
FEATCOLS_PATH = os.path.join(OUT_DIR, "feature_cols.txt")

SEQ_LEN     = 30
STEP        = 1

# -------------------- LOAD --------------------
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = load(SCALER_PATH)
classes = [line.strip() for line in open(CLASSES_PATH, encoding="utf-8").read().splitlines()]
feat_cols = [line.strip() for line in open(FEATCOLS_PATH, encoding="utf-8").read().splitlines()]
n_classes = len(classes)
print(f"✅ Loaded {n_classes} classes:", classes)
print(f"✅ Using {len(feat_cols)} feature columns from feature_cols.txt")

# -------------------- ĐỌC CSV --------------------
print("CSV shape:", df.shape)
# Sắp xếp theo thời gian nếu có
if "frame" in df.columns:
    df = df.sort_values("frame").reset_index(drop=True)
elif "time_s" in df.columns:
    df = df.sort_values("time_s").reset_index(drop=True)

# Kiểm tra cột đủ/thiếu
missing = [c for c in feat_cols if c not in df.columns]
extra   = [c for c in df.columns if c not in set(feat_cols) | {"label","video_path","frame","time_s"}]
if missing:
    raise ValueError(f"❌ CSV thiếu {len(missing)} cột feature so với lúc train.\nThiếu: {missing[:10]}{'...' if len(missing)>10 else ''}")
if extra:
    print(f"ℹ️ CSV có {len(extra)} cột dư (bỏ qua khi predict): {extra[:8]}{'...' if len(extra)>8 else ''}")

# -------------------- LÀM SẠCH NHẸ (như train) --------------------
# Nội suy + ffill/bfill để tránh NaN khi transform
df[feat_cols] = df[feat_cols].interpolate(limit_direction="both")
df[feat_cols] = df[feat_cols].fillna(method="bfill").fillna(method="ffill")
# Nếu vẫn còn NaN (video quá nhiều khung không detect), thay bằng median cột
if df[feat_cols].isna().sum().sum() > 0:
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

# -------------------- CHUẨN HOÁ --------------------
X = scaler.transform(df[feat_cols].values.astype(np.float32))
T = len(X)
if T < SEQ_LEN:
    raise ValueError(f"❌ Số frame ({T}) < SEQ_LEN ({SEQ_LEN}). Cần video/CSV dài hơn hoặc giảm SEQ_LEN.")

def make_windows(X, seq_len=30, step=1):
    out = []
    idx = []
    for i in range(0, len(X) - seq_len + 1, step):
        out.append(X[i:i+seq_len])
        idx.append(i)
    return np.array(out, dtype=np.float32), np.array(idx, dtype=np.int32)

X_seq, starts = make_windows(X, SEQ_LEN, STEP)
print("X_seq:", X_seq.shape)

# -------------------- DỰ ĐOÁN --------------------
probs = model.predict(X_seq, verbose=0)  # (num_windows, n_classes)
# Soft-vote per-frame: cộng dồn xác suất của các cửa sổ bao phủ frame đó
frame_probs = np.zeros((T, n_classes), dtype=np.float32)
frame_counts = np.zeros((T, 1), dtype=np.float32)

for w, s in enumerate(starts):
    e = s + SEQ_LEN
    frame_probs[s:e] += probs[w]
    frame_counts[s:e] += 1.0

# Tránh chia 0
mask = frame_counts.squeeze() > 0
frame_probs[mask] /= frame_counts[mask]
# Các frame đầu/cuối ngoài vùng trượt (nếu có) dùng nearest fill
if not np.all(mask):
    # fill theo nearest non-zero window
    last = None
    for t in range(T):
        if mask[t]:
            last = frame_probs[t]
        elif last is not None:
            frame_probs[t] = last
    last = None
    for t in range(T-1, -1, -1):
        if mask[t]:
            last = frame_probs[t]
        elif last is not None:
            frame_probs[t] = last

pred_ids_frame = np.argmax(frame_probs, axis=1)
pred_labels_frame = [classes[i] for i in pred_ids_frame]
pred_conf_frame = frame_probs[np.arange(T), pred_ids_frame]

# -------------------- THỐNG KÊ --------------------
counts = Counter(pred_labels_frame)
print("\n===== KẾT QUẢ DỰ ĐOÁN (per-frame) =====")
for k, v in counts.most_common():
    print(f"{k:<15}: {v:5d} ({v/len(pred_labels_frame)*100:.1f}%)")
top = counts.most_common(1)[0][0]
print("\n🧘‍♂️ Tư thế dự đoán chính:", top)


# ==================== HIỂN THỊ & GHI VIDEO KẾT QUẢ ====================
# (Đặt sau khi đã có pred_labels_frame, pred_conf_frame và df chứa keypoints)

# 1) Định nghĩa kết nối xương (COCO-17) và màu
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

# Map tên -> index cột trong df
name_to_idx = {name:i for i, name in enumerate(kpt_names)}
def get_xy(row, name):
    x = row[f"{name}_x"]; y = row[f"{name}_y"]
    return float(x), float(y)

# 2) Chuẩn bị VideoWriter
cap2 = cv2.VideoCapture(str(VIDEO_PATH))
assert cap2.isOpened(), f"Không mở được video: {VIDEO_PATH}"
fps2   = cap2.get(cv2.CAP_PROP_FPS) or fps or 30.0
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)  or width)
height2= int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)

# Tên file xuất
out_path = str(Path(OUT_DIR) / "prediction_overlay2.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps2, (width2, height2))

# 3) Tuỳ chọn hiển thị lên màn hình (True/False)
SHOW_WINDOW = True  # đặt False nếu chỉ muốn ghi file

# 4) Nếu bạn đã normalize toạ độ 0–1 khi tạo df, set cờ này = True để tự scale về pixel
WAS_NORMALIZED_01 = NORMALIZE_01  # dùng lại cờ bạn đã đặt ở trên

# 5) Bảng màu đơn giản cho label (ổn định theo lớp)
def color_for_label(label):
    import hashlib
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
    return (h % 255, (h//255) % 255, (h//65025) % 255)

frame_id = 0
while True:
    ok, frame0 = cap2.read()
    if not ok:
        break
    if frame_id >= len(pred_labels_frame):
        # đề phòng length lệch do cắt cửa sổ
        writer.write(frame0)
        frame_id += 1
        continue

    row = df.iloc[frame_id]
    frame = frame0.copy()

    # 6) Vẽ skeleton từ df
    # Lấy toạ độ từng keypoint; nếu normalized thì scale lại
    pts = {}
    for name in kpt_names:
        x, y = row[f"{name}_x"], row[f"{name}_y"]
        if np.isnan(x) or np.isnan(y):
            continue
        if WAS_NORMALIZED_01:
            x = x * width2
            y = y * height2
        pts[name] = (int(round(x)), int(round(y)))

    # Vẽ nodes
    for name, (x, y) in pts.items():
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    # Vẽ edges
    for a, b in pairs:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    # 7) Ghi nhãn dự đoán + độ tự tin
    label = pred_labels_frame[frame_id]
    conf  = float(pred_conf_frame[frame_id])
    color = color_for_label(label)
    text  = f"{label}  ({conf*100:.1f}%)"

    # Vẽ hộp nền text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x0, y0 = 10, 30
    cv2.rectangle(frame, (x0-5, y0-th-6), (x0+tw+5, y0+6), (0,0,0), -1)
    cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # (tuỳ chọn) thêm thông tin frame/time
    t_sec = row["time_s"] if "time_s" in row else frame_id / fps2
    info = f"Frame: {frame_id+1} / Time: {t_sec:.2f}s"
    cv2.putText(frame, info, (10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # 8) Ghi ra video
    writer.write(frame)

    # 9) Show trực tiếp
    if SHOW_WINDOW:
        cv2.imshow("Yoga Pose Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_id += 1

cap2.release()
writer.release()
cv2.destroyAllWindows()
print(f"✅ Đã ghi video có overlay: {out_path}")




