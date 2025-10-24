# -*- coding: utf-8 -*-
# =========================================================
# Dự đoán tư thế yoga từ CSV keypoints (không có _conf)
# - Đọc thứ tự cột từ out/feature_cols.txt để khớp scaler/model
# - Làm sạch nhẹ NaN: interpolate + ffill/bfill (theo cả file)
# - Dự đoán theo cửa sổ, rồi soft-vote xác suất ra nhãn từng frame
# =========================================================

import os
import numpy as np
import pandas as pd
from collections import Counter
from joblib import load
import tensorflow as tf

# -------------------- CẤU HÌNH --------------------
OUT_DIR     = "out"
MODEL_PATH  = os.path.join(OUT_DIR, "best_model.keras")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
CLASSES_PATH = os.path.join(OUT_DIR, "classes.txt")
FEATCOLS_PATH = os.path.join(OUT_DIR, "feature_cols.txt")

CSV_PATH    = "/home/B2205945-CT255/ProJect_yolo/data/test_csv/csv/test/Trikasana/video 12.csv"
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
df = pd.read_csv(CSV_PATH)
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

# -------------------- LƯU KẾT QUẢ --------------------
df_out = df.copy()
df_out["pred_label"] = pred_labels_frame
df_out["pred_conf"]  = pred_conf_frame.astype(np.float32)
out_csv = os.path.join(OUT_DIR, "predict_output.csv")
df_out.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\n✅ Đã lưu kết quả → {out_csv}")


