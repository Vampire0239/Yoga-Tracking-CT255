# -*- coding: utf-8 -*-
# =========================================================
# Train Conv1D + BiGRU cho chuỗi keypoints (yoga poses)
# - Dữ liệu mới: chỉ (x,y) + video_path,label (không có conf)
# - Làm sạch NaN theo từng video (interpolate + ffill/bfill + optional smoothing)
# - Khoá thứ tự cột feature; lưu feature_cols.txt để predict khớp
# =========================================================

# pip install numpy pandas scikit-learn "tensorflow==2.*" tqdm joblib

import os, glob
import numpy as np
import pandas as pd
from collections import Counter
from joblib import dump
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ===== CẤU HÌNH =====
CSV_DIR   = "data/csv"
OUT_DIR   = "out"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN         = 30
TRAIN_STEP      = 5
VAL_TEST_STEP   = 1
VAL_SIZE        = 0.15
TEST_SIZE       = 0.15
RANDOM_SEED     = 42
MAJORITY_RATIO_MIN = 0.8

# Regularization/kiến trúc
L2W = 1e-3
GAUSS_NOISE_STD = 0.02
SPATIAL_DROPOUT = 0.30
GRU_UNITS = 96
GRU_REC_DROPOUT = 0.25
DENSE_UNITS = 96

USE_CLASS_WEIGHT = False
BATCH_SIZE = 64
EPOCHS = 100

META_COLS = {"label", "video_path", "frame", "time_s"}

# ===== ENV INFO =====
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
print("TensorFlow:", tf.__version__)
print("Using tf.keras only ✅")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        print("GPU(s):", gpus)
    else:
        print("GPU: None (CPU mode)")
except Exception as e:
    print("GPU setup note:", e)

def set_global_seed(seed=42):
    import random
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
set_global_seed(RANDOM_SEED)

# ===== KEYPOINT ORDER (COCO-17) =====
COCO17_ORDER = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# ===== ĐỌC CSV =====
def load_all_csv(csv_dir):
    paths = sorted(glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True))
    assert paths, f"Không tìm thấy CSV trong {csv_dir}"
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        # Đảm bảo tồn tại video_path/label nếu chưa có (phòng hờ)
        if "video_path" not in df.columns:
            df["video_path"] = os.path.relpath(p, csv_dir)
        if "label" not in df.columns:
            # cố gắng lấy label từ đường dẫn cha (vd: .../Padamasana/video 9.csv)
            parts = os.path.normpath(p).split(os.sep)
            df["label"] = parts[-2] if len(parts) >= 2 else "Unknown"
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Chuẩn kiểu
    if "frame" in df.columns:  df["frame"]  = pd.to_numeric(df["frame"],  errors="coerce").fillna(0).astype(int)
    if "time_s" in df.columns: df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    return df

df = load_all_csv(CSV_DIR)
print("Tổng số dòng:", len(df))
print("Các cột đầu vào:", list(df.columns)[:14], " ...")
for col in ["label", "video_path"]:
    assert col in df.columns, f"Thiếu cột {col}"

# Sort theo thời gian
if "frame" in df.columns:
    df = df.sort_values(["video_path", "frame"], kind="mergesort")
elif "time_s" in df.columns:
    df = df.sort_values(["video_path", "time_s"], kind="mergesort")

# ===== XÂY DỰNG DANH SÁCH FEATURE (34 cột x,y) =====
def build_feature_cols(df):
    cols = []
    for kp in COCO17_ORDER:
        xcol, ycol = f"{kp}_x", f"{kp}_y"
        if xcol in df.columns and ycol in df.columns:
            cols += [xcol, ycol]
    assert cols, "Không tìm thấy cột keypoint (x,y) hợp lệ!"
    return cols

feature_cols = build_feature_cols(df)
print(f"Số feature (x,y) được chọn: {len(feature_cols)}  (expect 34)")
print("Ví dụ 10 cột đầu:", feature_cols[:10])

# ===== LÀM SẠCH THEO VIDEO: interpolate + ffill/bfill + (optional) smoothing =====
def clean_per_video(df_all, feat_cols, win=5):
    out = []
    for vid, g in df_all.groupby("video_path"):
        g = g.sort_values(["frame" if "frame" in g.columns else "time_s"]).reset_index(drop=True)
        # nội suy + ffill/bfill cho từng cột feature
        g[feat_cols] = g[feat_cols].interpolate(limit_direction="both")
        g[feat_cols] = g[feat_cols].fillna(method="bfill").fillna(method="ffill")
        # OPTIONAL: smoothing rolling median nhỏ để đỡ jitter (có thể tắt nếu không muốn)
        if win and win >= 3:
            g[feat_cols] = g[feat_cols].rolling(window=win, min_periods=1, center=True).median()
        out.append(g)
    return pd.concat(out, ignore_index=True)

df = clean_per_video(df, feature_cols, win=3)
# Sau clean, bảo đảm không còn NaN trong feature (nếu vẫn còn — loại dòng)
na_before = df[feature_cols].isna().sum().sum()
if na_before:
    print(f"⚠️ Còn {na_before} NaN sau clean → drop")
    df = df.dropna(subset=feature_cols)

# ===== ENCODE NHÃN =====
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"].astype(str))
classes = list(le.classes_)
print("Classes:", classes)

# ===== SPLIT THEO VIDEO =====
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
train_val_idx, test_idx = next(gss.split(df, groups=df["video_path"]))
df_train_val = df.iloc[train_val_idx].copy()
df_test      = df.iloc[test_idx].copy()

gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_SEED)
tr_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["video_path"]))
df_train = df_train_val.iloc[tr_idx].copy()
df_val   = df_train_val.iloc[val_idx].copy()

print("Số frame: train/val/test =", len(df_train), len(df_val), len(df_test))
print("Số video  :", df_train.video_path.nunique(), df_val.video_path.nunique(), df_test.video_path.nunique())

# ===== SCALER (fit trên TRAIN) =====
scaler = StandardScaler()
scaler.fit(df_train[feature_cols].values)
dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
with open(os.path.join(OUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(classes))
with open(os.path.join(OUT_DIR, "feature_cols.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(feature_cols))
print("✅ Saved scaler.pkl, classes.txt, feature_cols.txt")

def apply_scale(df_):
    return scaler.transform(df_[feature_cols].values)

# ===== TẠO CỬA SỔ CHUỖI (majority + bỏ mơ hồ) =====
def make_windows_from_df(df_part, seq_len=30, step=1):
    Xs, ys = [], []
    feat_dim = len(feature_cols)
    for vid, g in df_part.groupby("video_path"):
        g = g.reset_index(drop=True)
        feats = apply_scale(g)  # (T, feat_dim)
        labels = g["label_id"].values
        T = len(g)
        if T < seq_len:
            continue
        for i in range(0, T - seq_len + 1, step):
            win_lab = labels[i:i+seq_len]
            c = Counter(win_lab)
            lab_major, count_major = c.most_common(1)[0]
            if count_major / seq_len < MAJORITY_RATIO_MIN:
                continue
            Xs.append(feats[i:i+seq_len])
            ys.append(int(lab_major))
    if not Xs:
        return np.zeros((0, seq_len, feat_dim), np.float32), np.zeros((0,), np.int64)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

X_train, y_train = make_windows_from_df(df_train, SEQ_LEN, TRAIN_STEP)
X_val,   y_val   = make_windows_from_df(df_val,   SEQ_LEN, VAL_TEST_STEP)
X_test,  y_test  = make_windows_from_df(df_test,  SEQ_LEN, VAL_TEST_STEP)

print("Shapes:")
print("  X_train", X_train.shape, "y_train", y_train.shape)
print("  X_val  ", X_val.shape,   "y_val  ", y_val.shape)
print("  X_test ", X_test.shape,  "y_test ", y_test.shape)

np.savez_compressed(os.path.join(OUT_DIR, "dataset_seq.npz"),
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test,
                    classes=np.array(classes))

# ===== MODEL: Conv1D + BiGRU + L2 + Dropout =====
def build_model(seq_len, feat_dim, n_classes):
    reg = regularizers.l2(L2W)
    inp = layers.Input((seq_len, feat_dim))
    x = layers.GaussianNoise(GAUSS_NOISE_STD)(inp)
    x = layers.Conv1D(128, 5, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(SPATIAL_DROPOUT)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.GRU(GRU_UNITS, kernel_regularizer=reg, recurrent_dropout=GRU_REC_DROPOUT))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

n_classes = len(classes)
feat_dim  = X_train.shape[-1] if X_train.size else len(feature_cols)
model = build_model(SEQ_LEN, feat_dim, n_classes)
model.summary()

ckpt_path = os.path.join(OUT_DIR, "best_model.keras")
cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# ===== CLASS WEIGHT (tuỳ chọn) =====
fit_kwargs = {}
if USE_CLASS_WEIGHT and len(np.unique(y_train)) > 1:
    cw = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y_train)
    fit_kwargs["class_weight"] = {i: float(w) for i, w in enumerate(cw)}
    print("Class weight:", fit_kwargs["class_weight"])

# ===== TRAIN =====
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1,
    **fit_kwargs
)

# ===== SAVE =====
last_path = os.path.join(OUT_DIR, "last_model.keras")
model.save(last_path)  # Keras format

h5_path = os.path.join(OUT_DIR, "best_model_tf.h5")
model.save(h5_path, include_optimizer=False)  # optional legacy

savedmodel_dir = os.path.join(OUT_DIR, "best_model_savedmodel")
# Keras 3: export SavedModel (cho TF Serving/TFLite). Nếu môi trường không hỗ trợ, bỏ qua.
try:
    model.export(savedmodel_dir)
except Exception as e:
    print("Note (export):", e)

# ===== EVALUATE =====
print("\n== Evaluate on TEST ==")
test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
y_pred = np.argmax(test_probs, axis=1)
print(classification_report(y_test, y_pred, target_names=classes, digits=4))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
plt.matshow(cm)
plt.title('Confusion matrix rate')
plt.colorbar()
plt.ylabel(' Giá trị thực')
plt.xlabel(' Giá trị dự đoán')
plt.show()

print("\n✅ Saved:")
print(" -", ckpt_path)
print(" -", last_path)
print(" -", h5_path)
print(" -", savedmodel_dir)
print(" -", os.path.join(OUT_DIR, "classes.txt"))
print(" -", os.path.join(OUT_DIR, "scaler.pkl"))
print(" -", os.path.join(OUT_DIR, "feature_cols.txt"))
