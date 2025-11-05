# -*- coding: utf-8 -*-
# =========================================================
# Train Conv1D + BiGRU cho chuỗi keypoints (yoga poses)
# - Dữ liệu: chỉ (x,y) + video_path,label
# - Clean NaN theo từng video (interpolate + ffill/bfill + optional smoothing)
# - Khoá thứ tự cột feature; lưu feature_cols.txt để predict khớp
# - Lưu đầy đủ kết quả + CM train/test (gray_r) kèm số trong ô
# - KHÔNG dùng validation (chỉ train/test)
# =========================================================

# pip install numpy pandas scikit-learn "tensorflow==2.*" tqdm joblib matplotlib

import os, glob, json, io, sys, platform, datetime
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
from keras import layers, models, callbacks, regularizers
from keras.callbacks import CSVLogger, TensorBoard


# ===== CẤU HÌNH =====
CSV_DIR   = "data/csv"
OUT_DIR   = "out2d"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN         = 30
TRAIN_STEP      = 5
VAL_TEST_STEP   = 1
TEST_SIZE       = 0.15
RANDOM_SEED     = 42
MAJORITY_RATIO_MIN = 0.8

# Regularization/kiến trúc
# Learning_rate = 3e-4 
L2W = 1e-3
GAUSS_NOISE_STD = 0.02     # tăng độ nhiễu Gaussian (chỉ khi train)
SPATIAL_DROPOUT = 0.30     # tắt kênh đặc trưng theo trục thời gian
GRU_UNITS = 96             # số ô nhớ của GRU
GRU_REC_DROPOUT = 0.25     # dropout trong nhánh hồi tiếp GRU
DENSE_UNITS = 96

USE_CLASS_WEIGHT = False    # cân bằng lớp khi train
BATCH_SIZE = 64
EPOCHS = 100

META_COLS = {"label", "video_path", "frame", "time_s"}

# ===== ENV INFO =====
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
print("TensorFlow:", tf.__version__)
print("Using tf.keras only")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
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
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv"), recursive=True))
    assert paths, f"Không tìm thấy CSV trong {csv_dir}"
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        # Đảm bảo tồn tại video_path/label nếu chưa có (phòng hờ)
        if "video_path" not in df.columns:
            df["video_path"] = os.path.relpath(p, csv_dir)
        if "label" not in df.columns:
            parts = os.path.normpath(p).split(os.sep)
            df["label"] = parts[-2] if len(parts) >= 2 else "Unknown"
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Chuẩn kiểu
    if "frame" in df_all.columns:
        df_all["frame"] = pd.to_numeric(df_all["frame"], errors="coerce").fillna(0).astype(int)
    if "time_s" in df_all.columns:
        df_all["time_s"] = pd.to_numeric(df_all["time_s"], errors="coerce")
    return df_all

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
na_after = df[feature_cols].isna().sum().sum()
if na_after:
    print(f"⚠️ Còn {na_after} NaN sau clean → drop")
    df = df.dropna(subset=feature_cols)

# ===== ENCODE NHÃN =====
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"].astype(str))
classes = list(le.classes_)
print("Classes:", classes)

# ===== SPLIT THEO VIDEO (train 85% / test 15%) =====
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(df, groups=df["video_path"]))
df_train = df.iloc[train_idx].copy()
df_test  = df.iloc[test_idx].copy()

print("Số frame: train/test =", len(df_train), len(df_test))
print("Số video :", df_train.video_path.nunique(), df_test.video_path.nunique())

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
X_test,  y_test  = make_windows_from_df(df_test,  SEQ_LEN, VAL_TEST_STEP)

print("Shapes:")
print("  X_train", X_train.shape, "y_train", y_train.shape)
print("  X_test ", X_test.shape,  "y_test ", y_test.shape)


# ===== RESHAPE: (N, T, 34) -> (N, T, 17, 2, 1) cho Conv2D per-frame =====
feat_dim = X_train.shape[-1] if X_train.size else len(feature_cols)
assert feat_dim == 34, f"Conv2D (COCO-17) yêu cầu feat_dim=34 (17 keypoints * 2). Hiện tại: {feat_dim}"

NUM_KP, COORDS = 17, 2
X_train = X_train.reshape((-1, SEQ_LEN, NUM_KP, COORDS))[..., np.newaxis]  # (N, T, 17, 2, 1)
X_test  = X_test.reshape((-1, SEQ_LEN, NUM_KP, COORDS))[..., np.newaxis]   # (N, T, 17, 2, 1)

FRAME_INPUT_SHAPE = (SEQ_LEN, NUM_KP, COORDS, 1)

# Lưu dataset windows (tùy chọn)
np.savez_compressed(os.path.join(OUT_DIR, "dataset_seq.npz"),
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    classes=np.array(classes))

# ===== MODEL: Conv2D + BiGRU + L2 + Dropout =====
# ===== MODEL: TimeDistributed(Conv2D) + BiGRU =====
def build_model(frame_input_shape, n_classes):
    reg = regularizers.l2(L2W)

    inp = layers.Input(frame_input_shape)  # (T, 17, 2, 1)

    # Conv2D theo từng frame (TimeDistributed)
    x = layers.TimeDistributed(layers.GaussianNoise(GAUSS_NOISE_STD))(inp)

    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 2), padding="same", activation="relu", kernel_regularizer=reg)
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 1)))(x)  # giảm theo keypoint, giữ trục (x,y)

    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 2), padding="same", activation="relu", kernel_regularizer=reg)
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Dropout(0.4))(x)

    # (batch, T, H, W, C) -> (batch, T, F_flat)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # BiGRU gom theo thời gian
    x = layers.Bidirectional(
        layers.GRU(GRU_UNITS, kernel_regularizer=reg, recurrent_dropout=GRU_REC_DROPOUT)
    )(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Khởi tạo model
n_classes = len(classes)
model = build_model(FRAME_INPUT_SHAPE, n_classes)
model.summary()


# n_classes = len(classes)
# feat_dim  = X_train.shape[-1] if X_train.size else len(feature_cols)
# model = build_model(SEQ_LEN, feat_dim, n_classes)
# model.summary()

ckpt_path = os.path.join(OUT_DIR, "best_model.keras")

# ==== LOG TRAINING CSV + TENSORBOARD (monitor theo TRAIN, không dùng val) ====
log_csv = os.path.join(OUT_DIR, "training_log.csv")
tb_dir  = os.path.join(OUT_DIR, "tb_logs")

cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="accuracy", save_best_only=True, mode="max", verbose=1),
    callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True, mode="min"),
    callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger(log_csv, append=False),
    TensorBoard(log_dir=tb_dir)
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
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1,
    **fit_kwargs
)

# Lưu history ra CSV (phòng khi không dùng CSVLogger)
try:
    pd.DataFrame(history.history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)
except Exception:
    pass

# ===== SAVE =====
last_path = os.path.join(OUT_DIR, "last_model.keras")
model.save(last_path)  # Keras format

h5_path = os.path.join(OUT_DIR, "best_model_tf.h5")
model.save(h5_path, include_optimizer=False)  # optional legacy

savedmodel_dir = os.path.join(OUT_DIR, "best_model_savedmodel")
try:
    # Keras 3: export SavedModel (nếu hỗ trợ)
    model.export(savedmodel_dir)
except Exception as e:
    print("Note (export):", e)

# ===== HÀM VẼ CONFUSION MATRIX (gray_r + in số trong ô) =====
def plot_confusion_matrix(cm, classes, title, out_path, normalize=False):
    plt.figure(figsize=(0.8*len(classes)+3, 0.8*len(classes)+3))
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
        data_to_show = cm_disp
        fmt = ".2f"
    else:
        data_to_show = cm
        fmt = "d"

    # gray_r: trắng (0) → đen (max)
    im = plt.imshow(data_to_show, interpolation='nearest', cmap='gray_r')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = data_to_show.max() / 2.0 if data_to_show.size else 0.5
    for i in range(data_to_show.shape[0]):
        for j in range(data_to_show.shape[1]):
            val = data_to_show[i, j]
            txt = format(val, fmt)
            color = "white" if val > thresh else "black"
            plt.text(j, i, txt, ha="center", va="center", color=color)

    plt.ylabel('Giá trị thực')
    plt.xlabel('Giá trị dự đoán')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===== EVALUATE: trên TRAIN và TEST =====
def evaluate_split(name, X, y, classes, out_prefix):
    probs = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    rep_txt = classification_report(y, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y, y_pred)

    print(f"\n== {name} REPORT ==")
    print(rep_txt)
    print(f"{name} Confusion matrix:\n", cm)

    # Lưu báo cáo & ma trận
    with open(os.path.join(OUT_DIR, f"{out_prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt)
    with open(os.path.join(OUT_DIR, f"{out_prefix}_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(classification_report(y, y_pred, target_names=classes, digits=4, output_dict=True),
                  f, ensure_ascii=False, indent=2)
    np.savetxt(os.path.join(OUT_DIR, f"{out_prefix}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.save(os.path.join(OUT_DIR, f"{out_prefix}_y_true.npy"), y)
    np.save(os.path.join(OUT_DIR, f"{out_prefix}_y_pred.npy"), y_pred)
    np.save(os.path.join(OUT_DIR, f"{out_prefix}_probs.npy"), probs)

    # Vẽ CM (đếm) & (chuẩn hoá hàng)
    plot_confusion_matrix(cm, classes, title=f'Confusion Matrix ({name})', 
                          out_path=os.path.join(OUT_DIR, f"{out_prefix}_cm_gray.png"),
                          normalize=False)
    plot_confusion_matrix(cm, classes, title=f'Confusion Matrix Normalized ({name})', 
                          out_path=os.path.join(OUT_DIR, f"{out_prefix}_cm_gray_norm.png"),
                          normalize=True)

# Evaluate TRAIN
evaluate_split("TRAIN", X_train, y_train, classes, out_prefix="train")
# Evaluate TEST
evaluate_split("TEST",  X_test,  y_test,  classes, out_prefix="test")

# ===== TỔNG HỢP THÔNG TIN THỰC NGHIỆM =====
exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_txt_path  = os.path.join(OUT_DIR, f"experiment_summary_{exp_time}.txt")
summary_json_path = os.path.join(OUT_DIR, f"experiment_summary_{exp_time}.json")

# Thu thập kiến trúc model.summary()
stringio = io.StringIO()
model.summary(print_fn=lambda x: stringio.write(x + "\n"))
model_summary_str = stringio.getvalue()

exp_info = {
    "timestamp": exp_time,
    "python_version": sys.version,
    "platform": platform.platform(),
    "tensorflow_version": tf.__version__,
    "gpus": [str(g) for g in tf.config.list_physical_devices('GPU')],
    "paths": {"CSV_DIR": CSV_DIR, "OUT_DIR": OUT_DIR},
    "data_info": {
        "num_rows_total": int(len(df)),
        "num_videos_total": int(df.video_path.nunique()),
        "num_classes": int(len(classes)),
        "classes": classes,
        "feature_cols_count": int(len(feature_cols)),
        "feature_cols_sample": feature_cols[:10],
        "frames_split": {
            "train": int(len(df_train)),
            "test": int(len(df_test))
        },
        "videos_split": {
            "train": int(df_train.video_path.nunique()),
            "test": int(df_test.video_path.nunique())
        },
        "seq_settings": {
            "SEQ_LEN": SEQ_LEN,
            "TRAIN_STEP": TRAIN_STEP,
            "VAL_TEST_STEP": VAL_TEST_STEP,
            "MAJORITY_RATIO_MIN": MAJORITY_RATIO_MIN
        }
    },
    "hyperparams": {
        "TEST_SIZE": TEST_SIZE,
        "RANDOM_SEED": RANDOM_SEED,
        "L2W": L2W,
        "GAUSS_NOISE_STD": GAUSS_NOISE_STD,
        "SPATIAL_DROPOUT": SPATIAL_DROPOUT,
        "GRU_UNITS": GRU_UNITS,
        "GRU_REC_DROPOUT": GRU_REC_DROPOUT,
        "DENSE_UNITS": DENSE_UNITS,
        "USE_CLASS_WEIGHT": USE_CLASS_WEIGHT,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "optimizer": "Adam(3e-4)",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    },
    "artifacts": {
        "best_model": ckpt_path,
        "last_model": last_path,
        "h5_model": h5_path,
        "savedmodel_dir": savedmodel_dir,
        "scaler": os.path.join(OUT_DIR, "scaler.pkl"),
        "classes_txt": os.path.join(OUT_DIR, "classes.txt"),
        "feature_cols_txt": os.path.join(OUT_DIR, "feature_cols.txt"),
        "training_log_csv": log_csv,
        "tensorboard_dir": tb_dir,
        "history_csv": os.path.join(OUT_DIR, "history.csv"),
        # train
        "train_report_txt": os.path.join(OUT_DIR, "train_classification_report.txt"),
        "train_report_json": os.path.join(OUT_DIR, "train_classification_report.json"),
        "train_cm_csv": os.path.join(OUT_DIR, "train_confusion_matrix.csv"),
        "train_cm_png": os.path.join(OUT_DIR, "train_cm_gray.png"),
        "train_cm_norm_png": os.path.join(OUT_DIR, "train_cm_gray_norm.png"),
        "train_y_true": os.path.join(OUT_DIR, "train_y_true.npy"),
        "train_y_pred": os.path.join(OUT_DIR, "train_y_pred.npy"),
        "train_probs": os.path.join(OUT_DIR, "train_probs.npy"),
        # test
        "test_report_txt": os.path.join(OUT_DIR, "test_classification_report.txt"),
        "test_report_json": os.path.join(OUT_DIR, "test_classification_report.json"),
        "test_cm_csv": os.path.join(OUT_DIR, "test_confusion_matrix.csv"),
        "test_cm_png": os.path.join(OUT_DIR, "test_cm_gray.png"),
        "test_cm_norm_png": os.path.join(OUT_DIR, "test_cm_gray_norm.png"),
        "test_y_true": os.path.join(OUT_DIR, "test_y_true.npy"),
        "test_y_pred": os.path.join(OUT_DIR, "test_y_pred.npy"),
        "test_probs": os.path.join(OUT_DIR, "test_probs.npy"),
        "dataset_npz": os.path.join(OUT_DIR, "dataset_seq.npz"),
    },
    "model_summary": model_summary_str,
    "notes": "Thực nghiệm (train/test) + Confusion Matrix gray_r (kèm số) cho cả train & test."
}

# Ghi TXT (dễ đọc) + JSON (máy đọc)
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("=== EXPERIMENT SUMMARY ===\n")
    f.write(json.dumps(exp_info, ensure_ascii=False, indent=2))
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(exp_info, f, ensure_ascii=False, indent=2)

print("\n✅ Saved:")
for k, v in exp_info["artifacts"].items():
    print(" -", v)
