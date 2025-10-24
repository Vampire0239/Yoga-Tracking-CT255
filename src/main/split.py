# ===================== CNN+GRU PIPELINE (hardened, fixed) =====================
import os, numpy as np, pandas as pd, pickle, warnings
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models

warnings.filterwarnings("ignore")

# --------- Cấu hình ---------
IN_CSV   = "data/out/datanew/merged_landmarks_angles.csv"   # hoặc merged_landmarks_angles.csv
OUT_DIR  = "data/out/dataset_clean"                         # <== đồng bộ cùng thư mục
SEQ_LEN  = 30
STEP     = 1
MIN_PURE = 0.80
LR       = 1e-4
BATCH    = 16
EPOCHS   = 60
USE_CONF = False       # thêm *_conf nếu muốn
BALANCE_PER_CLIP = True    # giới hạn tối đa windows/clip
MAX_PER_CLIP     = 200
BALANCE_BY_CLASS = True    # undersample theo lớp sau khi cap/clip

# --------- Tái lập ngẫu nhiên ---------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# --------- 1) Đọc & chuẩn bị dữ liệu ---------
df = pd.read_csv(IN_CSV)

for c in ["label", "video_path"]:
    if c not in df.columns:
        raise ValueError(f"Thiếu cột bắt buộc: {c}")

df["clip_id"] = (df["label"].astype(str) + "::" + df["video_path"].astype(str))

# Chọn feature: *_x, *_y (+ *_conf nếu bật)
feat_cols_xy   = [c for c in df.columns if (c.endswith("_x") or c.endswith("_y")) and not c.endswith("_conf")]
feat_cols_conf = [c for c in df.columns if c.endswith("_conf")] if USE_CONF else []
# Nếu bạn dùng file merged (có cả góc), các cột góc đã là numeric khác *_x/_y/_conf -> sẽ vẫn nằm ngoài chọn lọc này.
# => Nếu muốn dùng góc trong merged, ta sẽ tự động thêm ở dưới bằng cách nhận diện numeric-không-phải-xy-conf:
meta_cols = {"label","video_path","clip_id","frame","time_s"}
num_cols = set(df.select_dtypes(include=[np.number]).columns)
angle_cols = sorted(list(num_cols - set(feat_cols_xy) - set(feat_cols_conf) - meta_cols))

feat_cols = sorted(feat_cols_xy + feat_cols_conf + angle_cols)

# Lưu danh sách cột để infer
np.save(os.path.join(OUT_DIR, "feat_cols.npy"), np.array(feat_cols, dtype=object))

# Xử lý NaN
df[feat_cols] = df[feat_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)

# Sắp xếp chuỗi theo clip + thời gian
sort_cols = ["clip_id"]
if "time_s" in df.columns: sort_cols.append("time_s")
if "frame"  in df.columns: sort_cols.append("frame")
df = df.sort_values(sort_cols)

# --------- 2) Tạo cửa sổ theo clip_id, lọc "độ thuần" nhãn ---------
def pick_label(segL, min_ratio=MIN_PURE):
    cnt = Counter(segL)
    lab, f = cnt.most_common(1)[0]
    return (lab, f/len(segL) >= min_ratio)

Xs, ys, clips = [], [], []

def make_windows(g, seq_len=SEQ_LEN, step=STEP):
    arr  = g[feat_cols].values
    labs = g["label"].values
    outX, outY = [], []
    for i in range(0, len(g) - seq_len + 1, step):
        segX = arr[i:i+seq_len]
        segL = labs[i:i+seq_len]
        lab, ok = pick_label(segL, MIN_PURE)
        if ok:
            outX.append(segX)
            outY.append(lab)
    return outX, outY

for cid, g in df.groupby("clip_id", sort=False):
    if len(g) < SEQ_LEN: 
        continue
    x, y = make_windows(g, SEQ_LEN, STEP)
    if not x: 
        continue
    Xs += x; ys += y; clips += [cid] * len(x)

if not Xs:
    raise RuntimeError("Không tạo được window nào (MIN_PURE quá cao hoặc dữ liệu quá ít).")

X = np.asarray(Xs, dtype=np.float32)   # (N, T, F)
y = np.asarray(ys)
clips = np.asarray(clips)

# --------- 2.1 Cân bằng (tùy chọn) trước khi split ---------
rng = np.random.RandomState(SEED)

if BALANCE_PER_CLIP:
    by_clip = defaultdict(list)
    for i, cid in enumerate(clips):
        by_clip[cid].append(i)
    keep = []
    for cid, idx in by_clip.items():
        if len(idx) > MAX_PER_CLIP:
            keep += list(rng.choice(idx, size=MAX_PER_CLIP, replace=False))
        else:
            keep += idx
    keep = np.array(sorted(keep))
    X, y, clips = X[keep], y[keep], clips[keep]

if BALANCE_BY_CLASS:
    cls_counts = Counter(y)
    K = min(max(50, int(np.median(list(cls_counts.values())))), 500)  # ngưỡng mềm
    by_cls = defaultdict(list)
    for i, lab in enumerate(y):
        by_cls[lab].append(i)
    keep2 = []
    for lab, idx in by_cls.items():
        if len(idx) > K:
            keep2 += list(rng.choice(idx, size=K, replace=False))
        else:
            keep2 += idx
    keep2 = np.array(sorted(keep2))
    X, y, clips = X[keep2], y[keep2], clips[keep2]

# Lưu để kiểm tra/repro
np.save(os.path.join(OUT_DIR, "X_seq.npy"), X)
pd.Series(y).to_csv(os.path.join(OUT_DIR, "y_seq.csv"), index=False, header=False)
pd.Series(clips).to_csv(os.path.join(OUT_DIR, "clip_seq.csv"), index=False, header=False)

print("Windows created:", X.shape, " (#seq, T, F)")
print("Overall label dist:", Counter(y))

# --------- 3) Chia train/val/test THEO CLIP ---------
clip_unique = np.unique(clips)
clip2label = {c: Counter(y[clips==c]).most_common(1)[0][0] for c in clip_unique}

le = LabelEncoder()
y_id = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", list(le.classes_), "| num_classes:", num_classes)

with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

clip_labels_enc = le.transform([clip2label[c] for c in clip_unique])

try:
    clip_train, clip_tmp, _, lab_tmp = train_test_split(
        clip_unique, clip_labels_enc, test_size=0.30, random_state=SEED, stratify=clip_labels_enc
    )
    clip_val, clip_test, _, _ = train_test_split(
        clip_tmp, lab_tmp, test_size=0.50, random_state=SEED, stratify=lab_tmp
    )
except ValueError:
    clip_train, clip_tmp = train_test_split(clip_unique, test_size=0.30, random_state=SEED, shuffle=True)
    clip_val, clip_test  = train_test_split(clip_tmp,     test_size=0.50, random_state=SEED, shuffle=True)
    print("[WARN] Không stratify được do một số lớp quá ít clip.")

# Lưu danh sách clip theo split & check leak
pd.Series(clip_train).to_csv(os.path.join(OUT_DIR, "clips_train.csv"), index=False, header=False)
pd.Series(clip_val).to_csv(os.path.join(OUT_DIR, "clips_val.csv"), index=False, header=False)
pd.Series(clip_test).to_csv(os.path.join(OUT_DIR, "clips_test.csv"), index=False, header=False)

def subset_by_clips(target_clips):
    m = np.isin(clips, target_clips)
    return X[m], y_id[m]

X_train, y_train = subset_by_clips(clip_train)
X_val,   y_val   = subset_by_clips(clip_val)
X_test,  y_test  = subset_by_clips(clip_test)

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

def inter(a,b): return set(a).intersection(set(b))
print("Split:")
print("  Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)
print("  train dist:", Counter(y_train))
print("  val   dist:", Counter(y_val))
print("  test  dist:", Counter(y_test))
print("Leak check train∩val:", inter(clip_train, clip_val), "| train∩test:", inter(clip_train, clip_test))

# --------- 3.1 Class weights ---------
w = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
class_weight = {i: float(w[i]) for i in range(num_classes)}
print("class_weight:", class_weight)

# --------- 4) Build mô hình CNN + (Bi)GRU ---------
T, F = X_train.shape[1], X_train.shape[2]

def build_cnn_gru(seq_len=T, feat_dim=F, num_classes=num_classes, lr=LR):
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.TimeDistributed(layers.Reshape((feat_dim, 1)))(inp)

    x = layers.TimeDistributed(layers.Conv1D(32, 3, activation='relu', padding='valid'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPool1D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)

    x = layers.TimeDistributed(layers.Conv1D(128, 3, activation='relu', padding='valid'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPool1D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.GRU(64)(x)

    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_cnn_gru()
model.summary()

# --------- 5) Train + Evaluate ---------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "best_cnn_gru.keras"),
                                       monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weight
)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"[TEST] loss={test_loss:.4f} | acc={test_acc:.4f}")

# Dự đoán & báo cáo
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred, target_names=list(le.classes_)))
print(confusion_matrix(y_test, y_pred))

# Debug: mean prob per class trên VAL để phát hiện dồn lớp
p_val = model.predict(X_val, verbose=0)
print("Mean prob per class (VAL):", dict(zip(le.classes_, p_val.mean(axis=0))))

# Lưu artifacts
with open(os.path.join(OUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
    for c in le.classes_:
        f.write(str(c) + "\n")
with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("✅ Done. Model & artifacts saved in:", OUT_DIR)
# =====================================================================
