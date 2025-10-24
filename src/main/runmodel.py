import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from pathlib import Path

# optional: try to use MediaPipe for real landmarks extraction
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

# ===== Đường dẫn model và classes =====
MODEL_PATH = "data/out/dataset_clean/best_cnn_gru.keras"
CLASSES_TXT = "data/out/dataset_clean/classes.txt"

# ===== Đường dẫn video cần kiểm tra =====
VIDEO_PATH = "/home/B2205945-CT255/ProJect_yolo/data/out/Padamasana/video 2.mp4"

# ===== Cấu hình =====
SEQ_LEN = 30   # phải khớp với model lúc train
model = load_model(MODEL_PATH)

# Đọc classes
with open(CLASSES_TXT, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

print("Model loaded ✅")
print("Classes:", classes)

# ===== 1) Hàm đọc landmarks từ video (đã chuẩn hóa YOLO pose trước đó) =====
# Trước tiên cố gắng load .npy đã sinh trước đó theo tên video.
# Nếu không có, sẽ cố gắng trích xuất bằng MediaPipe (nếu cài), nếu không có thì dùng fallback
LANDMARKS_DIR = Path("data/out/landmarks")
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

def _npy_path_for_video(video_path: str) -> Path:
    bn = Path(video_path).stem
    return LANDMARKS_DIR / f"{bn}.npy"

def extract_landmarks_from_video_mediapipe(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # use x,y,z,visibility for each landmark
            row = []
            for p in lm:
                row.extend([p.x, p.y, p.z, p.visibility])
        else:
            # if no detection, append zeros
            row = [0.0] * (33 * 4)
        feats.append(row)
    cap.release()
    pose.close()
    return np.array(feats, dtype=np.float32)

def extract_landmarks_from_video_fallback(video_path: str, target_size=(64,64)):
    # simple fallback: per-frame downscaled grayscale flattened (deterministic pseudo-features)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f = cv2.resize(frame, target_size)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        feats.append(g.flatten().tolist())
    cap.release()
    return np.array(feats, dtype=np.float32)

def load_or_compute_landmarks(video_path: str):
    npy = _npy_path_for_video(video_path)
    if npy.exists():
        lm = np.load(npy)
        print("Loaded landmarks from:", npy)
        return lm
    # else compute
    print("No .npy found for video, extracting landmarks...", "(MediaPipe enabled)" if _HAS_MEDIAPIPE else "(fallback)")
    if _HAS_MEDIAPIPE:
        lm = extract_landmarks_from_video_mediapipe(video_path)
    else:
        lm = extract_landmarks_from_video_fallback(video_path)
    # save for future runs
    np.save(npy, lm)
    print("Saved landmarks to:", npy)
    return lm

# ===== 2) Mô phỏng chuỗi landmarks cho video sẵn có (minh hoạ) =====
# Thay vì random, load hoặc trích xuất theo cách ở trên; sau đó chuẩn hoá chiều features theo model
F = model.input_shape[-1]

landmarks = load_or_compute_landmarks(VIDEO_PATH)  # shape: (n_frames, feat_dim)

# ensure feature dimension matches model's expected F
if landmarks.ndim != 2:
    landmarks = landmarks.reshape((landmarks.shape[0], -1))

feat_dim = landmarks.shape[1]
if feat_dim != F:
    if feat_dim > F:
        # trim features
        landmarks = landmarks[:, :F]
        print(f"Trimmed per-frame feature dim {feat_dim} -> {F}")
    else:
        # pad with zeros
        pad = np.zeros((landmarks.shape[0], F - feat_dim), dtype=landmarks.dtype)
        landmarks = np.concatenate([landmarks, pad], axis=1)
        print(f"Padded per-frame feature dim {feat_dim} -> {F}")

# ===== 3) Cắt thành các cửa sổ 30 frame =====
seqs = []
for i in range(0, len(landmarks)-SEQ_LEN+1, 1):
    seqs.append(landmarks[i:i+SEQ_LEN])
seqs = np.array(seqs)

# ===== 4) Dự đoán =====
preds = model.predict(seqs, verbose=0)
pred_idx = np.argmax(preds, axis=1)
pred_labels = [classes[i] for i in pred_idx]

# ===== 5) Hiển thị kết quả đơn giản =====
from collections import Counter
major = Counter(pred_labels).most_common(1)[0][0]
print(f"Kết quả dự đoán cho video {VIDEO_PATH}: {major}")

# ===== (Tuỳ chọn) Hiển thị từng frame kèm nhãn =====
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    label = pred_labels[min(frame_idx // 1, len(pred_labels)-1)]
    cv2.putText(frame, f"Pose: {label}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.imshow("Yoga Pose Tracking (offline)", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
