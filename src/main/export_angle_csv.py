import os
import numpy as np
import pandas as pd

in_csv  = "/home/B2205945-CT255/ProJect_yolo/data/out/datanew/landmarks_all_1.csv"
out_dir = "data/out/datanew"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(in_csv)

# ----- 1) TÍNH GÓC THEO TÊN KHỚP -----
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def row_angle_named(r, a, b, c):  # góc tại b (a-b-c)
    return angle((r[f"{a}_x"], r[f"{a}_y"]),
                 (r[f"{b}_x"], r[f"{b}_y"]),
                 (r[f"{c}_x"], r[f"{c}_y"]))

df_angles = pd.DataFrame({
    "left_elbow":  df.apply(lambda r: row_angle_named(r, "l_shoulder", "l_elbow", "l_wrist"), axis=1),
    "right_elbow": df.apply(lambda r: row_angle_named(r, "r_shoulder", "r_elbow", "r_wrist"), axis=1),
    "left_knee":   df.apply(lambda r: row_angle_named(r, "l_hip", "l_knee", "l_ankle"), axis=1),
    "right_knee":  df.apply(lambda r: row_angle_named(r, "r_hip", "r_knee", "r_ankle"), axis=1),
})

# Giữ meta để dùng sau (video_path, frame, time_s, label)
for col in ["label", "video_path", "frame", "time_s"]:
    if col in df.columns:
        df_angles[col] = df[col]

df_angles.to_csv(os.path.join(out_dir, "angles_dataset.csv"), index=False)

# ----- 2) LANDMARKS DATASET (xy đã chuẩn hoá 0..1) -----
# chọn toàn bộ *_x, *_y và loại *_conf
cols_xy = [c for c in df.columns
           if (c.endswith("_x") or c.endswith("_y")) and not c.endswith("_conf")]

X = df[cols_xy].copy()
# xử lý NaN nếu có (tuỳ chọn)
X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)
# clip đề phòng lỗi parse
X = X.clip(lower=0.0, upper=1.0)

df_landmarks = X.copy()
for col in ["label", "video_path", "frame", "time_s"]:
    if col in df.columns:
        df_landmarks[col] = df[col]

df_landmarks.to_csv(os.path.join(out_dir, "landmarks_dataset.csv"), index=False)

print("✅ angles_dataset.csv:", df_angles.shape, " | ✅ landmarks_dataset.csv:", df_landmarks.shape)
