import pandas as pd
import glob

# Đọc tất cả file CSV trong thư mục
files = glob.glob("/home/B2205945-CT255/ProJect_yolo/data/out/dataset_clean/*.csv")

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

# Gộp thành 1 dataset lớn
df_all = pd.concat(dfs, ignore_index=True)
print("Tổng số dòng:", len(df_all))
print("Các nhãn có:", df_all['label'].unique())

# Lấy danh sách cột keypoints (x, y)
cols_xy = [c for c in df_all.columns if c.endswith('_x') or c.endswith('_y')]
cols_conf = [c for c in df_all.columns if c.endswith('_conf')]

# Xử lý NaN nếu có
df_all[cols_xy + cols_conf] = df_all[cols_xy + cols_conf].fillna(0)

# Xuất file landmarks_dataset.csv
df_landmarks = df_all[cols_xy + ['label']]
df_landmarks.to_csv("data/out/datanew/landmarks_dataset.csv", index=False)

print("✅ Đã tạo file landmarks_dataset.csv")

import numpy as np

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def row_angle(r, a, b, c):
    return angle((r[a+'_x'], r[a+'_y']),
                 (r[b+'_x'], r[b+'_y']),
                 (r[c+'_x'], r[c+'_y']))

angles = []
for _, row in df_all.iterrows():
    ang = {
        'left_elbow':  row_angle(row, 'l_shoulder', 'l_elbow', 'l_wrist'),
        'right_elbow': row_angle(row, 'r_shoulder', 'r_elbow', 'r_wrist'),
        'left_knee':   row_angle(row, 'l_hip', 'l_knee', 'l_ankle'),
        'right_knee':  row_angle(row, 'r_hip', 'r_knee', 'r_ankle'),
        'label': row['label']
    }
    angles.append(ang)

df_angles = pd.DataFrame(angles)
df_angles.to_csv("data/out/datanew/angles_dataset.csv", index=False)

print("✅ Đã tạo file angles_dataset.csv")

