# -*- coding: utf-8 -*-
"""
Tạo đặc trưng chuẩn cho predict:
- Tính các joint angles từ keypoints 2D (COCO-17)
- Ghép vào DataFrame
- Căn cột theo out/feature_cols.txt (đúng như lúc train)
- Xuất CSV đã gộp để predict
"""

import os
import math
import numpy as np
import pandas as pd

OUT_DIR = "out"
FEATURE_COLS_PATH = os.path.join(OUT_DIR, "feature_cols.txt")

# === Sửa đường dẫn đầu vào/đầu ra tuỳ ý ===
RAW_CSV = "/home/B2205945-CT255/ProJect_yolo/data/test_csv/csv/video 15.csv"        # CSV đầu vào chỉ có keypoints
MERGED_CSV = "data/test_csv/test_video5_merged.csv"  # CSV đầu ra (đã gộp góc & căn cột)

# COCO-17 keypoints (đặt theo tên cột trong CSV của bạn)
KP = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# CSV kỳ vọng có cột: f"{name}_x", f"{name}_y" cho mỗi KP ở trên.

# Các bộ 3 điểm để tính góc tại điểm giữa: (A, B, C) -> góc ABC
ANGLE_TRIPLETS = [
    # tay
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_shoulder", "left_elbow"),
    ("right_hip", "right_shoulder", "right_elbow"),
    # chân
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    # thân người / cổ
    ("left_shoulder", "neck", "right_shoulder"),  # neck sẽ được tính trung điểm 2 vai
    ("left_hip", "pelvis", "right_hip"),          # pelvis: trung điểm 2 hông
    ("neck", "pelvis", "left_hip"),
    ("neck", "pelvis", "right_hip"),
]

def angle_between(ax, ay, bx, by, cx, cy, eps=1e-9):
    """
    Góc ABC (độ), A-B-C: B là đỉnh
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = math.hypot(v1x, v1y) + eps
    n2 = math.hypot(v2x, v2y) + eps
    dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def compute_virtual_points(row):
    """
    Tạo các điểm ảo 'neck' (trung điểm 2 vai) và 'pelvis' (trung điểm 2 hông)
    nếu đủ dữ liệu.
    """
    pts = {}
    # vai
    try:
        lsh = (row["left_shoulder_x"], row["left_shoulder_y"])
        rsh = (row["right_shoulder_x"], row["right_shoulder_y"])
        pts["neck_x"] = (lsh[0] + rsh[0]) / 2.0
        pts["neck_y"] = (lsh[1] + rsh[1]) / 2.0
    except Exception:
        pts["neck_x"] = np.nan
        pts["neck_y"] = np.nan

    # hông
    try:
        lhp = (row["left_hip_x"], row["left_hip_y"])
        rhp = (row["right_hip_x"], row["right_hip_y"])
        pts["pelvis_x"] = (lhp[0] + rhp[0]) / 2.0
        pts["pelvis_y"] = (lhp[1] + rhp[1]) / 2.0
    except Exception:
        pts["pelvis_x"] = np.nan
        pts["pelvis_y"] = np.nan

    return pts

def compute_angles(df):
    """
    Tính các góc theo ANGLE_TRIPLETS. Trả về DataFrame chỉ chứa các cột góc.
    Cột tên: angle_{A}_{B}_{C}
    """
    # đảm bảo có cột neck/pelvis nếu cần
    vir_cols = ["neck_x", "neck_y", "pelvis_x", "pelvis_y"]
    for c in vir_cols:
        if c not in df.columns:
            df[c] = np.nan
    # điền neck/pelvis cho từng frame
    for i in range(len(df)):
        pts = compute_virtual_points(df.iloc[i])
        for k, v in pts.items():
            df.iat[i, df.columns.get_loc(k)] = v

    angles = {}
    for (A, B, C) in ANGLE_TRIPLETS:
        colname = f"angle_{A}_{B}_{C}"
        vals = []
        for _, row in df.iterrows():
            def get_xy(name):
                if name in ("neck", "pelvis"):
                    return row[f"{name}_x"], row[f"{name}_y"]
                return row.get(f"{name}_x", np.nan), row.get(f"{name}_y", np.nan)

            ax, ay = get_xy(A)
            bx, by = get_xy(B)
            cx, cy = get_xy(C)
            if any(map(lambda v: (v is None) or (isinstance(v, float) and np.isnan(v)), [ax, ay, bx, by, cx, cy])):
                vals.append(np.nan)
            else:
                vals.append(angle_between(ax, ay, bx, by, cx, cy))
        angles[colname] = vals

    ang_df = pd.DataFrame(angles)
    # điền NaN thành 0 (hoặc có thể dùng forward-fill/mean tuỳ bạn)
    ang_df = ang_df.fillna(0.0)
    return ang_df

def main():
    assert os.path.exists(FEATURE_COLS_PATH), \
        f"Thiếu {FEATURE_COLS_PATH}. Hãy chạy train để sinh file này."
    feature_cols = [line.strip() for line in open(FEATURE_COLS_PATH, encoding="utf-8") if line.strip()]

    df = pd.read_csv(RAW_CSV)
    print("Raw CSV:", df.shape)

    # Tên cột keypoint kỳ vọng (x,y) cho tất cả KP
    expected_xy = []
    for k in KP:
        expected_xy += [f"{k}_x", f"{k}_y"]

    missing_xy = [c for c in expected_xy if c not in df.columns]
    if missing_xy:
        print("⚠️ Thiếu toạ độ KP so với kỳ vọng:", missing_xy[:10], "..." if len(missing_xy) > 10 else "")
        # Vẫn tiếp tục: cột góc liên quan sẽ thành 0 (do NaN→0), nhưng tốt nhất là bổ sung đầy đủ KP.

    # Tính góc
    ang_df = compute_angles(df.copy())

    # Ghép vào DataFrame gốc
    merged = pd.concat([df, ang_df], axis=1)

    # Căn theo feature_cols.txt (đúng thứ tự & đủ số lượng)
    # Nếu thiếu cột nào → thêm 0.0; thừa cột ngoài list → bỏ
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    merged = merged[feature_cols]

    # Gộp lại với các cột meta (nếu muốn giữ)
    META = [c for c in ["label", "video_path", "frame", "time_s"] if c in df.columns]
    final_df = pd.concat([df[META], merged], axis=1) if META else merged

    final_df.to_csv(MERGED_CSV, index=False, encoding="utf-8")
    print("✅ Đã xuất CSV đã gộp góc & căn cột →", MERGED_CSV)
    print("Cột đặc trưng cuối cùng:", len(feature_cols))

if __name__ == "__main__":
    main()
