import numpy as np

KPTS = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
        "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
        "l_knee","r_knee","l_ankle","r_ankle"]

def preprocess_frame(row):
    # lấy toạ độ
    pts = []
    for n in KPTS:
        x, y = row[f"{n}_x"], row[f"{n}_y"]
        pts.append([x, y])
    pts = np.array(pts, dtype=np.float32)  # shape (17,2)

    # nếu có NaN nhiều quá thì trả về None
    if np.isnan(pts).mean() > 0.4:
        return None

    # gốc tại mid-hip
    l_hip = pts[KPTS.index("l_hip")]
    r_hip = pts[KPTS.index("r_hip")]
    mid_hip = (l_hip + r_hip) / 2.0
    pts -= mid_hip

    # scale theo khoảng cách vai-hip trung bình
    l_sh, r_sh = pts[KPTS.index("l_shoulder")], pts[KPTS.index("r_shoulder")]
    shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6
    scale = shoulder_width
    pts /= scale

    return pts.flatten()  # (34,)
