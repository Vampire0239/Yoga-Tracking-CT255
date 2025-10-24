import os
import numpy as np
import pandas as pd
from collections import Counter

# ====== Cấu hình ======
IN_CSV  = "data/out/datanew/landmarks_dataset.csv"  # đường dẫn file bạn vừa xuất
OUT_DIR = "data/out/datanew"
SEQ_LEN = 30        # độ dài chuỗi
STEP    = 1         # bước trượt cửa sổ (1 = trượt từng frame; 2 = cách 1 frame, v.v.)

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Đọc dữ liệu
df = pd.read_csv(IN_CSV)

df["video_path"] = df.apply(
    lambda r: f"{r['label']}/{r['video_path']}", axis=1
)

# 2) Chọn cột feature: toàn bộ *_x, *_y (loại *_conf nếu lỡ còn sót)
feat_cols = [c for c in df.columns if (c.endswith("_x") or c.endswith("_y")) and not c.endswith("_conf")]

# 3) Xử lý NaN để không vỡ chuỗi
df[feat_cols] = df[feat_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)

# 4) Sắp xếp theo video + thời gian
sort_cols = []
if "time_s" in df.columns: sort_cols.append("time_s")
if "frame"  in df.columns: sort_cols.append("frame")

if "video_path" not in df.columns:
    raise ValueError("Thiếu cột 'video_path' trong CSV. Hãy thêm video_path khi gộp CSV.")

df = df.sort_values(["video_path"] + sort_cols)

# 5) Tạo cửa sổ chuỗi theo từng video
Xs, ys, vids = [], [], []

def make_windows(g, seq_len=SEQ_LEN, step=STEP):
    arr  = g[feat_cols].values              # (N, F)
    labs = g["label"].values
    outX, outY = [], []
    for i in range(0, len(g) - seq_len + 1, step):
        segX = arr[i:i+seq_len]             # (T, F)
        segL = labs[i:i+seq_len]            # (T,)
        # Lấy nhãn majority (ổn định hơn) — có thể thay bằng segL[seq_len//2]
        lab  = Counter(segL).most_common(1)[0][0]
        outX.append(segX)
        outY.append(lab)
    return outX, outY

for vid, g in df.groupby("video_path", sort=False):
    if len(g) < SEQ_LEN:
        continue
    x, y = make_windows(g, SEQ_LEN, STEP)
    Xs += x
    ys += y
    vids += [vid] * len(x)

X = np.asarray(Xs, dtype=np.float32)  # (N, T, F)
y = np.asarray(ys)                    # (N,)
vids = np.asarray(vids)               # (N,)

# 6) Lưu ra file để train CNN+GRU
np.save(os.path.join(OUT_DIR, "X_seq.npy"), X)
pd.Series(y).to_csv(os.path.join(OUT_DIR, "y_seq.csv"), index=False, header=False)
pd.Series(vids).to_csv(os.path.join(OUT_DIR, "vid_seq.csv"), index=False, header=False)

# 7) Lưu danh sách lớp (để dùng lại lúc infer)
classes = sorted(pd.unique(y).tolist())
with open(os.path.join(OUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
    for c in classes:
        f.write(str(c) + "\n")

print("✅ Done.")
print("X shape:", X.shape, "(N, T, F)  | T=", X.shape[1], "F=", X.shape[2])
print("Số cửa sổ:", len(X), " | Số video:", len(np.unique(vids)))
print("Lớp:", classes)
