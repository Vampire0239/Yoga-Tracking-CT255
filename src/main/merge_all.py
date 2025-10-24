import os, pandas as pd, numpy as np

BASE = "data/out/datanew"
LAND_CSV = f"{BASE}/landmarks_dataset.csv"
ANG_CSV  = f"{BASE}/angles_dataset.csv"
OUT_CSV  = f"{BASE}/merged_landmarks_angles.csv"

os.makedirs(BASE, exist_ok=True)

dfL = pd.read_csv(LAND_CSV)
dfA = pd.read_csv(ANG_CSV)

# ---- 1) Chọn khóa merge "an toàn"
# Ưu tiên (video_path, frame). Nếu thiếu frame, dùng (video_path, time_s).
merge_keys = [k for k in ["video_path", "frame"] if k in dfL.columns and k in dfA.columns]
if not merge_keys:
    merge_keys = [k for k in ["video_path", "time_s"] if k in dfL.columns and k in dfA.columns]
if not merge_keys:
    raise ValueError("Không tìm được khóa chung để merge. Cần 'video_path' và 'frame' hoặc 'time_s' ở cả 2 file.")

# Chuẩn hóa kiểu dữ liệu khóa (tránh lệch kiểu gây miss-join)
for k in merge_keys:
    if k == "frame":
        dfL[k] = pd.to_numeric(dfL[k], errors="coerce").astype("Int64")
        dfA[k] = pd.to_numeric(dfA[k], errors="coerce").astype("Int64")
    else:
        dfL[k] = dfL[k].astype(str)
        dfA[k] = dfA[k].astype(str)

# Nếu angles có 'label' thì bỏ để tránh đè (ưu tiên label của landmarks)
if "label" in dfA.columns:
    dfA = dfA.drop(columns=["label"])

# ---- 2) Xác định cột góc: numeric, KHÔNG phải *_x/*_y/*_conf và KHÔNG phải meta
META_COLS = {"label", "video_path", "frame", "time_s"}

# Lấy danh sách cột numeric an toàn (kể cả Int64 nullable)
num_cols_A = dfA.select_dtypes(include=[np.number]).columns.tolist()

angle_cols = [
    c for c in num_cols_A
    if (not c.endswith("_x"))
    and (not c.endswith("_y"))
    and (not c.endswith("_conf"))
    and (c not in META_COLS)
]


# Scale góc về [0,1] (giả sử đơn vị độ 0..180)
for c in angle_cols:
    dfA[c] = (dfA[c] / 180.0).clip(0.0, 1.0)

# Chống trùng tên cột (phòng xa)
dfA = dfA.loc[:, ~dfA.columns.duplicated()].copy()

# ---- 3) Kiểm tra/khử trùng khóa ở bảng góc (one_to_one yêu cầu khóa phải duy nhất)
if dfA.duplicated(subset=merge_keys).any():
    # Nếu có bản ghi trùng khóa, lấy bản ghi đầu tiên (hoặc bạn có thể groupby().mean())
    dfA = dfA.drop_duplicates(subset=merge_keys, keep="first")

# ---- 4) Merge (left join theo landmarks là “khung xương” chính)
print("merge_keys:", merge_keys)
print("angle_cols ({}):".format(len(angle_cols)), angle_cols[:10], "..." if len(angle_cols) > 10 else "")

dfM = pd.merge(
    dfL,
    dfA[merge_keys + angle_cols],
    on=merge_keys,
    how="left",
    validate="one_to_one"
)

# Fill NA cho các cột góc (trường hợp không tính được do thiếu khớp)
if angle_cols:
    dfM[angle_cols] = dfM[angle_cols].fillna(0.0)

dfM.to_csv(OUT_CSV, index=False)
print("✅ Merged to:", OUT_CSV)
