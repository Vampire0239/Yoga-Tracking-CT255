import pandas as pd
df = pd.read_csv("data/out/datanew/landmarks_all.csv")

print("Số clip duy nhất:", df["video_path"].nunique())
print(df[["video_path","label"]].drop_duplicates().head())

# Nếu muốn clip_id riêng:
df["clip_id"] = df["label"].astype(str) + "::" + df["video_path"].astype(str)
print("Số clip_id duy nhất:", df["clip_id"].nunique())
