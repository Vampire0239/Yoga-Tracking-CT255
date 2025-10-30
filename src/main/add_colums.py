import pandas as pd, glob, os
from pathlib import Path

files = glob.glob("/home/B2205945-CT255/ProJect_yolo/data/csv/csv/train/**/*.csv", recursive=True)


dfs = []
for f in files:
    df = pd.read_csv(f)
    path = Path(f)
    df["video_path"] = f"{path.parent.name}/{path.stem}"
    df["label"] = path.parent.name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv("data/csv/landmarks_all.csv", index=False)
print(df_all.head())
