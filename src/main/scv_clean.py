import pandas as pd, numpy as np
df = pd.read_csv("/home/B2205945-CT255/ProJect_yolo/data/out/dataset/Vrikshasana.csv")
# Bỏ frame có >40% NaN keypoints
feat = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
keep = df[feat].isna().mean(axis=1) < 0.4
df = df[keep].reset_index(drop=True)
df.to_csv("/home/B2205945-CT255/ProJect_yolo/data/out/dataset_clean/dataset_all_clean_Vrikshasana.csv", index=False)
