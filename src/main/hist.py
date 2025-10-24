

import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import confusion_matrix, classification_report

# ===================== Utils =====================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_lines(path: str):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def safe_load_npy(path: str):
    return np.load(path) if os.path.isfile(path) else None


def plot_hist_raw(out_dir, series, col, bins=60):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0: return
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.title(f"Histogram RAW (pixel) — {col}")
    plt.xlabel("Giá trị (pixel)"); plt.ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_raw_{col}.png"), dpi=150)
    plt.close()

def plot_hist_norm(out_dir, series, col, bins=60):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0: return
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.title(f"Histogram NORMALIZED (z-score) — {col}")
    plt.xlabel("Giá trị (z-score)"); plt.ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_norm_{col}.png"), dpi=150)
    plt.close()

# ===================== CSV Loader (RAW) =====================

COCO17_ORDER = [
    "nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
    "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"
]

def load_all_csv(csv_dir):
    paths = sorted(glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy CSV trong {csv_dir}")
    dfs=[]
    for p in paths:
        d = pd.read_csv(p)
        d.columns=[c.strip() for c in d.columns]
        if "video_path" not in d.columns:
            d["video_path"] = os.path.relpath(p, csv_dir)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    # sắp xếp nếu có frame/time_s
    if "frame" in df.columns:
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["video_path","frame"], kind="mergesort")
    elif "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df = df.sort_values(["video_path","time_s"], kind="mergesort")
    return df

# ===================== MAIN =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out", help="Thư mục artifacts đã lưu (mặc định: out)")
    ap.add_argument("--csv_dir", type=str, default="data/csv", help="Thư mục CSV thô (mặc định: data/csv)")
    ap.add_argument("--sample_feats", type=int, default=6, help="Số feature vẽ RAW/NORM (mặc định: 6)")
    ap.add_argument("--bins", type=int, default=60, help="Số bins cho histogram (mặc định: 60)")
    ap.add_argument("--sample_rows", type=int, default=0, help="Sample bớt rows CSV (0 = không)")
    args = ap.parse_args()

    out_dir = args.out_dir
    csv_dir = args.csv_dir
    bins = max(5, args.bins)

    eda_dir = os.path.join(out_dir, "eda_no_retrain")
    ensure_dir(eda_dir)

    # ---- Load artifacts đã lưu
    classes_path    = os.path.join(out_dir, "classes.txt")
    featcols_path   = os.path.join(out_dir, "feature_cols.txt")
    scaler_path     = os.path.join(out_dir, "scaler.pkl")

    if not os.path.isfile(classes_path):
        raise FileNotFoundError("Thiếu classes.txt trong <out_dir>.")
    classes = load_lines(classes_path)

    feature_cols = load_lines(featcols_path) if os.path.isfile(featcols_path) else None
    scaler = load(scaler_path) if os.path.isfile(scaler_path) else None

    # Nhãn & dự đoán
    y_test   = safe_load_npy(os.path.join(out_dir, "y_test.npy"))
    y_pred   = safe_load_npy(os.path.join(out_dir, "y_pred.npy"))
    probs_te = safe_load_npy(os.path.join(out_dir, "test_probs.npy"))

    y_train  = safe_load_npy(os.path.join(out_dir, "y_train.npy"))
    y_pred_tr= safe_load_npy(os.path.join(out_dir, "y_pred_train.npy"))
    probs_tr = safe_load_npy(os.path.join(out_dir, "train_probs.npy"))



    # ---- 4) RAW vs NORMALIZED (tách riêng), không train lại
    if scaler is not None and feature_cols is not None and os.path.isdir(csv_dir):
        try:
            df = load_all_csv(csv_dir)
            # Giảm tải nếu quá lớn
            if args.sample_rows and len(df) > args.sample_rows:
                df = df.sample(args.sample_rows, random_state=42)
            common_cols = [c for c in feature_cols if c in df.columns]
            if common_cols:
                # loại NaN tối thiểu
                df_clean = df.dropna(subset=common_cols, how="any")
                # chuẩn hoá bằng scaler đã lưu
                arr_norm = scaler.transform(df_clean[common_cols].values)
                df_norm = pd.DataFrame(arr_norm, columns=common_cols)
                # vẽ tách riêng
                subdir = os.path.join(eda_dir, "separate_raw_norm")
                ensure_dir(subdir)
                sample_cols = common_cols[:max(1, args.sample_feats)]
                for col in sample_cols:
                    plot_hist_raw(subdir, df_clean[col], col, bins=bins)
                    plot_hist_norm(subdir, df_norm[col],  col, bins=bins)
        except Exception as e:
            print("Bỏ qua phần RAW/NORMALIZED do lỗi:", e)

    print("✅ EDA đã lưu vào:", eda_dir)

if __name__ == "__main__":
    main()
