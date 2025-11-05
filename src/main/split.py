# -*- coding: utf-8 -*-
import os, shutil, argparse, random
from glob import glob
from math import floor

def split_class_files(files, train_ratio, val_ratio, seed=42):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = floor(n * train_ratio)
    n_val = floor(n * val_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    return train_files, val_files, test_files

def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Thư mục gốc hiện tại (chỉ có 'train' duy nhất, chia lớp bên trong)")
    ap.add_argument("--dst", default="data_split", help="Thư mục đích sẽ tạo train/val/test")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Mặc định copy. (Bạn có thể sửa thành symlink nếu muốn)")
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio - 1.0) <= 1e-6 or \
           (args.train_ratio + args.val_ratio) < 1.0, "Tỷ lệ không hợp lệ. Nếu muốn có test, train+val < 1.0."
    test_ratio = max(0.0, 1.0 - (args.train_ratio + args.val_ratio))
    print(f"Split ratios -> train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={test_ratio:.2f}")

    src_train = os.path.join(args.src, "train")
    if not os.path.isdir(src_train):
        raise FileNotFoundError(f"Không thấy {src_train}. Cấu trúc mong đợi: src/train/<class>/*.jpg")

    classes = [d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d))]
    classes.sort()
    print("Classes:", classes)

    for cls in classes:
        img_paths = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            img_paths += glob(os.path.join(src_train, cls, ext))
        img_paths.sort()

        if len(img_paths) == 0:
            print(f"[WARN] lớp '{cls}' không có ảnh, bỏ qua.")
            continue

        tr, va, te = split_class_files(img_paths, args.train_ratio, args.val_ratio, seed=args.seed)

        for src_path in tr:
            dst_path = os.path.join(args.dst, "train", cls, os.path.basename(src_path))
            safe_copy(src_path, dst_path)
        for src_path in va:
            dst_path = os.path.join(args.dst, "val", cls, os.path.basename(src_path))
            safe_copy(src_path, dst_path)
        if test_ratio > 0:
            for src_path in te:
                dst_path = os.path.join(args.dst, "test", cls, os.path.basename(src_path))
                safe_copy(src_path, dst_path)

        print(f"[{cls}] total={len(img_paths)}  -> train={len(tr)}, val={len(va)}, test={len(te)}")

    print(f"✅ Done. Thư mục đã tạo ở: {args.dst}")

if __name__ == "__main__":
    main()
