import argparse, os, sys, json, datetime, subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THIS_DIR = os.path.dirname(__file__)
def make_out_dir(base_out: str, exp_name: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_out, exp_name, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- handlers ----------
def cmd_preprocess(args):
    out_dir = make_out_dir(args.out_dir, args.exp_name)
    meta = {
        "action": "preprocess",
        "csv_dir": args.csv_dir,
        "out_dir": out_dir,
        "notes": "Place your real preprocess logic here."
    }
    save_json(os.path.join(out_dir, "preprocess_summary.json"), meta)
    print("Preprocess done ->", out_dir)

def cmd_train(args):
    out_dir = make_out_dir(args.out_dir, args.exp_name)
    script = os.path.join(THIS_DIR, "train_pose.py")
    cmd = [sys.executable, script]
    print(">> Running:", " ".join(cmd))
    ret = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if ret != 0:
        sys.exit(ret)
    params = vars(args); params["resolved_out_dir"] = out_dir
    save_json(os.path.join(out_dir, "train_params.json"), params)
    print("Train finished ->", out_dir)

def cmd_predict(args):
    out_dir = make_out_dir(args.out_dir, args.exp_name)
    script = os.path.join(os.path.dirname(__file__), "predict_v2.py")
    print(">> Running:", " ".join([sys.executable, script]))

    env = os.environ.copy()
    env["CSV_PATH"] = os.path.abspath(args.video_path or args.csv_dir)
    env["MODEL_PATH"] = os.path.join(PROJECT_ROOT, "out", "best_model.keras")

    ret = subprocess.call([sys.executable, script], cwd=PROJECT_ROOT, env=env)
    if ret != 0:
        sys.exit(ret)

    params = {
        "csv_dir": args.csv_dir,
        "video_path": args.video_path,
        "overlay": args.overlay,
        "show_window": args.show_window,
        "resolved_out_dir": out_dir,
    }
    save_json(os.path.join(out_dir, "predict_params.json"), params)
    print("Predict finished ->", out_dir)


def build_parser():
    p = argparse.ArgumentParser(description="Yoga Pipeline CLI (preprocess/train/predict)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # preprocess
    sp = sub.add_parser("preprocess", help="Tiền xử lý dữ liệu")
    sp.add_argument("--csv-dir", default="data/csv")
    sp.add_argument("--out-dir", default="out")
    sp.add_argument("--exp-name", default="exp")
    sp.set_defaults(func=cmd_preprocess)

    # train
    sp = sub.add_parser("train", help="Huấn luyện (gọi train_v3.py)")
    sp.add_argument("--csv-dir", default="data/csv")
    sp.add_argument("--out-dir", default="out")
    sp.add_argument("--exp-name", default="exp")
    sp.add_argument("--seq-len", type=int, default=30)
    sp.add_argument("--epochs", type=int, default=50)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.set_defaults(func=cmd_train)

    # predict
    sp = sub.add_parser("predict", help="Dự đoán (gọi predict_v2.py)")
    sp.add_argument("--csv-dir", default="data/csv")
    sp.add_argument("--out-dir", default="out")
    sp.add_argument("--exp-name", default="exp")
    sp.add_argument("--video-path", type=str, default=None)
    sp.add_argument("--overlay", action="store_true")
    sp.add_argument("--show-window", action="store_true")
    sp.set_defaults(func=cmd_predict)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
