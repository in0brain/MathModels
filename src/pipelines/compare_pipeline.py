# -*- coding: utf-8 -*-
import argparse, yaml, json, os, csv, subprocess, sys

def run_one(cfg_path: str):
    # 调用你的训练管线（回归示例）
    print(f"[compare_pipeline] run -> {cfg_path}")
    code = subprocess.call([sys.executable, "-m", "src.pipelines.reg_pipeline", "--config", cfg_path])
    if code != 0:
        print(f"[compare_pipeline] warn: run failed for {cfg_path} (exit={code})")

def find_report_from_cfg(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    tag = (cfg.get("outputs", {}) or {}).get("tag")
    if not tag:
        # 兜底取 model.name 当 tag
        tag = (cfg.get("model", {}) or {}).get("name", "model")
    return f"outputs/reports/{tag}_metrics.json", tag

def main(list_yaml: str, out_csv: str, rerun: bool):
    y = yaml.safe_load(open(list_yaml, "r", encoding="utf-8"))
    cfgs = y.get("configs", [])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows = []
    for cfg_path in cfgs:
        report_path, tag = find_report_from_cfg(cfg_path)

        if rerun:
            run_one(cfg_path)

        if not os.path.exists(report_path):
            print(f"[compare_pipeline] warn: report not found -> {report_path}")
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics", {})
        rows.append({"model": tag, "MAE": m.get("MAE"), "RMSE": m.get("RMSE"), "R2": m.get("R2")})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "MAE", "RMSE", "R2"])
        w.writeheader()
        w.writerows(rows)

    print(f"[compare_pipeline] saved -> {out_csv}")
    # 加在保存 CSV 后
    try:
        from src.core.viz import plot_reg_compare
        png_dir = "outputs/plots/compare"
        os.makedirs(png_dir, exist_ok=True)
        # 读回 CSV -> rows
        import pandas as pd
        df = pd.read_csv(out_csv)
        rows = df.to_dict(orient="records")
        for metric in ["MAE", "RMSE", "R2"]:
            plot_reg_compare(rows, metric, os.path.join(png_dir, f"compare_{metric}.png"), dpi=160)
        print(f"[compare_pipeline] plots -> {png_dir}")
    except Exception as e:
        print("[compare_pipeline] warn: plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="YAML with {configs: [path1, path2, ...]}")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--rerun", action="store_true", help="re-run each config via reg_pipeline before collecting metrics")
    args = ap.parse_args()
    main(args.list, args.out, args.rerun)
