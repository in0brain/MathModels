# -*- coding: utf-8 -*-
import argparse, os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.core import io  # 用你的 ensure_dir/save_csv 等:contentReference[oaicite:11]{index=11}

sns.set_theme(style="whitegrid")

def run(path: str, out_dir: str, target: str = None, topn: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path)
    # 基本统计
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(out_dir, "describe.csv"), encoding="utf-8")

    # 缺失率
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(os.path.join(out_dir, "missing_ratio.csv"), encoding="utf-8")

    # 数值分布（前 topn）
    num_cols = df.select_dtypes(include="number").columns.tolist()[:topn]
    for c in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c], ax=ax, kde=True)
        ax.set_title(f"Histogram: {c}")
        fig.savefig(os.path.join(out_dir, f"hist_{c}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 相关性热力图
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(min(12, 0.6*len(num_cols)+4), min(10, 0.6*len(num_cols)+3)))
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0)
        ax.set_title("Correlation Heatmap")
        fig.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 简单报告（Markdown）
    md = [
        f"# EDA Report",
        f"- Source: `{path}`",
        f"- Rows: {len(df)}, Cols: {len(df.columns)}",
        f"- Target: `{target}`" if target else "- Target: (not set)",
        f"- Saved stats in `describe.csv` and `missing_ratio.csv`",
        f"- Plots: histograms for first {len(num_cols)} numeric cols, correlation heatmap (if applicable)."
    ]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[eda] done -> {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()
    run(args.path, args.outdir, args.target, args.topn)
