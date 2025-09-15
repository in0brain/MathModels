# -*- coding: utf-8 -*-
"""
KMeans 聚类算法封装：
- 使用 sklearn.cluster.KMeans
- 支持 silhouette/inertia 评估
- 可视化：二维散点图、肘部法则曲线
"""

from typing import Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.core import io, viz

TASK = "clu"
ALGO = "KMeans"


def build(cfg: Dict[str, Any]):
    """构建模型对象（sklearn.KMeans）"""
    params = cfg["model"].get("params", {})
    return SKKMeans(**params, random_state=cfg.get("seed", 42))


def fit(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """训练并评估聚类模型"""
    base = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)

    # === 特征选择 ===
    feats = cfg["dataset"].get("features") or df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feats].values

    # === 标准化 ===
    if cfg["dataset"].get("standardize", True):
        X = StandardScaler().fit_transform(X)

    # === 训练 ===
    y_pred = model.fit_predict(X)

    # === 评估指标 ===
    metrics_res = {}
    if "silhouette" in cfg.get("eval", {}).get("metrics", []):
        metrics_res["silhouette"] = float(silhouette_score(X, y_pred)) if len(set(y_pred)) > 1 else -1
    if "inertia" in cfg.get("eval", {}).get("metrics", []):
        metrics_res["inertia"] = float(model.inertia_)

    # === 保存结果 ===
    out_df = df.copy()
    out_df["cluster"] = y_pred
    pred_path = io.out_path_predictions(base, ALGO, f"{tag}_preds.csv")
    io.save_csv(out_df, pred_path)

    # === 可视化 ===
    figs = []
    if cfg.get("viz", {}).get("enabled", True):
        dpi = cfg["viz"].get("dpi", 160)
        plots = cfg["viz"].get("plots", {})
        if plots.get("scatter", True) and X.shape[1] >= 2:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab10", alpha=0.7)
            ax.set_xlabel(feats[0]); ax.set_ylabel(feats[1])
            ax.set_title("KMeans Scatter")
            scatter_path = os.path.join(base, "figs", "clu", f"{tag}_scatter.png")
            io.ensure_dir(scatter_path)
            fig.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figs.append(scatter_path)

        if plots.get("elbow", True):
            inertias = []
            k_range = range(1, min(10, len(X)))
            for k in k_range:
                km = SKKMeans(n_clusters=k, n_init=5, random_state=cfg.get("seed", 42))
                km.fit(X)
                inertias.append(km.inertia_)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(list(k_range), inertias, "o-")
            ax.set_xlabel("k"); ax.set_ylabel("SSE (Inertia)")
            ax.set_title("Elbow Method")
            elbow_path = os.path.join(base, "figs", "clu", f"{tag}_elbow.png")
            io.ensure_dir(elbow_path)
            fig.savefig(elbow_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figs.append(elbow_path)

    # === 保存报告 ===
    rep_path = os.path.join(base, "reports", f"{tag}_metrics.json")
    io.save_json({"metrics": metrics_res, "n_samples": len(df)}, rep_path)

    return {
        "metrics": metrics_res,
        "artifacts": {
            "predictions_csv": pred_path,
            "figs": figs,
            "report_path": rep_path
        }
    }


def inference(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """对新数据打簇标签"""
    base = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO) + "_infer"

    feats = cfg["dataset"].get("features") or df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feats].values
    if cfg["dataset"].get("standardize", True):
        X = StandardScaler().fit_transform(X)

    y_pred = model.predict(X)
    out_df = df.copy()
    out_df["cluster"] = y_pred

    pred_path = io.out_path_predictions(base, ALGO, f"{tag}_infer_preds.csv")
    io.save_csv(out_df, pred_path)
    return {"predictions_csv": pred_path}
