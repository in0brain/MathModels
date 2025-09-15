# -*- coding: utf-8 -*-
"""
可视化：状态转移矩阵热力图、序列对比图；支持保存 PNG
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_transition_heatmap(P: np.ndarray, labels, out_png: str, dpi=160):
    # 中文注释：P 为 n×n 概率矩阵，labels 为状态名列表
    fig, ax = plt.subplots()
    im = ax.imshow(P, aspect="auto", origin="upper")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("Transition Matrix (Heatmap)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _ensure_dir(out_png); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out_png

def plot_sequence_compare(y_true, y_pred, out_png: str, dpi=160):
    # 中文注释：将离散状态映射为整数方便画图
    uniq = sorted(set(list(y_true) + list(y_pred)))
    idx = {s: i for i, s in enumerate(uniq)}
    yt = [idx[s] for s in y_true]
    yp = [idx[s] for s in y_pred]

    fig, ax = plt.subplots()
    ax.plot(yt, label="true", linewidth=1)
    ax.plot(yp, label="pred", linewidth=1)
    ax.set_title("Sequence (True vs Pred)")
    ax.set_xlabel("t"); ax.set_ylabel("state_id")
    ax.legend()
    _ensure_dir(out_png); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out_png

# -*- coding: utf-8 -*-
import numpy as np

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# —— 现有：时序热力图/序列对比（略） ——

# === 新增：特征重要性 ===
def plot_feature_importance(importances, feature_names, out_png, dpi=160, top=30):
    imp = np.asarray(importances)
    idx = np.argsort(imp)[::-1][:top]
    names = np.array(feature_names)[idx]
    vals = imp[idx]
    fig, ax = plt.subplots()
    ax.barh(range(len(vals)), vals[::-1])
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_title("Feature Importance")
    _ensure_dir(out_png); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out_png

# === 新增：残差直方图 ===
def plot_residuals(y_true, y_pred, out_png, dpi=160, bins=30):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.hist(res, bins=bins)
    ax.set_title("Residuals")
    _ensure_dir(out_png); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out_png

# === 新增：预测 vs 真实 散点 ===
def plot_pred_scatter(y_true, y_pred, out_png, dpi=160):
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(yt, yp, s=10)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    ax.set_title("Prediction vs True")
    _ensure_dir(out_png); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out_png
