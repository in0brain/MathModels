# -*- coding: utf-8 -*-
"""
可视化模块
---------
功能：
1. 状态转移矩阵热力图
2. 序列对比图（真实 vs 预测）
3. 回归可视化（特征重要性、残差直方图、预测散点）
4. 分类可视化（ROC 曲线、PR 曲线、混淆矩阵）
所有函数均支持保存为 PNG 文件。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# 工具函数：确保目录存在
def _ensure_dir(path: str):
    # 取输出路径的父目录，如果不存在则创建
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ========== 1) 时间序列可视化 ==========

def plot_transition_heatmap(P: np.ndarray, labels, out_png: str, dpi=160):
    """
    状态转移矩阵热力图
    P: n×n 概率矩阵
    labels: 状态名列表
    out_png: 输出文件路径
    """
    fig, ax = plt.subplots()
    # 显示概率矩阵为热力图
    im = ax.imshow(P, aspect="auto", origin="upper")
    # 设置坐标轴标签
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("Transition Matrix (Heatmap)")
    # 添加颜色条
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # 保存并关闭
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_sequence_compare(y_true, y_pred, out_png: str, dpi=160):
    """
    序列对比图（真实 vs 预测）
    将离散状态映射为整数，方便绘制折线对比
    """
    uniq = sorted(set(list(y_true) + list(y_pred)))   # 全部可能状态
    idx = {s: i for i, s in enumerate(uniq)}         # 状态->整数映射
    yt = [idx[s] for s in y_true]                    # 真实序列映射
    yp = [idx[s] for s in y_pred]                    # 预测序列映射

    fig, ax = plt.subplots()
    ax.plot(yt, label="true", linewidth=1)
    ax.plot(yp, label="pred", linewidth=1)
    ax.set_title("Sequence (True vs Pred)")
    ax.set_xlabel("t"); ax.set_ylabel("state_id")
    ax.legend()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ========== 2) 回归任务可视化 ==========

def plot_feature_importance(importances, feature_names, out_png, dpi=160, top=30):
    """
    特征重要性条形图
    importances: 特征重要性数组
    feature_names: 特征名列表
    top: 只取前 top 个
    """
    imp = np.asarray(importances)
    idx = np.argsort(imp)[::-1][:top]     # 排序取前 top
    names = np.array(feature_names)[idx]  # 对应特征名
    vals = imp[idx]                       # 对应重要性
    fig, ax = plt.subplots()
    ax.barh(range(len(vals)), vals[::-1])  # 反转画水平条形图
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_title("Feature Importance")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_residuals(y_true, y_pred, out_png, dpi=160, bins=30):
    """
    残差直方图
    y_true - y_pred
    """
    res = np.asarray(y_true) - np.asarray(y_pred)   # 计算残差
    fig, ax = plt.subplots()
    ax.hist(res, bins=bins)                         # 绘制直方图
    ax.set_title("Residuals")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_pred_scatter(y_true, y_pred, out_png, dpi=160):
    """
    预测值 vs 真实值散点图
    """
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(yt, yp, s=10)                        # 散点
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]  # 对角线范围
    ax.plot(lims, lims, linestyle="--")             # y=x 参考线
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    ax.set_title("Prediction vs True")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ========== 3) 分类任务可视化 ==========

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def plot_roc(y_true, proba, classes, out_png, dpi=160):
    """
    ROC 曲线
    - 二分类：绘制单条曲线
    - 多分类：一对多绘制多条
    """
    fig, ax = plt.subplots()
    y_true = np.asarray(y_true)
    if proba.shape[1] == 2:   # 二分类情况
        fpr, tpr, _ = roc_curve(y_true, proba[:,1])
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    else:                     # 多分类情况
        for i, c in enumerate(classes):
            y_bin = (y_true == c).astype(int)       # 当前类 vs 其他类
            fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
            ax.plot(fpr, tpr, label=f"class {c} AUC={auc(fpr,tpr):.3f}")
    ax.plot([0,1],[0,1],"--")                       # 随机分类参考线
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.legend(fontsize=8)
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_pr(y_true, proba, classes, out_png, dpi=160):
    """
    PR 曲线（精确率-召回率）
    """
    fig, ax = plt.subplots()
    y_true = np.asarray(y_true)
    if proba.shape[1] == 2:   # 二分类
        prec, rec, _ = precision_recall_curve(y_true, proba[:,1])
        ax.plot(rec, prec)
    else:                     # 多分类
        for i, c in enumerate(classes):
            y_bin = (y_true == c).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
            ax.plot(rec, prec, label=f"class {c}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curve")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_confusion_matrix(y_true, y_pred, classes, out_png, dpi=160):
    """
    混淆矩阵
    - 横轴：预测
    - 纵轴：真实
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)  # 计算混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    # 在每个格子中标数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png
