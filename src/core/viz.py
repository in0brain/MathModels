# -*- coding: utf-8 -*-
"""
可视化模块
---------
功能：
1. 状态转移矩阵热力图
2. 序列对比图（真实 vs 预测）
3. 回归可视化（特征重要性、残差直方图、预测散点）
4. 分类可视化（ROC 曲线、PR 曲线、混淆矩阵）
5. (新增) 可解释性可视化（SHAP 瀑布图）
6. (新增) 迁移学习可视化（t-SNE 分布）
所有函数均支持保存为 PNG 文件。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set_theme(style="whitegrid")

# 工具函数：确保目录存在
def _ensure_dir(path: str):
    # 取输出路径的父目录，如果不存在则创建
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# ========== 1) 时间序列可视化 ==========

def plot_waveform_grid(signals: dict, out_png: str, dpi=160):
    """
    (新增) 绘制2x2的信号时域波形图，风格与刘嘉欣论文对齐。

    参数:
        signals (dict): 一个字典，键为子图标题(如 "正常", "外圈故障")，值为一维信号数据 (numpy array)。
                       最多支持4个信号。
        out_png (str): 输出PNG文件路径。
        dpi (int): 图像分辨率。
    """
    if len(signals) > 4:
        print("Warning: Waveform grid plot supports a maximum of 4 signals. Taking the first 4.")
        signals = {k: v for i, (k, v) in enumerate(signals.items()) if i < 4}

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=dpi)
    axes = axes.flatten()  # 将2x2的axes数组展平为一维

    for i, (title, data) in enumerate(signals.items()):
        ax = axes[i]
        ax.plot(data)
        ax.set_title(title)
        ax.set_xlabel("采样点")
        ax.set_ylabel("幅值 (m/s²)")
        ax.autoscale(enable=True, axis='x', tight=True) # x轴紧凑

    # 隐藏多余的子图
    for i in range(len(signals), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"时域波形图已保存至: {out_png}")
    return out_png

def plot_transition_heatmap(P: np.ndarray, labels, out_png: str, dpi=160):
    fig, ax = plt.subplots()
    im = ax.imshow(P, aspect="auto", origin="upper")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("Transition Matrix (Heatmap)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_sequence_compare(y_true, y_pred, out_png: str, dpi=160):
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
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ========== 2) 回归任务可视化 ==========

def plot_feature_importance(importances, feature_names, out_png, dpi=160, top=30):
    imp = np.asarray(importances)
    idx = np.argsort(imp)[::-1][:top]
    names = np.array(feature_names)[idx]
    vals = imp[idx]
    fig, ax = plt.subplots()
    ax.barh(range(len(vals)), vals[::-1])
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_title("Feature Importance")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_residuals(y_true, y_pred, out_png, dpi=160, bins=30):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.hist(res, bins=bins)
    ax.set_title("Residuals")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_pred_scatter(y_true, y_pred, out_png, dpi=160):
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(yt, yp, s=10)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    ax.set_title("Prediction vs True")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ========== 3) 分类任务可视化 ==========

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def plot_roc(y_true, proba, classes, out_png, dpi=160):
    fig, ax = plt.subplots()
    y_true = np.asarray(y_true)
    if proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:,1])
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    else:
        for i, c in enumerate(classes):
            y_bin = (y_true == i).astype(int) # Assuming y_true is encoded
            fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
            ax.plot(fpr, tpr, label=f"class {c} AUC={auc(fpr,tpr):.3f}")
    ax.plot([0,1],[0,1],"--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.legend(fontsize=8)
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_pr(y_true, proba, classes, out_png, dpi=160):
    fig, ax = plt.subplots()
    y_true = np.asarray(y_true)
    if proba.shape[1] == 2:
        prec, rec, _ = precision_recall_curve(y_true, proba[:,1])
        ax.plot(rec, prec)
    else:
        for i, c in enumerate(classes):
            y_bin = (y_true == i).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
            ax.plot(rec, prec, label=f"class {c}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curve")
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png



def plot_confusion_matrix(y_true, y_pred, classes, out_png, dpi=160,
                          annot_size=14, cmap="YlGnBu", normalize=False):
    """
    (已修改) 混淆矩阵. 新增 normalize 参数以匹配论文风格。
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # --- 新增：归一化逻辑 ---
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # 格式化为两位小数
    else:
        fmt = 'd'  # 保持为整数

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    sns.heatmap(cm, annot=True, fmt=fmt, ax=ax, cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": annot_size})

    ax.set_xlabel("Predicted") # 预测标签
    ax.set_ylabel(" True") #真实标签
    ax.set_title("Confusion Matrix", fontsize=14, weight="bold") #混淆矩阵
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png

def plot_multi_roc_compare(results: dict, out_png: str, dpi=160):
    """
    多模型 ROC 对比
    results: {"ModelA": (y_true, proba_2col or proba_1col), "ModelB": ...}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, proba) in results.items():
        proba = np.asarray(proba)
        if proba.ndim == 1:  # 一列：正类概率
            fpr, tpr, _ = roc_curve(y_true, proba)
        else:                # 二列：取第二列为正类概率
            fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0,1],[0,1],"--", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve Comparison")
    ax.legend()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_reg_compare(rows, metric: str, out_png: str, dpi=160):
    models = [r['model'] for r in rows]
    values = [r.get(metric) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, values)
    ax.set_title(f"Model Comparison - {metric}")
    ax.set_ylabel(metric)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ========== 5) 可解释性可视化 ==========
def plot_shap_waterfall(shap_values_instance, out_png: str, max_display=15, dpi=160):
    """
    (新增) 绘制并保存单个样本的SHAP瀑布图。
    """
    plt.figure()
    shap.plots.waterfall(shap_values_instance, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_png

def plot_shap_summary_bar(shap_values, features_df, class_names, out_png: str, dpi=160):
    """(新增) 绘制并保存全局SHAP摘要图（条形图样式）。"""
    plt.figure()
    shap.summary_plot(shap_values, features_df, class_names=class_names, show=False, plot_type='bar')
    fig = plt.gcf()
    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_png

def plot_shap_waterfall_new(explanation_object, out_png: str, max_display=15, dpi=160):
    """绘制并保存单个样本的SHAP瀑布图，接收一个SHAP Explanation对象。"""
    plt.figure()
    shap.plots.waterfall(explanation_object, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_png

# ========== 6) 迁移学习可视化 ==========
import matplotlib.patches as mpatches  # 保留以兼容旧代码
from matplotlib.lines import Line2D    # 新增：用于自定义图例句柄
from sklearn.manifold import TSNE

from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


def plot_tsne_after_adaptation(source_latent: np.ndarray,
                               target_latent: np.ndarray,
                               source_labels: np.ndarray,
                               target_labels: np.ndarray,
                               out_png: str,
                               title: str,
                               dpi=300):
    """
    领域自适应后的 t-SNE 可视化：
    - 源域：实心圆（与真实标签同色），白色细描边
    - 目标域：实心三角形（与预测标签同色），白色细描边
    - 源/目标共用同一调色盘；一次性在拼接特征上拟合 t-SNE
    """
    print(f"Generating t-SNE plot for adapted space: {title}...")

    # 1) 共享嵌入
    combined_data = np.vstack((source_latent, target_latent))
    perplexity = float(max(5, min(40, len(combined_data) - 1)))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', verbose=0)
    Z_all = tsne.fit_transform(combined_data)
    n_src = len(source_latent)
    Zs, Zt = Z_all[:n_src], Z_all[n_src:]

    # 2) 统一颜色映射（同一类=同一颜色）
    all_labels = np.concatenate([source_labels, target_labels])
    uniq = np.unique(all_labels)
    cmap = plt.cm.get_cmap("tab20" if len(uniq) > 10 else "tab10", len(uniq))
    color_of = {lab: cmap(i) for i, lab in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # 源域：实心圆（下层）
    for lab in uniq:
        idx = (source_labels == lab)
        if np.any(idx):
            ax.scatter(
                Zs[idx, 0], Zs[idx, 1],
                s=24, marker="o",
                facecolors=[color_of[lab]], edgecolors="white",
                linewidths=0.6, alpha=0.95, zorder=2
            )

    # 目标域：实心三角形（上层）
    for lab in uniq:
        idx = (target_labels == lab)
        if np.any(idx):
            ax.scatter(
                Zt[idx, 0], Zt[idx, 1],
                s=36, marker="^",
                facecolors=[color_of[lab]], edgecolors="white",
                linewidths=0.6, alpha=0.98, zorder=3
            )

    # 清晰图例（代理句柄）
    legend_elems = [
        Line2D([0], [0], marker='o', color='white',
               markerfacecolor='gray', markeredgecolor='white',
               markeredgewidth=0.6, markersize=8, linewidth=0,
               label='Source Domain (true label)'),
        Line2D([0], [0], marker='^', color='white',
               markerfacecolor='gray', markeredgecolor='white',
               markeredgewidth=0.6, markersize=8, linewidth=0,
               label='Target Domain (pred label)'),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.25)

    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


# [建议新增] in src/core/viz.py at the end of the file

# def plot_tsne_by_class(source_latent: np.ndarray,
#                        target_latent: np.ndarray,
#                        source_labels: np.ndarray,
#                        target_labels: np.ndarray,
#                        class_names: list,
#                        out_png: str,
#                        title: str,
#                        dpi=160):
#     """
#     (修正版) 绘制按“故障类别”着色的t-SNE分布图。
#     - 颜色代表故障类别。
#     - 标记区分源域 (实心圆) 和目标域 (实心三角)。
#     - 使用面向对象的API，更稳健。
#     """
#     print(f"Generating class-colored t-SNE plot: {title}...")
#
#     combined_data = np.vstack((source_latent, target_latent))
#     perplexity = min(30.0, len(combined_data) - 1)
#
#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
#     tsne_results = tsne.fit_transform(combined_data)
#
#     num_source = len(source_latent)
#     Zs, Zt = tsne_results[:num_source], tsne_results[num_source:]
#
#     unique_class_indices = np.unique(np.hstack((source_labels, target_labels)))
#     colors = plt.cm.get_cmap('tab10', len(class_names))
#
#     # --- 核心改动：使用 fig, ax 对象 ---
#     fig, ax = plt.subplots(figsize=(12, 10))
#
#     # 绘制源域数据点
#     ax.scatter(Zs[:, 0], Zs[:, 1],
#                c=source_labels, cmap='tab10', marker='o', alpha=0.7,
#                s=30, linewidths=0.5, edgecolors='w', label="Source Domain")
#
#     # 绘制目标域数据点
#     ax.scatter(Zt[:, 0], Zt[:, 1],
#                c=target_labels, cmap='tab10', marker='^', alpha=0.9,
#                s=40, linewidths=0.5, edgecolors='w', label="Target Domain")
#
#     ax.set_title(title, fontsize=16)
#     ax.set_xlabel("t-SNE dimension 1")
#     ax.set_ylabel("t-SNE dimension 2")
#
#     # 创建一个清晰的图例
#     handles = []
#     for i, name in enumerate(class_names):
#         handles.append(mpatches.Patch(color=colors(i), label=name))
#
#     source_handle = plt.Line2D([0], [0], marker='o', color='w', label='Source Domain',
#                                markerfacecolor='grey', markersize=10)
#     target_handle = plt.Line2D([0], [0], marker='^', color='w', label='Target Domain',
#                                markerfacecolor='grey', markersize=10)
#     handles.extend([source_handle, target_handle])
#
#     ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes & Domains")
#
#     _ensure_dir(out_png)
#     fig.tight_layout(rect=[0, 0, 0.85, 1])
#     fig.savefig(out_png, dpi=dpi)
#     plt.close(fig)  # 确保关闭图形对象
#     return out_png

def plot_tsne_distribution(source_data: np.ndarray, target_data: np.ndarray, out_png: str, title: str, dpi=300):
    combined_data = np.vstack((source_data, target_data))
    perplexity = float(max(5, min(30, len(combined_data) - 1)))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', verbose=0, learning_rate='auto')
    tsne_results = tsne.fit_transform(combined_data)
    n_src = len(source_data)
    Zs, Zt = tsne_results[:n_src], tsne_results[n_src:]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.scatter(Zs[:, 0], Zs[:, 1], c='blue', alpha=0.5, label="source_domain")
    ax.scatter(Zt[:, 0], Zt[:, 1], c='red', alpha=0.5, label="target_domain")
    ax.set_title(title, fontsize=16)
    ax.legend()
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

def plot_tsne_by_class(source_latent: np.ndarray,
                       target_latent: np.ndarray,
                       source_labels: np.ndarray,
                       target_labels: np.ndarray,
                       class_names: list,
                       out_png: str,
                       title: str,
                       dpi=160):
    print(f"Generating class-colored t-SNE plot: {title}...")
    combined_data = np.vstack((source_latent, target_latent))
    perplexity = min(30.0, len(combined_data) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(combined_data)
    num_source = len(source_latent)
    Zs, Zt = tsne_results[:num_source], tsne_results[num_source:]
    unique_class_indices = np.unique(np.hstack((source_labels, target_labels)))
    # 使用 'tab10' colormap 匹配论文风格
    colors = plt.cm.get_cmap('tab10', len(class_names))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(Zs[:, 0], Zs[:, 1], c=colors(source_labels), marker='o', alpha=0.7, s=30, linewidths=0.5, edgecolors='w')
    ax.scatter(Zt[:, 0], Zt[:, 1], c=colors(target_labels), marker='^', alpha=0.9, s=40, linewidths=0.5, edgecolors='w')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    handles = [mpatches.Patch(color=colors(i), label=name) for i, name in enumerate(class_names)]
    source_handle = Line2D([0], [0], marker='o', color='w', label='Source Domain', markerfacecolor='grey', markersize=10)
    target_handle = Line2D([0], [0], marker='^', color='w', label='Target Domain', markerfacecolor='grey', markersize=10)
    handles.extend([source_handle, target_handle])
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes & Domains")
    _ensure_dir(out_png)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png