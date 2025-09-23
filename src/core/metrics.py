# -*- coding: utf-8 -*-
"""
metrics.py
---------
统一的指标计算模块：
- evaluate_markov     : 马尔可夫链（离散状态预测）的准确率
- evaluate_regression : 回归任务的 MAE / R2
- evaluate_classification : 分类任务的 ACC / F1 / ROC_AUC
"""
import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def evaluate_markov(y_true, y_pred, metrics=("acc",)):
    out = {}
    if "acc" in metrics:
        out["acc"] = accuracy(y_true, y_pred)
    return out


def evaluate_regression(y_true, y_pred, metrics=("MAE", "R2")):
    out = {}
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if "MAE" in metrics:
        out["MAE"] = float(mean_absolute_error(yt, yp))
    if "R2" in metrics:
        out["R2"] = float(r2_score(yt, yp))
    return out


def evaluate_classification(y_true, y_pred, proba=None, classes=None, metrics=("ACC", "F1", "ROC_AUC")):
    """
    分类任务指标计算（兼容二分类与多分类）。
    此版本已修正多分类ROC_AUC在验证集类别不全时的计算问题。
    """
    out = {}

    if "ACC" in metrics:
        out["ACC"] = float(accuracy_score(y_true, y_pred))

    if "F1" in metrics:
        out["F1"] = float(f1_score(y_true, y_pred, average="macro"))

    if "ROC_AUC" in metrics and proba is not None:
        proba = np.asarray(proba)
        n_classes = len(np.unique(y_true))

        # 二分类逻辑
        if proba.ndim == 1 or proba.shape[1] == 1:
            out["ROC_AUC"] = float(roc_auc_score(y_true, proba))
        # 多分类逻辑
        elif proba.shape[1] > 1:
            # 【核心修正】
            # 检查验证集中的类别数是否少于总类别数。
            # 如果是，我们需要明确告诉roc_auc_score所有可能的类别标签。
            if classes is not None and len(np.unique(y_true)) < len(classes):
                # 获取所有可能的类别标签的整数索引 [0, 1, 2, ...]
                all_labels = np.arange(len(classes))
                try:
                    out["ROC_AUC"] = float(
                        roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=all_labels))
                except ValueError as e:
                    print(
                        f"Warning: ROC_AUC calculation failed even with all labels. This might happen if proba columns don't match all_labels. Error: {e}")
                    out["ROC_AUC"] = None  # Or np.nan
            else:
                # 如果验证集类别齐全，正常计算
                out["ROC_AUC"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        else:
            out["ROC_AUC"] = None  # Or np.nan

    return out

def gaussian_kernel(x, y, sigma=1.0):
    # 计算高斯核
    beta = 1. / (2. * sigma)
    dist = torch.cdist(x, y)
    return torch.exp(-beta * dist.pow(2))

def mmd_loss(source_features, target_features, sigma=1.0):
    # 计算MMD损失
    xx = gaussian_kernel(source_features, source_features, sigma).mean()
    yy = gaussian_kernel(target_features, target_features, sigma).mean()
    xy = gaussian_kernel(source_features, target_features, sigma).mean()
    return xx + yy - 2 * xy