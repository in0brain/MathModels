# -*- coding: utf-8 -*-
"""
指标计算：此处为 MarkovChain 简单示例（准确率）
"""
import numpy as np

def accuracy(y_true, y_pred):
    # 中文注释：离散分类准确率
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def evaluate_markov(y_true, y_pred, metrics=("acc",)):
    # 中文注释：根据配置返回所需指标
    out = {}
    if "acc" in metrics:
        out["acc"] = accuracy(y_true, y_pred)
    return out
# -*- coding: utf-8 -*-
from sklearn.metrics import mean_absolute_error, r2_score

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def evaluate_markov(y_true, y_pred, metrics=("acc",)):
    out = {}
    if "acc" in metrics:
        out["acc"] = accuracy(y_true, y_pred)
    return out

# === 新增：回归指标 ===
def evaluate_regression(y_true, y_pred, metrics=("MAE","R2")):
    out = {}
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    if "MAE" in metrics:
        out["MAE"] = float(mean_absolute_error(yt, yp))
    if "R2" in metrics:
        out["R2"]  = float(r2_score(yt, yp))
    return out


# ====此处是神经网络====
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_classification(y_true, y_pred, proba=None, classes=None, metrics=("ACC","F1","ROC_AUC")):
    out = {}
    if "ACC" in metrics:
        out["ACC"] = float(accuracy_score(y_true, y_pred))
    if "F1" in metrics:
        # 二分类/多分类统一用宏平均（也可改 'weighted'）
        out["F1"] = float(f1_score(y_true, y_pred, average="macro"))
    if "ROC_AUC" in metrics and proba is not None:
        # 二分类：取正类概率；多类：一对多宏平均
        if proba.ndim == 1 or proba.shape[1] == 1:
            out["ROC_AUC"] = float(roc_auc_score(y_true, proba))
        else:
            out["ROC_AUC"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    return out
