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
