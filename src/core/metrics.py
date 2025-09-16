# -*- coding: utf-8 -*-  # 指定源码文件编码，避免中文注释乱码
"""
metrics.py
---------
统一的指标计算模块：
- evaluate_markov     : 马尔可夫链（离散状态预测）的准确率
- evaluate_regression : 回归任务的 MAE / R2
- evaluate_classification : 分类任务的 ACC / F1 / ROC_AUC
"""

# ===== 基础依赖 =====
import numpy as np  # 数值计算（数组、均值、比较等）

# 回归与分类常用指标从 sklearn 引入
from sklearn.metrics import (
    mean_absolute_error,  # MAE：平均绝对误差
    r2_score,             # R2：决定系数
    accuracy_score,       # ACC：分类准确率
    f1_score,             # F1：F1 分数
    roc_auc_score         # ROC_AUC：ROC 曲线下面积
)

# ===== 公用：离散准确率 =====
def accuracy(y_true, y_pred):
    """计算离散分类的准确率（= 预测与真实完全相等的比例）"""
    y_true = np.asarray(y_true)          # 保证输入为 numpy 数组
    y_pred = np.asarray(y_pred)          # 同上
    return float((y_true == y_pred).mean())  # 布尔数组取均值即为正确比例

# ===== 1) 马尔可夫链指标 =====
def evaluate_markov(y_true, y_pred, metrics=("acc",)):
    """
    根据请求的指标列表计算马尔可夫链任务的指标。
    参数：
      y_true  : 序列真实状态（可迭代）
      y_pred  : 序列预测状态（可迭代，长度与 y_true 对齐）
      metrics : 元组/列表，例如 ("acc",)
    返回：
      dict，例如 {"acc": 0.92}
    """
    out = {}                              # 用于收集各项指标
    if "acc" in metrics:                  # 如果请求了准确率
        out["acc"] = accuracy(y_true, y_pred)  # 计算并记录
    return out                             # 返回指标字典

# ===== 2) 回归指标 =====
def evaluate_regression(y_true, y_pred, metrics=("MAE", "R2")):
    """
    回归任务的常用指标。
    参数：
      y_true  : 真实值
      y_pred  : 预测值
      metrics : 需要计算的指标集合（大小写按此实现）
    返回：
      dict，例如 {"MAE": 1234.5, "R2": 0.87}
    """
    out = {}                               # 结果字典
    yt = np.asarray(y_true)                # 转 numpy 数组
    yp = np.asarray(y_pred)                # 转 numpy 数组
    if "MAE" in metrics:                   # 若需要 MAE
        out["MAE"] = float(mean_absolute_error(yt, yp))  # 计算 MAE
    if "R2" in metrics:                    # 若需要 R2
        out["R2"] = float(r2_score(yt, yp))              # 计算 R2
    return out

# ===== 3) 分类指标 =====
def evaluate_classification(y_true, y_pred, proba=None, classes=None, metrics=("ACC", "F1", "ROC_AUC")):
    """
    分类任务指标计算（兼容二分类与多分类）。
    参数：
      y_true  : 真实标签（形状 [N]）
      y_pred  : 预测标签（形状 [N]）
      proba   : 预测概率（可选）
                - 二分类：形状 [N] 或 [N,1]/[N,2]（正类概率在后一种需自行取列）
                - 多分类：形状 [N, C]，C 为类别数
      classes : 类别列表（可选，主要用于与你的外部代码对齐，不在此函数中强制使用）
      metrics : 需要计算的指标集合（"ACC"|"F1"|"ROC_AUC"）
    返回：
      dict，例如 {"ACC":0.91, "F1":0.88, "ROC_AUC":0.93}
    """
    out = {}                                # 结果字典

    # ---- ACC：准确率 ----
    if "ACC" in metrics:
        out["ACC"] = float(accuracy_score(y_true, y_pred))  # 直接用 sklearn 计算

    # ---- F1：宏平均（macro）----
    if "F1" in metrics:
        # 宏平均：各类别分别计算 F1 再取平均，类别不均衡时更稳健
        out["F1"] = float(f1_score(y_true, y_pred, average="macro"))

    # ---- ROC_AUC：需要概率 ----
    if "ROC_AUC" in metrics and proba is not None:
        proba = np.asarray(proba)                         # 转 numpy 数组
        # 二分类：若为一维概率（仅正类概率），直接用；若为二维需自行选取正类列
        if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
            out["ROC_AUC"] = float(roc_auc_score(y_true, proba))
        else:
            # 多分类：一对多（ovr）宏平均
            out["ROC_AUC"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))

    return out                              # 返回指标结果
