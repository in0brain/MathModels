# src/models/clf/XGBoost/build.py
from typing import Dict, Any, List, Tuple
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.core import io, viz, metrics

# build.py:
# 数据筛选: 明确只使用 domain == 'source' 的数据进行训练。
# 特征选择: 自动选择所有由上一步生成的时域 (td_)、频域 (fd_) 和包络 (env_) 特征。
# 标签编码: 使用 sklearn.preprocessing.LabelEncoder 将故障类型（如 "Normal", "Ball"）转换成模型可以理解的数字（0, 1, 2...）。
# 防泄漏划分: 严格按照方案要求，使用 StratifiedGroupKFold 进行数据划分。groups=source_df["original_file"] 是关键，它确保了来自同一个原始 .mat 文件的所有数据窗口要么都在训练集，要么都在验证集，有效防止了数据泄漏。
# 产出物: 保存模型、预测结果、评估报告以及一个可视化的混淆矩阵，直观地展示分类效果。
#
# params.yaml:
# dataset.path 直接指向我们之前生成的 features.parquet 文件。
# model.params 中设置了与您方案中相似的 XGBoost 超参数，并指定了多分类的目标函数 multi:softprob。
# split.n_splits 定义了交叉验证的折数。代码默认使用第一折作为验证集。

TASK = "clf"
ALGO = "XGBoost"

def build(cfg: Dict[str, Any]):
    """初始化XGBoost分类器模型"""
    params = cfg.get("model", {}).get("params", {})
    params.setdefault("seed", cfg.get("seed", 42))
    return XGBClassifier(**params)

def fit(model: XGBClassifier, df: pd.DataFrame, cfg: Dict[str, Any]):
    """训练、评估并保存XGBoost分类模型"""
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)
    target_col = cfg["dataset"]["target"]

    # 1. 筛选源域数据并准备特征和标签
    source_df = df[df["domain"] == "source"].copy()
    # 检查分割数据集不为空
    if source_df.empty:
        raise ValueError(
            f"在输入的特征文件 (features.parquet) 中没有找到源域数据。\n"
            f"加载的数据表形状为 {df.shape}, 但其中没有任何行的 'domain' 列是 'source'。\n"
            f"请重新运行数据加载和特征提取步骤，并确保YAML文件中的 'source_dir' 路径是正确的。"
        )

    initial_rows = len(source_df)
    source_df.dropna(subset=[target_col], inplace=True)
    rows_after_dropna = len(source_df)
    if initial_rows > rows_after_dropna:
        print(
            f"[XGBoost Fit] Dropped {initial_rows - rows_after_dropna} rows with missing target label ('{target_col}').")
    # 确定特征列 (所有td_, fd_, env_开头的列)
    feature_cols = [col for col in source_df.columns if col.startswith(('td_', 'fd_', 'env_'))]
    X = source_df[feature_cols]
    y_raw = source_df[target_col]

    # 标签编码 (将字符串标签如 'Normal', 'Ball' 转为 0, 1, 2...)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 保存标签编码器，推理时需要
    encoder_path = os.path.join(base_dir, "models", f"{tag}_label_encoder.pkl")
    io.ensure_dir(encoder_path)
    joblib.dump(le, encoder_path)

    # 2. 严格防泄漏的数据划分 (StratifiedGroupKFold)
    # 按 'original_file' 分组，确保来自同一文件的窗口不会同时出现在训练集和验证集
    # 同时保持各类别在划分中的比例
    splitter = StratifiedGroupKFold(n_splits=cfg["split"]["n_splits"], shuffle=True, random_state=cfg.get("seed", 42))
    train_idx, val_idx = next(splitter.split(X, y, groups=source_df["original_file"]))

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    y_val_raw = y_raw.iloc[val_idx] # 用于保存结果

    # 3. 训练模型
    print(f"[XGBoost Fit] Training on {len(X_tr)} samples, validating on {len(X_val)} samples.")
    model.fit(X_tr, y_tr)

    # 4. 预测与评估
    y_pred = model.predict(X_val)
    proba = model.predict_proba(X_val)

    # 使用 src.core.metrics 中的函数进行评估
    eval_metrics = cfg.get("eval", {}).get("metrics", ["ACC", "F1"])
    res = metrics.evaluate_classification(y_val, y_pred, proba, le.classes_, metrics=tuple(eval_metrics))
    print(f"[XGBoost Fit] Validation Metrics: {res}")

    # 5. 保存产出物
    # 保存预测结果
    out_df = pd.DataFrame({
        "true_label": y_val_raw,
        "pred_label": le.inverse_transform(y_pred)
    })
    for i, class_name in enumerate(le.classes_):
        out_df[f"proba_{class_name}"] = proba[:, i]
    pred_path = io.out_path_predictions(base_dir, ALGO, f"{tag}_preds.csv")
    io.save_csv(out_df, pred_path)

    # 保存模型
    model_path = os.path.join(base_dir, "models", f"{tag}.pkl")
    io.save_model(model, model_path)

    # 保存报告
    report_path = os.path.join(base_dir, "reports", f"{tag}_metrics.json")
    io.save_json({"metrics": res, "n_train": len(X_tr), "n_test": len(X_val)}, report_path)

    # 可视化 (混淆矩阵)
    fig_paths = []
    if cfg.get("viz", {}).get("enabled", True):
        dpi = cfg["viz"].get("dpi", 160)
        if cfg["viz"].get("plots", {}).get("cm", True):
            cm_path = os.path.join(base_dir, "figs", "clf", f"{tag}_cm.png")
            # viz.plot_confusion_matrix(y_val_raw, le.inverse_transform(y_pred), le.classes_, cm_path, dpi=dpi)
            if cfg.get("viz", {}).get("enabled", True):
                dpi = cfg["viz"].get("dpi", 160)
                if cfg["viz"].get("plots", {}).get("cm", True):
                    cm_path = os.path.join(base_dir, "figs", "clf", f"{tag}_cm.png")
                    # --- MODIFIED LINE ---
                    # Set normalize=True to get percentages, like in the thesis.
                    viz.plot_confusion_matrix(
                        y_val_raw,
                        le.inverse_transform(y_pred),
                        le.classes_,
                        cm_path,
                        dpi=dpi,
                        normalize=True,
                        cmap='YlGnBu'  # A colormap similar to the thesis
                    )
                    fig_paths.append(cm_path)
            fig_paths.append(cm_path)

    return {
        "metrics": res,
        "artifacts": {
            "predictions_csv": pred_path,
            "model_path": model_path,
            "label_encoder_path": encoder_path,
            "report_path": report_path,
            "figs": fig_paths
        }
    }