# src/models/clf/SVM/build.py
from typing import Dict, Any
import pandas as pd
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from src.core import io, metrics, viz

TASK = "clf"
ALGO = "SVM"


def build(cfg: Dict[str, Any]):
    params = cfg.get("model", {}).get("params", {})
    params.setdefault("random_state", cfg.get("seed", 42))
    params.setdefault("probability", True)  # 必须开启才能获取概率
    return SVC(**params)


def fit(model: SVC, df: pd.DataFrame, cfg: Dict[str, Any]):
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)
    target_col = cfg["dataset"]["target"]

    source_df = df[df["domain"] == "source"].copy()
    source_df.dropna(subset=[target_col], inplace=True)

    # 动态根据配置文件中的feature_prefixes来选择特征列
    feature_cols = [c for c in source_df.columns if c.startswith(tuple(cfg["dataset"]["feature_prefixes"]))]
    X_raw = source_df[feature_cols]
    y_raw = source_df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # SVM对特征尺度敏感，必须进行标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    splitter = StratifiedGroupKFold(n_splits=cfg["split"]["n_splits"], shuffle=True, random_state=cfg.get("seed", 42))
    train_idx, val_idx = next(splitter.split(X, y, groups=source_df["original_file"]))

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # 保留原始的文本标签用于保存结果
    y_val_raw = y_raw.iloc[val_idx]

    print(f"[SVM Fit] Training on {len(X_tr)} samples, validating on {len(X_val)} samples.")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    proba = model.predict_proba(X_val)

    res = metrics.evaluate_classification(y_val, y_pred, proba, le.classes_, metrics=tuple(cfg["eval"]["metrics"]))
    print(f"[SVM Fit] Validation Metrics: {res}")

    # --- 新增：保存预测结果到CSV文件 ---
    out_df = pd.DataFrame({
        "true_label": y_val_raw,
        "pred_label": le.inverse_transform(y_pred)
    })
    for i, class_name in enumerate(le.classes_):
        out_df[f"proba_{class_name}"] = proba[:, i]

    # 使用核心IO工具来生成标准路径并保存
    pred_path = io.out_path_predictions(base_dir, ALGO, f"{tag}_preds.csv")
    io.save_csv(out_df, pred_path)
    # ------------------------------------

    # 保存模型、编码器和标准化器
    model_path = os.path.join(base_dir, "models", f"{tag}.pkl")
    # 注意：现在我们将模型、scaler和encoder打包保存在一个pkl文件中
    joblib.dump({"model": model, "scaler": scaler, "encoder": le}, model_path)

    # 保存评估报告
    report_path = os.path.join(base_dir, "reports", f"{tag}_metrics.json")
    io.save_json({"metrics": res}, report_path)

    # 返回产出物路径字典
    return {
        "metrics": res,
        "artifacts": {
            "model_path": model_path,
            "report_path": report_path,
            "predictions_csv": pred_path  # 将预测文件路径也加入返回字典
        }
    }