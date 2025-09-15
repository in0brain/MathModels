# -*- coding: utf-8 -*-
"""
NeuralNet 分类（sklearn MLPClassifier）：
- 统一 IO / registry / outputs 规范
- 支持数值 + 类别特征（OHE），数值标准化
- 评估：ACC/F1/ROC_AUC；绘图：ROC/PR/混淆矩阵
- 推理：支持输出 pred 与（若可用）proba
"""
from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.core import io, metrics, viz

TASK = "clf"
ALGO = "NeuralNet"

# ---------- 列选择与预处理 ----------
def _select_columns(df: pd.DataFrame, cfg: Dict[str, Any]):
    feats = cfg["dataset"].get("features") or []
    target = cfg["dataset"]["target"]
    if feats:
        X = df[feats].copy()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        cat_cols = [c for c in df.columns if c not in num_cols + [target]]
        X = df[num_cols + cat_cols].copy()
    return X, num_cols, cat_cols

def _build_preprocessor(num_cols, cat_cols, cfg):
    transformers = []
    if num_cols:
        steps = [("imp", SimpleImputer(strategy=cfg["preprocess"].get("impute_num", "median")))]
        if cfg["preprocess"].get("scale_num", True):
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num_cols))
    if cat_cols and cfg["preprocess"].get("one_hot_cat", True):
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy=cfg["preprocess"].get("impute_cat", "most_frequent"))),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers)

# ---------- 接口 ----------
def build(cfg: Dict[str, Any]):
    return {"cfg": cfg}  # 占位，实际管线在 fit 中创建

def fit(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)
    target = cfg["dataset"]["target"]
    seed = int(cfg.get("seed", 42))

    if cfg.get("preprocess", {}).get("dropna", False):
        df = df.dropna(subset=[target])
    y = df[target].values
    X, num_cols, cat_cols = _select_columns(df, cfg)

    pre = _build_preprocessor(num_cols, cat_cols, cfg)
    mlp = MLPClassifier(random_state=seed, **cfg["model"].get("params", {}))
    pipe = Pipeline([("pre", pre), ("model", mlp)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["dataset"].get("test_size", 0.2), random_state=seed, stratify=y
    )

    pipe.fit(X_tr, y_tr)

    # 预测与概率
    y_pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
    classes_ = pipe.classes_

    # 评估
    met_list = cfg.get("eval", {}).get("metrics", ["ACC", "F1", "ROC_AUC"])
    res = metrics.evaluate_classification(y_te, y_pred, proba, classes_, metrics=tuple(met_list))

    # 可视化
    fig_paths = []
    if cfg.get("viz", {}).get("enabled", True):
        dpi = cfg["viz"].get("dpi", 160)
        plots = cfg["viz"].get("plots", {})
        if plots.get("roc", True) and proba is not None:
            fig_paths.append(viz.plot_roc(y_te, proba, classes_, os.path.join(base_dir, "figs", "clf", f"{tag}_roc.png"), dpi=dpi))
        if plots.get("pr", True) and proba is not None:
            fig_paths.append(viz.plot_pr(y_te, proba, classes_, os.path.join(base_dir, "figs", "clf", f"{tag}_pr.png"), dpi=dpi))
        if plots.get("cm", True):
            fig_paths.append(viz.plot_confusion_matrix(y_te, y_pred, classes_, os.path.join(base_dir, "figs", "clf", f"{tag}_cm.png"), dpi=dpi))

    # 保存预测
    out_df = pd.DataFrame({"true": y_te, "pred": y_pred})
    if proba is not None:
        for i, c in enumerate(classes_):
            out_df[f"proba_{c}"] = proba[:, i]
    pred_path = io.out_path_predictions(base_dir, ALGO, f"{tag}_preds.csv")
    io.save_csv(out_df, pred_path)

    # 保存模型与报告
    model_path = os.path.join(base_dir, "models", f"{tag}.pkl")
    io.save_model(pipe, model_path)
    rep_path = os.path.join(base_dir, "reports", f"{tag}_metrics.json")
    io.save_json({"metrics": res, "n_train": len(X_tr), "n_test": len(X_te)}, rep_path)

    return {"metrics": res,
            "artifacts": {"predictions_csv": pred_path, "model_path": model_path, "report_path": rep_path, "figs": fig_paths}}

def inference(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", cfg["model"]["name"])
    algo = cfg["model"]["name"]

    target = cfg["dataset"].get("target")
    X = df.drop(columns=[target]) if (target and target in df.columns) else df.copy()

    y_pred = model.predict(X)
    out = pd.DataFrame({"pred": y_pred})

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for i, c in enumerate(model.classes_):
            out[f"proba_{c}"] = proba[:, i]

    pred_path = io.out_path_predictions(base_dir, algo, f"{tag}_infer_preds.csv")
    io.save_csv(out, pred_path)
    return {"predictions_csv": pred_path}
