# -*- coding: utf-8 -*-
"""
XGBoost 回归：构建/训练/评估/作图/推理
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

from src.core import io, metrics, viz

TASK = "reg"            # —— 供注册中心识别：任务类型（回归）
ALGO = "XGBoost"        # —— 供注册中心识别：算法别名（与 params.yaml 一致）

# ========= 内部：特征列选择 & 预处理 =========
def _select_columns(df: pd.DataFrame, cfg: Dict[str, Any]):
    """中文注释：根据配置决定使用的特征列；若未指定则自动数值/类别列"""
    feats: List[str] = cfg["dataset"].get("features") or []
    if feats:
        X = df[feats].copy()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
    else:
        # 自动：数值列 + 类别列
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 目标列不应出现在特征中
        target = cfg["dataset"]["target"]
        if target in num_cols:
            num_cols.remove(target)
        cat_cols = [c for c in df.columns if c not in num_cols + [target]]
        X = df[num_cols + cat_cols].copy()
    return X, num_cols, cat_cols

def _build_preprocessor(num_cols, cat_cols, cfg):
    """中文注释：依据配置创建 ColumnTransformer"""
    transformers = []
    if num_cols:
        num_steps = [("imp", SimpleImputer(strategy=cfg["preprocess"].get("impute_num", "median")))]
        if cfg["preprocess"].get("scale_num", True):
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols and cfg["preprocess"].get("one_hot_cat", True):
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy=cfg["preprocess"].get("impute_cat", "most_frequent"))),
            # sklearn>=1.2: sparse_output；旧版本请改为 sparse=False
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers)

def _feature_names_after_pre(pre: ColumnTransformer):
    """中文注释：从预处理器中获取变换后的特征名（用于画特征重要性）"""
    feat_names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            feat_names += list(cols)
        elif name == "cat":
            ohe = trans.named_steps["ohe"]
            feat_names += ohe.get_feature_names_out(cols).tolist()
    return feat_names

# ========= 对外接口 =========
def build(cfg: Dict[str, Any]):
    """中文注释：仅返回配置（实际 Pipeline 在 fit 中创建并保存）"""
    return {"cfg": cfg}

def fit(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """中文注释：训练 + 评估 + 作图 + 落盘"""
    from xgboost import XGBRegressor

    # 0) 基本元素
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)
    target = cfg["dataset"]["target"]
    seed = int(cfg.get("seed", 42))

    # 1) 准备数据
    if cfg.get("preprocess", {}).get("dropna", False):
        df = df.dropna(subset=[target])
    y = df[target].values
    X, num_cols, cat_cols = _select_columns(df, cfg)

    # 2) 预处理器 & 模型
    pre = _build_preprocessor(num_cols, cat_cols, cfg)
    xgb = XGBRegressor(**cfg["model"].get("params", {}), n_jobs=-1, random_state=seed)
    pipe = Pipeline([("pre", pre), ("model", xgb)])

    # 3) 切分 & 训练
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["dataset"].get("test_size", 0.2), random_state=seed
    )
    pipe.fit(X_tr, y_tr)

    # 4) 评估
    preds = pipe.predict(X_te)
    met_list = cfg.get("eval", {}).get("metrics", ["MAE", "R2"])
    res = metrics.evaluate_regression(y_te, preds, metrics=tuple(met_list))

    # 5) 作图（特征重要性/残差/散点）
    fig_paths = []
    if cfg.get("viz", {}).get("enabled", True):
        dpi = cfg["viz"].get("dpi", 160)
        plots = cfg["viz"].get("plots", {})
        names = _feature_names_after_pre(pipe.named_steps["pre"])
        if plots.get("feat_importance", True) and hasattr(pipe.named_steps["model"], "feature_importances_"):
            fig_paths.append(
                viz.plot_feature_importance(pipe.named_steps["model"].feature_importances_, names,
                                            os.path.join(base_dir, "figs", "reg", f"{tag}_featimp.png"), dpi=dpi)
            )
        if plots.get("residuals", True):
            fig_paths.append(
                viz.plot_residuals(y_te, preds, os.path.join(base_dir, "figs", "reg", f"{tag}_residuals.png"), dpi=dpi)
            )
        if plots.get("pred_scatter", True):
            fig_paths.append(
                viz.plot_pred_scatter(y_te, preds, os.path.join(base_dir, "figs", "reg", f"{tag}_pred_scatter.png"), dpi=dpi)
            )

    # 6) 保存预测/模型/报告
    pred_path = io.out_path_predictions(base_dir, ALGO, f"{tag}_preds.csv")
    io.save_csv(pd.DataFrame({"true": y_te, "pred": preds}), pred_path)

    model_path = os.path.join(base_dir, "models", f"{tag}.pkl")
    io.save_model(pipe, model_path)

    rep_path = os.path.join(base_dir, "reports", f"{tag}_metrics.json")
    io.save_json({"metrics": res, "n_train": len(X_tr), "n_test": len(X_te)}, rep_path)

    return {"metrics": res,
            "artifacts": {
                "predictions_csv": pred_path,
                "model_path": model_path,
                "report_path": rep_path,
                "figs": fig_paths}}

def inference(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """中文注释：通用推理（加载好的 Pipeline + 新数据 -> 预测）"""
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", cfg["model"]["name"])
    algo = cfg["model"]["name"]

    target = cfg["dataset"].get("target")  # 可能没有真值
    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df.copy()

    y_pred = model.predict(X)
    out = pd.DataFrame({"pred": y_pred})
    if target and target in df.columns:
        out.insert(0, "true", df[target].values)

    pred_path = io.out_path_predictions(base_dir, algo, f"{tag}_infer_preds.csv")
    io.save_csv(out, pred_path)
    return {"predictions_csv": pred_path}
