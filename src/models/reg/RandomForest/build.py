# -*- coding: utf-8 -*-
"""
RandomForest 回归：构建/训练/评估/推理
"""
from typing import Dict, Any, Tuple, List
import os, json, joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

TASK = "reg"
ALGO = "RandomForest"

def _safe_cfg(cfg: Dict[str, Any]):
    out_cfg = cfg.get("outputs", {}) or {}
    base_dir = out_cfg.get("base_dir", "outputs")
    tag = out_cfg.get("tag", cfg.get("model", {}).get("name", ALGO))
    prep = cfg.get("preprocess", {}) or {}
    viz_cfg = cfg.get("viz", {}) or {}
    seed = int(cfg.get("seed", 42))
    return base_dir, tag, prep, viz_cfg, seed

def _select_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    target = cfg["dataset"]["target"]
    feats = cfg["dataset"].get("features")
    if feats:
        X = df[feats].copy()
    else:
        X = df.drop(columns=[target]) if target in df.columns else df.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, num_cols, cat_cols, target

def _build_preprocessor(num_cols, cat_cols, prep: Dict[str, Any]):
    impute_num = prep.get("impute_num", "median")
    scale_num = bool(prep.get("scale_num", True))
    one_hot_cat = bool(prep.get("one_hot_cat", True))
    impute_cat = prep.get("impute_cat", "most_frequent")

    transformers = []
    if num_cols:
        steps = [("imp", SimpleImputer(strategy=impute_num))]
        if scale_num:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num_cols))
    if cat_cols and one_hot_cat:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy=impute_cat)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers)

def _ensure_dir(path: str):
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _save_json(obj: dict, path: str):
    _ensure_dir(path)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build(cfg: Dict[str, Any]):
    return {"cfg": cfg}

def fit(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    base_dir, tag, prep, viz_cfg, seed = _safe_cfg(cfg)
    X, num_cols, cat_cols, target = _select_columns(df, cfg)
    y = df[target].values

    pre = _build_preprocessor(num_cols, cat_cols, prep)

    params = dict(cfg.get("model", {}).get("params", {}))
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", -1)

    if params.get("max_features", None) == "auto":
        params["max_features"] = None

    rf = RandomForestRegressor(**params)
    pipe = Pipeline([("pre", pre), ("model", rf)])

    test_size = cfg.get("split", {}).get("test_size", 0.2)
    rnd = cfg.get("split", {}).get("random_state", seed)
    shuffle = bool(cfg.get("split", {}).get("shuffle", True))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rnd, shuffle=shuffle)

    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae = float(mean_absolute_error(y_te, preds))
    r2 = float(r2_score(y_te, preds))
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    preds_path = f"{base_dir}/predictions/{tag}_preds.csv"
    _ensure_dir(preds_path)
    pd.DataFrame({"true": y_te, "pred": preds}).to_csv(preds_path, index=False, encoding="utf-8")

    model_path = f"{base_dir}/models/{tag}.pkl"
    _ensure_dir(model_path)
    joblib.dump(pipe, model_path)

    report_path = f"{base_dir}/reports/{tag}_metrics.json"
    _save_json({"metrics": metrics}, report_path)

    return {"metrics": metrics,
            "artifacts": {"predictions_csv": preds_path, "model_path": model_path, "report_path": report_path}}

def inference(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    target = cfg["dataset"].get("target")
    X = df.drop(columns=[target]) if target and target in df.columns else df.copy()
    y_pred = model.predict(X)
    return {"predictions": pd.DataFrame({"pred": y_pred})}
