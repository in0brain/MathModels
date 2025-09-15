# -*- coding: utf-8 -*-
import os
import json
import joblib
import pandas as pd  # 在顶部统一导入

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        # 移除这里的重复导入，使用顶部已导入的pd
        return pd.read_excel(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def save_json(obj, path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(path)
    df.to_csv(path, index=False, encoding="utf-8")

def save_model(model, path: str):
    ensure_dir(path)
    joblib.dump(model, path)

# === 统一生成 outputs 下的分类路径 ===
def out_path_predictions(base_dir: str, algo: str, filename: str) -> str:
    """预测结果：outputs/data/predictions/<Algo>/<filename>"""
    path = os.path.join(base_dir, "data", "predictions", algo, filename)
    ensure_dir(path)
    return path

def out_path_artifacts(base_dir: str, algo: str, filename: str) -> str:
    """训练工件：outputs/data/artifacts/<Algo>/<filename>"""
    path = os.path.join(base_dir, "data", "artifacts", algo, filename)
    ensure_dir(path)
    return path