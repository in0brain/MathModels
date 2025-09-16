# -*- coding: utf-8 -*-
import os
import json
import joblib
import pandas as pd  # 在顶部统一导入

#  """确保目标文件的父目录存在"""
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
    # if path.lower().endswith(".parquet"): return read_parquet(path)
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

# --- 新增：保存 DataFrame 为 Parquet ---
def save_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> str:
    """
    将 DataFrame 保存为 Parquet 文件（默认 snappy 压缩）。
    依赖 pyarrow：请确保 requirements.txt 已包含 `pyarrow`。
    返回：写入的文件路径
    """
    ensure_dir(path)
    try:
        df.to_parquet(path, engine="pyarrow", index=False, compression=compression)
    except ImportError as e:
        raise ImportError(
            "缺少依赖 pyarrow。请在 requirements.txt 添加 `pyarrow` 并执行 `pip install -r requirements.txt`。"
        ) from e
    return path

def read_parquet(path: str) -> pd.DataFrame:
    """读取 Parquet（pyarrow 引擎）。"""
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except ImportError as e:
        raise ImportError(
            "如果缺少依赖 pyarrow。在 requirements.txt 添加 `pyarrow` 并执行 `pip install -r requirements.txt`。"
        ) from e
