# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

def _num_cols(df: pd.DataFrame, cols: Optional[List[str]]):
    return cols or df.select_dtypes(include="number").columns.tolist()

class CreatePolynomialFeaturesTask(PreprocessTask):
    """多项式特征：对指定数值列生成 2..degree 次幂"""
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        columns: List[str] = self.cfg["columns"]
        degree = int(self.cfg.get("degree", 2))
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                print(f"[poly] warn: column '{col}' not in data, skip")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"[poly] warn: column '{col}' is not numeric, skip")
                continue
            for d in range(2, degree + 1):
                df[f"{col}_pow{d}"] = df[col] ** d
        df.to_csv(out, index=False)
        return {"out": out, "columns": columns, "degree": degree}

register_task("create_polynomial_features", CreatePolynomialFeaturesTask)

class CreateInteractionTask(PreprocessTask):
    """交互项：col1 * col2"""
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        col1, col2 = self.cfg["col1"], self.cfg["col2"]
        df = pd.read_csv(path)
        if col1 in df.columns and col2 in df.columns:
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            else:
                print(f"[inter] warn: {col1} or {col2} not numeric, skip")
        else:
            print(f"[inter] warn: missing {col1} or {col2}, skip")
        df.to_csv(out, index=False)
        return {"out": out, "col1": col1, "col2": col2}

register_task("create_interaction_features", CreateInteractionTask)

class CreateTimeFeaturesTask(PreprocessTask):
    """从时间列提取年月日/星期/季度"""
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        time_col = self.cfg["time_col"]
        df = pd.read_csv(path)
        if time_col not in df.columns:
            print(f"[time] warn: '{time_col}' not found, keep original")
            df.to_csv(out, index=False)
            return {"out": out, "time_col": time_col, "note": "column not found"}
        dt = pd.to_datetime(df[time_col], errors="coerce")
        df[f"{time_col}_year"] = dt.dt.year
        df[f"{time_col}_month"] = dt.dt.month
        df[f"{time_col}_day"] = dt.dt.day
        df[f"{time_col}_dayofweek"] = dt.dt.dayofweek
        df[f"{time_col}_quarter"] = dt.dt.quarter
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "time_col": time_col}

register_task("create_time_features", CreateTimeFeaturesTask)
