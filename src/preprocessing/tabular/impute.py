# src/preprocessing/tabular/impute.py
from typing import Dict, Any, Optional, List
import pandas as pd

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

def _filter_existing_cols(df: pd.DataFrame, cols: Optional[List[str]]):
    if cols is None:
        return df.columns.tolist(), []
    missing = [c for c in cols if c not in df.columns]
    exists = [c for c in cols if c in df.columns]
    return exists, missing

class ImputeSimpleTask(PreprocessTask):
    """
    strategy: mean | median | mode
    - 数值列: mean/median/mode 按设置
    - 非数值列: 强制使用 mode（即使设置为 mean/median 也不会报错）
    - YAML 中不存在的列会被忽略并警告
    """
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols: Optional[List[str]] = self.cfg.get("cols")
        strategy = self.cfg.get("strategy", "median").lower()

        df = pd.read_csv(path)
        cols, missing = _filter_existing_cols(df, cols)
        if missing:
            print(f"[impute_simple] warn: drop missing columns {missing} (not in file)")
        if not cols:
            df.to_csv(out, index=False)
            return {"out": out, "cols": [], "strategy": strategy, "note": "no valid columns"}

        for c in cols:
            is_num = pd.api.types.is_numeric_dtype(df[c])
            if is_num:
                if strategy == "mean":
                    val = df[c].mean()
                elif strategy == "median":
                    val = df[c].median()
                elif strategy == "mode":
                    mode = df[c].mode()
                    val = mode.iloc[0] if not mode.empty else df[c].dropna().iloc[0] if df[c].notna().any() else None
                else:
                    raise ValueError("strategy must be mean|median|mode")
            else:
                # 非数值列一律用众数，避免 mean/median 报错
                mode = df[c].mode()
                val = mode.iloc[0] if not mode.empty else df[c].dropna().iloc[0] if df[c].notna().any() else None
                if strategy in ("mean", "median"):
                    print(f"[impute_simple] info: '{c}' is non-numeric, fallback to mode")

            if val is not None:
                df[c] = df[c].fillna(val)

        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols, "strategy": strategy, "dropped": missing}

register_task("impute_simple", ImputeSimpleTask)
