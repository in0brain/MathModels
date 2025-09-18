# src/preprocessing/tabular/encode.py
from typing import Dict, Any, Optional, List
import json
import numpy as np
import pandas as pd
from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

class OneHotEncodeTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols: Optional[List[str]] = self.cfg.get("cols")
        drop_first = bool(self.cfg.get("drop_first", False))
        dtype = getattr(np, self.cfg.get("dtype", "uint8"))
        artifacts_out = self.cfg.get("artifacts_out")  # 可选：保存映射
        df = pd.read_csv(path)
        if cols is None:
            cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        res = pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=dtype)
        io.ensure_dir(out)
        res.to_csv(out, index=False)
        mapping = {}
        if artifacts_out:
            for c in cols:
                mapping[c] = [col for col in res.columns if col.startswith(c + "_")]
            with open(artifacts_out, "w", encoding="utf-8") as f:
                json.dump({"one_hot_columns": mapping}, f, ensure_ascii=False, indent=2)
        return {"out": out, "encoded_cols": cols, "drop_first": drop_first, "artifacts_out": artifacts_out}
register_task("one_hot_encode", OneHotEncodeTask)
