# src/preprocessing/tabular/discretize.py
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

class BinEqualWidthTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        col = self.cfg["col"]
        bins = int(self.cfg["bins"])
        labels = self.cfg.get("labels")
        right = bool(self.cfg.get("right", True))
        df = pd.read_csv(path)
        cat = pd.cut(df[col], bins=bins, labels=labels, right=right)
        df[col + "_bin"] = cat.astype(str)
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "col": col, "bins": bins}
register_task("bin_equal_width", BinEqualWidthTask)

class BinEqualFreqTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        col = self.cfg["col"]
        q = int(self.cfg["q"])
        labels = self.cfg.get("labels")
        duplicates = self.cfg.get("duplicates", "drop")
        df = pd.read_csv(path)
        cat = pd.qcut(df[col], q=q, labels=labels, duplicates=duplicates)
        df[col + "_bin"] = cat.astype(str)
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "col": col, "q": q}
register_task("bin_equal_freq", BinEqualFreqTask)

# 信息增益单阈二分（y 需为 0..K-1 的整数标签）
class BinInfoGainOnceTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        x_col = self.cfg["x_col"]; y_col = self.cfg["y_col"]
        df = pd.read_csv(path)
        x = df[x_col].values
        y = df[y_col].values.astype(int)
        sort_idx = np.argsort(x)
        xs = x[sort_idx]; ys = y[sort_idx]
        # 候选切点
        cand = (xs[:-1] + xs[1:]) / 2.0
        def entropy(yv):
            if len(yv) == 0: return 0.0
            p = np.bincount(yv, minlength=(yv.max()+1)) / len(yv)
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        base = entropy(ys)
        best_gain, best_t = -1.0, None
        for t in np.unique(cand):
            mask = xs <= t
            left, right = ys[mask], ys[~mask]
            gain = base - (len(left)/len(ys))*entropy(left) - (len(right)/len(ys))*entropy(right)
            if gain > best_gain:
                best_gain, best_t = gain, float(t)
        df[f"{x_col}_bin"] = (df[x_col] <= best_t).astype(int)
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "x_col": x_col, "y_col": y_col, "threshold": best_t, "info_gain": best_gain}
register_task("bin_info_gain_once", BinInfoGainOnceTask)
