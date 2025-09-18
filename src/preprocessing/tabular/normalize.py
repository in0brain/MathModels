# src/preprocessing/tabular/normalize.py
import math, json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

def _num_cols(df: pd.DataFrame, cols: Optional[List[str]]):
    return cols or df.select_dtypes(include="number").columns.tolist()

# 1.1 Min-Max
class MinMaxScaleTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        a, b = self.cfg.get("feature_range", (0.0, 1.0))
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        for c in cols:
            vmin, vmax = df[c].min(), df[c].max()
            if vmax > vmin:
                df[c] = (df[c] - vmin) / (vmax - vmin)
                df[c] = df[c] * (b - a) + a
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols, "range": (a, b)}
register_task("minmax_scale", MinMaxScaleTask)

# 1.2 Z-Score
class ZScoreScaleTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        ddof = int(self.cfg.get("ddof", 0))
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        for c in cols:
            mu, sigma = df[c].mean(), df[c].std(ddof=ddof)
            if sigma > 0:
                df[c] = (df[c] - mu) / sigma
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols, "ddof": ddof}
register_task("zscore_scale", ZScoreScaleTask)

# Robust: (x - median) / IQR
class RobustScaleTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        for c in cols:
            q1, med, q3 = df[c].quantile(0.25), df[c].median(), df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[c] = (df[c] - med) / iqr
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols}
register_task("robust_scale", RobustScaleTask)

# MaxAbs: x / max(|x|)
class MaxAbsScaleTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        for c in cols:
            m = np.abs(df[c]).max()
            if m > 0:
                df[c] = df[c] / m
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols}
register_task("maxabs_scale", MaxAbsScaleTask)

# Log: 默认 log1p；如含非正值自动平移或用自定义 shift
class LogTransformTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        shift = self.cfg.get("shift")  # None 则自动平移到 >0
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        for c in cols:
            s = df[c]
            sh = shift
            if (s <= 0).any():
                sh = sh if sh is not None else (1 - s.min())
            if sh:
                s = s + sh
            df[c] = np.log1p(s)
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols, "shift": shift}
register_task("log_transform", LogTransformTask)

# Box-Cox：正值要求；保存每列 lambda 便于再预测
class BoxCoxTransformTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        from scipy.stats import boxcox
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        lmbda = self.cfg.get("lmbda")
        artifacts_out = self.cfg.get("artifacts_out")
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        lambdas = {}
        for c in cols:
            s = df[c]
            if (s <= 0).any():
                raise ValueError(f"Box-Cox requires positive values, column '{c}' has non-positive.")
            if lmbda is None:
                y, lam = boxcox(s.values)
            else:
                lam = float(lmbda)
                y = boxcox(s.values, lmbda=lam)
            df[c] = y
            lambdas[c] = lam

        io.ensure_dir(out)
        df.to_csv(out, index=False)
        if artifacts_out:
            with open(artifacts_out, "w", encoding="utf-8") as f:
                json.dump({"boxcox_lambdas": lambdas}, f, ensure_ascii=False, indent=2)
        return {"out": out, "cols": cols, "lambdas": lambdas, "artifacts_out": artifacts_out}
register_task("boxcox_transform", BoxCoxTransformTask)

# QuantileTransformer：映射到均匀/高斯
class QuantileTransformTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        from sklearn.preprocessing import QuantileTransformer
        path, out = self.cfg["path"], self.cfg["out"]
        cols = self.cfg.get("cols")
        n_quantiles = int(self.cfg.get("n_quantiles", 1000))
        output_distribution = self.cfg.get("output_distribution", "uniform")  # "uniform"|"normal"
        random_state = int(self.cfg.get("random_state", 42))
        df = pd.read_csv(path)
        cols = _num_cols(df, cols)
        qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution,
                                 random_state=random_state, subsample=int(1e9))
        df[cols] = qt.fit_transform(df[cols])
        io.ensure_dir(out)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols, "n_quantiles": n_quantiles, "output_distribution": output_distribution}
register_task("quantile_transform", QuantileTransformTask)
