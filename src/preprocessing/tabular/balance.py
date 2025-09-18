# src/preprocessing/tabular/balance.py
from typing import Dict, Any, List, Optional
import pandas as pd

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

def _check_y(df: pd.DataFrame, y_col: str):
    if y_col not in df.columns:
        raise KeyError(f"y_col '{y_col}' not found")
    return y_col

# 欠采样：多数类降到≈ratio*少数类
class UndersampleRandomTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        y_col = _check_y(pd.read_csv(path), self.cfg["y_col"])
        ratio = float(self.cfg.get("ratio", 1.0))
        rs = int(self.cfg.get("random_state", 42))
        df = pd.read_csv(path)
        vc = df[y_col].value_counts()
        min_n = vc.min()
        frames = []
        for cls, n in vc.items():
            sub = df[df[y_col] == cls]
            target = min(min_n * ratio, n) if n > min_n else min_n
            target = int(target)
            frames.append(sub.sample(n=target, random_state=rs) if n > target else sub)
        out_df = pd.concat(frames, axis=0).sample(frac=1.0, random_state=rs).reset_index(drop=True)
        io.ensure_dir(out)
        out_df.to_csv(out, index=False)
        return {"out": out, "y_col": y_col,
                "counts_before": vc.to_dict(),
                "counts_after": out_df[y_col].value_counts().to_dict()}
register_task("undersample_random", UndersampleRandomTask)

# 过采样：少数类升到≈ratio*多数类
class OversampleRandomTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        y_col = _check_y(pd.read_csv(path), self.cfg["y_col"])
        ratio = float(self.cfg.get("ratio", 1.0))
        rs = int(self.cfg.get("random_state", 42))
        df = pd.read_csv(path)
        vc = df[y_col].value_counts()
        max_n = vc.max()
        target = int(max_n * ratio)
        frames = []
        for cls, n in vc.items():
            sub = df[df[y_col] == cls]
            if n < target:
                extras = sub.sample(n=target - n, replace=True, random_state=rs)
                sub = pd.concat([sub, extras], axis=0)
            frames.append(sub)
        out_df = pd.concat(frames, axis=0).sample(frac=1.0, random_state=rs).reset_index(drop=True)
        io.ensure_dir(out)
        out_df.to_csv(out, index=False)
        return {"out": out, "y_col": y_col,
                "counts_before": vc.to_dict(),
                "counts_after": out_df[y_col].value_counts().to_dict()}
register_task("oversample_random", OversampleRandomTask)

# SMOTE（需要 imbalanced-learn）
class SmoteSampleTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        try:
            from imblearn.over_sampling import SMOTE
        except Exception as e:
            raise ImportError("SMOTE requires 'imbalanced-learn'. Please add it to requirements.txt and install.") from e
        path, out = self.cfg["path"], self.cfg["out"]
        y_col = _check_y(pd.read_csv(path), self.cfg["y_col"])
        k = int(self.cfg.get("k_neighbors", 5))
        rs = int(self.cfg.get("random_state", 42))
        df = pd.read_csv(path)
        X = df.drop(columns=[y_col])
        y = df[y_col]
        sm = SMOTE(k_neighbors=k, random_state=rs)
        X_res, y_res = sm.fit_resample(X, y)
        out_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y_col)], axis=1)
        io.ensure_dir(out)
        out_df.to_csv(out, index=False)
        return {"out": out, "y_col": y_col}
register_task("smote_sample", SmoteSampleTask)
