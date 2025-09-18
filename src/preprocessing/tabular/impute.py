# src/preprocessing/tabular/impute.py
from typing import Dict, Any, Optional, List
import pandas as pd
from src.preprocessing.base import PreprocessTask, register_task

class ImputeSimpleTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        path, out = self.cfg["path"], self.cfg["out"]
        cols: Optional[List[str]] = self.cfg.get("cols")
        strategy = self.cfg.get("strategy", "median")  # mean|median|mode
        df = pd.read_csv(path)
        cols = cols or df.columns.tolist()
        for c in cols:
            if strategy == "mean":
                val = df[c].mean()
            elif strategy == "median":
                val = df[c].median()
            elif strategy == "mode":
                mode = df[c].mode()
                val = mode.iloc[0] if not mode.empty else df[c].dropna().iloc[0]
            else:
                raise ValueError("strategy must be mean|median|mode")
            df[c] = df[c].fillna(val)
        df.to_csv(out, index=False)
        return {"out": out, "cols": cols, "strategy": strategy}
register_task("impute_simple", ImputeSimpleTask)

class ImputeKNNTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        from sklearn.impute import KNNImputer
        path, out = self.cfg["path"], self.cfg["out"]
        cols: Optional[List[str]] = self.cfg.get("cols")
        n_neighbors = int(self.cfg.get("n_neighbors", 5))
        df = pd.read_csv(path)
        num_cols = cols or df.select_dtypes(include="number").columns.tolist()
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[num_cols] = imputer.fit_transform(df[num_cols])
        df.to_csv(out, index=False)
        return {"out": out, "cols": num_cols, "n_neighbors": n_neighbors}
register_task("impute_knn", ImputeKNNTask)
