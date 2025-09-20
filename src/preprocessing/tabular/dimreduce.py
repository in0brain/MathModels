# src/preprocessing/tabular/dimreduce.py
from typing import Dict, Any, Optional, List
import joblib
import pandas as pd

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

class PcaReduceTask(PreprocessTask):
    def run(self) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        path, out = self.cfg["path"], self.cfg["out"]
        n_components: Optional[int] = self.cfg.get("n_components")
        variance_ratio: Optional[float] = self.cfg.get("variance_ratio")
        pca_model_out: Optional[str] = self.cfg.get("pca_model_out")
        cols: Optional[List[str]] = self.cfg.get("cols")
        random_state = int(self.cfg.get("random_state", 42))

        df = pd.read_csv(path)
        X_cols = cols or df.select_dtypes(include="number").columns.tolist()

        if variance_ratio is not None and n_components is None:
            pca = PCA(n_components=float(variance_ratio), svd_solver="full", random_state=random_state)
        else:
            pca = PCA(n_components=int(n_components) if n_components is not None else None,
                      svd_solver="full", random_state=random_state)

        X_new = pca.fit_transform(df[X_cols])
        comp_cols = [f"pca_{i+1}" for i in range(X_new.shape[1])]
        out_df = df.drop(columns=X_cols).copy()
        for i, name in enumerate(comp_cols):
            out_df[name] = X_new[:, i]
        io.ensure_dir(out)
        out_df.to_csv(out, index=False)
        if pca_model_out:
            joblib.dump({"pca": pca, "cols": X_cols}, pca_model_out)
        return {
            "out": out,
            "n_components_": getattr(pca, "n_components_", None),
            "explained_variance_ratio": getattr(pca, "explained_variance_ratio_", None).tolist()
                if getattr(pca, "explained_variance_ratio_", None) is not None else None,
            "pca_model_out": pca_model_out
        }
register_task("pca_reduce", PcaReduceTask)
