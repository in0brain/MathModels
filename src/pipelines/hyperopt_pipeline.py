# -*- coding: utf-8 -*-
import argparse, json, os
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
from xgboost import XGBRegressor  # 例子：回归（可按需换成分类）
from src.core import io  # 你的 IO 工具:contentReference[oaicite:14]{index=14}

def run(cfg_path: str):
    import yaml
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    path = cfg["dataset"]["path"]
    target = cfg["dataset"]["target"]
    out_dir = cfg["report"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(path)
    y = df[target].values
    X = df.drop(columns=[target]).values

    n_splits = int(cfg.get("cv", {}).get("n_splits", 5))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "random_state": 42,
            "tree_method": "hist",
        }
        model = XGBRegressor(**params)
        # 评价用 MAE（越小越好），Optuna 默认最小化
        scores = -cross_val_score(
            model, X, y,
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            cv=cv, n_jobs=-1
        )
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(cfg.get("trials", 30)))

    # 保存结果
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }
    with open(os.path.join(out_dir, "hyperopt_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print("[hyperopt] best:", best)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
