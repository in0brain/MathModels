# src/pipelines/hyperopt_pipeline.py
import argparse, json, os
import pandas as pd
import optuna
import yaml
# 【FIX】 Import StratifiedGroupKFold instead of StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.core import io


def run(cfg_path: str):
    """
    Runs hyperparameter optimization for a CLASSIFICATION task using Optuna.
    """
    # 1. Load Configuration
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    path = cfg["dataset"]["path"]
    target = cfg["dataset"]["target"]
    out_dir = cfg["report"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 2. Load and Prepare Data
    print(f"Loading data from: {path}")
    df_full = io.read_table(path)

    df = df_full[df_full['domain'] == 'source'].copy()
    feature_cols = [col for col in df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]
    X = df[feature_cols]  # Keep as DataFrame for now

    # 【FIX】 Keep original_file for grouping
    groups = df["original_file"]

    le = LabelEncoder()
    y = le.fit_transform(df[target])

    print(f"Data prepared for hyperparameter optimization. Feature shape: {X.shape}")

    # 3. Configure Cross-Validation
    n_splits = int(cfg.get("cv", {}).get("n_splits", 5))
    # 【FIX】 Use the correct, non-leaky cross-validator
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 4. Define the Objective Function for Optuna
    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "use_label_encoder": False,
            "eval_metric": 'mlogloss',
            "random_state": 42,
            "tree_method": "gpu_hist",
        }

        model = XGBClassifier(**params)

        # 【FIX】 Pass the groups to cross_val_score
        scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1, groups=groups)

        return scores.mean()

    # 5. Run the Optimization
    study = optuna.create_study(direction="maximize")
    print(f"Starting Optuna optimization with {cfg.get('trials', 30)} trials...")
    study.optimize(objective, n_trials=int(cfg.get('trials', 30)))

    # 6. Save the Results
    best = {
        "best_value (accuracy)": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }

    report_path = os.path.join(out_dir, "hyperopt_best.json")
    io.save_json(best, report_path)

    print(f"\nOptimization finished! Best parameters found:")
    print(best)
    print(f"Results saved to: {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)