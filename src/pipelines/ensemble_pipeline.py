# src/pipelines/ensemble_pipeline.py
import argparse

import yaml
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys
import os
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from src.core import io, metrics

def train_single_model(config_path: str):
    """调用clf_pipeline来训练一个独立的模型"""
    print(f"\n--- Training model with config: {config_path} ---")
    # 使用 subprocess 调用独立的训练进程
    subprocess.run([sys.executable, "-m", "src.pipelines.clf_pipeline", "--config", config_path], check=True)

def run(config_path: str):
    print(f"[Ensemble Pipeline] Starting with config: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_output_dir = "outputs/ensemble_temp"
    os.makedirs(base_output_dir, exist_ok=True)

    # --- 1. 独立训练三个模型 ---
    model_configs = {}
    for domain, model_cfg in cfg["models"].items():
        # 动态生成每个模型的配置文件
        temp_cfg = {
            "task": "clf",
            "dataset": {
                "path": model_cfg["features_path"],
                "target": "fault_type",
                "feature_prefixes": model_cfg["feature_prefixes"]
            },
            "split": cfg["split"],
            "model": {
                "name": model_cfg["name"],
                "params": model_cfg["params"]
            },
            "eval": cfg["eval"],
            "outputs": {
                "base_dir": "outputs",
                "tag": f"ensemble_{domain}_{model_cfg['name']}"
            },
            "seed": cfg["seed"]
        }

        temp_config_path = os.path.join(base_output_dir, f"params_{domain}.yaml")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_cfg, f)

        model_configs[domain] = temp_config_path
        # 执行训练
        train_single_model(temp_config_path)

    # --- 2. 加载预测结果并进行投票 ---
    print("\n--- Performing Ensemble Voting ---")

    # 加载特征文件以获取验证集索引和真实标签
    any_feature_path = cfg["models"]["time_domain"]["features_path"]
    features_df = pd.read_parquet(any_feature_path)
    source_df = features_df[features_df['domain'] == 'source'].copy().dropna(subset=['fault_type'])

    from sklearn.model_selection import StratifiedGroupKFold
    le = LabelEncoder()
    y = le.fit_transform(source_df['fault_type'])
    splitter = StratifiedGroupKFold(n_splits=cfg["split"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    _, val_idx = next(splitter.split(source_df, y, groups=source_df["original_file"]))

    y_true_encoded = y[val_idx]
    y_true_labels = le.inverse_transform(y_true_encoded)

    # 收集每个模型的预测
    predictions = {}
    for domain, model_cfg in cfg["models"].items():
        tag = f"ensemble_{domain}_{model_cfg['name']}"
        preds_path = f"outputs/data/predictions/{model_cfg['name']}/{tag}_preds.csv"
        preds_df = pd.read_csv(preds_path)
        predictions[domain] = le.transform(preds_df['pred_label'])

    # 硬投票
    pred_array = np.array(list(predictions.values()))
    ensemble_preds_encoded, _ = mode(pred_array, axis=0)
    ensemble_preds_encoded = ensemble_preds_encoded.flatten()

    # --- 3. 评估所有模型 ---
    print("\n--- Final Performance Evaluation ---")
    report = {}

    # 评估独立模型
    for domain, preds_encoded in predictions.items():
        res = metrics.evaluate_classification(y_true_encoded, preds_encoded, metrics=tuple(cfg["eval"]["metrics"]))
        print(f"Model ({domain}): {res}")
        report[f"model_{domain}"] = res

    # 评估融合模型
    ensemble_res = metrics.evaluate_classification(y_true_encoded, ensemble_preds_encoded, metrics=tuple(cfg["eval"]["metrics"]))
    print(f"Ensemble Model (Hard Voting): {ensemble_res}")
    report["model_ensemble"] = ensemble_res

    # 保存最终对比报告
    report_path = cfg["outputs"]["report_path"]
    io.ensure_dir(report_path)
    io.save_json(report, report_path)
    print(f"\nEnsemble comparison report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ensemble Comparison Pipeline")
    parser.add_argument("--config", required=True, help="Path to the ensemble config YAML")
    args = parser.parse_args()
    run(args.config)