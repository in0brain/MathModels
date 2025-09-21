# src/pipelines/interpretability_pipeline.py
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

from src.core import io


def run(config_path: str):
    print(f"[Interpretability Pipeline] Starting with config: {config_path}")
    # --- FIX: Explicitly specify UTF-8 encoding ---
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 1. 加载模型、数据和编码器
    print("Loading model and data...")
    model = joblib.load(cfg["model_path"])
    le = joblib.load(cfg["label_encoder_path"])
    features_df = pd.read_parquet(cfg["features_path"])

    feature_cols = [col for col in features_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    source_df = features_df[features_df['domain'] == 'source']
    target_df = features_df[features_df['domain'] == 'target']

    X_source = source_df[feature_cols]
    X_target = target_df[feature_cols]

    # 2. 初始化SHAP解释器
    print("Initializing SHAP explainer...")
    # 使用源数据的一个子集作为背景数据，这对于TreeExplainer是推荐的做法
    background_data = shap.sample(X_source, cfg["shap_params"]["background_samples"])
    explainer = shap.TreeExplainer(model, background_data)

    # 3. 计算目标数据的SHAP值
    print("Calculating SHAP values for target data...")
    shap_values = explainer(X_target)

    # 4. 生成并保存全局可解释性图 (事后可解释性 - 整体)
    print(f"Saving global summary plot to {cfg['outputs']['global_plot_path']}...")
    plt.figure()
    shap.summary_plot(shap_values, X_target, class_names=le.classes_, show=False)
    io.ensure_dir(cfg['outputs']['global_plot_path'])
    plt.savefig(cfg['outputs']['global_plot_path'], bbox_inches='tight')
    plt.close()

    # 5. 生成并保存局部可解释性图 (事后可解释性 - 个体)
    print("Generating local explanation plots...")
    output_dir = cfg['outputs']['base_dir']
    io.ensure_dir(output_dir)

    num_examples = min(cfg["shap_params"]["num_local_explanations"], len(X_target))

    for i in range(num_examples):
        # 为每个类别生成一个瀑布图
        for c in range(len(le.classes_)):
            plt.figure()
            shap.plots.waterfall(shap_values[i, :, c], max_display=15, show=False)
            fig = plt.gcf()
            fig.tight_layout()
            plot_path = os.path.join(output_dir, f"local_explanation_sample_{i}_class_{le.classes_[c]}.png")
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close()

    print(f"Saved {num_examples} local explanation plots to {output_dir}")
    print("[Interpretability Pipeline] Finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP Interpretability Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    run(args.config)