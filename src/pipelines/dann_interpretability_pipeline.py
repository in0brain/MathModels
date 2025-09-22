# src/pipelines/dann_interpretability_pipeline.py
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
import os

from src.models.clf.DANN import build as dann_builder
from src.core import io, viz  # <--- 核心：我们将调用viz中的函数
from sklearn.preprocessing import StandardScaler


def run(config_path: str):
    print(f"[DANN可解释性流水线] 启动，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    with open(cfg['dann_config_path'], 'r', encoding='utf-8') as f:
        dann_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # --- 1. 加载所有必需文件和数据 ---
    print("步骤1: 加载模型、数据和编码器...")
    le = joblib.load(cfg["label_encoder_path"])
    num_classes = len(le.classes_)

    model = dann_builder.build(dann_cfg, num_classes)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.to(device)
    model.eval()

    features_df = pd.read_parquet(cfg["features_path"])
    feature_cols = [col for col in features_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    source_df = features_df[features_df['domain'] == 'source']
    target_df = features_df[features_df['domain'] == 'target']

    scaler = StandardScaler()
    Xs_raw = source_df[feature_cols].values
    Xt_raw = target_df[feature_cols].values
    scaler.fit(Xs_raw)
    Xs = scaler.transform(Xs_raw)
    Xt = scaler.transform(Xt_raw)

    # --- 2. 初始化SHAP DeepExplainer ---
    print("步骤2: 初始化SHAP DeepExplainer...")
    background_data_np = shap.sample(Xs, cfg["shap_params"]["background_samples"])
    background_data_tensor = torch.tensor(background_data_np).float().to(device)

    samples_to_explain_np = Xt[:cfg["shap_params"]["num_local_explanations"]]
    samples_to_explain_tensor = torch.tensor(samples_to_explain_np).float().to(device)

    model_to_explain = nn.Sequential(model.feature_extractor, model.label_predictor).to(device)
    explainer = shap.DeepExplainer(model_to_explain, background_data_tensor)

    # --- 3. 计算目标样本的SHAP值 ---
    print("步骤3: 计算目标样本的SHAP值...")
    shap_values = explainer.shap_values(samples_to_explain_tensor)

    # --- 4. (解耦) 调用viz模块生成并保存全局可解释性图 ---
    print(f"步骤4: 保存全局SHAP摘要图...")
    all_target_sample_np = shap.sample(Xt, 1000)
    all_target_tensor = torch.tensor(all_target_sample_np).float().to(device)
    shap_values_global = explainer.shap_values(all_target_tensor)

    X_target_df_sample = pd.DataFrame(all_target_sample_np, columns=feature_cols)

    viz.plot_shap_summary_bar(
        shap_values=shap_values_global,
        features_df=X_target_df_sample,
        class_names=le.classes_,
        out_png=cfg['outputs']['global_plot_path']
    )
    print(f"全局摘要图已保存至: {cfg['outputs']['global_plot_path']}")

    # --- 5. (解耦) 调用viz模块生成并保存局部可解释性图 ---
    print("步骤5: 生成局部样本解释图...")
    output_dir = cfg['outputs']['base_dir']

    samples_to_explain_df = pd.DataFrame(samples_to_explain_np, columns=feature_cols)
    base_values_numpy = explainer.expected_value

    for i in range(cfg["shap_params"]["num_local_explanations"]):
        for c_idx, class_name in enumerate(le.classes_):
            exp = shap.Explanation(
                values=shap_values[c_idx][i],
                base_values=base_values_numpy[c_idx],
                data=samples_to_explain_df.iloc[i],
                feature_names=feature_cols
            )
            plot_path = os.path.join(output_dir, f"local_explanation_sample_{i}_class_{class_name}.png")
            viz.plot_shap_waterfall_new(
                explanation_object=exp,
                out_png=plot_path,
                max_display=15
            )

    print(f"已保存 {cfg['shap_params']['num_local_explanations']} 个样本的局部解释图至: {output_dir}")
    print("[DANN可解释性流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP Interpretability Pipeline for DANN model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    run(args.config)