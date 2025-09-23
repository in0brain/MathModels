# src/pipelines/source_domain_compare_pipeline.py
import argparse
import yaml
import json
import os
import pandas as pd
import subprocess
import sys
from src.core import viz, io


def run_single_training(config_path: str):
    """调用clf_pipeline来训练一个独立的分类模型"""
    print(f"[Comparison Pipeline] Training model with config: {config_path}")
    try:
        subprocess.run(
            [sys.executable, "-m", "src.pipelines.clf_pipeline", "--config", config_path],
            check=True,
            text=True,
            encoding='utf-8'
        )
        print(f"[Comparison Pipeline] Successfully trained: {config_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Comparison Pipeline] ERROR: Training failed for {config_path}. Exit code: {e.returncode}")
        # 即使单个模型训练失败，也继续尝试下一个，而不是终止整个对比流程
    except FileNotFoundError:
        print(f"ERROR: '{sys.executable}' not found. Ensure Python is in your PATH.")
        sys.exit(1)


def get_report_path_from_config(config_path: str) -> (str, str):
    """从模型的配置文件中解析出其评估报告的路径和模型标签"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tag = cfg.get("outputs", {}).get("tag", cfg.get("model", {}).get("name", "untagged"))
    report_path = os.path.join(cfg.get("outputs", {}).get("base_dir", "outputs"), "reports", f"{tag}_metrics.json")
    return report_path, tag


def main(list_yaml_path: str, output_csv_path: str, should_rerun: bool):
    """主函数：执行模型对比流程"""
    with open(list_yaml_path, 'r', encoding='utf-8') as f:
        compare_config = yaml.safe_load(f)

    model_config_paths = compare_config.get("configs", [])
    if not model_config_paths:
        print("Warning: No model configurations found in the list file. Exiting.")
        return

    # 1. (可选) 重新训练所有模型
    if should_rerun:
        print("\n--- Rerunning training for all models ---")
        for cfg_path in model_config_paths:
            run_single_training(cfg_path)
        print("--- Finished retraining all models ---\n")

    # 2. 收集所有模型的评估结果
    print("--- Collecting metrics from all model reports ---")
    results_rows = []
    all_metrics = set()
    for cfg_path in model_config_paths:
        report_path, model_tag = get_report_path_from_config(cfg_path)
        if not os.path.exists(report_path):
            print(f"Warning: Report not found for model '{model_tag}' at '{report_path}'. Skipping.")
            continue

        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        metrics_data = report_data.get("metrics", {})
        metrics_data['model'] = model_tag  # 将模型标签也加入字典
        results_rows.append(metrics_data)
        all_metrics.update(metrics_data.keys())
        print(f"Successfully collected metrics for model '{model_tag}'.")

    if not results_rows:
        print("Error: No valid metrics reports could be found. Cannot generate comparison. Aborting.")
        return

    # 3. 保存对比报告到CSV
    fieldnames = ['model'] + sorted([m for m in all_metrics if m != 'model'])
    df_results = pd.DataFrame(results_rows)
    df_results = df_results[fieldnames]  # 确保列的顺序一致
    io.save_csv(df_results, output_csv_path)
    print(f"\nComparison report saved to: {output_csv_path}")
    print("Report content:")
    print(df_results.to_string())

    # 4. 生成可视化对比图
    print("\n--- Generating comparison plots ---")
    plots_dir = os.path.join(os.path.dirname(output_csv_path), "comparison_plots_clf")
    metrics_to_plot = [m for m in fieldnames if m not in ['model', 'ROC_AUC']]  # 排除非0-1范围的指标

    for metric in metrics_to_plot:
        if all(metric in row for row in results_rows):
            plot_path = os.path.join(plots_dir, f"compare_{metric}.png")
            viz.plot_clf_compare(results_rows, metric, plot_path)
            print(f"Generated comparison plot for '{metric}' at: {plot_path}")
        else:
            print(f"Skipping plot for '{metric}' as some models are missing this metric.")

    print("--- Comparison pipeline finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a comparison pipeline for classification models.")
    parser.add_argument("--list", required=True, help="Path to the YAML file listing the model configs to compare.")
    parser.add_argument("--out", required=True, help="Path to save the output comparison CSV report.")
    parser.add_argument("--rerun", action="store_true", help="If set, retrain all models before comparing.")
    args = parser.parse_args()

    main(args.list, args.out, args.rerun)