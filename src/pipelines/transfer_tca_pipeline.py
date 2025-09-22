# src/pipelines/transfer_tca_pipeline.py
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.transfer.tca import TCA
from src.core import io, viz


def run(config_path: str):
    print(f"[最终修正版迁移诊断流水线] 开始运行，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # === 1. 加载所有必需文件 ===
    print("步骤1: 加载数据、预训练模型和标签编码器...")
    source_model = joblib.load(cfg["source_model_path"])
    le = joblib.load(cfg["label_encoder_path"])
    features_df = pd.read_parquet(cfg["features_path"])

    # === 2. 准备数据并定义两种特征集 ===
    print("步骤2: 准备源域和目标域数据...")
    source_df = features_df[features_df['domain'] == 'source'].copy()
    target_df = features_df[features_df['domain'] == 'target'].copy()
    source_df.dropna(subset=['fault_type'], inplace=True)

    # 定义源模型使用的特征集 (td_, fd_, env_)
    source_model_feature_cols = [col for col in features_df.columns if col.startswith(('td_', 'fd_', 'env_'))]

    # 定义用于TCA和最终模型训练的全特征集 (td_, fd_, env_, cwt_)
    transfer_feature_cols = [col for col in features_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    print(f"源模型使用 {len(source_model_feature_cols)} 个特征进行预测。")
    print(f"TCA迁移过程将使用 {len(transfer_feature_cols)} 个特征进行对齐。")

    # 准备用于初步预测的数据 (使用源模型特征)
    Xt_raw_for_initial_pred = target_df[source_model_feature_cols].values

    # 准备用于TCA的全特征数据
    Xs_raw_for_transfer = source_df[transfer_feature_cols].values
    Xt_raw_for_transfer = target_df[transfer_feature_cols].values
    ys = le.transform(source_df['fault_type'])

    # 对全特征数据进行标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs_raw_for_transfer)
    Xt = scaler.transform(Xt_raw_for_transfer)

    # === 3. 使用【正确的特征集】对目标域进行初步预测 ===
    print("步骤3: 对目标域进行初步预测以生成伪标签...")
    # 核心修正：使用 Xt_raw_for_initial_pred (15个特征)
    target_initial_proba = source_model.predict_proba(Xt_raw_for_initial_pred)
    target_initial_preds = np.argmax(target_initial_proba, axis=1)

    # === 4. 根据置信度阈值筛选高质量的伪标签样本 ===
    threshold = cfg.get("semi_supervised_params", {}).get("pseudo_label_threshold", 0.90)
    print(f"步骤4: 使用置信度阈值 {threshold} 筛选高质量伪标签...")
    high_confidence_indices = np.where(np.max(target_initial_proba, axis=1) >= threshold)[0]

    if len(high_confidence_indices) == 0:
        print(f"警告：在阈值 {threshold} 下没有找到任何高置信度的目标域样本。将使用所有预测样本中最可靠的10%作为替代。")
        # 备用策略：如果没有任何样本超过阈值，则取置信度最高的10%
        top_10_percent_idx = np.argsort(np.max(target_initial_proba, axis=1))[-int(0.1 * len(target_initial_proba)):]
        high_confidence_indices = top_10_percent_idx
        if len(high_confidence_indices) == 0:
            raise ValueError("错误：目标域数据为空或模型预测完全失败，无法继续。")

    # 从【全特征】数据集中选出这些高置信度样本
    Xt_pseudo_labeled = Xt[high_confidence_indices]
    yt_pseudo_labeled = target_initial_preds[high_confidence_indices]
    print(f"筛选出 {len(Xt_pseudo_labeled)} 个高置信度目标域样本用于指导TCA。")

    # === 5. 使用源域数据和带伪标签的目标域数据共同拟合TCA ===
    print("步骤5: 使用源域和高置信度目标域样本共同拟合TCA...")
    sample_size = min(len(Xs), len(Xt_pseudo_labeled) * 2, 4000)
    np.random.seed(42)
    source_sample_indices = np.random.choice(len(Xs), sample_size, replace=False)
    Xs_sample_for_fit = Xs[source_sample_indices]

    tca = TCA(**cfg["tca_params"])
    tca.fit(Xs_sample_for_fit, Xt_pseudo_labeled)

    # === 6. 转换完整的【全特征】数据集到新空间 ===
    print("步骤6: 转换全部数据集到新的特征空间...")
    Xs_tca = tca.transform(Xs)
    Xt_tca = tca.transform(Xt)

    # === 7. 在对齐后的新空间中，训练最终的分类器 ===
    print("步骤7: 在对齐后的空间中训练最终分类器...")
    classifier_in_tca_space = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    classifier_in_tca_space.fit(Xs_tca, ys)

    target_pred_labels_encoded = classifier_in_tca_space.predict(Xt_tca)
    target_pred_labels = le.inverse_transform(target_pred_labels_encoded)
    target_pred_proba = classifier_in_tca_space.predict_proba(Xt_tca)

    # === 8. 保存最终的预测结果 ===
    print(f"步骤8: 保存目标域最终预测结果至 {cfg['outputs']['target_predictions_path']}...")
    results_df = target_df[['original_file', 'window_id']].copy()
    results_df['predicted_fault_type'] = target_pred_labels
    for i, class_name in enumerate(le.classes_):
        results_df[f'proba_{class_name}'] = target_pred_proba[:, i]
    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])

    # === 9. 可视化对比 ===
    print("步骤9: 生成t-SNE降维可视化图...")
    # TCA处理前的分布
    initial_source_sample_viz = Xs[np.random.choice(len(Xs), 1000, replace=False)]
    initial_target_sample_viz = Xt[np.random.choice(len(Xt), 1000, replace=False)]
    viz.plot_tsne_distribution(
        source_data=initial_source_sample_viz,
        target_data=initial_target_sample_viz,
        out_png=cfg["outputs"]["visualization_path_before"],
        title='Data Distribution Before Adaptation (t-SNE Viz)'
    )

    # TCA处理后的分布
    viz.plot_tsne_after_adaptation(
        source_latent=Xs_tca[source_sample_indices],
        target_latent=Xt_tca[high_confidence_indices],
        source_labels=ys[source_sample_indices],
        target_labels=yt_pseudo_labeled,
        out_png=cfg["outputs"]["visualization_path_after"],
        title='Data Distribution After Guided Adaptation (t-SNE Viz)'
    )

    print("[最终修正版迁移诊断流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行最终修正版的TCA迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)