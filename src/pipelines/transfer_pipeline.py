# src/pipelines/transfer_pipeline.py
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.transfer.tca import TCA
from src.core import io


def run(config_path: str):
    print(f"[迁移诊断流水线] 开始运行，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # === 1. 加载数据、预训练的模型和标签编码器 ===
    print("加载数据和预训练模型...")
    model = joblib.load(cfg["source_model_path"])
    le = joblib.load(cfg["label_encoder_path"])
    features_df = pd.read_parquet(cfg["features_path"])

    # === 2. 准备源域和目标域数据 ===
    source_df = features_df[features_df['domain'] == 'source'].copy()
    target_df = features_df[features_df['domain'] == 'target'].copy()

    # 清洗源域数据，丢弃标签为空的行，防止后续编码出错
    source_df.dropna(subset=['fault_type'], inplace=True)

    # 筛选出用于建模的特征列 (所有以 'td_', 'fd_', 'env_', 'cwt_' 开头的列)
    feature_cols = [col for col in source_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    Xs_raw = source_df[feature_cols].values
    Xt_raw = target_df[feature_cols].values
    ys = le.transform(source_df['fault_type'])

    # 特征标准化，确保所有特征在同一尺度上
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs_raw)
    Xt = scaler.transform(Xt_raw)

    print(f"源域样本数: {Xs.shape[0]}, 目标域样本数: {Xt.shape[0]}, 特征维度: {Xs.shape[1]}")

    # === 解决方案：为避免内存溢出，在数据的随机子集上拟合TCA ===
    # 选取一个合理的样本量，既能代表数据分布，又不会耗尽内存
    sample_size = min(2000, len(Xs), len(Xt))
    np.random.seed(42)  # 固定随机种子以保证结果可复现
    source_sample_indices = np.random.choice(len(Xs), sample_size, replace=False)
    target_sample_indices = np.random.choice(len(Xt), sample_size, replace=False)

    Xs_sample = Xs[source_sample_indices]
    Xt_sample = Xt[target_sample_indices]
    print(f"为避免内存问题，将在 {len(Xs_sample)} 个源域和 {len(Xt_sample)} 个目标域样本上拟合TCA。")

    # === 4. 在数据子集上拟合TCA，然后用学习到的映射转换完整数据集 ===
    print("正在应用TCA进行领域自适应...")
    tca = TCA(**cfg["tca_params"])
    tca.fit(Xs_sample, Xt_sample)  # 在小样本上进行拟合

    print("正在使用学习到的映射转换整个数据集...")
    Xs_tca = tca.transform(Xs)  # 转换完整的源域数据
    Xt_tca = tca.transform(Xt)  # 转换完整的目标域数据

    # === 5. 在TCA对齐后的新空间中，训练分类器并进行预测 ===
    # 注意：源域模型是在原始特征空间训练的，不能直接用于TCA空间。
    # 正确做法是在转换后的源域数据(Xs_tca)上重新训练一个分类器。
    print("正在TCA对齐空间中训练分类器...")
    from xgboost import XGBClassifier
    classifier_in_tca_space = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    classifier_in_tca_space.fit(Xs_tca, ys)  # 在转换后的源域数据上训练

    # 在转换后的目标域数据上进行预测
    target_pred_labels_encoded = classifier_in_tca_space.predict(Xt_tca)
    target_pred_labels = le.inverse_transform(target_pred_labels_encoded)
    target_pred_proba = classifier_in_tca_space.predict_proba(Xt_tca)

    # === 6. 保存目标域的预测结果 ===
    print(f"正在保存目标域预测结果至 {cfg['outputs']['target_predictions_path']}...")
    results_df = target_df[['original_file', 'window_id']].copy()
    results_df['predicted_fault_type'] = target_pred_labels
    for i, class_name in enumerate(le.classes_):
        results_df[f'proba_{class_name}'] = target_pred_proba[:, i]
    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])

    # === 7. 可视化 (使用相同的样本进行对比，以保证公平性) ===
    print("正在生成t-SNE降维可视化图...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size - 1), max_iter=1000, init='pca')

    # TCA处理前的分布
    X_combined_before = np.vstack((Xs_sample, Xt_sample))
    X_tsne_before = tsne.fit_transform(X_combined_before)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne_before[:sample_size, 0], X_tsne_before[:sample_size, 1], c='blue', label='源域 (Source)',
                alpha=0.5)
    plt.scatter(X_tsne_before[sample_size:, 0], X_tsne_before[sample_size:, 1], c='red', label='目标域 (Target)',
                alpha=0.5)
    plt.title('Data Distribution Before Adaptation in TCA(t-SNE_Viz)')
    plt.legend()
    io.ensure_dir(cfg["outputs"]["visualization_path_before"])
    plt.savefig(cfg["outputs"]["visualization_path_before"])
    plt.close()

    # TCA处理后的分布
    X_combined_after = np.vstack((Xs_tca[source_sample_indices], Xt_tca[target_sample_indices]))
    X_tsne_after = tsne.fit_transform(X_combined_after)
    plt.figure(figsize=(10, 8))
    sampled_ys = ys[source_sample_indices]
    sampled_target_preds = target_pred_labels_encoded[target_sample_indices]
    # 绘制源域点，按真实标签着色
    plt.scatter(X_tsne_after[:sample_size, 0], X_tsne_after[:sample_size, 1], c=sampled_ys, cmap='viridis', alpha=0.5,
                s=10)
    # 绘制目标域点，按预测标签着色
    plt.scatter(X_tsne_after[sample_size:, 0], X_tsne_after[sample_size:, 1], c=sampled_target_preds, cmap='cool',
                marker='x', alpha=0.7, s=20)
    plt.title('Data Distribution After Adaptation in TCA(t-SNE_Viz)') #TCA领域自适应后的数据分布
    # 创建图例
    source_patch = mpatches.Patch(color='purple', label='source_domain (label_Coloring)')
    target_patch = plt.Line2D([0], [0], marker='x', color='w', label='target_domain (label_Coloring)', markerfacecolor='red',
                              markersize=10)
    plt.legend(handles=[source_patch, target_patch])
    io.ensure_dir(cfg["outputs"]["visualization_path_after"])
    plt.savefig(cfg["outputs"]["visualization_path_after"])
    plt.close()

    print("[迁移诊断流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行TCA迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)