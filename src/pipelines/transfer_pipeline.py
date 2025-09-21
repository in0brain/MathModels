import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.transfer.tca import TCA
from src.core import io

def run(config_path: str):
    print(f"[Transfer Pipeline] Starting with config: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 加载数据和模型
    print("Loading data and pre-trained model...")
    model = joblib.load(cfg["source_model_path"])
    le = joblib.load(cfg["label_encoder_path"])
    features_df = pd.read_parquet(cfg["features_path"])

    # 2. 准备源域和目标域数据
    source_df = features_df[features_df['domain'] == 'source'].copy()
    target_df = features_df[features_df['domain'] == 'target'].copy()

    feature_cols = [col for col in source_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    Xs_raw = source_df[feature_cols].values
    Xt_raw = target_df[feature_cols].values
    ys = le.transform(source_df['fault_type'])

    # 数据标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs_raw)
    Xt = scaler.transform(Xt_raw)

    print(f"Source samples: {Xs.shape[0]}, Target samples: {Xt.shape[0]}, Features: {Xs.shape[1]}")

    # 3. 可视化：迁移前
    print("Generating t-SNE visualization before TCA...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_combined_before = np.vstack((Xs, Xt))
    X_tsne_before = tsne.fit_transform(X_combined_before)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne_before[:len(Xs)], X_tsne_before[len(Xs):, 1], c='blue', label='Source', alpha=0.5)
    plt.scatter(X_tsne_before[len(Xs):], X_tsne_before[len(Xs):, 1], c='red', label='Target', alpha=0.5)
    plt.title('t-SNE Visualization Before TCA Domain Adaptation')
    plt.legend()
    io.ensure_dir(cfg["outputs"]["visualization_path_before"])
    plt.savefig(cfg["outputs"]["visualization_path_before"])
    plt.close()


    # 4. 应用TCA进行域自适应
    print("Applying TCA for domain adaptation...")
    tca = TCA(**cfg["tca_params"])
    Xs_tca, Xt_tca = tca.fit_transform(Xs, Xt)

    # 5. 在转换后的数据上进行预测
    print("Predicting labels on transformed target data...")
    target_pred_labels_encoded = model.predict(Xt_tca)
    target_pred_labels = le.inverse_transform(target_pred_labels_encoded)
    target_pred_proba = model.predict_proba(Xt_tca)

    # 6. 保存目标域预测结果
    print(f"Saving target predictions to {cfg['outputs']['target_predictions_path']}...")
    results_df = target_df[['original_file', 'window_id']].copy()
    results_df['predicted_fault_type'] = target_pred_labels
    for i, class_name in enumerate(le.classes_):
        results_df[f'proba_{class_name}'] = target_pred_proba[:, i]

    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])

    # 7. 可视化：迁移后
    print("Generating t-SNE visualization after TCA...")
    X_combined_after = np.vstack((Xs_tca, Xt_tca))
    X_tsne_after = tsne.fit_transform(X_combined_after)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne_after[:len(Xs_tca)], X_tsne_after[len(Xs_tca):, 1], c=ys, cmap='viridis', label='Source', alpha=0.5, s=10)
    plt.scatter(X_tsne_after[len(Xs_tca):], X_tsne_after[len(Xs_tca):, 1], c=target_pred_labels_encoded, cmap='cool', marker='x', label='Target (Predicted)', alpha=0.7, s=20)
    plt.title('t-SNE Visualization After TCA Domain Adaptation')
    plt.legend()
    io.ensure_dir(cfg["outputs"]["visualization_path_after"])
    plt.savefig(cfg["outputs"]["visualization_path_after"])
    plt.close()

    print("[Transfer Pipeline] Finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TCA Transfer Learning Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    run(args.config)