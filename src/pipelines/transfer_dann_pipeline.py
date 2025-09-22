# src/pipelines/transfer_dann_pipeline.py
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.clf.DANN import build as dann_builder
from src.core import io, viz


def run(config_path: str):
    print(f"[DANN迁移诊断流水线] 开始运行，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- 1. 加载和准备数据 ---
    print("步骤1: 加载和准备数据...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    le = joblib.load(cfg["label_encoder_path"])
    features_df = pd.read_parquet(cfg["features_path"])

    source_df = features_df[features_df['domain'] == 'source'].copy().dropna(subset=['fault_type'])
    target_df = features_df[features_df['domain'] == 'target'].copy()

    feature_cols = [col for col in features_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    Xs_raw = source_df[feature_cols].values
    Xt_raw = target_df[feature_cols].values
    ys = le.transform(source_df['fault_type'])
    num_classes = len(le.classes_)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs_raw)
    Xt = scaler.transform(Xt_raw)

    # --- 2. 创建PyTorch DataLoaders ---
    print("步骤2: 创建PyTorch DataLoaders...")
    batch_size = cfg['training_params']['batch_size']
    source_dataset = TensorDataset(torch.tensor(Xs).float(), torch.tensor(ys).long())
    target_dataset = TensorDataset(torch.tensor(Xt).float())
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- 3. 构建DANN模型和优化器 ---
    print("步骤3: 构建DANN模型和优化器...")
    model = dann_builder.build(cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training_params']['learning_rate'])
    criterion_label = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # --- 4. 执行对抗训练 ---
    print("步骤4: 开始对抗训练...")
    num_epochs = cfg['training_params']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        # 使用tqdm创建进度条
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        for (source_data, source_labels), (target_data,) in progress_bar:
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)

            optimizer.zero_grad()

            # 对源域数据进行正向传播
            label_output, domain_output_source = model(source_data,
                                                       lambda_=cfg['training_params']['adversarial_lambda'])
            loss_label = criterion_label(label_output, source_labels)
            loss_domain_source = criterion_domain(domain_output_source,
                                                  torch.zeros(batch_size, dtype=torch.long, device=device))

            # 对目标域数据进行正向传播
            _, domain_output_target = model(target_data, lambda_=cfg['training_params']['adversarial_lambda'])
            loss_domain_target = criterion_domain(domain_output_target,
                                                  torch.ones(batch_size, dtype=torch.long, device=device))

            # 计算总损失
            total_loss = loss_label + loss_domain_source + loss_domain_target

            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=total_loss.item(), loss_label=loss_label.item(),
                                     loss_domain=(loss_domain_source + loss_domain_target).item())

    print("训练完成。")

    # --- 5. 对目标域进行预测 ---
    print("步骤5: 对目标域进行预测...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in DataLoader(target_dataset, batch_size=batch_size):
            inputs = data[0].to(device)
            label_output, _ = model(inputs, lambda_=0)  # 预测时lambda=0
            preds = torch.argmax(label_output, dim=1)
            all_preds.extend(preds.cpu().numpy())

    target_pred_labels = le.inverse_transform(all_preds)

    # --- 6. 保存结果 ---
    print(f"步骤6: 保存预测结果...")
    results_df = target_df[['original_file', 'window_id']].copy()
    results_df['predicted_fault_type'] = target_pred_labels
    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])
    torch.save(model.state_dict(), cfg['outputs']['model_path'])

    # --- 7. 可视化 ---
    print("步骤7: 生成t-SNE可视化图...")
    model.eval()
    with torch.no_grad():
        # 获取源域和目标域在训练后的特征空间中的表示
        source_features = model.feature_extractor(torch.tensor(Xs).float().to(device)).cpu().numpy()
        target_features = model.feature_extractor(torch.tensor(Xt).float().to(device)).cpu().numpy()

    viz.plot_tsne_after_adaptation(
        source_latent=source_features,
        target_latent=target_features,
        source_labels=ys,
        target_labels=np.array(all_preds),
        out_png=cfg["outputs"]["visualization_path"],
        title='Data Distribution After DANN Adaptation (t-SNE Viz)'
    )

    print(f"[DANN迁移诊断流水线] 成功运行完毕。所有产出物已保存在 '{cfg['outputs']['base_dir']}' 目录中。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行DANN迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)