# [新文件] src/pipelines/transfer_dann_1d_pipeline.py
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
    print(f"[DANN 1D特征迁移流水线] 开始运行，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device.type.upper()}")

    # --- 1. 加载和准备1D特征数据 ---
    print("步骤1: 加载1D特征数据 (features.parquet)...")
    le = joblib.load(cfg["label_encoder_path"])
    features_df = pd.read_parquet(cfg["features_path"])

    # 从配置文件中获取模型所需的输入维度，并进行验证
    expected_input_dim = cfg['model']['input_dim']
    feature_cols = [col for col in features_df.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]
    if len(feature_cols) != expected_input_dim:
        raise ValueError(
            f"配置文件中的 input_dim ({expected_input_dim}) 与特征文件中的特征数量 ({len(feature_cols)}) 不匹配!")

    source_df = features_df[features_df['domain'] == 'source'].copy().dropna(subset=['fault_type'])
    target_df = features_df[features_df['domain'] == 'target'].copy()

    scaler = StandardScaler()
    Xs_raw = source_df[feature_cols].values
    Xt_raw = target_df[feature_cols].values

    scaler.fit(Xs_raw)
    Xs = scaler.transform(Xs_raw)
    Xt = scaler.transform(Xt_raw)
    ys_labels = le.transform(source_df['fault_type'])

    source_dataset = TensorDataset(torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys_labels, dtype=torch.long))
    target_dataset = TensorDataset(torch.tensor(Xt, dtype=torch.float32))

    # --- 2. 创建PyTorch DataLoaders ---
    print("步骤2: 创建PyTorch DataLoaders...")
    batch_size = cfg['training_params']['batch_size']
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- 3. 构建DANN模型和优化器 ---
    print("步骤3: 构建DANN模型和优化器...")
    num_classes = len(le.classes_)
    model = dann_builder.build(cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training_params']['learning_rate'])

    class_counts = np.bincount(ys_labels)
    class_weights = 1. / torch.tensor(np.where(class_counts == 0, 1, class_counts), dtype=torch.float)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
    print(f"计算出的类别权重: {class_weights.cpu().numpy()}")

    criterion_label = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()

    # --- 4. 执行对抗训练 ---
    print("步骤4: 开始对抗训练...")
    num_epochs = cfg['training_params']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        for (source_data, source_labels), (target_data,) in progress_bar:
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()

            # 使用动态lambda进行更稳定的训练
            p = float(epoch) / num_epochs
            lambda_coeff = 2. / (1. + np.exp(-10. * p)) - 1
            adversarial_lambda = lambda_coeff * cfg['training_params']['adversarial_lambda']

            label_output, domain_output_source = model(source_data, lambda_=adversarial_lambda)
            loss_label = criterion_label(label_output, source_labels)
            loss_domain_source = criterion_domain(domain_output_source,
                                                  torch.zeros(len(source_data), dtype=torch.long, device=device))

            _, domain_output_target = model(target_data, lambda_=adversarial_lambda)
            loss_domain_target = criterion_domain(domain_output_target,
                                                  torch.ones(len(target_data), dtype=torch.long, device=device))

            total_loss = loss_label + loss_domain_source + loss_domain_target
            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=total_loss.item(), loss_label=loss_label.item(),
                                     loss_domain=(loss_domain_source + loss_domain_target).item())

    # --- 5. 对目标域进行预测 ---
    print("步骤5: 对目标域进行预测...")
    model.eval()
    all_preds, all_features = [], []
    with torch.no_grad():
        target_loader_eval = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
        for (inputs,) in target_loader_eval:
            inputs = inputs.to(device)
            features = model.feature_extractor(inputs)
            label_output, _ = model(inputs, lambda_=0)
            preds = torch.argmax(label_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_features.append(features.cpu().numpy())
    target_pred_labels = le.inverse_transform(all_preds)
    target_features = np.vstack(all_features)

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
        source_features_list = []
        source_loader_eval = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
        for (inputs, _) in source_loader_eval:
            inputs = inputs.to(device)
            source_features_list.append(model.feature_extractor(inputs).cpu().numpy())
        source_features = np.vstack(source_features_list)

    viz.plot_tsne_by_class(
        source_latent=source_features,
        target_latent=target_features,
        source_labels=ys_labels,
        target_labels=np.array(all_preds),
        class_names=list(le.classes_),
        out_png=cfg["outputs"]["visualization_path"],
        title=f'Class Distribution After {cfg["model"].get("feature_extractor_type", "DANN")} Adaptation'
    )

    print(f"[DANN 1D特征迁移流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于1D特征的DANN迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)