# 文件路径: src/pipelines/transfer_dann_pipeline.py
# 描述: 最终版 - 结合了领域对抗(DANN)与显式距离度量(MMD)的混合迁移学习流水线，用于处理二维时频图。

import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from src.core.metrics import mmd_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.models.clf.DANN import build as dann_builder
from src.core import io, viz

# --- 自定义Dataset类用于加载图像 (保持不变) ---
class SpectrogramDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.manifest = manifest_df
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('L')  # 读取灰度图

        label = row.get('encoded_label', -1)  # -1 表示目标域无标签

        if self.transform:
            image = self.transform(image)

        if label == -1:
            return (image,)
        else:
            return image, torch.tensor(label, dtype=torch.long)


def run(config_path: str):
    print(f"[DANN-2D + MMD 混合迁移流水线] 开始运行，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- 1. 加载和准备数据 ---
    print("步骤1: 加载图像清单和数据...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device.type.upper()}")

    le = joblib.load(cfg["label_encoder_path"])
    image_manifest = pd.read_csv(cfg["image_manifest_path"])

    source_domain_for_encoding = image_manifest[image_manifest['domain'] == 'source'].copy()
    source_domain_for_encoding.dropna(subset=['fault_type'], inplace=True)

    image_manifest['encoded_label'] = -1
    image_manifest.loc[source_domain_for_encoding.index, 'encoded_label'] = le.fit_transform(
        source_domain_for_encoding['fault_type'])

    source_df = image_manifest[image_manifest['domain'] == 'source']
    target_df = image_manifest[image_manifest['domain'] == 'target']
    ys = source_df['encoded_label'].values
    num_classes = len(le.classes_)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- 2. 创建PyTorch DataLoaders ---
    print("步骤2: 创建PyTorch DataLoaders...")
    batch_size = cfg['training_params']['batch_size']
    source_dataset = SpectrogramDataset(source_df, transform=transform)
    target_dataset = SpectrogramDataset(target_df, transform=transform)

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- 3. 构建DANN模型和优化器 ---
    print("步骤3: 构建DANN模型和优化器...")
    model = dann_builder.build(cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training_params']['learning_rate'])

    class_counts = np.bincount(ys)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
    print(f"计算出的类别权重: {class_weights.cpu().numpy()}")

    criterion_label = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()

    # --- 4. 执行混合对抗训练 ---
    print("步骤4: 开始混合对抗训练 (DANN + MMD)...")
    num_epochs = cfg['training_params']['num_epochs']
    lambda_adv = cfg['training_params']['adversarial_lambda']
    lambda_mmd = cfg['training_params'].get('mmd_lambda', 1.0)  # 从配置读取MMD权重

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        for (source_data, source_labels), (target_data,) in progress_bar:
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()

            # --- 核心修改：模型现在返回三项输出 ---
            label_output, domain_output_source, source_features = model(source_data, lambda_=lambda_adv)
            _, domain_output_target, target_features = model(target_data, lambda_=lambda_adv)

            # 计算三部分损失
            loss_label = criterion_label(label_output, source_labels)
            loss_domain_source = criterion_domain(domain_output_source,
                                                  torch.zeros(len(source_data), dtype=torch.long, device=device))
            loss_domain_target = criterion_domain(domain_output_target,
                                                  torch.ones(len(target_data), dtype=torch.long, device=device))
            loss_mmd_val = mmd_loss(source_features, target_features)  # 计算MMD损失

            # 组合成最终的混合损失函数
            total_loss = loss_label + (loss_domain_source + loss_domain_target) + (lambda_mmd * loss_mmd_val)

            total_loss.backward()
            optimizer.step()

            # 更新进度条以显示所有损失
            progress_bar.set_postfix(
                total_loss=total_loss.item(),
                loss_label=loss_label.item(),
                loss_domain=(loss_domain_source + loss_domain_target).item(),
                loss_mmd=loss_mmd_val.item()
            )

    # --- 5. 对目标域进行预测 (保持不变) ---
    print("步骤5: 对目标域进行预测...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for (inputs,) in DataLoader(target_dataset, batch_size=batch_size):
            inputs = inputs.to(device)
            # --- 注意：预测时不再需要特征，模型调用保持不变 ---
            label_output, _, _ = model(inputs, lambda_=0)
            preds = torch.argmax(label_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
    target_pred_labels = le.inverse_transform(all_preds)

    # --- 6. 保存结果 (保持不变) ---
    print(f"步骤6: 保存预测结果...")
    results_df = target_df[['original_file']].copy()
    # 确保预测数量和元数据行数一致
    if len(target_pred_labels) != len(results_df):
        print(f"警告：预测数量({len(target_pred_labels)})与目标域样本数({len(results_df)})不匹配。将截断以匹配预测数量。")
        results_df = results_df.iloc[:len(target_pred_labels)]
    results_df['predicted_fault_type'] = target_pred_labels
    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])
    torch.save(model.state_dict(), cfg['outputs']['model_path'])

    # --- 7. 可视化 (保持不变) ---
    print("步骤7: 生成t-SNE可视化图...")
    model.eval()
    with torch.no_grad():
        source_features_list = []
        for (inputs, _) in DataLoader(source_dataset, batch_size=batch_size):
            inputs = inputs.to(device)
            # --- 注意：可视化时需要提取特征 ---
            _, _, features = model(inputs, lambda_=0)
            source_features_list.append(features.cpu().numpy())
        source_features = np.vstack(source_features_list)

        target_features_list = []
        for (inputs,) in DataLoader(target_dataset, batch_size=batch_size):
            inputs = inputs.to(device)
            _, _, features = model(inputs, lambda_=0)
            target_features_list.append(features.cpu().numpy())
        target_features = np.vstack(target_features_list)

    viz.plot_tsne_by_class(
        source_latent=source_features,
        target_latent=target_features,
        source_labels=ys,
        target_labels=np.array(all_preds),
        class_names=list(le.classes_),
        out_png=cfg["outputs"]["visualization_path"],
        title='Class Distribution After DANN+MMD Adaptation'
    )

    print(f"[DANN-2D + MMD 混合迁移流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于图像和混合损失的DANN迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)