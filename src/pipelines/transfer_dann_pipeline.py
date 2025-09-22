# [图像版本] src/pipelines/transfer_dann_pipeline.py
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from PIL import Image

from src.models.clf.DANN import build as dann_builder
from src.core import io, viz


# --- 新增：自定义Dataset类用于加载图像 ---
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

        # 对于目标域，我们只返回图像
        if label == -1:
            return (image,)
        # 对于源域，返回图像和标签
        else:
            return image, torch.tensor(label, dtype=torch.long)


def run(config_path: str):
    print(f"[DANN-2D迁移诊断流水线] 开始运行，配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- 1. 加载和准备数据 (已修改) ---
    print("步骤1: 加载图像清单和数据...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    le = joblib.load(cfg["label_encoder_path"])
    image_manifest = pd.read_csv(cfg["image_manifest_path"])

    # 筛选出有标签的源域数据用于编码
    source_domain_for_encoding = image_manifest[image_manifest['domain'] == 'source'].copy()
    source_domain_for_encoding.dropna(subset=['fault_type'], inplace=True)

    # 使用fit_transform确保所有标签都被编码
    image_manifest['encoded_label'] = -1  # 初始化
    image_manifest.loc[source_domain_for_encoding.index, 'encoded_label'] = le.fit_transform(
        source_domain_for_encoding['fault_type'])

    source_df = image_manifest[image_manifest['domain'] == 'source']
    target_df = image_manifest[image_manifest['domain'] == 'target']
    ys = source_df['encoded_label'].values
    num_classes = len(le.classes_)

    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])

    # --- 2. 创建PyTorch DataLoaders (已修改) ---
    print("步骤2: 创建PyTorch DataLoaders...")
    batch_size = cfg['training_params']['batch_size']
    source_dataset = SpectrogramDataset(source_df, transform=transform)
    target_dataset = SpectrogramDataset(target_df, transform=transform)

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- 后续步骤与之前类似，但数据流已变为图像 ---
    # 步骤3: 构建DANN模型和优化器
    print("步骤3: 构建DANN模型和优化器...")
    model = dann_builder.build(cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training_params']['learning_rate'])

    class_counts = np.bincount(ys)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"Calculated Class Weights: {class_weights}")

    criterion_label = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion_domain = nn.CrossEntropyLoss()

    # 步骤4: 执行对抗训练
    print("步骤4: 开始对抗训练...")
    # ... (训练循环代码与之前完全相同)
    num_epochs = cfg['training_params']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        for (source_data, source_labels), (target_data,) in progress_bar:
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()
            label_output, domain_output_source = model(source_data,
                                                       lambda_=cfg['training_params']['adversarial_lambda'])
            loss_label = criterion_label(label_output, source_labels)
            loss_domain_source = criterion_domain(domain_output_source,
                                                  torch.zeros(len(source_data), dtype=torch.long, device=device))
            _, domain_output_target = model(target_data, lambda_=cfg['training_params']['adversarial_lambda'])
            loss_domain_target = criterion_domain(domain_output_target,
                                                  torch.ones(len(target_data), dtype=torch.long, device=device))
            total_loss = loss_label + loss_domain_source + loss_domain_target
            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=total_loss.item(), loss_label=loss_label.item(),
                                     loss_domain=(loss_domain_source + loss_domain_target).item())

    # 步骤5: 对目标域进行预测
    print("步骤5: 对目标域进行预测...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        # 注意：这里的DataLoader现在返回的是元组(image,)
        for (inputs,) in DataLoader(target_dataset, batch_size=batch_size):
            inputs = inputs.to(device)
            label_output, _ = model(inputs, lambda_=0)
            preds = torch.argmax(label_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
    target_pred_labels = le.inverse_transform(all_preds)

    # 步骤6: 保存结果
    print(f"步骤6: 保存预测结果...")
    results_df = target_df[['original_file']].copy()
    results_df['predicted_fault_type'] = target_pred_labels
    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])
    torch.save(model.state_dict(), cfg['outputs']['model_path'])

    # 步骤7: 可视化
    print("步骤7: 生成t-SNE可视化图...")
    model.eval()
    with torch.no_grad():
        source_features_list = []
        for (inputs, _) in DataLoader(source_dataset, batch_size=batch_size):
            inputs = inputs.to(device)
            source_features_list.append(model.feature_extractor(inputs).cpu().numpy())
        source_features = np.vstack(source_features_list)

        target_features_list = []
        for (inputs,) in DataLoader(target_dataset, batch_size=batch_size):
            inputs = inputs.to(device)
            target_features_list.append(model.feature_extractor(inputs).cpu().numpy())
        target_features = np.vstack(target_features_list)

    viz.plot_tsne_by_class(
        source_latent=source_features,
        target_latent=target_features,
        source_labels=ys,
        target_labels=np.array(all_preds),
        class_names=list(le.classes_),
        out_png=cfg["outputs"]["visualization_path"],
        title='Class Distribution After 2D-DANN Adaptation (t-SNE Viz)'
    )

    print(f"[DANN-2D迁移诊断流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于图像的DANN迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)