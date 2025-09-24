# 文件路径: src/pipelines/transfer_dann_pipeline.py
# 描述: 最终版 - 修正了NaN标签导致的类型转换错误

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
import matplotlib.pyplot as plt
import os

from src.models.clf.DANN import build as dann_builder
from src.core import io, viz


# --- Dataset类保持不变 ---
class SpectrogramDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.manifest = manifest_df
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('L')
        # The label is now guaranteed to be an integer or -1
        label = row['encoded_label']
        if self.transform:
            image = self.transform(image)
        if label == -1:
            return (image,)
        else:
            # The label is already an int, so this conversion is safe
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

    # 仅在源域子集上进行标签编码
    source_df_subset = image_manifest[image_manifest['domain'] == 'source'].copy().dropna(subset=['fault_type'])
    source_df_subset['encoded_label'] = le.transform(source_df_subset['fault_type'])

    # 将编码后的标签合并回主清单
    image_manifest = image_manifest.merge(source_df_subset[['image_path', 'encoded_label']], on='image_path',
                                          how='left')

    # --- 【核心修正】用-1填充目标域产生的NaN标签 ---
    image_manifest['encoded_label'].fillna(-1, inplace=True)
    image_manifest['encoded_label'] = image_manifest['encoded_label'].astype(int)
    # ---------------------------------------------

    source_df = image_manifest[image_manifest['domain'] == 'source']
    target_df = image_manifest[image_manifest['domain'] == 'target']
    ys = source_df['encoded_label'].values
    num_classes = len(le.classes_)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # --- 2. 创建DataLoaders ---
    print("步骤2: 创建PyTorch DataLoaders...")
    batch_size = cfg['training_params']['batch_size']
    source_dataset = SpectrogramDataset(source_df, transform=transform)
    target_dataset = SpectrogramDataset(target_df, transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- 3. 构建模型和优化器 ---
    print("步骤3: 构建DANN模型和优化器...")
    model = dann_builder.build(cfg, num_classes).to(device)
    # 采用之前的差分学习率
    base_lr = cfg['training_params']['learning_rate']
    optimizer = torch.optim.Adam([
        # 为特征提取器设置一个学习率
        {'params': model.feature_extractor.parameters(), 'lr': base_lr},
        # 为标签分类器设置同样学习率
        {'params': model.label_predictor.parameters(), 'lr': base_lr},
        # 【关键】为领域判别器设置一个更慢的学习率，例如0.1倍
        {'params': model.domain_discriminator.parameters(), 'lr': base_lr * 0.1},
    ])

    class_counts = np.bincount(ys, minlength=num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
    criterion_label = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()

    history = {'epoch': [], 'domain_loss': [], 'domain_accuracy': []}

    # --- 4. 执行混合对抗训练 ---
    print("步骤4: 开始混合对抗训练 (DANN + MMD)...")
    num_epochs = cfg['training_params']['num_epochs']
    lambda_adv_max = cfg['training_params']['adversarial_lambda']
    lambda_mmd = cfg['training_params'].get('mmd_lambda', 1.0)

    total_steps = num_epochs * min(len(source_loader), len(target_loader))
    current_step = 0

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        epoch_domain_losses, epoch_domain_corrects, epoch_domain_total = [], 0, 0

        for (source_data, source_labels), (target_data,) in progress_bar:
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()

            p = float(current_step) / total_steps if total_steps > 0 else 0
            dynamic_lambda = 2. / (1. + np.exp(-10. * p)) - 1
            current_adv_lambda = dynamic_lambda * lambda_adv_max

            label_output, domain_output_source, source_features = model(source_data, lambda_=current_adv_lambda)
            _, domain_output_target, target_features = model(target_data, lambda_=current_adv_lambda)

            loss_label = criterion_label(label_output, source_labels)
            domain_source_labels = torch.zeros(len(source_data), dtype=torch.long, device=device)
            domain_target_labels = torch.ones(len(target_data), dtype=torch.long, device=device)
            loss_domain_source = criterion_domain(domain_output_source, domain_source_labels)
            loss_domain_target = criterion_domain(domain_output_target, domain_target_labels)
            loss_domain = loss_domain_source + loss_domain_target
            loss_mmd_val = mmd_loss(source_features, target_features)
            total_loss = loss_label + loss_domain + (lambda_mmd * loss_mmd_val)
            total_loss.backward()
            optimizer.step()

            current_step += 1

            epoch_domain_losses.append(loss_domain.item())
            _, source_preds = torch.max(domain_output_source, 1)
            _, target_preds = torch.max(domain_output_target, 1)
            epoch_domain_corrects += torch.sum(source_preds == domain_source_labels).item()
            epoch_domain_corrects += torch.sum(target_preds == domain_target_labels).item()
            epoch_domain_total += len(source_data) + len(target_data)
            progress_bar.set_postfix(loss=total_loss.item(),
                                     domain_acc=epoch_domain_corrects / epoch_domain_total if epoch_domain_total > 0 else 0)

        avg_domain_loss = np.mean(epoch_domain_losses) if epoch_domain_losses else 0
        domain_accuracy = epoch_domain_corrects / epoch_domain_total if epoch_domain_total > 0 else 0
        history['epoch'].append(epoch + 1)
        history['domain_loss'].append(avg_domain_loss)
        history['domain_accuracy'].append(domain_accuracy)
        print(
            f"Epoch {epoch + 1} Summary: Avg Domain Loss = {avg_domain_loss:.4f}, Domain Accuracy = {domain_accuracy:.4f}")

    # --- 步骤 5, 6, 7, 8 保持不变 ---
    # 5. 预测
    print("步骤5: 对目标域进行预测...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        target_loader_eval = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
        for (inputs,) in target_loader_eval:
            inputs = inputs.to(device)
            label_output, _, _ = model(inputs, lambda_=0)
            preds = torch.argmax(label_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
    target_pred_labels = le.inverse_transform(all_preds)

    # 6. 保存结果
    print(f"步骤6: 保存预测结果...")
    results_df = target_df[['original_file']].copy()
    results_df = results_df.iloc[:len(target_pred_labels)].copy()
    results_df['predicted_fault_type'] = target_pred_labels
    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])
    torch.save(model.state_dict(), cfg['outputs']['model_path'])

    # 7. 绘制迁移过程图
    print("步骤7: 生成迁移过程可解释性图表...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Domain Discriminator Loss', color=color)
    ax1.plot(history['epoch'], history['domain_loss'], color=color, marker='o', label='Domain Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Domain Discriminator Accuracy', color=color)
    ax2.plot(history['epoch'], history['domain_accuracy'], color=color, marker='x', label='Domain Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.title('Interpretability of Migration Process')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    plot_path = os.path.join(os.path.dirname(cfg['outputs']['visualization_path']),
                             "interpretability_migration_process.png")
    io.ensure_dir(plot_path)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"迁移过程图表已保存至: {plot_path}")

    # 8. 绘制t-SNE图
    print("步骤8: 生成t-SNE可视化图 (通过抽样)...")
    model.eval()

    # 定义一个合理的样本数量，用于可视化
    SAMPLE_SIZE = 2000  # 从每个域中抽取2000个样本，足以看清分布

    with torch.no_grad():
        # --- 安全地提取源域特征 ---
        source_features_list = []
        # 【修改1】使用 torch.utils.data.SubsetRandomSampler 进行高效随机抽样
        source_sampler = torch.utils.data.SubsetRandomSampler(
            np.random.choice(len(source_dataset), min(SAMPLE_SIZE, len(source_dataset)), replace=False)
        )
        source_loader_eval_sampled = DataLoader(source_dataset, batch_size=batch_size, sampler=source_sampler)

        # 我们还需要对应样本的标签
        ys_sampled = []

        # 这个循环现在只会迭代少量批次
        for (inputs, labels) in tqdm(source_loader_eval_sampled, desc="Extracting Source Samples"):
            inputs = inputs.to(device)
            _, _, features = model(inputs, lambda_=0)
            source_features_list.append(features.cpu().numpy())
            ys_sampled.append(labels.cpu().numpy())  # 收集标签

        source_features = np.vstack(source_features_list)
        ys_labels_sampled = np.hstack(ys_sampled)  # 拼接标签

        # --- 安全地提取目标域特征 ---
        target_features_list = []
        # 【修改2】同样对目标域进行抽样
        target_sampler = torch.utils.data.SubsetRandomSampler(
            np.random.choice(len(target_dataset), min(SAMPLE_SIZE, len(target_dataset)), replace=False)
        )
        target_loader_eval_sampled = DataLoader(target_dataset, batch_size=batch_size, sampler=target_sampler)

        # 同样，我们需要对应样本的预测标签
        all_preds_sampled_indices = target_sampler.indices
        all_preds_sampled = np.array(all_preds)[all_preds_sampled_indices]  # 从之前的完整预测中按索引取出

        for (inputs,) in tqdm(target_loader_eval_sampled, desc="Extracting Target Samples"):
            inputs = inputs.to(device)
            _, _, features = model(inputs, lambda_=0)
            target_features_list.append(features.cpu().numpy())

        target_features = np.vstack(target_features_list)

    # 【修改3】使用抽样后的数据进行可视化
    viz.plot_tsne_by_class(
        source_latent=source_features,
        target_latent=target_features,
        source_labels=ys_labels_sampled,  # 使用抽样样本的真实标签
        target_labels=all_preds_sampled,  # 使用抽样样本的预测标签
        class_names=list(le.classes_),
        out_png=cfg["outputs"]["visualization_path"],
        title='Class Distribution After DANN+MMD Adaptation (Sampled)'
    )
    print(f"[DANN-2D + MMD 混合迁移流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于图像和混合损失的DANN迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)