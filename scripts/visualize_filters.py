# scripts/visualize_filters.py
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 确保项目根目录在sys.path中，以便导入src模块
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.clf.DANN import build as dann_builder
from src.core import io


def visualize_first_layer_filters(config_path: str):
    """
    加载训练好的DANN模型，并可视化其特征提取器的第一个卷积层的卷积核。
    """
    print(f"[Filter Visualization] 启动，配置文件: {config_path}")

    # 1. 加载DANN模型配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg['model']
    outputs_cfg = cfg['outputs']

    # 确保模型是为2D图像设计的
    if model_cfg.get("feature_extractor_type") != "CNN2D":
        raise ValueError("此脚本仅适用于 feature_extractor_type 为 'CNN2D' 的模型。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device.type.upper()}")

    # 2. 加载训练好的模型
    # 假设num_classes为4 (Normal, Ball, InnerRace, OuterRace)，如果您的类别数不同，需要调整
    # 更好的方式是从label_encoder加载，这里为简化脚本，直接硬编码
    num_classes = 4
    model = dann_builder.build(cfg, num_classes)

    model_path = outputs_cfg['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"错误：找不到模型文件 '{model_path}'。请确保您已经成功运行了DANN 2D迁移学习流水线。")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"成功从 '{model_path}' 加载模型。")

    # 3. 提取第一个卷积层的权重
    # 根据 src/models/clf/DANN/build.py 的结构，第一个卷积层在 feature_extractor.conv_block[0]
    first_conv_layer = model.feature_extractor.conv_block[0]
    weights = first_conv_layer.weight.data.cpu().numpy()

    # weights 的形状通常是 (out_channels, in_channels, kernel_height, kernel_width)
    # 对于灰度图输入，in_channels=1
    num_filters = weights.shape[0]
    kernel_size = (weights.shape[2], weights.shape[3])
    print(f"提取到 {num_filters} 个卷积核，每个尺寸为 {kernel_size}")

    # 4. 可视化卷积核
    # 动态计算网格布局
    num_cols = int(np.ceil(np.sqrt(num_filters)))
    num_rows = int(np.ceil(num_filters / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
    axes = axes.flatten()

    for i in range(num_filters):
        kernel = weights[i, 0, :, :]
        ax = axes[i]
        ax.imshow(kernel, cmap='viridis', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余的子图
    for i in range(num_filters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("A-priori Interpretability: First Layer Convolutional Filters", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 5. 保存图像
    output_path = os.path.join(os.path.dirname(outputs_cfg['visualization_path']), "apriori_convolutional_filters.png")
    io.ensure_dir(output_path)
    plt.savefig(output_path, dpi=200)
    print(f"卷积核可视化图像已保存至: {output_path}")


if __name__ == "__main__":
    # 使用与最终方案相同的配置文件
    # python scripts/visualize_filters.py --config runs/transfer_dann_2d.yaml
    parser = argparse.ArgumentParser(description="事前可解释性：可视化DANN模型的第一个卷积层")
    parser.add_argument("--config", type=str, required=True,
                        help="DANN 2D模型的配置文件路径 (e.g., runs/transfer_dann_2d.yaml)")
    args = parser.parse_args()
    visualize_first_layer_filters(args.config)