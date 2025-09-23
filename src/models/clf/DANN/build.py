# [FINAL CORRECTED VERSION - 22/09/2025]
# File Path: src/models/clf/DANN/build.py
import torch
import torch.nn as nn
from torch.autograd import Function
import math


# ----------------- DANN 核心组件：梯度反转层 (保持不变) -----------------
class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunc.apply(x, lambda_)


# ----------------- 1. [原始方案] 1D-CNN 特征提取器 (保留) -----------------
class CNN1DFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(CNN1DFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_output = self.conv_block(dummy_input)
            self.flattened_dim = dummy_output.view(1, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return x

# 定义一个SE（注意力）模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



# ----------------- 2. [图像方案] 2D-CNN 特征提取器 (保留) -----------------
class CNN2DFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNN2DFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        # --在最后一个卷积块后加入SELayer--
        self.se_block = SELayer(channel=64)  # 最后一层卷积输出通道是64
        #----
        self.flattened_dim = 64 * 8 * 8

    def forward(self, x):
        x = self.conv_block(x)
        return x.view(x.size(0), -1)


# ----------------- 3. [已修正的] MHDCNN 特征提取器 -----------------
class MHDCNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim):  # input_dim 现在是窗口长度, e.g., 4096
        super(MHDCNNFeatureExtractor, self).__init__()

        # --- FIX: 使用大卷积核处理长序列信号 ---
        self.multiscale_conv1 = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=64, stride=2, padding=31),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=128, stride=2, padding=63),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=256, stride=2, padding=127)
        ])
        # 拼接后通道数为 4 * 3 = 12

        # 混合空洞卷积层
        self.hybrid_dilated_conv_block = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=16, dilation=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16, dilation=2, padding='same'),
        )

        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.flattened_dim = 16

    def forward(self, x):
        x = x.unsqueeze(1)
        multiscale_outputs = [conv(x) for conv in self.multiscale_conv1]

        # 找到多尺度卷积输出的最小长度并池化
        min_len = min([out.size(2) for out in multiscale_outputs])
        multiscale_pooled = [
            nn.functional.adaptive_avg_pool1d(out, min_len) for out in multiscale_outputs
        ]

        x = torch.cat(multiscale_pooled, dim=1)
        x = self.hybrid_dilated_conv_block(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        return x

# ----------------- MLP 构建器 (通用，保持不变) -----------------
def _build_net(input_dim, layer_dims, output_dim=None, dropout=0.5):
    layers = []
    current_dim = input_dim
    for dim in layer_dims:
        layers.append(nn.Linear(current_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current_dim = dim
    if output_dim:
        layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


# ----------------- DANN 模型整体结构 (动态选择特征提取器, 保持不变) -----------------
class DANNModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DANNModel, self).__init__()
        model_cfg = cfg['model']

        extractor_type = model_cfg.get("feature_extractor_type", "CNN1D")
        print(f"--- INFO: Building DANN with '{extractor_type}' feature extractor. ---")

        if extractor_type == "MHDCNN":
            self.feature_extractor = MHDCNNFeatureExtractor(model_cfg['input_dim'])
        elif extractor_type == "CNN2D":
            self.feature_extractor = CNN2DFeatureExtractor()
        else:
            self.feature_extractor = CNN1DFeatureExtractor(model_cfg['input_dim'])

        feature_output_dim = self.feature_extractor.flattened_dim
        lp_arch = model_cfg['label_predictor_arch']
        self.label_predictor = _build_net(feature_output_dim, lp_arch, output_dim=num_classes)
        dd_arch = model_cfg['domain_discriminator_arch']
        self.domain_discriminator = _build_net(feature_output_dim, dd_arch, output_dim=2)

    def forward(self, input_data, lambda_=1.0):
        features = self.feature_extractor(input_data)
        label_output = self.label_predictor(features)
        reversed_features = grad_reverse(features, lambda_)
        domain_output = self.domain_discriminator(reversed_features)
        # --- 新增返回项 ---
        return label_output, domain_output, features


# ----------------- 框架集成接口 (保持不变) -----------------
TASK = "clf"
ALGO = "DANN"


def build(cfg: dict, num_classes: int):
    return DANNModel(cfg, num_classes)