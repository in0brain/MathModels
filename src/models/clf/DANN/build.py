# [FINAL CORRECTED VERSION - 23/09/2025]
# File Path: src/models/clf/DANN/build.py
# FIX: Correctly implement GradientReversalFunc to pass lambda_

import torch
import torch.nn as nn
from torch.autograd import Function

# ----------------- DANN 核心组件：梯度反转层 (已修正) -----------------
class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        # --- 核心修正: 使用 save_for_backward 保存 lambda_ ---
        # ctx.lambda_ = lambda_ # 这种方式是无效的
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # --- 核心修正: 从 ctx.saved_tensors 中恢复 lambda_ ---
        lambda_, = ctx.saved_tensors
        # 梯度反转
        output = grad_output.neg() * lambda_
        # 第一个返回值对应 forward 的第一个输入 (x)
        # 第二个返回值对应 forward 的第二个输入 (lambda_)，由于它不需要梯度，返回None
        return output, None

def grad_reverse(x, lambda_):
    # 我们需要确保 lambda_ 是一个tensor，才能被 save_for_backward 保存
    return GradientReversalFunc.apply(x, torch.tensor(lambda_, device=x.device))


# ----------------- 2D-CNN 特征提取器 (保持不变) -----------------
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
        # 注意：SELayer部分为可选优化，为确保稳定性，此处暂时注释
        # self.se_block = SELayer(channel=64)
        self.flattened_dim = 64 * 8 * 8

    def forward(self, x):
        x = self.conv_block(x)
        # x = self.se_block(x)
        return x.view(x.size(0), -1)

# (其他特征提取器和_build_net函数保持不变)
# ...
class CNN1DFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(CNN1DFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_output = self.conv_block(dummy_input)
            self.flattened_dim = dummy_output.view(1, -1).size(1)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        return x.view(x.size(0), -1)

class MHDCNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(MHDCNNFeatureExtractor, self).__init__()
        self.multiscale_conv1 = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=64, stride=2, padding=31),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=128, stride=2, padding=63),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=256, stride=2, padding=127)
        ])
        self.hybrid_dilated_conv_block = nn.Sequential(
            nn.BatchNorm1d(12), nn.ReLU(),
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=16, dilation=1, padding='same'),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16, dilation=2, padding='same'),
        )
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.flattened_dim = 16
    def forward(self, x):
        x = x.unsqueeze(1)
        multiscale_outputs = [conv(x) for conv in self.multiscale_conv1]
        min_len = min([out.size(2) for out in multiscale_outputs])
        multiscale_pooled = [nn.functional.adaptive_avg_pool1d(out, min_len) for out in multiscale_outputs]
        x = torch.cat(multiscale_pooled, dim=1)
        x = self.hybrid_dilated_conv_block(x)
        x = self.final_pool(x)
        return x.view(x.size(0), -1)

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
# ...


# ----------------- DANN 模型整体结构 (保持不变) -----------------
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
        reversed_features = grad_reverse(features, lambda_=lambda_)
        domain_output = self.domain_discriminator(reversed_features)
        return label_output, domain_output, features

# ----------------- 框架集成接口 (保持不变) -----------------
TASK = "clf"
ALGO = "DANN"

def build(cfg: dict, num_classes: int):
    return DANNModel(cfg, num_classes)