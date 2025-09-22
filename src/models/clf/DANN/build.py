# # [最终修正版] in src/models/clf/DANN/build.py
# import torch
# import torch.nn as nn
# from torch.autograd import Function
# import math
#
# # ----------------- DANN 核心组件：梯度反转层 -----------------
# class GradientReversalFunc(Function):
#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.lambda_
#         return output, None
#
# def grad_reverse(x, lambda_=1.0):
#     """梯度反转层的包裹函数"""
#     return GradientReversalFunc.apply(x, lambda_)
#
# # ----------------- 1D-CNN 特征提取器 -----------------
# class CNNFeatureExtractor(nn.Module):
#     def __init__(self, input_dim):
#         super(CNNFeatureExtractor, self).__init__()
#         # 定义卷积模块
#         self.conv_block = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#         )
#         # 动态计算展平后的维度
#         # 创建一个虚拟输入来确定输出维度
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 1, input_dim)
#             dummy_output = self.conv_block(dummy_input)
#             self.flattened_dim = dummy_output.view(1, -1).size(1)
#
#     def forward(self, x):
#         # CNN需要一个通道维度, (batch_size, features) -> (batch_size, 1, features)
#         x = x.unsqueeze(1)
#         x = self.conv_block(x)
#         x = x.view(x.size(0), -1) # 展平
#         return x
#
# # ----------------- MLP 构建器 (用于分类器和判别器) -----------------
# def _build_net(input_dim, layer_dims, output_dim=None, dropout=0.5):
#     layers = []
#     current_dim = input_dim
#     for dim in layer_dims:
#         layers.append(nn.Linear(current_dim, dim))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout))
#         current_dim = dim
#     if output_dim:
#         layers.append(nn.Linear(current_dim, output_dim))
#     return nn.Sequential(*layers)
#
# # ----------------- DANN 模型整体结构 (修正版) -----------------
# class DANNModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super(DANNModel, self).__init__()
#         self.cfg = cfg
#         input_dim = cfg['model']['input_dim']
#
#         # 1. 特征提取器 (Feature Extractor) - 使用新的CNN模块
#         self.feature_extractor = CNNFeatureExtractor(input_dim)
#         feature_output_dim = self.feature_extractor.flattened_dim
#
#         # 2. 标签预测器 (Label Predictor)
#         lp_arch = cfg['model']['label_predictor_arch']
#         self.label_predictor = _build_net(feature_output_dim, lp_arch, output_dim=num_classes)
#
#         # 3. 领域判别器 (Domain Discriminator)
#         dd_arch = cfg['model']['domain_discriminator_arch']
#         self.domain_discriminator = _build_net(feature_output_dim, dd_arch, output_dim=2)
#
#     def forward(self, input_data, lambda_=1.0):
#         features = self.feature_extractor(input_data)
#         label_output = self.label_predictor(features)
#         reversed_features = grad_reverse(features, lambda_)
#         domain_output = self.domain_discriminator(reversed_features)
#         return label_output, domain_output
#
# # ----------------- 框架集成接口 -----------------
# TASK = "clf"
# ALGO = "DANN"
#
# def build(cfg: dict, num_classes: int):
#     """构建DANN模型实例"""
#     return DANNModel(cfg, num_classes)
# [MHDCNN最终优化版] src/models/clf/DANN/build.py
# [最终修正版 v2] src/models/clf/DANN/build.py
# [2D-CNN版本] src/models/clf/DANN/build.py
import torch
import torch.nn as nn
from torch.autograd import Function


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


# ----------------- 2D-CNN 特征提取器 -----------------
class CNN2DFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNN2DFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # in: 64x64
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 32x32
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 16x16
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # -> 8x8
        )
        self.flattened_dim = 64 * 8 * 8

    def forward(self, x):
        x = self.conv_block(x)
        return x.view(x.size(0), -1)


# ----------------- MLP 构建器 (保持不变) -----------------
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


# ----------------- DANN 模型整体结构 (使用2D-CNN) -----------------
class DANNModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DANNModel, self).__init__()
        self.feature_extractor = CNN2DFeatureExtractor()
        feature_output_dim = self.feature_extractor.flattened_dim

        lp_arch = cfg['model']['label_predictor_arch']
        self.label_predictor = _build_net(feature_output_dim, lp_arch, output_dim=num_classes)

        dd_arch = cfg['model']['domain_discriminator_arch']
        self.domain_discriminator = _build_net(feature_output_dim, dd_arch, output_dim=2)

    def forward(self, input_data, lambda_=1.0):
        features = self.feature_extractor(input_data)
        label_output = self.label_predictor(features)
        reversed_features = grad_reverse(features, lambda_)
        domain_output = self.domain_discriminator(reversed_features)
        return label_output, domain_output


# ----------------- 框架集成接口 (保持不变) -----------------
TASK = "clf"
ALGO = "DANN"


def build(cfg: dict, num_classes: int):
    return DANNModel(cfg, num_classes)
