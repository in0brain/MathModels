# src/models/clf/DANN/build.py
import torch
import torch.nn as nn
from torch.autograd import Function


# ----------------- DANN 核心组件：梯度反转层 -----------------
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
    """梯度反转层的包裹函数"""
    return GradientReversalFunc.apply(x, lambda_)


# ----------------- 动态网络构建器 (核心修正) -----------------
def _build_net(input_dim, layer_dims, output_dim=None, dropout=0.5):
    """一个辅助函数，根据配置动态构建全连接网络层"""
    layers = []
    current_dim = input_dim
    for dim in layer_dims:
        layers.append(nn.Linear(current_dim, dim))
        # ******************** 核心修正 ********************
        # 将 nn.ReLU(True) 修改为 nn.ReLU()，禁用inplace操作
        layers.append(nn.ReLU())
        # *************************************************
        layers.append(nn.Dropout(dropout))
        current_dim = dim

    if output_dim:
        layers.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*layers)


# ----------------- DANN 模型整体结构 -----------------
class DANNModel(nn.Module):
    """领域对抗神经网络 (DANN) 的完整模型"""

    def __init__(self, cfg, num_classes):
        super(DANNModel, self).__init__()
        self.cfg = cfg
        input_dim = cfg['model']['input_dim']

        # 1. 特征提取器 (Feature Extractor)
        fe_arch = cfg['model']['feature_extractor_arch']
        self.feature_extractor = _build_net(input_dim, fe_arch)

        # 2. 标签预测器 (Label Predictor)
        lp_input_dim = fe_arch[-1]
        lp_arch = cfg['model']['label_predictor_arch']
        self.label_predictor = _build_net(lp_input_dim, lp_arch, output_dim=num_classes)

        # 3. 领域判别器 (Domain Discriminator)
        dd_input_dim = fe_arch[-1]
        dd_arch = cfg['model']['domain_discriminator_arch']
        self.domain_discriminator = _build_net(dd_input_dim, dd_arch, output_dim=2)

    def forward(self, input_data, lambda_=1.0):
        """模型的前向传播"""
        features = self.feature_extractor(input_data)
        label_output = self.label_predictor(features)
        reversed_features = grad_reverse(features, lambda_)
        domain_output = self.domain_discriminator(reversed_features)
        return label_output, domain_output


# --- 为了与项目框架集成 ---
TASK = "clf"
ALGO = "DANN"


def build(cfg: dict, num_classes: int):
    """构建DANN模型实例"""
    return DANNModel(cfg, num_classes)