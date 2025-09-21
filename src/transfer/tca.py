# src/transfer/tca.py
import numpy as np
from scipy.linalg import eigh


def kernel(ker, X1, X2, gamma):
    """
    计算核矩阵。核函数用于将数据映射到高维空间，以便更好地捕捉非线性关系。

    参数:
        ker (str): 核函数类型 ('linear' 或 'rbf').
        X1 (np.ndarray): 第一个数据集.
        X2 (np.ndarray): 第二个数据集.
        gamma (float): RBF核的参数.

    返回:
        np.ndarray: 计算出的核矩阵 K.
    """
    K = None
    if ker == 'linear':
        # 线性核：直接计算内积
        K = X1 @ X2.T
    elif ker == 'rbf':
        # 径向基函数(RBF)核：一种常用的非线性核
        n1, n2 = X1.shape[0], X2.shape[0]
        # 计算欧氏距离的平方
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        D = X1_sq @ np.ones((1, n2)) + np.ones((n1, 1)) @ X2_sq.T - 2 * X1 @ X2.T
        # 应用RBF公式
        K = np.exp(-gamma * D)
    return K


class TCA:
    def __init__(self, n_components, kernel_type='rbf', gamma=1.0):
        """
        迁移成分分析 (TCA) 算法的实现。

        参数:
            n_components (int): 降维后的目标维度.
            kernel_type (str): 使用的核函数类型.
            gamma (float): RBF核的gamma参数.
        """
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.W = None  # 存储学习到的转换矩阵
        self.X_fit = None  # 存储用于拟合模型的数据

    def fit(self, Xs, Xt):
        """
        从源域和目标域的样本中，学习TCA的转换矩阵W。
        这一步是计算密集型的，并且有内存瓶颈，因此通常在一个数据子集上执行。

        参数:
            Xs (np.ndarray): 源域特征数据.
            Xt (np.ndarray): 目标域特征数据.
        """
        # 将源域和目标域数据合并
        self.X_fit = np.vstack((Xs, Xt))
        n = self.X_fit.shape[0]
        ns, nt = Xs.shape[0], Xt.shape[0]

        # 计算MMD(最大均值差异)矩阵 L
        # 这个矩阵用于度量源域和目标域分布的差异
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e @ e.T

        # 计算中心化矩阵 H
        # 用于对数据进行中心化，移除均值的影响
        H = np.eye(n) - 1 / n * np.ones((n, n))

        # 计算核矩阵 K
        K = kernel(self.kernel_type, self.X_fit, self.X_fit, gamma=self.gamma)

        # 求解广义特征值问题: (K @ L @ K) W = lambda * (K @ H @ K) W
        # 我们的目标是找到一个投影W，使得投影后域间差异(分子)最小化，同时数据方差(分母)最大化
        term_a = K @ L @ K
        # 添加一个小的正则项(0.1 * np.eye(n))来增加数值稳定性，避免矩阵奇异
        term_b = K @ H @ K + 0.1 * np.eye(n)

        # 特征值分解
        vals, vecs = eigh(term_a, term_b)

        # 按特征值降序对特征向量进行排序
        indices = np.argsort(vals)[::-1]
        # 选择前 n_components 个特征向量作为我们的转换矩阵
        self.W = vecs[:, indices[:self.n_components]]

        return self

    def transform(self, X):
        """
        将学习到的TCA转换应用到新的数据X上。

        参数:
            X (np.ndarray): 需要转换的新数据.

        返回:
            np.ndarray: 转换后的数据.
        """
        if self.W is None or self.X_fit is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法。")

        # 计算新数据X与拟合时使用的数据(self.X_fit)之间的核矩阵
        K = kernel(self.kernel_type, X, self.X_fit, gamma=self.gamma)

        # 应用转换矩阵W进行投影
        return K @ self.W