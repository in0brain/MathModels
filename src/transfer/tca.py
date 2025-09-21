import numpy as np
from scipy.linalg import eigh

def kernel(ker, X1, X2, gamma):
    """计算核矩阵"""
    K = None
    if ker == 'linear':
        K = X1 @ X2.T
    elif ker == 'rbf':
        n1, n2 = X1.shape[0], X2.shape[0]
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        D = X1_sq @ np.ones((1, n2)) + np.ones((n1, 1)) @ X2_sq.T - 2 * X1 @ X2.T
        K = np.exp(-gamma * D)
    return K

class TCA:
    def __init__(self, n_components, kernel_type='linear', gamma=1.0):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.W = None # 转换矩阵

    def fit_transform(self, Xs, Xt):
        """学习并应用转换"""
        X = np.vstack((Xs, Xt))
        n, m = X.shape
        ns, nt = Xs.shape[0], Xt.shape[0]

        # 计算MMD矩阵L
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e @ e.T

        # 计算中心化矩阵H
        H = np.eye(n) - 1 / n * np.ones((n, n))

        # 计算核矩阵K
        K = kernel(self.kernel_type, X, X, gamma=self.gamma)

        # 求解广义特征值问题
        # (K @ L @ K.T) W = lambda * (K @ H @ K.T) W
        # 为简化和稳定，我们求解 (KHK)^-1 * (KLK) W = lambda W
        # 添加正则项避免奇异矩阵
        term_a = K @ L @ K.T
        term_b = K @ H @ K.T + 0.1 * np.eye(n) # 正则化

        # 特征值分解
        vals, vecs = eigh(term_a, term_b)

        # 特征向量按特征值降序排序
        indices = np.argsort(vals)[::-1]
        self.W = vecs[:, indices[:self.n_components]]

        # 返回转换后的数据
        X_transformed = K @ self.W
        return X_transformed[:ns, :], X_transformed[ns:, :]