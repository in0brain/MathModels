# -*- coding: utf-8 -*-
"""
make_toy_data.py
----------------
生成一个最小可跑的房价预测数据集（house.csv），方便快速验证流水线。
默认输出到 data/house.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression


def main():
    # 随机生成基础特征数据（数值型）
    X, y = make_regression(
        n_samples=300,
        n_features=5,
        noise=15.0,
        random_state=42
    )

    df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3", "feat4", "feat5"])

    # 加入一些“伪房产属性”模拟分类变量
    rng = np.random.default_rng(42)
    df["city"] = rng.choice(["NY", "LA", "SF", "CHI"], size=len(df))
    df["type"] = rng.choice(["Apt", "House", "Town"], size=len(df))

    # 加入目标列
    df["price"] = y + rng.normal(0, 5, size=len(y)) + \
                  (df["city"].map({"NY": 50, "LA": 30, "SF": 40, "CHI": 20}).values) * 100

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "house.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[make_toy_data] generated -> {out_path}, shape={df.shape}")


if __name__ == "__main__":
    main()
