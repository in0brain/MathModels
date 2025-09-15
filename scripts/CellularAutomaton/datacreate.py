# -*- coding: utf-8 -*-
"""
生成一个 0/1 矩阵 CSV 作为 CA 初始状态（可编辑后再用）
"""
import numpy as np, pandas as pd, os
np.random.seed(42)

rows, cols = 60, 80
density = 0.25
grid = (np.random.rand(rows, cols) < density).astype(int)

os.makedirs("data", exist_ok=True)
pd.DataFrame(grid).to_csv("data/ca_init.csv", index=False)
print("[OK] saved data/ca_init.csv")
