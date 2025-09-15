# -*- coding: utf-8 -*-
"""
生成一个马尔可夫链演示数据（CSV），包含两列：
- t: 序号/时间步
- state: 离散状态（字符串或类别编号）
"""
import os
import numpy as np
import pandas as pd

def main():
    np.random.seed(42)

    # --- 定义状态空间与状态转移矩阵 ---
    states = ["A", "B", "C"]  # 状态集合
    P = np.array([
        [0.80, 0.15, 0.05],  # A -> A/B/C
        [0.20, 0.60, 0.20],  # B -> A/B/C
        [0.10, 0.30, 0.60],  # C -> A/B/C
    ])

    n = 300  # 序列长度
    seq = []
    cur = np.random.choice(len(states))     # 随机初始状态
    seq.append(states[cur])

    for _ in range(n - 1):                  # 逐步采样
        cur = np.random.choice(len(states), p=P[cur])
        seq.append(states[cur])

    df = pd.DataFrame({"t": np.arange(n), "state": seq})

    # 👉【需要修改】如需更换文件名/路径，改这里
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "markov_demo.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved to {out_path}")

if __name__ == "__main__":
    main()
