# -*- coding: utf-8 -*-
"""
生成示例数据：house.csv（训练集，含 price）
和 house_new.csv（预测用，不含 price）
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_house_data(n=200, with_price=True):
    # 基础特征
    area = np.random.randint(50, 200, size=n)              # 面积
    bedrooms = np.random.randint(1, 6, size=n)             # 卧室数
    bathrooms = np.random.randint(1, 4, size=n)            # 卫生间数
    location = np.random.choice(["A", "B", "C"], size=n)   # 地段
    year_built = np.random.randint(1990, 2023, size=n)     # 建造年份

    data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "year_built": year_built,
    }

    if with_price:
        # price 计算逻辑（带噪声）
        base = 50000 + area * 2000 + bedrooms * 30000 + bathrooms * 20000
        loc_factor = np.where(location == "A", 1.3, np.where(location == "B", 1.1, 0.9))
        year_factor = 1 + (year_built - 1990) * 0.005
        noise = np.random.normal(0, 20000, size=n)
        price = base * loc_factor * year_factor + noise
        data["price"] = price.astype(int)

    return pd.DataFrame(data)

def main():
    os.makedirs("data", exist_ok=True)

    # 生成训练数据（含 price）
    df_train = generate_house_data(200, with_price=True)
    df_train.to_csv("data/house.csv", index=False, encoding="utf-8")
    print("[OK] 生成 data/house.csv (200 条)")

    # 生成预测数据（不含 price）
    df_new = generate_house_data(20, with_price=False)   # 20 条新数据
    df_new.to_csv("data/house_new.csv", index=False, encoding="utf-8")
    print("[OK] 生成 data/house_new.csv (20 条)")

if __name__ == "__main__":
    main()
