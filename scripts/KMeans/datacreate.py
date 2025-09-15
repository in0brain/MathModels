# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def main():
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    df = pd.DataFrame(X, columns=["f1", "f2"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/clu_demo.csv", index=False, encoding="utf-8")
    print("[OK] 生成 data/clu_demo.csv")

if __name__ == "__main__":
    main()
