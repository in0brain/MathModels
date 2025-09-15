# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
from sklearn.datasets import make_classification
np.random.seed(42)

def main():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, class_sep=1.0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["label"] = y
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/clf_demo.csv", index=False, encoding="utf-8")
    print("[OK] data/clf_demo.csv")

if __name__ == "__main__":
    main()
