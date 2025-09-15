# -*- coding: utf-8 -*-
from typing import Dict, Any, List
import os, numpy as np, pandas as pd
from src.core import io, metrics, viz

TASK = "ts"
ALGO = "MarkovChain"

def _fit_transition_matrix(states: List, smoothing: float):
    labels = sorted(list(set(states)))
    idx = {s: i for i, s in enumerate(labels)}
    n = len(labels)
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(len(states) - 1):
        a, b = idx[states[i]], idx[states[i + 1]]
        C[a, b] += 1.0
    P = C + 1e-12 + float(smoothing)                       # 中文注释：加性平滑
    P /= P.sum(axis=1, keepdims=True, where=P.sum(axis=1, keepdims=True) > 0)
    return P, labels

def _predict_next(P: np.ndarray, labels: List, state):
    idx = {s: i for i, s in enumerate(labels)}
    if state not in idx:  # 未见过状态：回退到全局最常去的列
        j = int(np.argmax(P.sum(axis=0)))
        return labels[j]
    i = idx[state]
    j = int(np.argmax(P[i]))
    return labels[j]

def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = cfg.get("model", {}).get("params", {})
    return {"params": {
                "order": int(params.get("order", 1)),
                "smoothing": float(params.get("smoothing", 1e-6)),
                "topk_eval": int(params.get("topk_eval", 1)),
            },
            "P": None, "labels": None}

def fit(model: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # === 读取列名 & 切分比例 ===
    state_col = cfg["dataset"]["state_col"]
    test_ratio = float(cfg["dataset"].get("test_ratio", 0.2))
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)

    if cfg.get("preprocess", {}).get("dropna", True):
        df = df.dropna(subset=[state_col])

    states = df[state_col].astype(str).tolist()
    n = len(states)
    split = int(n * (1 - test_ratio))
    train_seq, test_seq = states[:split], states[split:]

    # === 训练转移矩阵 ===
    P, labels = _fit_transition_matrix(train_seq, model["params"]["smoothing"])
    model.update({"P": P, "labels": labels})

    # === 保存训练工件（转移矩阵 CSV）到 artifacts/MarkovChain ===
    tm_path = io.out_path_artifacts(base_dir, ALGO, f"{tag}_transition_matrix.csv")
    io.save_csv(pd.DataFrame(P, index=labels, columns=labels), tm_path)

    # === 测试集逐步 next-state 预测 ===
    preds = [_predict_next(P, labels, s) for s in test_seq[:-1]]
    y_true = test_seq[1:]

    # === 指标 ===
    met_keys = cfg.get("eval", {}).get("metrics", ["acc"])
    result = metrics.evaluate_markov(y_true, preds, metrics=tuple(met_keys))

    # === 可视化 ===
    fig_paths = []
    if cfg.get("viz", {}).get("enabled", True):
        dpi = cfg["viz"].get("dpi", 160)
        figs_base = os.path.join(base_dir, "figs", "ts")
        fig_paths.append(
            viz.plot_transition_heatmap(P, labels, os.path.join(figs_base, f"{tag}_trans_heatmap.png"), dpi=dpi)
        )
        if len(preds) > 0:
            fig_paths.append(
                viz.plot_sequence_compare(y_true, preds, os.path.join(figs_base, f"{tag}_seq_compare.png"), dpi=dpi)
            )

    # === 预测结果保存到 predictions/MarkovChain ===
    pred_path = io.out_path_predictions(base_dir, ALGO, f"{tag}_preds.csv")
    io.save_csv(pd.DataFrame({"true_next": y_true, "pred_next": preds}), pred_path)

    # === 模型与报告 ===
    model_path = os.path.join(base_dir, "models", f"{tag}.pkl")
    io.save_model({"P": P, "labels": labels, "params": model["params"]}, model_path)
    rep_path = os.path.join(base_dir, "reports", f"{tag}_metrics.json")
    io.save_json({"metrics": result, "n_train": len(train_seq), "n_test": len(test_seq)}, rep_path)

    return {"metrics": result,
            "artifacts": {
                "transition_matrix_csv": tm_path,
                "predictions_csv": pred_path,
                "model_path": model_path,
                "report_path": rep_path,
                "figs": fig_paths}}

# === 新增：通用推理适配（供 src/inference/runner.py 调用） ===
def inference(model: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """中文注释：从 DataFrame 中取 state 列，逐步预测 next-state；不做训练"""
    state_col = cfg["dataset"]["state_col"]
    base_dir = cfg["outputs"]["base_dir"]
    tag = cfg["outputs"].get("tag", ALGO)

    states = df[state_col].astype(str).tolist()
    preds = [_predict_next(model["P"], model["labels"], s) for s in states[:-1]]

    # 保存到 predictions/MarkovChain/
    pred_path = io.out_path_predictions(base_dir, ALGO, f"{tag}_infer_preds.csv")
    out = pd.DataFrame({"input_state": states[:-1], "pred_next": preds})
    io.save_csv(out, pred_path)
    return {"predictions_csv": pred_path}
