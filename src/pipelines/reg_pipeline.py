# -*- coding: utf-8 -*-
"""
reg_pipeline.py
---------------
回归任务的流水线（Pipeline）示例。
负责整个“训练一次模型”的工作流：
1. 读取配置文件（params.yaml）
2. 根据配置加载数据
3. 从 registry 获取对应算法模块（如 XGBoost）
4. 调用算法的 build / fit 接口完成训练与评估
5. 输出指标与产物路径
"""

import argparse, traceback, yaml
from src.core import io, registry


def run(config_path: str):
    """
    主流程函数：
    参数：
        config_path: str —— 配置文件路径（params.yaml）
    """

    # ===== 1. 打印启动信息 =====
    print(f"[reg_pipeline] start, config={config_path}")

    # ===== 2. 读取配置文件 =====
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 确认任务类型正确（这里只支持回归）
    assert cfg.get("task") == "reg", "此流水线仅处理 task=reg"

    # 从配置里获取算法名称（如 XGBoost）
    algo_name = cfg["model"]["name"]
    print(f"[reg_pipeline] task=reg, algo={algo_name}")

    # ===== 3. 获取算法模块 =====
    # registry 会根据 task+algo_name 找到对应的 build.py
    algo = registry.get_algo("reg", algo_name)

    # ===== 4. 读取数据 =====
    print(f"[reg_pipeline] reading dataset: {cfg['dataset']['path']}")
    df = io.read_table(cfg["dataset"]["path"])
    print(f"[reg_pipeline] data shape={df.shape}")

    # ===== 5. 构建模型（占位对象） =====
    model = algo.module.build(cfg)

    # ===== 6. 调用 fit 进行训练与评估 =====
    print("[reg_pipeline] fitting...")
    result = algo.module.fit(model, df, cfg)

    if result is None:
        print("[reg_pipeline] warn: fit() returned None; continuing without result dict")
        result = {}
    # ===== 7. 打印训练完成信息 =====
    print("[reg_pipeline] done.")
    print("[reg_pipeline] metrics:", result.get("metrics"))

    # 打印产物路径（预测结果、模型文件、图表等）
    print("[reg_pipeline] artifacts:")
    for k, v in result.get("artifacts", {}).items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    try:
        # ===== 命令行接口 =====
        # 用法示例：
        # python -m src.pipelines.reg_pipeline --config src/models/XGBoost/params.yaml
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="path to params.yaml")
        args = parser.parse_args()

        # 调用主流程
        run(args.config)

    except Exception as e:
        # ===== 错误处理 =====
        print("[reg_pipeline] ERROR:", e)
        traceback.print_exc()
