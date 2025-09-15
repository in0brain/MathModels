# -*- coding: utf-8 -*-
"""
通用推理入口：
- 通过 registry 定位算法模块
- 加载已训练的 .pkl 模型
- 读取新数据（--data）
- 调用算法模块的 inference() 完成推理
用法：
python -m src.inference.runner --task ts --algo MarkovChain \
    --model outputs/models/demo_markov.pkl \
    --data data/new_states.csv \
    --config src/models/MarkovChain/params.yaml \
    --tag new_markov
"""
import argparse, yaml, joblib
from src.core import io, registry

def run(args):
    # 1) 读取 config（用于列名/输出目录等）
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 允许用命令行覆盖数据路径和 tag
    if args.data:
        cfg.setdefault("dataset", {})["path"] = args.data
    if args.tag:
        cfg.setdefault("outputs", {})["tag"] = args.tag

    # 2) 获取算法模块 & 加载模型
    algo = registry.get_algo(args.task, args.algo)
    model = joblib.load(args.model)

    # 3) 读取新数据并推理
    df = io.read_table(cfg["dataset"]["path"])
    result = algo.module.inference(model, df, cfg)

    print("[inference] done. artifacts:")
    for k, v in result.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["reg","clf","ts","clu"], help="任务类型")
    p.add_argument("--algo", required=True, help="算法别名，如 MarkovChain")
    p.add_argument("--model", required=True, help="已训练模型 .pkl 路径")
    p.add_argument("--data", default=None, help="新数据路径（可覆盖 config.dataset.path）")
    p.add_argument("--config", required=True, help="算法 config（用于列名/输出目录）")
    p.add_argument("--tag", default=None, help="输出文件命名标签（可覆盖 config.outputs.tag）")
    args = p.parse_args()
    run(args)
