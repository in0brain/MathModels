# -*- coding: utf-8 -*-
import argparse, traceback, yaml
from src.core import io, registry

def run(config_path: str):
    print(f"[ts_pipeline] start, config={config_path}")  # <— 启动日志
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("task") == "ts", "此流水线仅处理 task=ts"
    algo_name = cfg["model"]["name"]
    print(f"[ts_pipeline] task=ts, algo={algo_name}")

    algo = registry.get_algo("ts", algo_name)

    print(f"[ts_pipeline] reading dataset: {cfg['dataset']['path']}")
    df = io.read_table(cfg["dataset"]["path"])
    print(f"[ts_pipeline] data shape={df.shape}")

    model = algo.module.build(cfg)
    print("[ts_pipeline] fitting...")
    result = algo.module.fit(model, df, cfg)

    print("[ts_pipeline] done.")
    print("[ts_pipeline] metrics:", result.get("metrics"))
    print("[ts_pipeline] artifacts:")
    for k, v in result.get("artifacts", {}).items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="path to params.yaml")
        args = parser.parse_args()
        run(args.config)
    except Exception as e:
        print("[ts_pipeline] ERROR:", e)
        traceback.print_exc()
