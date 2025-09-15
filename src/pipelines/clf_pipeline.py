# -*- coding: utf-8 -*-
import argparse, traceback, yaml
from src.core import io, registry

def run(config_path: str):
    print(f"[clf_pipeline] start, config={config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("task") == "clf", "此流水线仅处理 task=clf"
    algo_name = cfg["model"]["name"]
    print(f"[clf_pipeline] task=clf, algo={algo_name}")

    algo = registry.get_algo("clf", algo_name)

    print(f"[clf_pipeline] reading dataset: {cfg['dataset']['path']}")
    df = io.read_table(cfg["dataset"]["path"])
    print(f"[clf_pipeline] data shape={df.shape}")

    model = algo.module.build(cfg)
    print("[clf_pipeline] fitting...")
    result = algo.module.fit(model, df, cfg)

    print("[clf_pipeline] done.")
    print("[clf_pipeline] metrics:", result.get("metrics"))
    print("[clf_pipeline] artifacts:")
    for k, v in result.get("artifacts", {}).items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="path to params.yaml")
        args = parser.parse_args()
        run(args.config)
    except Exception as e:
        print("[clf_pipeline] ERROR:", e)
        traceback.print_exc()
