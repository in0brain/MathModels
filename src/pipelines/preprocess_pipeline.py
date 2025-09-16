# -*- coding: utf-8 -*-
import argparse, yaml
from src.preprocessing.base import build_task

def run(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    for step in cfg["steps"]:
        name = step["name"]; params = step.get("params", {})
        task = build_task(name, params)
        task.setup(); result = task.run(); task.teardown()
        print(f"[preprocess] step={name} -> {result}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a steps yaml")
    args = ap.parse_args()
    run(args.config)
