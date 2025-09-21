# -*- coding: utf-8 -*-
import argparse, yaml
from src.preprocessing.base import build_task

# src/pipelines/preprocess_pipeline.py
import argparse, yaml
from src.preprocessing.base import build_task

def _load_all_tasks():
    import importlib, pkgutil
    for pkg in ("src.preprocessing.tabular", "src.preprocessing.vision", "src.preprocessing.signal"):
        try:
            m = importlib.import_module(pkg)
        except Exception:
            continue
        if not hasattr(m, "__path__"):
            continue
        for _, modname, _ in pkgutil.walk_packages(m.__path__, m.__name__ + "."):
            try:
                importlib.import_module(modname)
            except Exception as e:
                print(f"[preprocess] skip {modname}: {e}")

def run(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    _load_all_tasks()  # <<< 关键：先加载并注册所有任务
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
