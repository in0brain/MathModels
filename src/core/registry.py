# -*- coding: utf-8 -*-
"""
注册中心：自动发现 models/*/build.py 并注册算法
约定：每个 build.py 至少提供常量 TASK, ALGO，以及函数 build/fit/predict/plot（plot 可选）
"""
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class AlgoSpec:
    task: str                  # "reg" | "clf" | "ts" | "clu"
    name: str                  # 算法显示名/别名（如 "MarkovChain"）
    module: Any                # 模块对象（含 build/fit/predict/plot）

REGISTRY: Dict[str, AlgoSpec] = {}

def _discover_models():
    # 中文注释：扫描 src.models 包下的所有子模块中的 build.py
    package = importlib.import_module("src.models")
    for finder, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if modname.endswith(".build"):
            m = importlib.import_module(modname)
            task = getattr(m, "TASK", None)
            algo = getattr(m, "ALGO", None)
            if not task or not algo:
                continue
            key = f"{task}:{algo}".lower()
            REGISTRY[key] = AlgoSpec(task=task, name=algo, module=m)

# 在导入时即完成注册
_discover_models()

def get_algo(task: str, name: str) -> AlgoSpec:
    """中文注释：按任务与算法名获取注册项"""
    key = f"{task}:{name}".lower()
    if key not in REGISTRY:
        raise KeyError(f"Algorithm not registered: {key}")
    return REGISTRY[key]
