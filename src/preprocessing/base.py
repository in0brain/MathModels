# -*- coding: utf-8 -*-
# 定义任务基类与简易注册器，让不同预处理子模块（vision/tabular/…）都能以“任务”的形式被 pipeline 调用。
#
# 关键接口：
#
# class PreprocessTask: setup(cfg), run() -> Dict[str, Any], teardown()
#
# register_task(name: str, cls: Type[PreprocessTask])
#
# build_task(name, **kwargs)

# -*- coding: utf-8 -*-  # 指定源码文件的编码为 UTF-8，避免中文注释乱码
from typing import Dict, Any, Type  # 导入类型注解用的别名（字典/任意/类型）

_TASKS = {}  # 全局字典：名字 -> 任务类，用于注册与构建

class PreprocessTask:
    """预处理任务的基类，所有自定义任务都应继承它"""
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg  # 保存任务的配置（字典）

    def setup(self):
        """前置步骤：资源初始化/检查（子类可覆盖）"""
        pass

    def run(self) -> Dict[str, Any]:
        """核心执行逻辑（必须由子类实现），返回结果字典"""
        raise NotImplementedError

    def teardown(self):
        """收尾步骤：资源释放（子类可覆盖）"""
        pass

def register_task(name: str, cls: Type[PreprocessTask]):
    """把任务类注册到全局字典中，供名字查找"""
    _TASKS[name] = cls  # 存一份映射

def build_task(name: str, cfg: Dict[str, Any]) -> PreprocessTask:
    """通过名字和配置构建具体任务实例"""
    return _TASKS[name](cfg)  # 调用对应类的构造函数
