# -*- coding: utf-8 -*-
"""
Cellular Automaton（元胞自动机）：
- 通用B*/S*外总和规则（如 "B3/S23" 生命游戏）
- 支持 moore(8邻域)/neumann(4邻域)，wrap/constant 边界
- 支持随机初始化或从CSV读取初始状态
- 产出：演化GIF、最终状态PNG、初末状态CSV
调参教学见文末“# === 调参指引 ===”
"""
from typing import Dict, Any, Tuple, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.core import io

TASK = "ts"
ALGO = "CellularAutomaton"

# ---------- 工具：规则解析 ----------
def parse_rule(rule: str) -> Tuple[set, set]:
    """将 'B3/S23' 解析为 (birth_set, survive_set)"""
    rule = rule.upper().replace(" ", "")
    if "/" not in rule:
        raise ValueError("规则需形如 'B3/S23'")
    bpart, spart = rule.split("/")
    birth = set(int(c) for c in bpart.replace("B", "") if c.isdigit())
    surv  = set(int(c) for c in spart.replace("S", "") if c.isdigit())
    return birth, surv

# ---------- 工具：邻域计数 ----------
def neighbor_count(grid: np.ndarray, neighborhood: str, boundary: str) -> np.ndarray:
    """计算每个细胞的活邻居数量"""
    if neighborhood not in {"moore", "neumann"}:
        raise ValueError("neighborhood 必须为 'moore' 或 'neumann'")
    # 卷积实现更快，这里用移位叠加，直观易读
    shifts = []
    if neighborhood == "moore":
        shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    else:  # neumann
        shifts = [(-1,0), (0,-1), (0,1), (1,0)]

    H, W = grid.shape
    total = np.zeros_like(grid, dtype=np.int16)
    for dx, dy in shifts:
        if boundary == "wrap":
            total += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
        else:
            # constant 边界：用零填充后切片叠加
            pad = ((1,1),(1,1))
            g = np.pad(grid, pad, mode="constant")
            total += g[1+dx:H+1+dx, 1+dy:W+1+dy]
    return total

# ---------- 初始化 ----------
def init_grid(cfg: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    ds = cfg["dataset"]
    if ds.get("init_random", True) or not ds.get("path"):
        rows = int(ds.get("grid_rows", 60))
        cols = int(ds.get("grid_cols", 80))
        density = float(ds.get("density", 0.25))
        grid = (rng.random((rows, cols)) < density).astype(np.uint8)
    else:
        df = io.read_table(ds["path"])
        # 允许csv为0/1矩阵或单列展开；这里假设0/1矩阵
        grid = df.values.astype(np.float32)
        thr = float(cfg.get("preprocess", {}).get("binarize_threshold", 0.5))
        grid = (grid > thr).astype(np.uint8)
    return grid

# ---------- 渲染 ----------
def render_grid(grid: np.ndarray, out_png: str, dpi: int = 120, cmap: str = "gray"):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    io.ensure_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ---------- 对外接口 ----------
def build(cfg: Dict[str, Any]):
    """构建“模型对象”——此处仅保存规则等参数"""
    params = cfg["model"]["params"]
    birth, survive = parse_rule(params.get("rule", "B3/S23"))
    return {
        "params": {
            "birth": birth,
            "survive": survive,
            "neighborhood": params.get("neighborhood", "moore"),
            "boundary": params.get("boundary", "wrap"),
            "steps": int(params.get("steps", 120)),
            "fps": int(params.get("fps", 8)),
            "render_stride": int(params.get("render_stride", 1)),
            "cmap": params.get("cmap", "gray"),
            "save_frames": bool(params.get("save_frames", False)),
        }
    }

def step_once(grid: np.ndarray, model) -> np.ndarray:
    """演化一步"""
    p = model["params"]
    nbh = neighbor_count(grid, p["neighborhood"], p["boundary"])
    birth, survive = p["birth"], p["survive"]
    born  = (grid == 0) & np.isin(nbh, list(birth))
    stay  = (grid == 1) & np.isin(nbh, list(survive))
    nxt = np.zeros_like(grid)
    nxt[born | stay] = 1
    return nxt

def fit(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    “训练”在CA里即为：根据初始状态演化 steps 步并产出可视化/工件。
    df 在这里不使用（保留签名统一性）。
    """
    base = cfg["outputs"]["base_dir"]; tag = cfg["outputs"].get("tag", ALGO)
    dpi = int(cfg["viz"].get("dpi", 120)); seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    # 初始化网格
    grid0 = init_grid(cfg, rng)

    # 保存初始状态
    init_csv = io.out_path_artifacts(base, ALGO, f"{tag}_init.csv")
    io.save_csv(pd.DataFrame(grid0), init_csv)

    # 演化 & 渲染
    steps = model["params"]["steps"]; stride = model["params"]["render_stride"]
    cmap = model["params"]["cmap"]; fps = model["params"]["fps"]
    frames: List[np.ndarray] = []
    grid = grid0.copy()

    # 每步可插入“探针”统计：活细胞数量等（可写到 artifacts）
    alive_series = []

    for t in range(steps + 1):
        if t % stride == 0:
            # 渲染这一帧到内存（合成GIF）并可选保存PNG
            fig_path = os.path.join(base, "figs", "ts", f"{tag}_frame_{t:04d}.png")
            render_grid(grid, fig_path, dpi=dpi, cmap=cmap)
            if not model["params"]["save_frames"]:
                # 不保留PNG则读取为帧后删除文件，节省空间（简单起见，我们直接读图像）
                img = imageio.imread(fig_path)
                frames.append(img)
                try:
                    os.remove(fig_path)
                except OSError:
                    pass
            else:
                frames.append(imageio.imread(fig_path))

        alive_series.append(int(grid.sum()))
        if t < steps:
            grid = step_once(grid, model)

    # 最终状态保存
    final_csv = io.out_path_artifacts(base, ALGO, f"{tag}_final.csv")
    io.save_csv(pd.DataFrame(grid), final_csv)

    # 合成GIF
    gif_path = os.path.join(base, "figs", "ts", f"{tag}_evolution.gif")
    imageio.mimsave(gif_path, frames, fps=fps)

    # 最终快照
    final_png = os.path.join(base, "figs", "ts", f"{tag}_final.png")
    render_grid(grid, final_png, dpi=dpi, cmap=cmap)

    # 报告（这里写入简单统计）
    rep_path = os.path.join(base, "reports", f"{tag}_metrics.json")
    io.save_json({
        "steps": steps,
        "alive_series_first": alive_series[:10],
        "alive_last": alive_series[-1],
        "grid_shape": list(grid.shape),
        "rule": list(model["params"]["birth"]),  # 仅示意
    }, rep_path)

    # 返回统一结构
    return {
        "metrics": {},   # CA 无“预测误差”，此处留空
        "artifacts": {
            "init_csv": init_csv,
            "final_csv": final_csv,
            "gif": gif_path,
            "final_png": final_png,
            "report_path": rep_path,
        }
    }

def inference(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    再预测（给定新的初始状态或配置，重复演化并产出GIF）
    - 行为与 fit 类似，但文件名带 *_infer，且不覆盖训练产物
    """
    base = cfg["outputs"]["base_dir"]; tag = cfg["outputs"].get("tag", ALGO) + "_infer"
    dpi = int(cfg["viz"].get("dpi", 120)); seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    grid = init_grid(cfg, rng)
    steps = model["params"]["steps"]; stride = model["params"]["render_stride"]
    cmap = model["params"]["cmap"]; fps = model["params"]["fps"]

    frames: List[np.ndarray] = []
    for t in range(steps + 1):
        if t % stride == 0:
            fig_path = os.path.join(base, "figs", "ts", f"{tag}_frame_{t:04d}.png")
            render_grid(grid, fig_path, dpi=dpi, cmap=cmap)
            img = imageio.imread(fig_path)
            frames.append(img)
            try:
                os.remove(fig_path)
            except OSError:
                pass
        if t < steps:
            grid = step_once(grid, model)

    gif_path = os.path.join(base, "figs", "ts", f"{tag}_evolution.gif")
    imageio.mimsave(gif_path, frames, fps=fps)
    final_png = os.path.join(base, "figs", "ts", f"{tag}_final.png")
    render_grid(grid, final_png, dpi=dpi, cmap=cmap)

    return {"gif": gif_path, "final_png": final_png}

# === 调参指引 ===
# 1) 形态变化幅度不明显（像“静止/振荡”太快结束）：
#    - steps ↑（演化更久）
#    - density ↑ 或 ↓（改变初始活细胞比例）
#    - rule 改成更“活跃”的，如 B36/S23（HighLife）
# 2) 花样太噪、看不清：
#    - render_stride ↑（每隔更多步渲染，GIF短一点）
#    - cmap 改为 "Greys"、"viridis" 等
# 3) 边界效应明显：
#    - boundary 换 "wrap"（环绕）避免边界吸收
# 4) 规则族：
#    - 生命游戏：B3/S23
#    - HighLife：B36/S23
#    - Seeds：B2/S
#    - Day&Night：B3678/S34678
# 5) 邻域：
#    - moore：8邻域（更丰富）
#    - neumann：4邻域（结构更硬朗）
