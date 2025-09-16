# -*- coding: utf-8 -*-
"""
schemas.py
---------
定义视觉预处理阶段的中间数据表结构（列名/含义）+ 轻量校验工具。
"""

from typing import Dict, List
import pandas as pd
import numpy as np

# 1) 检测框逐帧明细（tracks.parquet）
TRACKS_COLUMNS: Dict[str, str] = {
    "frame": "帧号（int）",
    "x1": "bbox左上x（float）",
    "y1": "bbox左上y（float）",
    "x2": "bbox右下x（float）",
    "y2": "bbox右下y（float）",
    "cls": "类别id（int）",
    "conf": "置信度（float,0~1）",
    "site": "观测点（str）",
    "chunk": "切片编号（str, 3位）",
}

# 2) 单切片秒级聚合（agg.csv）
AGG_COLUMNS: Dict[str, str] = {
    "sec": "片内秒（int）",
    "q": "该秒过线车辆数（int）",
    "v": "该秒平均速度m/s（float，可NaN）",
    "k": "该秒密度veh/m（float，可NaN）",
    "site": "观测点（str）",
    "chunk": "切片编号（str）",
}

# 3) 站点级汇总（traffic_site*.csv）
TRAFFIC_COLUMNS: Dict[str, str] = {
    "timestamp": "时间戳（pd.Timedelta 或 pd.Timestamp）",
    "q": "窗口流量（int/float）",
    "v": "窗口速度（float，可NaN）",
    "k": "窗口密度（float，可NaN）",
    "site": "观测点（str）",
}

# ---------- 轻量校验工具 ----------

def _require_columns(df: pd.DataFrame, required: List[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[schema:{name}] 缺失列: {missing}，实际列={list(df.columns)}")

def validate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """校验/尽量整理 tracks 表"""
    _require_columns(df, list(TRACKS_COLUMNS.keys()), "tracks")
    # 基本类型/范围温和检查（尽量不抛错，只在明显异常时报错）
    # 帧号
    if not np.issubdtype(df["frame"].dtype, np.integer):
        try: df["frame"] = df["frame"].astype(int)
        except Exception as e: raise TypeError(f"[schema:tracks] frame 需为int: {e}")
    # 坐标/置信度
    for c in ["x1","y1","x2","y2","conf"]:
        if not np.issubdtype(df[c].dtype, np.floating):
            try: df[c] = df[c].astype(float)
            except Exception as e: raise TypeError(f"[schema:tracks] {c} 需为float: {e}")
    # 类别
    if not np.issubdtype(df["cls"].dtype, np.integer):
        try: df["cls"] = df["cls"].astype(int)
        except Exception as e: raise TypeError(f"[schema:tracks] cls 需为int: {e}")
    # site/chunk 转字符串
    for c in ["site","chunk"]:
        df[c] = df[c].astype(str)
    return df

def validate_agg(df: pd.DataFrame) -> pd.DataFrame:
    """校验/尽量整理 agg 表"""
    _require_columns(df, list(AGG_COLUMNS.keys()), "agg")
    # sec / q
    for c in ["sec","q"]:
        if not np.issubdtype(df[c].dtype, np.integer):
            try: df[c] = df[c].astype(int)
            except Exception as e: raise TypeError(f"[schema:agg] {c} 需为int: {e}")
    # v/k
    for c in ["v","k"]:
        if not (np.issubdtype(df[c].dtype, np.floating) or np.issubdtype(df[c].dtype, np.integer)):
            try: df[c] = df[c].astype(float)
            except Exception as e: raise TypeError(f"[schema:agg] {c} 需为float: {e}")
    for c in ["site","chunk"]:
        df[c] = df[c].astype(str)
    return df

def validate_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """校验/尽量整理 站点级 traffic 表"""
    _require_columns(df, list(TRAFFIC_COLUMNS.keys()), "traffic")
    # 时间戳列：尽量转为 pandas 时间类型
    if not (np.issubdtype(df["timestamp"].dtype, np.datetime64) or isinstance(df["timestamp"].dtype, pd.TimedeltaDtype)):
        try:
            # 优先转 Timestamp；失败则尝试 Timedelta
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="ignore")
            if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                df["timestamp"] = pd.to_timedelta(df["timestamp"], errors="coerce")
        except Exception as e:
            raise TypeError(f"[schema:traffic] timestamp 需为时间类型: {e}")
    for c in ["q","v","k"]:
        if c in df.columns:
            try: df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception as e: raise TypeError(f"[schema:traffic] {c} 数值列: {e}")
    df["site"] = df["site"].astype(str)
    return df
