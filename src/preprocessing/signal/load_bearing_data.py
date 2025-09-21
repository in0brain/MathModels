# src/preprocessing/signal/load_bearing_data.py
import os
import re
import pandas as pd
from scipy.io import loadmat
from typing import Dict, Any, List
from tqdm import tqdm

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

def _parse_filename(filename: str) -> Dict[str, Any]:
    """
    从源域数据文件名中解析出故障类型、故障直径、载荷等元信息。
    此版本已更新，以兼容 Normal 和 OR@... 格式。
    """
    filename_no_ext = os.path.splitext(filename)[0]

    # 1. 优先处理正常样本
    if "normal" in filename.lower():
        try:
            # 尝试从文件名末尾提取载荷，如 "Normal_0.mat"
            load = int(filename_no_ext.split('_')[-1])
        except (ValueError, IndexError):
            load = -1 # 如果无法解析载荷，标记为-1
        return {"fault_type": "Normal", "fault_size": 0.0, "load": load}

    # 2. 处理故障样本 (更新版正则表达式)
    # 这个正则表达式可以匹配 B007_0, IR014_1, OR007@3_2 等格式
    match = re.match(r"([A-Z]+)(\d{3})(?:@\d+)?_(\d+)", filename_no_ext)
    if not match:
        return {} # 如果不匹配任何规则，返回空字典

    fault_map = {"OR": "OuterRace", "IR": "InnerRace", "B": "Ball"}
    fault_type_code, fault_size_code, load_str = match.groups()

    return {
        "fault_type": fault_map.get(fault_type_code, "Unknown"),
        "fault_size": float(f"0.0{fault_size_code}"),
        "load": int(load_str)
    }


class LoadBearingDataTask(PreprocessTask):
    """
    读取源域和目标域的 .mat 文件，提取振动信号和元数据，
    并将每个信号保存为独立的 Parquet 文件，同时生成一个总的元数据清单。
    此版本使用 os.walk 进行递归文件搜索。
    """
    def run(self) -> Dict[str, Any]:
        source_dir = self.cfg["source_dir"]
        target_dir = self.cfg["target_dir"]
        out_dir = self.cfg["out_dir"]

        meta_records = []

        # --- 使用 os.walk 进行递归搜索 ---
        print(f"Recursively searching for .mat files in source domain: {source_dir}")
        source_filepaths = []
        if os.path.exists(source_dir):
            for root, _, files in os.walk(source_dir):
                for name in files:
                    if name.endswith(".mat"):
                        source_filepaths.append(os.path.join(root, name))

        if not source_filepaths:
            print(f"Warning: No .mat files found in source directory: {source_dir}")
        else:
            print(f"Found {len(source_filepaths)} source files.")

        for filepath in tqdm(source_filepaths, desc="Source Domain"):
            filename = os.path.basename(filepath)
            mat_data = loadmat(filepath)

            rpm_key = next((key for key in mat_data if key.endswith("RPM")), None)
            rpm = float(mat_data[rpm_key][0][0]) if rpm_key else -1.0

            file_meta = _parse_filename(filename)

            for key in mat_data:
                if "_time" in key:
                    sensor = key.split("_")[1]
                    signal_data = mat_data[key].flatten()

                    record = {
                        "domain": "source",
                        "original_file": filename,
                        "sensor": sensor,
                        "rpm": rpm,
                        **file_meta
                    }

                    out_path = os.path.join(out_dir, f"source_{os.path.splitext(filename)[0]}_{sensor}.parquet")
                    io.save_parquet(pd.DataFrame({"signal": signal_data}), out_path)
                    record["signal_path"] = out_path
                    meta_records.append(record)

        # 对目标目录也应用相同的递归搜索逻辑
        print(f"Recursively searching for .mat files in target domain: {target_dir}")
        target_filepaths = []
        if target_dir and os.path.exists(target_dir):
            for root, _, files in os.walk(target_dir):
                for name in files:
                    if name.endswith(".mat"):
                        target_filepaths.append(os.path.join(root, name))

        if not target_filepaths:
             print(f"Warning: No .mat files found in target directory: {target_dir}")
        else:
            print(f"Found {len(target_filepaths)} target files.")


        for filepath in tqdm(target_filepaths, desc="Target Domain"):
            filename = os.path.basename(filepath)
            mat_data = loadmat(filepath)

            signal_key = next((k for k in mat_data if not k.startswith("__")), None)
            if signal_key:
                signal_data = mat_data[signal_key].flatten()

                record = {
                    "domain": "target",
                    "original_file": filename,
                    "sensor": "Unknown",
                    "rpm": 600,
                    "fault_type": "Unknown",
                    "fault_size": -1.0,
                    "load": "Unknown"
                }

                out_path = os.path.join(out_dir, f"target_{os.path.splitext(filename)[0]}.parquet")
                io.save_parquet(pd.DataFrame({"signal": signal_data}), out_path)
                record["signal_path"] = out_path
                meta_records.append(record)

        # --- 保存总的元数据清单 ---
        manifest_path = os.path.join(os.path.dirname(out_dir), "manifest.csv")
        meta_df = pd.DataFrame(meta_records)
        io.save_csv(meta_df, manifest_path)

        return {
            "manifest_path": manifest_path,
            "output_dir": out_dir,
            "source_files_processed": len(source_filepaths),
            "target_files_processed": len(target_filepaths)
        }


register_task("load_bearing_data", LoadBearingDataTask)