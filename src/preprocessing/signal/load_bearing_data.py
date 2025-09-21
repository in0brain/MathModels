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
    """从源域数据文件名中解析出故障类型、故障直径、载荷等元信息"""
    # 正常样本
    if "N" in filename:
        return {"fault_type": "Normal", "fault_size": 0.0, "load": int(filename.split("_")[-1])}

    # 故障样本
    match = re.match(r"([A-Z]+)(\d+)_(\d+)", filename)
    if not match:
        return {}

    fault_map = {"OR": "OuterRace", "IR": "InnerRace", "B": "Ball"}
    fault_type, fault_size_code, load = match.groups()

    return {
        "fault_type": fault_map.get(fault_type, "Unknown"),
        "fault_size": float(f"0.0{fault_size_code}"),
        "load": int(load)
    }


class LoadBearingDataTask(PreprocessTask):
    """
    读取源域和目标域的 .mat 文件，提取振动信号和元数据，
    并将每个信号保存为独立的 Parquet 文件，同时生成一个总的元数据清单。
    """

    def run(self) -> Dict[str, Any]:
        source_dir = self.cfg["source_dir"]
        target_dir = self.cfg.get("target_dir")
        out_dir = self.cfg["out_dir"]

        meta_records = []

        # --- 处理源域数据 ---
        print(f"Processing source domain data from: {source_dir}")
        source_files = [f for f in os.listdir(source_dir) if f.endswith(".mat")]
        for filename in tqdm(source_files, desc="Source Domain"):
            filepath = os.path.join(source_dir, filename)
            mat_data = loadmat(filepath)

            rpm_key = next((key for key in mat_data if key.endswith("RPM")), None)
            rpm = float(mat_data[rpm_key][0][0]) if rpm_key else -1.0

            file_meta = _parse_filename(os.path.splitext(filename)[0])

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

                    # 保存信号数据
                    out_path = os.path.join(out_dir, f"source_{os.path.splitext(filename)[0]}_{sensor}.parquet")
                    io.save_parquet(pd.DataFrame({"signal": signal_data}), out_path)
                    record["signal_path"] = out_path
                    meta_records.append(record)

        # --- 处理目标域数据 (如果提供了) ---
        if target_dir and os.path.exists(target_dir):
            print(f"Processing target domain data from: {target_dir}")
            target_files = [f for f in os.listdir(target_dir) if f.endswith(".mat")]
            for filename in tqdm(target_files, desc="Target Domain"):
                filepath = os.path.join(target_dir, filename)
                mat_data = loadmat(filepath)

                # 目标域数据结构可能更简单，假设只有一个key包含信号
                signal_key = next((k for k in mat_data if not k.startswith("__")), None)
                if signal_key:
                    signal_data = mat_data[signal_key].flatten()

                    record = {
                        "domain": "target",
                        "original_file": filename,
                        "sensor": "Unknown",  # 目标域传感器位置未知
                        "rpm": 600,  # 根据文档约为600rpm
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
            "source_files_processed": len(source_files),
            "target_files_processed": len(target_files) if target_dir else 0
        }


register_task("load_bearing_data", LoadBearingDataTask)