# scripts/visualize_waveforms.py
import pandas as pd
import numpy as np
from scipy import signal
import os

from src.core import io, viz


def main():
    """
    (修正版)
    加载CWRU数据集中的四种典型健康状态信号，并绘制一个2x2的时域波形图，
    风格与刘嘉欣论文中的图2-7一致。
    此版本逻辑更健壮，会自动寻找每种故障类型的第一个可用样本，而不是依赖硬编码的工况。
    """
    print("开始生成时域波形对比图...")

    # --- 1. 定义所需参数 ---
    manifest_path = "outputs/data/artifacts/manifest.csv"
    window_size = 1024  # 论文中单个波形图的采样点长度
    target_fs = 12000.0  # 我们分析12k采样率的数据

    # 定义我们要查找的四种状态和对应的中文标签
    states_to_plot = {
        "正常": "Normal",
        "外圈故障": "OuterRace",
        "内圈故障": "InnerRace",
        "滚动体故障": "Ball",
    }

    # --- 2. 加载数据清单 ---
    if not os.path.exists(manifest_path):
        print(f"错误: manifest.csv 未找到于 {manifest_path}")
        print(
            "请先运行数据加载流水线: python -m src.pipelines.preprocess_pipeline --config src/preprocessing/signal/steps_load_data.yaml")
        return

    manifest_df = pd.read_csv(manifest_path)

    # 仅使用源域数据进行绘图
    source_manifest = manifest_df[manifest_df['domain'] == 'source'].copy()

    signals_for_plotting = {}

    # --- 3. (修正逻辑) 为每种状态找到第一个代表性信号 ---
    for title, fault_type in states_to_plot.items():
        print(f"正在查找 '{title}' (type: {fault_type}) 状态的信号...")

        # 筛选出该故障类型的所有记录
        candidate_rows = source_manifest[source_manifest['fault_type'] == fault_type]

        if candidate_rows.empty:
            print(f"警告: 未能在 manifest.csv 中找到任何 '{fault_type}' 类型的记录，将跳过。")
            continue

        # 使用第一个找到的记录作为代表
        target_row = candidate_rows.iloc[0]
        print(f"  > 已选择文件: {target_row['original_file']} 作为代表。")

        # --- 4. 加载并处理信号 ---
        signal_path = target_row['signal_path']
        raw_signal = io.read_parquet(signal_path)["signal"].values

        # 检查并进行重采样 (如果需要)
        current_fs = 12000.0  # 默认
        if "48K" in target_row["original_file"].upper():
            current_fs = 48000.0

        if current_fs != target_fs:
            num_samples = int(len(raw_signal) * target_fs / current_fs)
            resampled_signal = signal.resample(raw_signal, num_samples)
        else:
            resampled_signal = raw_signal

        # 提取第一个窗口作为代表
        if len(resampled_signal) >= window_size:
            signals_for_plotting[title] = resampled_signal[:window_size]
        else:
            print(f"警告: '{title}' 的信号长度不足 {window_size}，跳过。")

    # --- 5. 调用新的绘图函数 ---
    if len(signals_for_plotting) == 4:
        output_path = "outputs/figs/custom/waveform_comparison.png"
        viz.plot_waveform_grid(
            signals=signals_for_plotting,
            out_png=output_path,
            dpi=200
        )
    else:
        print(
            "错误: 未能凑齐4种状态的信号，无法生成对比图。请检查 manifest.csv 是否包含了 'Normal', 'OuterRace', 'InnerRace', 'Ball' 四种类型的数据。")


if __name__ == "__main__":
    main()