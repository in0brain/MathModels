# src/preprocessing/signal/feature_extraction.py
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Dict, Any, List
from tqdm import tqdm
import os
from src.core import io
from src.preprocessing.base import PreprocessTask, register_task

# 新增：导入 pywt 用于连续小波变换
# 请确保已安装此库: pip install PyWavelets
try:
    import pywt
except ImportError:
    raise ImportError(
        "CWT feature extraction requires the 'PyWavelets' library. Please install it using 'pip install PyWavelets' and add 'PyWavelets' to your requirements.txt")


def get_time_domain_features(window: np.ndarray) -> Dict[str, float]:
    """提取单个信号窗口的时域特征"""
    rms = np.sqrt(np.mean(window ** 2))
    return {
        'td_mean': np.mean(window),
        'td_std': np.std(window),
        'td_rms': rms,
        'td_skew': skew(window),
        'td_kurtosis': kurtosis(window),
        'td_max': np.max(window),
        'td_min': np.min(window),
        'td_peak_to_peak': np.max(window) - np.min(window),
        'td_crest_factor': np.max(np.abs(window)) / (rms + 1e-9),
        'td_shape_factor': rms / (np.mean(np.abs(window)) + 1e-9),
    }


def get_freq_domain_features(window: np.ndarray, fs: float) -> Dict[str, float]:
    """提取单个信号窗口的频域特征"""
    n = len(window)
    fft_vals = np.fft.rfft(window)
    fft_freq = np.fft.rfftfreq(n, 1.0 / fs)
    fft_mag = np.abs(fft_vals) / n

    # 找到峰值频率
    peak_freq_index = np.argmax(fft_mag)
    peak_freq = fft_freq[peak_freq_index]

    return {
        'fd_peak_freq': peak_freq,
        'fd_peak_mag': fft_mag[peak_freq_index],
        'fd_mean_mag': np.mean(fft_mag),
    }


def get_envelope_features(window: np.ndarray, fs: float) -> Dict[str, float]:
    """提取信号包络谱的特征"""
    analytic_signal = signal.hilbert(window)
    envelope = np.abs(analytic_signal)

    # 移除直流分量
    envelope = envelope - np.mean(envelope)

    n = len(envelope)
    fft_vals = np.fft.rfft(envelope)
    fft_freq = np.fft.rfftfreq(n, 1.0 / fs)
    fft_mag = np.abs(fft_vals) / n

    peak_freq_index = np.argmax(fft_mag)
    peak_freq = fft_freq[peak_freq_index]

    return {
        'env_peak_freq': peak_freq,
        'env_peak_mag': fft_mag[peak_freq_index]
    }


def get_cwt_features(window: np.ndarray, fs: float, wavelet_name: str = 'morl', num_scales: int = 64) -> Dict[
    str, float]:
    """
    (新增) 提取单个信号窗口的连续小波变换（CWT）特征。
    """
    # 定义尺度，通常与小波和信号长度有关
    scales = np.arange(1, num_scales + 1)

    # 执行CWT
    # sampling_period = 1.0 / fs
    coefficients, frequencies = pywt.cwt(window, scales, wavelet_name, sampling_period=1.0 / fs)

    # CWT系数是复数，我们通常使用其幅度的平方（能量）
    cwt_energy = np.abs(coefficients) ** 2

    # 提取特征
    total_energy = np.sum(cwt_energy)
    mean_energy_per_scale = np.mean(cwt_energy, axis=1)

    # 定义频带（示例，可以根据物理先验进行调整）
    low_freq_band = (frequencies > 0) & (frequencies <= fs / 8)
    mid_freq_band = (frequencies > fs / 8) & (frequencies <= fs / 4)
    high_freq_band = (frequencies > fs / 4) & (frequencies <= fs / 2)

    # 计算不同频带的能量
    energy_low = np.sum(cwt_energy[low_freq_band, :]) if np.any(low_freq_band) else 0
    energy_mid = np.sum(cwt_energy[mid_freq_band, :]) if np.any(mid_freq_band) else 0
    energy_high = np.sum(cwt_energy[high_freq_band, :]) if np.any(high_freq_band) else 0

    return {
        'cwt_total_energy': total_energy,
        'cwt_mean_energy': np.mean(mean_energy_per_scale),
        'cwt_std_energy': np.std(mean_energy_per_scale),
        'cwt_energy_low_freq_band': energy_low,
        'cwt_energy_mid_freq_band': energy_mid,
        'cwt_energy_high_freq_band': energy_high
    }


class ExtractSignalFeaturesTask(PreprocessTask):
    """
    从信号数据中分窗并提取时域、频域、包络谱和CWT特征。
    """

    def run(self) -> Dict[str, Any]:
        manifest_path = self.cfg["manifest_path"]
        target_fs = self.cfg["target_fs"]
        window_size = self.cfg["window_size"]
        overlap = self.cfg["overlap"]

        manifest_df = pd.read_csv(manifest_path)
        all_features = []

        step = int(window_size * (1 - overlap))

        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Features"):
            signal_path = row["signal_path"]
            df_signal = io.read_parquet(signal_path)
            raw_signal = df_signal["signal"].values

            # 1. 重采样 (Resample)
            current_fs = self.get_sampling_rate(row)
            if current_fs != target_fs:
                num_samples = int(len(raw_signal) * target_fs / current_fs)
                resampled_signal = signal.resample(raw_signal, num_samples)
            else:
                resampled_signal = raw_signal

            # 2. 分窗 (Windowing)
            num_windows = (len(resampled_signal) - window_size) // step + 1
            for i in range(num_windows):
                start = i * step
                end = start + window_size
                window = resampled_signal[start:end]

                # 3. 特征提取 (Feature Extraction)
                features = {"window_id": f"{os.path.basename(signal_path)}_{i}"}
                features.update(get_time_domain_features(window))
                features.update(get_freq_domain_features(window, target_fs))
                features.update(get_envelope_features(window, target_fs))
                # --- 新增：调用CWT特征提取 ---
                features.update(get_cwt_features(window, target_fs))

                # 合并元数据
                features.update(row.to_dict())
                all_features.append(features)

        # 保存所有特征
        features_df = pd.DataFrame(all_features)
        out_path = self.cfg["out_path"]
        io.ensure_dir(out_path)
        io.save_parquet(features_df, out_path)

        return {
            "features_path": out_path,
            "total_windows": len(all_features),
            "total_features": features_df.shape[1]
        }

    def get_sampling_rate(self, meta_row: pd.Series) -> float:
        """根据元数据判断原始采样率"""
        if meta_row["domain"] == "target":
            return 32000.0

        filename = meta_row["original_file"]
        if "48K" in filename.upper() or meta_row["sensor"] == 'DE' and ('_2' in filename or '_3' in filename):
            return 48000.0
        else:
            return 12000.0


register_task("extract_signal_features", ExtractSignalFeaturesTask)