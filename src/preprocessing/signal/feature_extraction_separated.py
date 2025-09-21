# src/preprocessing/signal/feature_extraction_separated.py
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Dict, Any
from tqdm import tqdm
import os
import pywt

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task
from src.preprocessing.signal.feature_extraction import (
    get_time_domain_features,
    get_freq_domain_features,
    get_envelope_features,
    get_cwt_features
)

class ExtractAndSeparateSignalFeaturesTask(PreprocessTask):
    """
    从信号数据中分窗提取特征，并按域（时域、频域、时频域）保存到不同的文件中。
    """

    def run(self) -> Dict[str, Any]:
        manifest_path = self.cfg["manifest_path"]
        target_fs = self.cfg["target_fs"]
        window_size = self.cfg["window_size"]
        overlap = self.cfg["overlap"]

        manifest_df = pd.read_csv(manifest_path)
        all_features = []

        step = int(window_size * (1 - overlap))

        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Separated Features"):
            signal_path = row["signal_path"]
            raw_signal = io.read_parquet(signal_path)["signal"].values

            current_fs = self.get_sampling_rate(row)
            if current_fs != target_fs:
                num_samples = int(len(raw_signal) * target_fs / current_fs)
                resampled_signal = signal.resample(raw_signal, num_samples)
            else:
                resampled_signal = raw_signal

            num_windows = (len(resampled_signal) - window_size) // step + 1
            for i in range(num_windows):
                start = i * step
                end = start + window_size
                window = resampled_signal[start:end]

                features = {"window_id": f"{os.path.basename(signal_path)}_{i}"}
                features.update(get_time_domain_features(window))
                features.update(get_freq_domain_features(window, target_fs))
                features.update(get_envelope_features(window, target_fs))
                features.update(get_cwt_features(window, target_fs))
                features.update(row.to_dict())
                all_features.append(features)

        features_df = pd.DataFrame(all_features)

        # --- 核心区别：分离特征并保存 ---
        meta_cols = ['window_id', 'domain', 'original_file', 'sensor', 'rpm', 'fault_type', 'fault_size', 'load', 'signal_path']

        # 1. 时域特征
        td_cols = [col for col in features_df.columns if col.startswith('td_')]
        df_td = features_df[meta_cols + td_cols]
        out_td = self.cfg["out_paths"]["time_domain"]
        io.save_parquet(df_td, out_td)

        # 2. 频域特征 (FFT + 包络)
        fd_cols = [col for col in features_df.columns if col.startswith(('fd_', 'env_'))]
        df_fd = features_df[meta_cols + fd_cols]
        out_fd = self.cfg["out_paths"]["freq_domain"]
        io.save_parquet(df_fd, out_fd)

        # 3. 时频域特征 (CWT)
        cwt_cols = [col for col in features_df.columns if col.startswith('cwt_')]
        df_cwt = features_df[meta_cols + cwt_cols]
        out_cwt = self.cfg["out_paths"]["cwt_domain"]
        io.save_parquet(df_cwt, out_cwt)

        print("Separated feature files have been successfully generated.")

        return {
            "features_td_path": out_td,
            "features_fd_path": out_fd,
            "features_cwt_path": out_cwt,
        }

    def get_sampling_rate(self, meta_row: pd.Series) -> float:
        if meta_row["domain"] == "target": return 32000.0
        filename = meta_row["original_file"]
        if "48K" in filename.upper() or (meta_row["sensor"] == 'DE' and ('_2' in filename or '_3' in filename)):
            return 48000.0
        return 12000.0

register_task("extract_signal_features_separated", ExtractAndSeparateSignalFeaturesTask)