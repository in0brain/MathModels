# [新文件] src/preprocessing/signal/feature_extraction_to_image.py
import os
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task


def create_spectrogram(window: np.ndarray, fs: float, out_path: str, img_size=(64, 64)):
    """将信号窗口转换为时频图并保存为灰度图像"""
    # 计算STFT
    f, t, Zxx = signal.stft(window, fs, nperseg=128)

    # 取对数幅值谱，并处理零值
    Sxx = np.abs(Zxx)
    Sxx = np.where(Sxx == 0, 1e-10, Sxx)
    log_Sxx = np.log(Sxx)

    # 绘制图像，不带坐标轴和边框
    fig, ax = plt.subplots(figsize=(img_size[0] / 100.0, img_size[1] / 100.0), dpi=100)
    ax.pcolormesh(t, f, log_Sxx, shading='gouraud', cmap='gray')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存为临时文件再用Pillow处理
    temp_path = out_path + ".temp.png"
    fig.savefig(temp_path)
    plt.close(fig)

    # 用Pillow打开，转换为灰度图，调整大小，并保存
    img = Image.open(temp_path).convert('L').resize(img_size)
    img.save(out_path)

    # 清理临时文件
    os.remove(temp_path)


class SignalToImageTask(PreprocessTask):
    def run(self) -> dict:
        manifest_path = self.cfg["manifest_path"]
        target_fs = self.cfg["target_fs"]
        window_size = self.cfg["window_size"]
        overlap = self.cfg["overlap"]
        image_out_dir = self.cfg["image_out_dir"]

        manifest_df = pd.read_csv(manifest_path)
        new_meta_records = []
        step = int(window_size * (1 - overlap))

        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Signal to Spectrogram"):
            signal_path = row["signal_path"]
            raw_signal = io.read_parquet(signal_path)["signal"].values

            # 重采样
            current_fs = 12000.0  # 简化处理，假设源域主要是12k，可按之前逻辑修改
            if '48K' in row['original_file'].upper(): current_fs = 48000.0
            if row['domain'] == 'target': current_fs = 32000.0

            if current_fs != target_fs:
                num_samples = int(len(raw_signal) * target_fs / current_fs)
                resampled_signal = signal.resample(raw_signal, num_samples)
            else:
                resampled_signal = raw_signal

            # 分窗并生成图像
            num_windows = (len(resampled_signal) - window_size) // step + 1
            for i in range(num_windows):
                start = i * step
                end = start + window_size
                window = resampled_signal[start:end]

                # 为每个窗口创建独立的图像文件
                img_filename = f"{row['domain']}_{os.path.splitext(os.path.basename(row['original_file']))[0]}_win{i}.png"
                img_path = os.path.join(image_out_dir, img_filename)
                io.ensure_dir(img_path)

                create_spectrogram(window, target_fs, img_path)

                # 记录新的元数据
                new_record = row.to_dict()
                new_record['image_path'] = img_path
                new_meta_records.append(new_record)

        # 保存新的图像清单
        new_manifest_path = os.path.join(os.path.dirname(image_out_dir), "image_manifest.csv")
        pd.DataFrame(new_meta_records).to_csv(new_manifest_path, index=False)

        return {"new_manifest_path": new_manifest_path, "image_count": len(new_meta_records)}


register_task("signal_to_image", SignalToImageTask)