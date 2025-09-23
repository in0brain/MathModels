# [新文件] src/preprocessing/signal/feature_extraction_to_image.py
import os
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch  # 导入 PyTorch

from src.core import io
from src.preprocessing.base import PreprocessTask, register_task


def create_spectrogram_gpu(window: np.ndarray, fs: float, out_path: str, device: torch.device, img_size=(64, 64)):
    """
    (GPU版本) 将信号窗口转换为时频图并保存为灰度图像。
    使用PyTorch在GPU上计算STFT。
    """
    # 1. 将Numpy数组转换为PyTorch张量，并移动到GPU
    window_tensor = torch.from_numpy(window.astype(np.float32)).to(device)

    # 2. 在GPU上执行STFT计算
    # torch.stft 需要一个window张量，我们使用 hann_window
    n_fft = 256  # STFT的窗口大小
    hop_length = 64  # 帧移
    win = torch.hann_window(n_fft).to(device)

    stft_result = torch.stft(
        window_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )

    # 3. 计算幅值谱，并移动回CPU以便绘图
    Sxx = stft_result.abs().cpu().numpy()

    # 4. 取对数幅值谱，并处理零值
    Sxx = np.where(Sxx == 0, 1e-10, Sxx)
    log_Sxx = np.log(Sxx)

    # --- 后续绘图和保存部分与原版类似，在CPU上执行 ---

    # 5. 绘制图像，不带坐标轴和边框
    fig, ax = plt.subplots(figsize=(img_size[0] / 100.0, img_size[1] / 100.0), dpi=100)
    # 注意：这里的t和f是示意性的，因为我们不需要坐标轴
    ax.imshow(log_Sxx, aspect='auto', origin='lower', cmap='gray')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 6. 保存为临时文件再用Pillow处理
    temp_path = out_path + ".temp.png"
    fig.savefig(temp_path)
    plt.close(fig)

    # 7. 用Pillow打开，转换为灰度图，调整大小，并保存
    img = Image.open(temp_path).convert('L').resize(img_size)
    img.save(out_path)

    # 8. 清理临时文件
    os.remove(temp_path)


class SignalToImageTask(PreprocessTask):
    def run(self) -> dict:
        manifest_path = self.cfg["manifest_path"]
        target_fs = self.cfg["target_fs"]
        window_size = self.cfg["window_size"]
        overlap = self.cfg["overlap"]
        image_out_dir = self.cfg["image_out_dir"]

        # --- 新增：自动选择设备 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- INFO: Spectrogram generation will run on: {device.type.upper()} ---")

        manifest_df = pd.read_csv(manifest_path)
        new_meta_records = []
        step = int(window_size * (1 - overlap))

        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Signal to Spectrogram (GPU)"):
            signal_path = row["signal_path"]
            raw_signal = io.read_parquet(signal_path)["signal"].values

            # 重采样 (此部分仍在CPU上执行，因为通常不是性能瓶颈)
            current_fs = 12000.0
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

                img_filename = f"{row['domain']}_{os.path.splitext(os.path.basename(row['original_file']))[0]}_win{i}.png"
                img_path = os.path.join(image_out_dir, img_filename)
                io.ensure_dir(img_path)

                # --- 调用GPU版本的函数 ---
                create_spectrogram_gpu(window, target_fs, img_path, device)

                new_record = row.to_dict()
                new_record['image_path'] = img_path
                new_meta_records.append(new_record)

        # 保存新的图像清单
        new_manifest_path = os.path.join(os.path.dirname(image_out_dir), "image_manifest.csv")
        pd.DataFrame(new_meta_records).to_csv(new_manifest_path, index=False)

        return {"new_manifest_path": new_manifest_path, "image_count": len(new_meta_records)}


register_task("signal_to_image", SignalToImageTask)