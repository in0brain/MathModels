# [Corrected Version] src/pipelines/transfer_dann_raw_signal_pipeline.py
import argparse, yaml, pandas as pd, numpy as np, joblib
import os

from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from src.models.clf.DANN import build as dann_builder
from src.core import io, viz
from scipy import signal


# --- 1. Custom Dataset for Raw Signals (Modified to store metadata) ---
class RawSignalDataset(Dataset):
    def __init__(self, manifest_df, cfg, is_train=True):
        self.manifest = manifest_df
        self.window_size = cfg['data_params']['window_size']
        self.overlap = cfg['data_params']['overlap']
        self.target_fs = cfg['data_params']['target_fs']

        self.samples = []
        print(f"Loading and windowing data for {'training' if is_train else 'inference'}...")
        for _, row in tqdm(self.manifest.iterrows(), total=len(self.manifest)):
            raw_signal = io.read_parquet(row["signal_path"])["signal"].values

            current_fs = 12000.0
            if '48K' in row['original_file'].upper(): current_fs = 48000.0
            if row['domain'] == 'target': current_fs = 32000.0

            if current_fs != self.target_fs:
                num_samples = int(len(raw_signal) * self.target_fs / current_fs)
                resampled_signal = signal.resample(raw_signal, num_samples)
            else:
                resampled_signal = raw_signal

            step = int(self.window_size * (1 - self.overlap))
            num_windows = (len(resampled_signal) - self.window_size) // step + 1
            for i in range(num_windows):
                start = i * step
                window = resampled_signal[start: start + self.window_size]

                # --- FIX: Store metadata along with the window ---
                sample_info = {
                    "window": window.astype(np.float32),
                    "label": row.get('encoded_label', -1),
                    "original_file": row["original_file"],
                    "window_id": f"{os.path.basename(row['signal_path'])}_{i}"
                }
                self.samples.append(sample_info)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        window_tensor = torch.tensor(sample_info["window"], dtype=torch.float32)
        label = sample_info["label"]

        if label == -1:
            return (window_tensor,)
        else:
            return window_tensor, torch.tensor(label, dtype=torch.long)


# --- 2. Main Run Function (with corrected results saving) ---
def run(config_path: str):
    print(f"[DANN 原始信号迁移流水线] 开始运行，配置文件: {config_path}")
    # ... [Code from here to Step 5 is unchanged] ...
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device.type.upper()}")

    le = joblib.load(cfg["label_encoder_path"])
    manifest = pd.read_csv(cfg["manifest_path"])

    source_df = manifest[manifest['domain'] == 'source'].copy().dropna(subset=['fault_type'])
    target_df = manifest[manifest['domain'] == 'target'].copy()
    source_df['encoded_label'] = le.transform(source_df['fault_type'])

    source_dataset = RawSignalDataset(source_df, cfg)
    target_dataset = RawSignalDataset(target_df, cfg, is_train=False)
    ys_labels = np.array([s["label"] for s in source_dataset.samples])

    batch_size = cfg['training_params']['batch_size']
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- 3. Build Model ---
    num_classes = len(le.classes_)
    model = dann_builder.build(cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training_params']['learning_rate'])

    class_counts = np.bincount(ys_labels)
    class_weights = 1. / torch.tensor(np.where(class_counts == 0, 1, class_counts), dtype=torch.float)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
    criterion_label = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    print("步骤4: 开始对抗训练...")
    num_epochs = cfg['training_params']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")
        for source_batch, target_batch in progress_bar:
            source_data, source_labels = source_batch
            target_data = target_batch[0]
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()
            p = float(epoch) / num_epochs
            lambda_ = 2. / (1. + np.exp(-10. * p)) - 1
            adversarial_lambda = lambda_ * cfg['training_params']['adversarial_lambda']
            label_output, domain_output_source = model(source_data, lambda_=adversarial_lambda)
            loss_label = criterion_label(label_output, source_labels)
            loss_domain_source = criterion_domain(domain_output_source,
                                                  torch.zeros(len(source_data), dtype=torch.long, device=device))
            _, domain_output_target = model(target_data, lambda_=adversarial_lambda)
            loss_domain_target = criterion_domain(domain_output_target,
                                                  torch.ones(len(target_data), dtype=torch.long, device=device))
            total_loss = loss_label + loss_domain_source + loss_domain_target
            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=total_loss.item(), loss_label=loss_label.item(),
                                     loss_domain=(loss_domain_source + loss_domain_target).item())

    # --- 5. Prediction ---
    print("步骤5: 对目标域进行预测...")
    model.eval()
    all_preds, all_features = [], []
    with torch.no_grad():
        target_loader_eval = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
        for (inputs,) in target_loader_eval:
            inputs = inputs.to(device)
            features = model.feature_extractor(inputs)
            label_output, _ = model(inputs, lambda_=0)
            preds = torch.argmax(label_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_features.append(features.cpu().numpy())
    target_pred_labels = le.inverse_transform(all_preds)
    target_features = np.vstack(all_features)

    # --- 6. Save Results (Corrected) ---
    print(f"步骤6: 保存预测结果...")

    # --- FIX: Build results_df from the dataset's sample metadata ---
    target_metadata = [
        {"original_file": s["original_file"], "window_id": s["window_id"]}
        for s in target_dataset.samples
    ]
    results_df = pd.DataFrame(target_metadata)
    # Ensure the number of predictions matches the number of windows
    results_df = results_df.iloc[:len(target_pred_labels)].copy()
    results_df['predicted_fault_type'] = target_pred_labels

    io.save_csv(results_df, cfg['outputs']['target_predictions_path'])
    torch.save(model.state_dict(), cfg['outputs']['model_path'])

    # --- 7. Visualization ---
    print("步骤7: 生成t-SNE可视化图...")
    model.eval()
    with torch.no_grad():
        source_features_list = []
        source_loader_eval = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
        for (inputs, _) in source_loader_eval:
            inputs = inputs.to(device)
            source_features_list.append(model.feature_extractor(inputs).cpu().numpy())
        source_features = np.vstack(source_features_list)

    viz.plot_tsne_by_class(
        source_latent=source_features,
        target_latent=target_features,
        source_labels=ys_labels,
        target_labels=np.array(all_preds),
        class_names=list(le.classes_),
        out_png=cfg["outputs"]["visualization_path"],
        title=f'Class Distribution After {cfg["model"].get("feature_extractor_type", "DANN")} Adaptation'
    )

    print(f"[DANN 原始信号迁移流水线] 成功运行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于原始信号的DANN+MHDCNN迁移学习流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (YAML) 的路径")
    args = parser.parse_args()
    run(args.config)