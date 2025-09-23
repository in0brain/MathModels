import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# --- Use before running to ensure Chinese font support ---
# It's recommended to set up the plotting style and font beforehand for Chinese character support.
sns.set_theme(style="whitegrid")
try:
    # For Windows/Linux, common Chinese fonts are 'SimHei', 'Microsoft YaHei'. For macOS, 'Hiragino Sans GB' is a good choice.
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Warning: Could not set Chinese font. Please install a compatible font (e.g., SimHei). Error: {e}")


def plot_feature_distribution(df: pd.DataFrame, features: list, out_dir: str):
    """
    Draws box plots for specified key features to compare their distributions across different fault categories.
    """
    print(f"Generating feature distribution plots for: {features}...")
    fault_order = ['Normal', 'Ball', 'InnerRace', 'OuterRace']
    df['fault_type'] = pd.Categorical(df['fault_type'], categories=fault_order, ordered=True)

    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='fault_type', y=feature, data=df)
        plt.title(f'Distribution of Feature "{feature}" Across Fault Types', fontsize=16)
        plt.xlabel("Fault Type", fontsize=12)
        plt.ylabel("Feature Value", fontsize=12)

        output_path = os.path.join(out_dir, f"distribution_{feature}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Chart saved to: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list, out_dir: str):
    """
    Calculates and plots a correlation heatmap for all numerical features.
    """
    print("Generating feature correlation heatmap...")
    plt.figure(figsize=(18, 15))
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, cmap='vlag', annot=False)
    plt.title("Correlation Heatmap of All Extracted Features", fontsize=16)

    output_path = os.path.join(out_dir, "correlation_heatmap.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Chart saved to: {output_path}")


def plot_tsne_features(df: pd.DataFrame, feature_cols: list, out_dir: str, sample_size: int = 2000):
    """
    Performs t-SNE dimensionality reduction on the feature space and plots a 2D scatter plot.
    """
    print("Generating t-SNE scatter plot of the feature space...")

    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    X = df_sample[feature_cols].values
    y = df_sample['fault_type'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- CORE FIX: Removed the 'n_iter' argument ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_sample) - 1))
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y,
        palette="tab10",
        alpha=0.8,
        s=50,
        hue_order=['Normal', 'Ball', 'InnerRace', 'OuterRace']
    )
    plt.title("t-SNE Visualization of Feature Space", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Fault Type")

    output_path = os.path.join(out_dir, "tsne_feature_space.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for extracted bearing fault features.")
    parser.add_argument("--features_path", type=str, default="outputs/data/artifacts/features_32k.parquet",
                        help="Path to the input features.parquet file.")
    parser.add_argument("--out_dir", type=str, default="outputs/figs/feature_analysis",
                        help="Directory to save the output plots.")
    args = parser.parse_args()

    key_features_for_distribution = [
        'td_kurtosis',
        'env_peak_freq',
        'cwt_total_energy'
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading features from {args.features_path}...")
    df_full = pd.read_parquet(args.features_path)

    df_source = df_full[df_full['domain'] == 'source'].copy()
    if df_source.empty:
        raise ValueError("Error: No data found for the 'source' domain in the feature file.")

    feature_cols = [col for col in df_source.columns if col.startswith(('td_', 'fd_', 'env_', 'cwt_'))]

    print("\n--- Starting Chart Generation ---")
    plot_feature_distribution(df_source, key_features_for_distribution, args.out_dir)
    plot_correlation_heatmap(df_source, feature_cols, args.out_dir)
    plot_tsne_features(df_source, feature_cols, args.out_dir)

    print("\nAll feature visualizations have been generated successfully!")


if __name__ == "__main__":
    main()