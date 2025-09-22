import pandas as pd

# 定义要读取的预测结果文件路径
predictions_file = "outputs/predictions/XGBoost/target_preds_tca.csv"

# 定义我们想要锁定的预测标签
# 根据我们的推断，左侧的聚类可能主要由 'Ball' 和 'OuterRace' 或 'Normal' 组成
# 我们可以先筛选这两个，如果聚类还包含其他颜色，再把其他标签也加进来
labels_to_lock = ['Ball', 'OuterRace', 'Normal'] # 请根据您补全数据后的实际类别调整

print(f"正在从文件 '{predictions_file}' 中加载预测结果...")

# 使用 pandas 读取CSV文件
try:
    df = pd.read_csv(predictions_file)
    print("文件加载成功。")
except FileNotFoundError:
    print(f"错误：找不到文件 '{predictions_file}'。请确保您已经成功运行了迁移学习流水线。")
    exit()

print(f"\n正在筛选出所有被预测为 {labels_to_lock} 的样本...")

# 执行筛选操作
# 这行代码会选出 'predicted_fault_type' 这一列的值是 'Ball' 或 'OuterRace' 或 'Normal' 的所有行
anomalous_clusters_df = df[df['predicted_fault_type'].isin(labels_to_lock)]

# 打印筛选结果的头部几行，进行预览
print("\n已成功锁定'异类'聚类中的样本，以下是前20个样本的预览：")
print(anomalous_clusters_df.head(20))

# 打印筛选出的样本数量
print(f"\n总共锁定了 {len(anomalous_clusters_df)} 个'异类'样本。")

# （可选）将筛选结果保存到一个新的CSV文件中，方便后续使用
output_file = "outputs/analysis/anomalous_clusters_locked.csv"
# 确保输出目录存在
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)
anomalous_clusters_df.to_csv(output_file, index=False)
print(f"详细的锁定结果已保存到：'{output_file}'")