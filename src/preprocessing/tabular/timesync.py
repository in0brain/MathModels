import pandas as pd

def resample_uniform(csv_path: str, out_csv: str, freq: str = "10s", method: str = "ffill"):
    """
    把带有 timestamp 列的 CSV 重采样为固定时间步（默认 10s）：
      - 先把 'timestamp' 转为 pandas 的时间索引
      - .resample(freq).mean() 进行聚合
      - 用 method 做缺失插值（前向/后向）
      - 保存到 out_csv
    """
    df = pd.read_csv(csv_path)                 # 读取原始表
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # 转为真正时间
    df = df.set_index("timestamp").sort_index()        # 设为索引并按时间排序

    df_res = df.resample(freq).mean()          # 以固定频率聚合（均值）
    if method == "ffill":
        df_res = df_res.ffill()                # 前向填充
    elif method == "bfill":
        df_res = df_res.bfill()                # 后向填充

    df_res.to_csv(out_csv)                     # 导出
    return out_csv
