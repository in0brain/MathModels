import os, pandas as pd, glob  # glob 用于按模式匹配文件列表
from src.core.io import ensure_dir, save_csv  # 自家IO工具

def aggregate_site(agg_dir: str, out_csv: str, site: str, win_sec: int):
    """
    把某站点的所有切片级 agg.csv 合并为一个站点总表：
      - 纵向拼接
      - 生成统一时间轴（这里用片内“相对秒”做演示）
      - 排序去重（如有必要）
    """
    files = sorted(glob.glob(os.path.join(agg_dir, site, "*.csv")))  # 找到该站点所有切片聚合文件
    if not files:
        # 若无文件，输出空表骨架
        ensure_dir(out_csv)
        empty = pd.DataFrame(columns=["timestamp", "q", "v", "k", "site"])
        empty = validate_traffic(empty.assign(timestamp=pd.to_timedelta([])))  # 过形状校验
        save_csv(empty, out_csv)
        return out_csv

    # 读入并拼接
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 片内秒 → Timedelta；如后续有绝对时间可在这里加上 start_ts 偏移
    df["timestamp"] = pd.to_timedelta(df["sec"], unit="s")

    # 只保留关心列，并按时间排序
    df = df[["timestamp","q","v","k"]].sort_values("timestamp")

    # 补上站点字段
    df["site"] = site

    # 保存
    df = validate_traffic(df)

    ensure_dir(out_csv);
    save_csv(df, out_csv)
    return out_csv

