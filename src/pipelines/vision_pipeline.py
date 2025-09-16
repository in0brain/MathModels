# -*- coding: utf-8 -*-  # 中文注释安全
import argparse, yaml, os, pandas as pd    # argparse: 命令行参数；yaml: 读配置；os: 路径；pd: 表格
from src.preprocessing.vision import slicer, yolo_extract, aggregator  # 引入三步工具

def run(cfg_path: str, resume: bool = True):
    """
    视觉预处理流水线：
      1) 按 manifest 逐视频转码+切片（支持断点续跑）
      2) 对所有切片执行 YOLO+跟踪+计数，落盘中间产物
      3) 按站点聚合为 traffic_siteX_*.csv
    """
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))  # 加载 YAML 配置
    mani = pd.read_csv(cfg["videos"]["manifest"])                 # 读取视频清单

    out_tmp = cfg["videos"]["out_tmp"]                            # 切片输出根目录
    tcfg = cfg["videos"].get("transcode", {"enable":True,"scale_h":720,"fps":15,"gop":30})  # 转码配置

    for _, row in mani.iterrows():                                # 遍历每个视频
        path, site = row["path"], row["site"]                     # 取路径/站点名
        # 切片目录：按站点/源文件名分组，避免冲突
        site_dir = os.path.join(out_tmp, site, os.path.splitext(os.path.basename(path))[0])

        # 如果不是 resume 或目录不存在/空目录，则重新切片；否则直接复用已有切片
        if not resume or not os.path.isdir(site_dir) or not any(f.endswith(".mp4") for f in os.listdir(site_dir)):
            chunks = slicer.transcode_and_slice(
                path, site_dir, cfg["videos"]["chunk_sec"], tcfg["scale_h"], tcfg["fps"], tcfg["gop"]
            )
        else:
            chunks = sorted([os.path.join(site_dir, f) for f in os.listdir(site_dir) if f.endswith(".mp4")])

        # 逐切片跑提参
        for i, ck in enumerate(chunks):
            yolo_extract.process_chunk(ck, cfg, site, f"{i:03d}")

        # 切片级聚合 → 站点总表
        out_csv = os.path.join(cfg["aggregate"]["out_dir"], f"traffic_{site}_{cfg['aggregate']['tag']}.csv")
        aggregator.aggregate_site(os.path.join(cfg["aggregate"]["out_dir"], "agg"), out_csv, site, cfg["aggregate"]["win_sec"])
        print(f"[vision] site={site} -> {out_csv}")  # 打印产出路径，方便检查

if __name__ == "__main__":
    ap = argparse.ArgumentParser()                    # 创建命令行解析器
    ap.add_argument("--config", required=True)        # 必填：配置文件路径
    ap.add_argument("--no-resume", action="store_true")  # 可选：强制重新切片
    args = ap.parse_args()                            # 解析参数
    run(args.config, resume=not args.no_resume)       # 调用主流程（no-resume 取反）
