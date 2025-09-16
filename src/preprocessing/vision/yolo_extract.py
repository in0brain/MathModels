import os, cv2, numpy as np, pandas as pd               # cv2: 读视频/画图；np/pd: 数值与表格
from ultralytics import YOLO                             # YOLO 推理入口
from tqdm import tqdm                                     # 进度条
from src.core.io import ensure_dir, save_parquet, save_csv # 自有IO工具
from .tracker import CentroidTracker                      # 轻量跟踪器（质心）
from .schemas import validate_tracks, validate_agg

def _denorm_line(w,h, line):
    """把 0~1 的归一化线段坐标转为像素坐标"""
    x1,y1,x2,y2 = line
    return int(x1*w),int(y1*h),int(x2*w),int(y2*h)

def _intersect(p, q, a, b):
    """
    判断线段 pq 是否与线段 ab 相交（用于“过线计数”）
    采用向量叉积法，t/u 在 [0,1] 内即相交
    """
    def cross(u,v): return u[0]*v[1]-u[1]*v[0]
    r = (q[0]-p[0], q[1]-p[1]); s=(b[0]-a[0], b[1]-a[1])
    denom = cross(r,s)
    if denom == 0: return False
    t = cross((a[0]-p[0], a[1]-p[1]), s)/denom
    u = cross((a[0]-p[0], a[1]-p[1]), r)/denom
    return (0<=t<=1) and (0<=u<=1)

def process_chunk(chunk_path, cfg, site, chunk_id) -> dict:
    """
    处理单个切片：
      - 逐帧YOLO检测（可抽帧）
      - 质心跟踪，估算速度
      - 过线计数，得到流量 q
      - 按秒聚合，导出 q/v/k（密度近似）为 agg.csv
      - 同步落盘检测明细 tracks.parquet
    """
    cap = cv2.VideoCapture(chunk_path)        # 打开视频
    if not cap.isOpened(): raise RuntimeError(f"open fail: {chunk_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 视频宽度
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0      # 帧率（可能为0，用默认15）
    stride = int(cfg["infer"].get("frame_stride", 1))  # 抽帧间隔

    # 计数线（像素坐标）
    line = _denorm_line(W,H, cfg["roi"].get("count_line",[0.1,0.8,0.9,0.8]))
    p1=(line[0],line[1]); p2=(line[2],line[3])

    # YOLO 模型载入（一次）
    model = YOLO(cfg["yolo"]["model"])
    classes = cfg["yolo"].get("classes", None)  # 只检测指定类别（车类）

    # 轻量跟踪器
    tracker = CentroidTracker(
        match_thresh=cfg["tracking"]["match_thresh"],
        max_age=cfg["tracking"]["max_age"]
    )

    rows = []       # 保存每个检测框的明细（用于 tracks.parquet）
    agg_rows = []   # 每秒的汇总（q/v/k）
    frame_idx=0     # 当前帧号
    crossed=set()   # 已计数的轨迹ID集合（防止重复过线）
    q_count=0       # 当前秒内的过线数量（流量计数）
    last_pos={}     # 轨迹最近一次的位置：tid -> (cx,cy)

    # 标定参数（像素 -> 米；密度估计需要ROI长度）
    pixels_per_meter = float(cfg["calib"]["pixels_per_meter"])
    roi_len_m = float(cfg["calib"]["roi_length_m"])

    # 每秒为单位的聚合窗口长度；这里我们按“整秒”记录
    win = int(cfg["aggregate"]["win_sec"])

    # tqdm 进度条；估计需要处理的帧数 = 总帧/抽帧间隔
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//stride, desc=f"{site}:{chunk_id}") as bar:
        while True:
            ret = cap.grab()           # 抓取一帧（更高效）
            if not ret: break
            if frame_idx % stride != 0:
                frame_idx += 1; bar.update(1); continue  # 非抽样帧跳过，仍更新进度

            ret, frame = cap.retrieve()  # 取回当前帧图像
            if not ret: break

            # YOLO 推理（直接给帧对象），使用 conf/iou/device/half 等参数；限制 classes
            res = model.predict(
                source=frame,
                conf=cfg["yolo"]["conf"],
                iou=cfg["yolo"]["iou"],
                device=cfg["infer"]["device"],
                half=cfg["infer"]["half"],
                verbose=False,
                classes=classes
            )

            # 把结果转换成简单的 (x1,y1,x2,y2,cls,conf) 列表
            dets=[]
            for r in res:
                if r.boxes is None: continue
                for b in r.boxes:
                    x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                    cls = int(b.cls[0].item()); conf = float(b.conf[0].item())
                    dets.append((x1,y1,x2,y2,cls,conf))

            # 送入质心跟踪器，得到 {tid: (cx,cy)}
            tids = tracker.update([d[:4] for d in dets])

            # 估计速度：根据相邻抽样帧的位移（像素）/ 时间，换算为 m/s
            dt = stride/ fps
            speeds=[]
            for tid,(cx,cy) in tids.items():
                if tid in last_pos:
                    dx = cx - last_pos[tid][0]; dy = cy - last_pos[tid][1]
                    v_mps = (np.hypot(dx,dy)/max(pixels_per_meter,1e-6))/max(dt,1e-6)
                    speeds.append(v_mps)
                last_pos[tid]=(cx,cy)
            v_est = float(np.nanmedian(speeds)) if speeds else np.nan  # 用中位数更稳

            # 过线计数：上一位置 -> 当前位置 的线段是否与计数线相交
            for tid,(cx,cy) in tids.items():
                if tid in crossed: continue  # 一个轨迹只计一次
                prev = last_pos.get(tid, (cx,cy))
                if _intersect(prev,(cx,cy), p1,p2):
                    q_count += 1
                    crossed.add(tid)

            # 保存当前帧的所有检测明细（可用于回溯/调试或更精准统计）
            for (x1,y1,x2,y2,cls,conf) in dets:
                rows.append(dict(
                    frame=frame_idx, x1=x1,y1=y1,x2=x2,y2=y2, cls=cls, conf=conf,
                    site=site, chunk=chunk_id
                ))

            # 每满 1 秒输出一次汇总（以帧率整除判断）
            if int(fps) > 0 and frame_idx % int(fps) == 0:
                tsec = frame_idx//int(fps)             # 相对片内秒数
                k_approx = len(tids) / max(roi_len_m,1e-6)  # 简化密度：活跃轨迹/ROI长度
                agg_rows.append(dict(sec=tsec, q=q_count, v=v_est, k=k_approx, site=site, chunk=chunk_id))
                q_count = 0  # 重置下一秒计数

            frame_idx += 1
            bar.update(1)

    cap.release()  # 释放视频资源

    # 保存明细为 Parquet（体积更小、读写更快）
    out_parquet = os.path.join(cfg["aggregate"]["out_dir"], "tracks", site, f"{chunk_id}.parquet")
    ensure_dir(out_parquet)
    save_parquet(pd.DataFrame(rows), out_parquet)

    # 保存每秒聚合为 CSV（便于后续聚合成站点级别）
    out_csv = os.path.join(cfg["aggregate"]["out_dir"], "agg", site, f"{chunk_id}.csv")
    ensure_dir(out_csv)
    save_csv(pd.DataFrame(agg_rows), out_csv)

    # 返回产物路径（供上层 pipeline 记录/汇总）
    return {"tracks": out_parquet, "agg": out_csv}


def process_chunk(chunk_path, cfg, site, chunk_id) -> dict:
    # ... 上文逻辑同前略 ...
    cap.release()

    # DataFrame 构造
    tracks_df = pd.DataFrame(rows)
    agg_df = pd.DataFrame(agg_rows)

    # ---- 新增：schema 校验（保存前）----
    if not tracks_df.empty:
        tracks_df = validate_tracks(tracks_df)
    if not agg_df.empty:
        agg_df = validate_agg(agg_df)

    # 保存明细为 Parquet
    out_parquet = os.path.join(cfg["aggregate"]["out_dir"], "tracks", site, f"{chunk_id}.parquet")
    ensure_dir(out_parquet); save_parquet(tracks_df, out_parquet)

    # 保存每秒聚合为 CSV
    out_csv = os.path.join(cfg["aggregate"]["out_dir"], "agg", site, f"{chunk_id}.csv")
    ensure_dir(out_csv); save_csv(agg_df, out_csv)

    return {"tracks": out_parquet, "agg": out_csv}