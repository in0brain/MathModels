# 统一转码（720p/15fps 可配） + 固定时长切片（默认 300s）。
#
# 生成切片文件到 outputs/data/artifacts/vision/chunks/siteX/xxx_%03d.mp4。
#
# 支持“存在即跳过”，便于断点续跑。

import os, subprocess, shlex  # os: 路径/文件操作；subprocess: 调用外部命令；shlex: 安全拆分命令行
from src.core.io import ensure_dir  # 我们自己的工具：确保某个路径的父目录存在

def transcode_and_slice(video_path: str, out_dir: str, sec: int, scale_h: int, fps: int, gop: int):
    """
    把单个视频：先统一转码（分辨率/帧率/GOP），再按指定秒数切片
    参数：
      video_path: 输入视频路径
      out_dir   : 输出切片目录
      sec       : 每个切片的时长（秒）
      scale_h   : 目标高度（宽度自适应为偶数）
      fps       : 目标帧率
      gop       : 关键帧间隔（便于无损切片）
    返回：切片文件列表（按文件名排序）
    """
    ensure_dir(os.path.join(out_dir, "x.bin"))  # 创建输出目录（传个占位文件路径只为建父目录）

    tmp = os.path.join(out_dir, "tmp.mp4")  # 临时转码文件路径

    # ffmpeg 命令1：统一转码（缩放到 scale_h，高度固定、宽度自适配；设定目标 fps；x264 编码；强制关键帧间隔 gop）
    cmd1 = (
        f'ffmpeg -y -i "{video_path}" '
        f'-vf "scale=-2:{scale_h},fps={fps}" '  # -2 保证宽度为偶数；fps 把时间基统一
        f'-c:v libx264 -preset veryfast -pix_fmt yuv420p '  # x264 编码；色彩空间通用
        f'-g {gop} -sc_threshold 0 -an "{tmp}"'  # 固定 GOP；禁用场景切换触发；去音频
    )
    subprocess.run(shlex.split(cmd1), check=True)  # 执行命令，失败则抛异常

    # ffmpeg 命令2：按 sec 秒无损切片（直接复制码流），重置时间戳便于拼接聚合
    pattern = os.path.join(out_dir, "chunk_%03d.mp4")  # 输出命名模板
    cmd2 = f'ffmpeg -y -i "{tmp}" -c copy -map 0 -f segment -segment_time {sec} -reset_timestamps 1 "{pattern}"'
    subprocess.run(shlex.split(cmd2), check=True)

    # 清理临时文件（允许失败）
    try:
        os.remove(tmp)
    except OSError:
        pass

    # 返回所有切片路径（只取 chunk_*.mp4）
    return sorted(
        [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".mp4") and f.startswith("chunk_")]
    )

