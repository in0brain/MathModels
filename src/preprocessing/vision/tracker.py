import math  # 用于计算欧氏距离

class CentroidTracker:
    """
    极简多目标“质心”跟踪：
    - 每帧把检测框的中心点与上一帧已存在的轨迹进行最近邻匹配
    - 超过距离阈值不匹配则新建轨迹
    - 连续 max_age 帧未匹配到则移除轨迹
    """
    def __init__(self, match_thresh=80.0, max_age=30):
        self.next_id = 1         # 下一个可用的轨迹ID
        self.objs = {}           # 轨迹ID -> (cx, cy)
        self.age = {}            # 轨迹ID -> 未匹配计数
        self.th = match_thresh   # 匹配距离阈值（像素）
        self.max_age = max_age   # 允许未匹配的最大帧数

    @staticmethod
    def _centroid(x1,y1,x2,y2):  # 计算框的中心点
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    @staticmethod
    def _dist(a,b):             # 两点的欧氏距离
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, detections):
        """
        输入：当前帧的检测框列表 [(x1,y1,x2,y2), ...]
        输出：当前帧的轨迹字典 {track_id: (cx,cy)}
        """
        det_cent = [self._centroid(*d) for d in detections]  # 所有检测的中心点
        assigned = set()   # 已匹配的检测索引
        updates = {}       # 本帧更新后的轨迹位置

        # 先尝试给历史轨迹找最近的检测（小于阈值才算匹配）
        for tid, (cx,cy) in list(self.objs.items()):
            self.age[tid] += 1  # 默认本帧先记一次“未匹配”
            best = None; bestd = 1e9
            for i, dc in enumerate(det_cent):
                if i in assigned: continue
                d = self._dist((cx,cy), dc)
                if d < bestd:
                    bestd, best = d, i
            if best is not None and bestd <= self.th:
                assigned.add(best)      # 该检测被占用
                updates[tid] = det_cent[best]  # 轨迹位置更新为该检测中心
                self.age[tid] = 0       # 匹配成功，清零“未匹配计数”

        # 对剩余未被匹配的检测，创建新轨迹
        for i, dc in enumerate(det_cent):
            if i in assigned: continue
            tid = self.next_id; self.next_id += 1
            updates[tid] = dc
            self.age[tid] = 0

        # 删除老化轨迹（持续 max_age 帧未匹配）
        for tid in list(self.objs.keys()):
            if tid not in updates and self.age.get(tid,0) > self.max_age:
                self.objs.pop(tid, None)
                self.age.pop(tid, None)

        # 提交更新
        self.objs.update(updates)
        return self.objs.copy()  # 返回当前帧的轨迹位置快照
