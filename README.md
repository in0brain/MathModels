# 项目操作手册（pip 版）

## 一、项目结构与模块作用

```
mcm-ml/
├─ data/                               # 数据目录
│   ├─ markov_demo.csv                 # 示例：马尔科夫链训练数据
│   ├─ house.csv                       # 示例：XGBoost 回归数据
│   └─ ...                             # 新数据统一放这里
├─ outputs/                            # 自动生成的结果，不用手动修改
│  ├─ data/
│  │  ├─ artifacts/                    # 工件（转移矩阵/特征重要性等）
│  │  └─ predictions/                  # 预测结果（按算法分类）
│  ├─ figs/                            # 图表输出（按任务类型分类）
│  ├─ models/                          # 训练好的模型（.pkl）
│  └─ reports/                         # JSON 报告（指标等）
├─ scripts/                            # 辅助脚本（造数/批处理）
│  ├─ MarkovChain/datacreate.py
│  └─ NeuralNet/datacreate.py
├─ src/
│  ├─ core/                            # 公共工具
│  │   ├─ io.py        # IO 工具：读写 CSV/Parquet、保存模型
│  │   ├─ metrics.py   # 指标函数：ACC/F1/MAE/R2...
│  │   ├─ registry.py  # 自动注册算法 (task:algo -> build.py)
│  │   └─ viz.py       # 可视化工具：特征重要性、残差图等
│  ├─ inference/                       # 再预测逻辑
│  │   └─ runner.py    # 通用推理入口
│  ├─ models/                          # 算法集合（按任务分组）
│  │   ├─ ts/          # 时间序列
│  │   ├─ reg/         # 回归
│  │   ├─ clf/         # 分类
│  │   └─ clu/         # 聚类
│  │       └─ 每个算法子目录：__init__.py, build.py, params.yaml
│  ├─ pipelines/                       # 训练流水线
│  │   ├─ ts_pipeline.py   # 时间序列任务
│  │   ├─ reg_pipeline.py  # 回归任务
│  │   ├─ clf_pipeline.py  # 分类任务
│  │   └─ clu_pipeline.py  # 聚类任务
│  └─ preprocessing/                   # 预处理（视频/表格等）
│      ├─ base.py        # 通用任务基类
│      ├─ vision/        # 视频数据处理（YOLO+跟踪+聚合）
│      └─ tabular/       # 表格清洗与时间对齐
├─ requirements.txt                    # pip 依赖清单
└─ README.md                           # 使用文档
```

**核心关系**：

* **models/** 定义算法（build.py + params.yaml）
* **pipelines/** 负责训练流程
* **inference/** 负责再预测
* **core/** 提供通用工具
* **preprocessing/** 负责原始数据清洗/特征提取

---

## 二、如何运行（示例：XGBoost 回归）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包含：

* 科学计算：numpy, pandas, scipy, statsmodels
* 机器学习：scikit-learn, xgboost, lightgbm, catboost
* 时间序列：sktime, prophet
* 优化/实验：optuna, mlflow
* 可视化：matplotlib, seaborn
* 其他：PyYAML, joblib, imageio, ultralytics, opencv-python, pyarrow, tqdm

### 2. 准备数据

示例数据：`data/house.csv`
> 可以直接执行脚本生成,命令如下
>  
`python scripts/XGBoost/datacreate.py`

生成csv表如下

```csv
area,rooms,age,price
120,3,5,200000
80,2,10,120000
...
```

### 3. 训练模型

```bash
python -m src.pipelines.reg_pipeline --config src/models/reg/XGBoost/params.yaml
```

执行流程：

1. `reg_pipeline.py` 加载配置
2. `registry` 找到 `XGBoost/build.py`
3. 读取数据 → 训练模型 → 评估 → 可视化 → 保存

生成文件：

* `outputs/models/house_xgb.pkl` （训练好的模型）
* `outputs/data/artifacts/XGBoost/house_xgb_*.csv` （工件）
* `outputs/data/predictions/XGBoost/house_xgb_preds.csv` （预测结果）
* `outputs/reports/house_xgb_metrics.json` （指标报告）
* `outputs/figs/reg/house_xgb_feat_importance.png` 等图表

### 4. 再预测

新数据：`data/house_new.csv`

```bash
python -m src.inference.runner --task reg --algo XGBoost --model outputs/models/house_xgb.pkl --data data/house_new.csv --config src/models/reg/XGBoost/params.yaml --tag house_xgb_new
```

生成：

* `outputs/data/predictions/XGBoost/house_xgb_new_preds.csv`

### 5. 重新训练

删除旧结果再执行：

* `outputs/models/<tag>.pkl`
* `outputs/data/artifacts/XGBoost/*`
* `outputs/data/predictions/XGBoost/*`
* `outputs/reports/<tag>_metrics.json`
* `outputs/figs/reg/*`

---

## 三、如何扩展新算法模块

### 1. 创建目录

以分类任务新算法 `MyAlgo` 为例：

```
src/models/clf/MyAlgo/
├─ __init__.py(为空就行，将目录标记为Python包。通过在目录中包含这个文件，Python解释器会将该目录及其包含的文件视为一个包)
├─ build.py
└─ params.yaml
```

### 2. 编写配置（params.yaml）

```yaml
task: clf                  # 任务类型 (ts/reg/clf/clu)，不要盲目填，按算法分类填写
dataset:
  path: data/myalgo.csv    # 输入数据表
  target: label            # 标签列
  features: []             # 特征列，空=自动推断
model:
  name: MyAlgo             # 算法名（必须与 build.py 中 ALGO 一致）
  params:                  # 算法所需参数
    param1: 10
    param2: 0.1
eval:
  metrics: [ACC, F1]       # 指标（需在 core/metrics.py 实现）
viz:
  enabled: true
  plots:
    confusion_matrix: true # 可视化（需在 core/viz.py 实现）
outputs:
  base_dir: outputs
  tag: myalgo_demo
seed: 42
```

### 3. 编写算法逻辑（build.py）

```python
TASK = "clf"
ALGO = "MyAlgo"

def build(cfg):
    # 初始化模型对象（如 sklearn 或自定义类）
    ...

def fit(model, df, cfg):
    # 训练 → 评估 → 保存工件
    return {"metrics": {...}, "artifacts": {...}}

def inference(model, df, cfg):
    # 用已训练模型预测新数据
    return {"predictions_csv": "路径"}
```

### 4. 注册机制

无需手动改 `registry.py`。
系统会自动扫描所有 `build.py` 并注册为 `"{task}:{algo}"`。

如：`"clf:myalgo"`

### 5. 可选扩展

* 在 `src/core/metrics.py` 中增加新指标函数（如 G-Mean）。
* 在 `src/core/viz.py` 中增加新可视化函数（如 ROC 曲线）。
* 在 `requirements.txt` 中添加新依赖。

### 6. 运行新算法

* 训练：

```bash
python -m src.pipelines.clf_pipeline --config src/models/clf/MyAlgo/params.yaml
```

* 再预测：

```bash
python -m src.inference.runner \
  --task clf \
  --algo MyAlgo \
  --model outputs/models/myalgo_demo.pkl \
  --data data/myalgo_new.csv \
  --config src/models/clf/MyAlgo/params.yaml \
  --tag myalgo_new
```

---

### 7) 扩展预处理模块（preprocessing）

预处理模块用于把**原始数据**（视频、日志、IoT、表格等）转为算法可直接使用的**标准化表**（如 `traffic_site*.csv`、干净的 tabular CSV/Parquet）。整体约定：

```
src/
└─ preprocessing/
   ├─ base.py                 # 任务基类/注册器（可选）
   ├─ vision/                 # 视觉预处理（例：YOLO+跟踪+聚合）
   │  ├─ params.yaml          # 预处理参数（输入清单/ROI/抽帧/输出目录等）
   │  ├─ slicer.py            # ffmpeg 转码+切片
   │  ├─ tracker.py           # 轻量跟踪（或接 DeepSORT）
   │  ├─ yolo_extract.py      # 检测+过线计数+速度估计
   │  ├─ aggregator.py        # 切片聚合→站点级 traffic CSV
   │  └─ schemas.py           # 中间产物 schema + 校验
   └─ tabular/
      ├─ clean.py             # 表格清洗（去重/缺失/类型/列名统一）
      └─ timesync.py          # 时间对齐/重采样（与 vision 输出合并）
pipelines/
└─ vision_pipeline.py         # 调度：切片→检测→聚合
```

#### 7.1 参数文件规范（以 `src/preprocessing/vision/params.yaml` 为例）

```yaml
videos:
  manifest: data/videos/manifest.csv   # 列：path,site,roi_poly,start_ts,end_ts
  chunk_sec: 300                       # 切片长度（秒）
  transcode: { enable: true, scale_h: 720, fps: 15, gop: 30 }
  out_tmp: outputs/data/artifacts/vision/chunks
yolo: { model: yolov8n.pt, conf: 0.4, iou: 0.5, classes: [2,3,5,7] }
infer: { device: "0", batch: 16, half: true, frame_stride: 2 }
tracking: { max_age: 30, match_thresh: 80.0 }
roi: { use_polygon: false, poly: [], count_line: [0.1,0.8, 0.9,0.8] }   # 归一化
calib: { pixels_per_meter: 8.0, roi_length_m: 80.0 }
aggregate: { win_sec: 10, out_dir: outputs/data/artifacts/vision, tag: vision_v1 }
logging: { save_debug_frames: false, report_dir: outputs/reports/vision }
```

> 约定输出（按 schemas）：
>
> * 切片明细：`outputs/data/artifacts/vision/tracks/<site>/<chunk>.parquet`
> * 切片秒级聚合：`outputs/data/artifacts/vision/agg/<site>/<chunk>.csv`
> * 站点级：`outputs/data/artifacts/vision/traffic_<site>_<tag>.csv`（列：`timestamp,q,v,k,site`）

#### 7.2 新建一个预处理任务的步骤（模板）

1. **确定输入/输出**：

   * 输入：原始数据位置、清单（如 manifest）、必要的标定/ROI。
   * 输出：给 models/\* 使用的标准 CSV/Parquet（列名固定，见 `schemas.py`）。

2. **写参数文件**：放在 `src/preprocessing/<module>/params.yaml`，字段清晰、可复用。

3. **实现任务脚本**：

   * 简单任务可直接写成**独立 pipeline**（如 `vision_pipeline.py`）。
   * 通用化可继承基类：在 `preprocessing/base.py` 中使用 `PreprocessTask` 与注册器。

最小基类（已内置）：

```python
# src/preprocessing/base.py
class PreprocessTask:
    def __init__(self, cfg): self.cfg = cfg
    def setup(self): pass
    def run(self) -> dict: raise NotImplementedError
    def teardown(self): pass
```

4. **实现核心逻辑**（示例：tabular 清洗）

```python
# src/preprocessing/tabular/clean.py
import pandas as pd
def clean_table(df, dropna_cols=None, dedup=True, rename_map=None):
    if rename_map: df = df.rename(columns=rename_map)
    if dropna_cols: df = df.dropna(subset=dropna_cols)
    if dedup: df = df.drop_duplicates()
    return df
```

5. **schema 与校验**（推荐）：
   在 `schemas.py` 定义列名与 `validate_*` 函数，保存前调用，避免下游读取时报错。

6. **把任务接到 pipeline**：

   * 独立跑：`python -m src.pipelines.vision_pipeline --config src/preprocessing/vision/params.yaml`
   * 或在通用 `preprocess_pipeline.py` 里按步骤队列执行（需把任务注册成 `PreprocessTask`）。

#### 7.3 如何与训练流程对接

* 预处理产出（例如 `traffic_site*.csv`）**就是训练数据**：
  在对应算法的 `params.yaml` 中，直接把 `dataset.path` 指向这些 CSV：

  ```yaml
  # 例：LWR 时间序列模型
  task: ts
  dataset:
    path: outputs/data/artifacts/vision/traffic_site3_vision_v1.csv
    target: q           # 或者 v/k，视模型而定
  model: { name: LWR, params: {...} }
  outputs: { base_dir: outputs, tag: lwr_v1 }
  ```
* 其余一致：`python -m src.pipelines.ts_pipeline --config src/models/ts/LWR/params.yaml`

#### 7.4 一个“表格→训练”的最小闭环（示例）

1. **预处理**（清洗+重采样）：

```python
# 清洗
python - <<'PY'
import pandas as pd
from src.preprocessing.tabular.clean import clean_table
df = pd.read_csv("data/raw_sensors.csv")
df = clean_table(df, dropna_cols=["timestamp","flow"])
df.to_csv("outputs/data/artifacts/tabular/clean.csv", index=False)
PY

# 重采样为 10s
python - <<'PY'
from src.preprocessing.tabular.timesync import resample_uniform
resample_uniform("outputs/data/artifacts/tabular/clean.csv",
                 "outputs/data/artifacts/tabular/clean_10s.csv",
                 freq="10s")
PY
```

2. **训练**（以 XGBoost 为例，修改 `src/models/reg/XGBoost/params.yaml`）：

```yaml
dataset:
  path: outputs/data/artifacts/tabular/clean_10s.csv
  target: price
  features: []
```

```bash
python -m src.pipelines.reg_pipeline --config src/models/reg/XGBoost/params.yaml
```

#### 7.5 预处理模块的最佳实践

* **解耦**：预处理与训练分离；预处理可重复执行、断点续跑（存在即跳过）。
* **显式参数**：所有可变项入 `params.yaml`（输入位置、窗口、抽帧、阈值、输出目录）。
* **中间件落盘**：保存中间结果（Parquet/CSV），利于调试与审计。
* **统一 schema**：在 `schemas.py` 统一字段；保存前 `validate_*`。
* **可视化质检**：抽样保存调试帧/图（如画框、计数线、分布图），快速发现问题。
* **日志与报告**：在 `outputs/reports/xxx.json` 记录运行参数、耗时、输入/输出路径，便于复现。

---

