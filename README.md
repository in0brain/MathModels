# 项目操作手册（pip 版）

## 一、项目结构与模块作用

```
mcm-ml/
├── data/                               # 存放原始数据
│   ├── house.csv
│   └── ...
├── outputs/                            # 所有自动生成的产出 (被 .gitignore 忽略)
│   ├── data/
│   │   ├── artifacts/                  # 预处理/训练的中间产物
│   │   └── predictions/                # 模型的预测结果
│   ├── models/                         # 训练好的模型 (.pkl)
│   ├── plots/                          # 生成的图表
│   └── reports/                        # 评估指标报告 (.json, .csv)
├── runs/                               # (新增) 可复用的实验配置文件集合
│   ├── house_lgbm.yaml
│   ├── house_rf.yaml
│   ├── house_xgb.yaml
│   └── to_compare.yaml
├── scripts/                            # 数据生成等辅助脚本
│   ├── KMeans/datacreate.py
│   └── XGBoost/datacreate.py
├── src/
│   ├── core/                           # 核心公共工具
│   │   ├── io.py                       # IO 工具：读写 CSV/Parquet、保存模型
│   │   ├── metrics.py                  # 指标函数：ACC/F1/MAE/R2...
│   │   ├── registry.py                 # 自动注册算法 (task:algo -> build.py)
│   │   └── viz.py                      # 可视化工具：特征重要性、残差图等
│   ├── inference/                      # 使用已训练模型进行再预测
│   │   └── runner.py                   # 通用推理入口
│   ├── models/                         # 各类算法模块 (核心)
│   │   ├── clf/                        # 分类模型
│   │   ├── clu/                        # 聚类模型
│   │   ├── reg/                        # --- 回归模型 ---
│   │   │   ├── LightGBM/
│   │   │   │   ├── build.py
│   │   │   │   └── params.yaml
│   │   │   ├── RandomForest/
│   │   │   │   ├── build.py
│   │   │   │   └── params.yaml
│   │   │   └── XGBoost/
│   │   │       ├── build.py
│   │   │       └── params.yaml
│   │   └── ts/                         # 时间序列模型
│   ├── pipelines/                      # (新增与优化) 核心工作流
│   │   ├── compare_pipeline.py         # (新增) 多模型对比流水线
│   │   ├── eda_pipeline.py             # (新增) 探索性数据分析流水线
│   │   ├── hyperopt_pipeline.py        # (新增) Optuna超参数优化流水线
│   │   ├── preprocess_pipeline.py      # (优化) 模块化预处理流水线
│   │   ├─ ts_pipeline.py   # 时间序列任务
│   │   ├─ reg_pipeline.py  # 回归任务
│   │   ├─ clf_pipeline.py  # 分类任务
│   │   └─ clu_pipeline.py  # 聚类任务
│   └─ preprocessing/                  # (已重构) 数据预处理模块
│       ├── base.py
│       ├── tabular/                    # 表格数据处理任务
│       │   ├── balance.py
│       │   ├── dimreduce.py
│       │   ├── encode.py
│       │   ├── feature_engineering.py
│       │   ├── impute.py
│       │   ├── normalize.py
│       │   └── steps_house.yaml
│       └── vision/                     # 视觉数据处理任务
├── requirements.txt                    # Python 依赖清单
└── README.md                           # 本文档
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

## 三、对比XGBoost, RandomForest, LightGBM模型

### Quick Start（最小数据集生成）

本项目已在 `.gitignore` 中忽略了 `data/` 文件夹，开发者本地需要先生成一份最小可运行的数据集。

执行以下命令：

```bash
python scripts/make_toy_data.py
```

该脚本会在 `data/house.csv` 生成一份 300×8 的示例数据，包含数值特征、类别特征与目标价格列：

- 数值特征：`feat1, feat2, feat3, feat4, feat5`
- 类别特征：`city, type`
- 目标列：`price`

随后可运行预处理流水线：

```bash
python -m src.pipelines.preprocess_pipeline --config src/preprocessing/tabular/steps_house.yaml
```

产物会保存在：

```
outputs/data/artifacts/tabular/house_enc.csv
```

------

### 多模型对比教程

本项目支持对 **XGBoost / RandomForest / LightGBM** 等模型进行横向对比，流程完全由 YAML 配置驱动。

#### 1. 配置文件说明

- `runs/house_xgb.yaml`：XGBoost 模型配置
- `runs/house_rf.yaml`：RandomForest 模型配置
- `runs/house_lgbm.yaml`：LightGBM 模型配置
- `runs/to_compare.yaml`：对比清单（列出要比较的配置）

配置文件基本结构：

```yaml
task: reg

dataset:
  path: outputs/data/artifacts/tabular/house_enc.csv
  target: price

split:
  test_size: 0.2
  random_state: 42
  shuffle: true

preprocess:
  impute_num: median
  scale_num: true
  one_hot_cat: true
  impute_cat: most_frequent

model:
  name: XGBoost          # 或 RandomForest / LightGBM
  params:
    n_estimators: 400
    max_depth: 6
    learning_rate: 0.05

outputs:
  base_dir: outputs
  tag: xgb_baseline

eval:
  metrics: ["MAE","RMSE","R2"]

viz:
  enabled: true
  dpi: 160
```

对比清单 `runs/to_compare.yaml`：

```yaml
configs:
  - runs/house_xgb.yaml
  - runs/house_rf.yaml
  - runs/house_lgbm.yaml
```

------

#### 2. 运行命令

**单模型训练**

```bash
# XGBoost
python -m src.pipelines.reg_pipeline --config runs/house_xgb.yaml

# RandomForest
python -m src.pipelines.reg_pipeline --config runs/house_rf.yaml

# LightGBM（需先安装）
pip install lightgbm
python -m src.pipelines.reg_pipeline --config runs/house_lgbm.yaml
```

**一键对比**

```bash
# 汇总已有结果
python -m src.pipelines.compare_pipeline --list runs/to_compare.yaml --out outputs/reports/comparison_house.csv

# 重训并汇总（推荐）
python -m src.pipelines.compare_pipeline --list runs/to_compare.yaml --out outputs/reports/comparison_house.csv --rerun
```

产物：

- **CSV**：`outputs/reports/comparison_house.csv`（横向指标表）
- **图表**：`outputs/plots/compare/compare_MAE.png`, `compare_RMSE.png`, `compare_R2.png`

------

#### 3. 常见坑与解决方案

- `KeyError: 'outputs'` → YAML 缺少 `outputs:`，需补齐。
- `got multiple values for keyword argument 'random_state'` → 代码已改为 `params.setdefault()`，避免冲突。
- `RandomForestRegressor(max_features='auto')` 报错 → 改成 `null` 或 `"sqrt"`。
- 类别列做均值时报错 → 已区分数值列与类别列，类别列用众数填补 + OneHot。
- LightGBM 未安装 → `pip install lightgbm`。

------

```bash
git clone <repo-url>
cd MathModels
pip install -r requirements.txt
python scripts/make_toy_data.py
python -m src.pipelines.preprocess_pipeline --config src/preprocessing/tabular/steps_house.yaml
python -m src.pipelines.compare_pipeline --list runs/to_compare.yaml --out outputs/reports/comparison_house.csv --rerun
```


## 四、如何扩展新算法模块

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

