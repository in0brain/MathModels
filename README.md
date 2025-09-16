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
├─ __init__.py
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
