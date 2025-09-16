# 项目操作手册（pip 版）

## 1) 项目结构与作用

```
mcm-ml/
├─ data/                               # 原始数据 & 新数据
│   ├─ markov_demo.csv
│   └─ house.csv
├─ outputs/                            # 运行结果（自动生成）
│  ├─ data/
│  │  ├─ artifacts/                    # 训练工件（如转移矩阵）
│  │  │   ├─ MarkovChain/
│  │  │   └─ XGBoost/
│  │  └─ predictions/                  # 预测结果（按算法分类）
│  │      ├─ MarkovChain/
│  │      └─ XGBoost/
│  ├─ figs/
│  │  ├─ ts/                           # 时间序列图表
│  │  ├─ reg/                          # 回归图表
│  │  ├─ clf/                          # 分类图表
│  │  └─ clu/                          # 聚类图表
│  ├─ models/                          # 训练好的模型（.pkl）
│  └─ reports/                         # 运行报告（.json）
├─ scripts/                            # 辅助脚本（造数/批量测试）
│  └─ MarkovChain/
│      └─ datacreate.py
├─ src/
│  ├─ __init__.py
│  ├─ core/                            # 公共工具模块
│  │   ├─ __init__.py
│  │   ├─ io.py                        # 统一输入输出
│  │   ├─ metrics.py                   # 指标计算（acc/MAE/R2/等）
│  │   ├─ registry.py                  # 自动注册算法
│  │   └─ viz.py                       # 可视化工具
│  ├─ inference/                       # 再预测逻辑
│  │   ├─ __init__.py
│  │   └─ runner.py                    # 通用推理入口
│  ├─ models/                          # 算法集合（按任务分组）
│  │   ├─ __init__.py
│  │   ├─ ts/                          # 时间序列类算法
│  │   │   ├─ __init__.py
│  │   │   └─ MarkovChain/
│  │   │       ├─ __init__.py
│  │   │       ├─ build.py
│  │   │       └─ params.yaml
│  │   ├─ reg/                         # 回归类算法
│  │   │   ├─ __init__.py
│  │   │   └─ XGBoost/
│  │   │       ├─ __init__.py
│  │   │       ├─ build.py
│  │   │       └─ params.yaml
│  │   ├─ clf/                         # 分类类算法
│  │   │   ├─ __init__.py
│  │   │   └─ NeuralNet/
│  │   │       ├─ __init__.py
│  │   │       ├─ build.py
│  │   │       └─ params.yaml
│  │   └─ clu/                         # 聚类类算法
│  │       ├─ __init__.py
│  │       └─ KMeans/                  # （后续扩展）
│  │           ├─ __init__.py
│  │           ├─ build.py
│  │           └─ params.yaml
│  └─ pipelines/                       # 训练流水线（按任务）
│      ├─ __init__.py
│      ├─ ts_pipeline.py
│      ├─ reg_pipeline.py
│      ├─ clf_pipeline.py
│      └─ clu_pipeline.py
├─ requirements.txt                    # pip 依赖清单
└─ README.md                           # 操作手册
```

> 为保证 `python -m src.xxx` 可用，**src、core、models、pipelines、四个子目录、每个算法目录都需有空的 `__init__.py`**。

---

## 2) 依赖安装（requirements.txt）

**requirements.txt：**

```
pandas
numpy
matplotlib
PyYAML
joblib
scikit-learn
xgboost
```

**安装命令：**

```bash
pip install -r requirements.txt
```

---

## 3) 配置文件（\*.yaml）逐行解释

### 3.1 MarkovChain（`src/models/ts/MarkovChain/params.yaml`）

```yaml
task: ts                        # 任务类型：ts=时间序列
dataset:
  path: data/markov_demo.csv    # 训练数据路径
  time_col: t                   # 时间/序号列名
  state_col: state              # 状态列名
  test_ratio: 0.2               # 测试集比例=20%
preprocess:
  dropna: true                  # 是否丢弃缺失
model:
  name: MarkovChain
  params:
    order: 1
    smoothing: 1e-6
    topk_eval: 1
eval:
  metrics: [acc]
viz:
  enabled: true
  dpi: 160
  plots:
    trans_heatmap: true
    seq_compare: true
outputs:
  base_dir: outputs
  tag: demo_markov
seed: 42
```

### 3.2 XGBoost 回归（`src/models/reg/XGBoost/params.yaml`）

```yaml
task: reg
dataset:
  path: data/house.csv
  target: price
  features: []
  test_size: 0.2
preprocess:
  dropna: false
  impute_num: median
  impute_cat: most_frequent
  scale_num: true
  one_hot_cat: true
model:
  name: XGBoost
  params:
    n_estimators: 800
    max_depth: 8
    learning_rate: 0.05
    subsample: 0.9
    colsample_bytree: 0.9
    tree_method: hist
eval:
  metrics: [MAE, R2]
viz:
  enabled: true
  dpi: 160
  plots:
    feat_importance: true
    residuals: true
    pred_scatter: true
outputs:
  base_dir: outputs
  tag: house_xgb
seed: 42
```

---

## 4) 工作流（以马尔科夫链为例）

### 4.1 生成示例数据

```bash
python scripts/MarkovChain/datacreate.py
```

### 4.2 训练与产出

```bash
python -m src.pipelines.ts_pipeline --config src/models/ts/MarkovChain/params.yaml
```

* 产出示例：

  * `outputs/data/artifacts/MarkovChain/demo_markov_transition_matrix.csv`
  * `outputs/data/predictions/MarkovChain/demo_markov_preds.csv`
  * `outputs/reports/demo_markov_metrics.json`
  * `outputs/figs/ts/demo_markov_trans_heatmap.png`
  * `outputs/figs/ts/demo_markov_seq_compare.png`
  * `outputs/models/demo_markov.pkl`

---

## 5) 再预测

---

### 5.1 准备新数据

放入 `data/`，如 `data/new_states.csv`，保证表头与配置一致。

---

### 5.2 执行命令

```bash
python -m src.inference.runner \
  --task ts \
  --algo MarkovChain \
  --model outputs/models/demo_markov.pkl \
  --data data/new_states.csv \
  --config src/models/ts/MarkovChain/params.yaml \
  --tag new_markov
```

---

>【**注意：不要全部复制，斜杠和换行会被复制进去**】，
>正确的为：`python -m src.inference.runner --task ts --algo MarkovChain --model outputs/models/demo_markov.pkl --data data/new_states.csv --config src/models/ts/MarkovChain/params.yaml --tag new_markov`


产出：`outputs/data/predictions/MarkovChain/new_markov_infer_preds.csv`

---

## 6) 重新训练需要清理什么

删除：

* `outputs/models/<tag>.pkl`
* `outputs/data/artifacts/<Algo>/*`
* `outputs/data/predictions/<Algo>/*`
* `outputs/reports/<tag>_metrics.json`
* `outputs/figs/<task>/*`

然后重新执行训练命令：

```bash
python -m src.pipelines.ts_pipeline --config src/models/ts/MarkovChain/params.yaml
```

---

## 7) 常见问题（FAQ）

* **Q：为什么要有 `__init__.py`？** <br>
  A：保证 Python 能把目录识别为包，`python -m src.xxx` 和 registry 才能找到。

* **Q：XGBoost 回归如何运行？** <br>
  A：

  ```bash
  python -m src.pipelines.reg_pipeline --config src/models/reg/XGBoost/params.yaml
  ```

  再预测：

  ```bash
  python -m src.inference.runner \
    --task reg \
    --algo XGBoost \
    --model outputs/models/house_xgb.pkl \
    --data data/house_new.csv \
    --config src/models/reg/XGBoost/params.yaml \
    --tag house_xgb_new
  ```

## 8） 快速上手命令（总结）

以下命令需在 **项目根目录** 执行。【~~在控制台输入pwd看看~~】

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 可选：生成示例数据

（不同算法的脚本放在 `scripts/` 下）

```bash
# 分类 - 神经网络
python scripts/NeuralNet/datacreate.py

# 时间序列 - 马尔科夫链
python scripts/MarkovChain/datacreate.py
```

### 3) 训练模型

```bash
# 时间序列 - 马尔科夫链
python -m src.pipelines.ts_pipeline  --config src/models/ts/MarkovChain/params.yaml

# 回归 - XGBoost
python -m src.pipelines.reg_pipeline --config src/models/reg/XGBoost/params.yaml

# 分类 - 神经网络
python -m src.pipelines.clf_pipeline --config src/models/clf/NeuralNet/params.yaml

# 聚类 - KMeans
python -m src.pipelines.clu_pipeline --config src/models/clu/KMeans/params.yaml
```

### 4) 再预测（使用训练好的模型）

```bash
python -m src.inference.runner \
  --task <任务类型: ts|reg|clf|clu> \
  --algo <算法名: MarkovChain|XGBoost|NeuralNet|KMeans> \
  --model outputs/models/<tag>.pkl \
  --data data/<new_data>.csv \
  --config src/models/<子目录>/<算法名>/params.yaml \
  --tag <新实验标签>
```

**示例（神经网络分类）：**

```bash
python -m src.inference.runner \
  --task clf \
  --algo NeuralNet \
  --model outputs/models/nn_clf_demo.pkl \
  --data data/clf_new.csv \
  --config src/models/clf/NeuralNet/params.yaml \
  --tag nn_clf_new
```

------

快速定位 **安装 → 造数 → 训练 → 再预测**四步操作。



## 9）任务–算法–命令对照表

| 任务类型      | 算法名称        | 训练命令                                                     | 再预测命令示例                                               |
| ------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 时间序列 (ts) | MarkovChain     | `bash<br>python -m src.pipelines.ts_pipeline --config src/models/ts/MarkovChain/params.yaml<br>` | `bash<br>python -m src.inference.runner \ <br> --task ts \ <br> --algo MarkovChain \ <br> --model outputs/models/demo_markov.pkl \ <br> --data data/new_states.csv \ <br> --config src/models/ts/MarkovChain/params.yaml \ <br> --tag new_markov<br>` |
| 回归 (reg)    | XGBoost         | `bash<br>python -m src.pipelines.reg_pipeline --config src/models/reg/XGBoost/params.yaml<br>` | `bash<br>python -m src.inference.runner \ <br> --task reg \ <br> --algo XGBoost \ <br> --model outputs/models/house_xgb.pkl \ <br> --data data/house_new.csv \ <br> --config src/models/reg/XGBoost/params.yaml \ <br> --tag house_xgb_new<br>` |
| 分类 (clf)    | NeuralNet (MLP) | `bash<br>python -m src.pipelines.clf_pipeline --config src/models/clf/NeuralNet/params.yaml<br>` | `bash<br>python -m src.inference.runner \ <br> --task clf \ <br> --algo NeuralNet \ <br> --model outputs/models/nn_clf_demo.pkl \ <br> --data data/clf_new.csv \ <br> --config src/models/clf/NeuralNet/params.yaml \ <br> --tag nn_clf_new<br>` |
| 聚类 (clu)    | KMeans          | `bash<br>python -m src.pipelines.clu_pipeline --config src/models/clu/KMeans/params.yaml<br>` | `bash<br>python -m src.inference.runner \ <br> --task clu \ <br> --algo KMeans \ <br> --model outputs/models/kmeans_demo.pkl \ <br> --data data/clu_new.csv \ <br> --config src/models/clu/KMeans/params.yaml \ <br> --tag kmeans_new<br>` |


## 10）扩展新算法步骤

----
### 0)
为了主分支不被影响，可以在git中创建新的版本分支`dev`
1. 在本地创建分支
```bash
# 确认在项目目录下
git checkout main   # 切到主分支（或 master）
git pull origin main   # 确保代码是最新的
git checkout -b dev # 创建并切换到新分支

```
2. 将分支推送到 GitHub**

```bash
git push origin dev

```

3. 其他人想在这个分支上开发，可以这样：

git fetch origin
git checkout dev

4. 提交代码

在分支里正常提交即可：
```bash
git add .
git commit -m "xx算法"
git push origin dev
```


5. 合并到主分支（推荐用 Pull Request）

在 GitHub 网页上，进入仓库后点 Pull requests → New pull request。

选择目标分支（通常是 main），源分支（如 dev）。

提交 PR，让其他人 review 后再合并。

---


为了保证新算法能够无缝接入 **pipeline / inference / registry**，需要按以下规范添加。假设我们要扩展一个新算法 `MyAlgo` 到 **分类任务 (clf)**：

### 1) 新建目录

在对应任务子目录下新建文件夹：

```
src/models/clf/MyAlgo/
├─ __init__.py        # 空文件，保证 Python 包识别
├─ build.py           # 算法核心逻辑（训练/评估/推理）
└─ params.yaml        # 配置文件（超参数、数据字段等）
```

### 2) 编写配置文件（params.yaml）

最小模板：

```yaml
task: clf                  # 任务类型：ts | reg | clf | clu
dataset:
  path: data/myalgo.csv    # 输入数据
  target: label            # 标签列名（分类/回归时）
  features: []             # 特征列，空=自动选择
model:
  name: MyAlgo             # 算法别名（必须与 build.py 中 ALGO 一致）
  params:
    param1: value1         # 算法超参数
    param2: value2
eval:
  metrics: [ACC, F1]       # 指标选择，具体由 metrics.py 实现
viz:
  enabled: true
outputs:
  base_dir: outputs
  tag: myalgo_demo
seed: 42
```

### 3) 编写 build.py

最小接口规范：

```python
# -*- coding: utf-8 -*-
from typing import Dict, Any
import pandas as pd

TASK = "clf"       # 与 params.yaml 中 task 一致
ALGO = "MyAlgo"    # 与 params.yaml 中 model.name 一致

def build(cfg: Dict[str, Any]):
    """构建模型对象（可返回 sklearn/自定义对象）"""
    ...

def fit(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    训练 + 评估 + 保存结果
    返回字典：{"metrics": {...}, "artifacts": {...}}
    """
    ...

def inference(model, df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    加载已训练模型，对新数据进行预测
    返回字典：{"predictions_csv": 路径}
    """
    ...
```

### 4) 自动注册

无需手动修改 `registry.py`。
 因为 `registry` 会递归扫描 `src/models/` 下的 `build.py`，并自动注册键：

```
"{task}:{algo}".lower()
```

例如：

```
"clf:myalgo"
```

### 5) 运行方式

- 训练：

  ```bash
  python -m src.pipelines.clf_pipeline --config src/models/clf/MyAlgo/params.yaml
  ```

- 再预测：

  ```bash
  python -m src.inference.runner \
    --task clf \
    --algo MyAlgo \
    --model outputs/models/myalgo_demo.pkl \
    --data data/myalgo_new.csv \
    --config src/models/clf/MyAlgo/params.yaml \
    --tag myalgo_new
  ```

### 6) 建议

- 若算法需要特定评估指标或可视化方法，请在 `src/core/metrics.py` / `src/core/viz.py` 中 **追加函数**，不要覆盖已有。
- 输出路径统一通过 `io.out_path_predictions` / `io.save_model` 等接口，保持与现有算法一致。
- 若算法依赖额外库，请在 `requirements.txt` 中补充。