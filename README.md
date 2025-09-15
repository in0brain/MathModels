# 📘 项目操作手册（pip 版）

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
│  ├─ models/                          # 算法集合
│  │   ├─ __init__.py
│  │   ├─ MarkovChain/
│  │   │   ├─ __init__.py
│  │   │   ├─ build.py                 # 训练/预测/作图/推理
│  │   │   └─ params.yaml              # 马尔科夫链配置
│  │   ├─ XGBoost/
│  │   │   ├─ __init__.py
│  │   │   ├─ build.py                 # 回归模型逻辑
│  │   │   └─ params.yaml              # XGBoost 配置
│  │   ├─ LightGBM/                    # （后续扩展）
│  │   └─ KMeans/                      # （后续扩展）
│  └─ pipelines/                       # 训练流水线（按任务）
│      ├─ __init__.py
│      ├─ ts_pipeline.py               # 时间序列
│      ├─ reg_pipeline.py              # 回归
│      ├─ clf_pipeline.py              # 分类
│      └─ clu_pipeline.py              # 聚类
├─ requirements.txt                    # pip 依赖清单
└─ README.md                           # 操作手册
```

> 为保证 `python -m src.xxx` 可用，**src、core、models、pipelines、各算法目录都需有空的 `__init__.py`**。

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

**安装命令（在项目根目录执行）：**

```bash
pip install -r requirements.txt
```

---

## 3) 配置文件（\*.yaml）逐行解释

### 3.1 MarkovChain（`src/models/MarkovChain/params.yaml`）

```yaml
task: ts                        # 任务类型：ts=时间序列
dataset:
  path: data/markov_demo.csv    # 训练数据路径（CSV）
  time_col: t                   # 时间/序号列名（用于绘图标注）
  state_col: state              # 状态列名（模型使用的离散状态）
  test_ratio: 0.2               # 测试集比例=20%（从序列尾部分割）
preprocess:
  dropna: true                  # 训练前是否丢弃缺失（针对 state_col）
model:
  name: MarkovChain             # 算法别名（必须等于 build.py 中 ALGO）
  params:
    order: 1                    # 马尔可夫阶数（当前实现用于一阶）
    smoothing: 1e-6             # 加性平滑，避免零概率
    topk_eval: 1                # 评估时取概率最大的1类，计算acc
eval:
  metrics: [acc]                # 评估指标：准确率
viz:
  enabled: true                 # 是否生成图像
  dpi: 160                      # 图片分辨率
  plots:
    trans_heatmap: true         # 输出转移矩阵热力图
    seq_compare: true           # 输出真实/预测序列对比图
outputs:
  base_dir: outputs             # 所有产物的根目录
  tag: demo_markov              # 实验标签（用于输出文件命名）
seed: 42                        # 随机种子
```

### 3.2 XGBoost 回归（`src/models/XGBoost/params.yaml`）

```yaml
task: reg                       # 任务类型：reg=回归
dataset:
  path: data/house.csv          # 训练数据路径（CSV）
  target: price                 # 目标列（回归标签）
  features: []                  # 特征列清单；为空=自动取数值列（排除 target）
  test_size: 0.2                # 测试集比例（随机划分）
preprocess:
  dropna: false                 # 是否先丢弃含 target 缺失的行
  impute_num: median            # 数值缺失填充策略（中位数）
  impute_cat: most_frequent     # 类别缺失填充策略（众数）
  scale_num: true               # 数值特征是否标准化
  one_hot_cat: true             # 类别特征是否 OneHot 编码
model:
  name: XGBoost                 # 算法别名（必须等于 build.py 中 ALGO）
  params:                       # 传入 XGBRegressor 的超参数
    n_estimators: 800
    max_depth: 8
    learning_rate: 0.05
    subsample: 0.9
    colsample_bytree: 0.9
    tree_method: hist
eval:
  metrics: [MAE, R2]            # 回归评估指标：MAE、R2
viz:
  enabled: true                 # 是否生成图像
  dpi: 160
  plots:
    feat_importance: true       # 特征重要性条形图
    residuals: true             # 残差直方图
    pred_scatter: true          # 真实 vs 预测 散点
outputs:
  base_dir: outputs
  tag: house_xgb                # 实验标签
seed: 42
```

---

## 4) 工作流（以马尔科夫链为例）

### 4.1 生成示例数据（可跳过，若你已有数据）

```bash
python scripts/MarkovChain/datacreate.py
```

* 作用：生成 `data/markov_demo.csv`，含两列：`t`（时间步）、`state`（离散状态）。

### 4.2 训练与产出（必须从**项目根目录**执行）

```bash
python -m src.pipelines.ts_pipeline --config src/models/MarkovChain/params.yaml
```

* **做了什么**：

  1. 读取 `params.yaml`；
  2. 加载 `data/markov_demo.csv`；
  3. 训练 MarkovChain；
  4. 评估 & 产出图像/表格/模型。
* **产出文件说明**：

  * `outputs/data/artifacts/MarkovChain/demo_markov_transition_matrix.csv`
    → 训练得到的状态转移矩阵（行=当前状态，列=下一状态概率）
  * `outputs/data/predictions/MarkovChain/demo_markov_preds.csv`
    → 测试段逐步 next-state 的真实与预测对齐表
  * `outputs/reports/demo_markov_metrics.json`
    → 指标（如 acc）与样本数量
  * `outputs/figs/ts/demo_markov_trans_heatmap.png`
    → 转移矩阵热力图
  * `outputs/figs/ts/demo_markov_seq_compare.png`
    → 序列对比图（真实 vs 预测）
  * `outputs/models/demo_markov.pkl`
    → 训练好的模型（持久化保存，便于复用）

> 若**无输出**，多半是包路径问题：确保 `src` 及其子目录都有空的 `__init__.py`；并从项目根执行命令。

---

## 5) 再预测（不用重新训练）

### 5.1 准备新数据

* 放入 `data/`，例如 `data/new_states.csv`
* **必须包含表头**：与配置一致

  * `state_col`（默认 `state`）必需；
  * `time_col`（默认 `t`）可选，仅用于绘图时的横轴标注；
* 如列名不同，请修改 `src/models/MarkovChain/params.yaml` 中：

```yaml
dataset:
  path: data/new_states.csv
  time_col: your_time_col_name
  state_col: your_state_col_name
```

### 5.2 执行再预测命令

```bash
python -m src.inference.runner \
  --task ts \
  --algo MarkovChain \
  --model outputs/models/demo_markov.pkl \
  --data data/new_states.csv \
  --config src/models/MarkovChain/params.yaml \
  --tag new_markov
```

* **产出**：`outputs/data/predictions/MarkovChain/new_markov_infer_preds.csv`（输入状态与预测的下一状态）

> 回归/分类/聚类的再预测命令同理，只需把 `--task/--algo/--model/--data/--config/--tag` 换成对应算法与数据即可。

---

## 6) 重新训练：需要清理什么

若想“从零开始”重新训练，建议删除旧产物（不删也行，但易混淆）：

* **模型文件**：`outputs/models/<tag>.pkl`
* **训练工件**：`outputs/data/artifacts/<Algo>/*`
* **预测结果**：`outputs/data/predictions/<Algo>/*`
* **指标报告**：`outputs/reports/<tag>_metrics.json`
* **图表**：`outputs/figs/<task>/*`

然后重新执行训练命令：

```bash
python -m src.pipelines.ts_pipeline --config src/models/MarkovChain/params.yaml
```

---

## 7) 常见问题（FAQ）

* **Q：为什么一定要有 `__init__.py`？**
  A：让目录被 Python 当作“包”识别，`python -m src.xxx` 和自动注册（registry）才能找到模块。

* **Q：我只用 CSV，可以删 xlsx/parquet 支持吗？**
  A：可以。把 `core/io.py` 中的非 CSV 分支删掉，`requirements.txt` 也无需 `openpyxl`。

* **Q：XGBoost 回归如何运行？**
  A：确保 `data/house.csv` 和 `XGBoost/params.yaml` 的列名一致，然后执行：

  ```bash
  python -m src.pipelines.reg_pipeline --config src/models/XGBoost/params.yaml
  ```

  再预测（用训练好的模型）：

  ```bash
  python -m src.inference.runner \
    --task reg \
    --algo XGBoost \
    --model outputs/models/house_xgb.pkl \
    --data data/house_new.csv \
    --config src/models/XGBoost/params.yaml \
    --tag house_xgb_new
  ```

---
