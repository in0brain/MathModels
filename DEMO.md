# 2025年华为杯E题

本文档详细介绍了为解决“2025年中国研究生数学建模竞赛E题 - 高铁轴承智能故障诊断问题”而构建的全新数据处理与建模工作流。该工作流基于本项目的模块化设计，新增了针对振动信号数据的预处理模块和相应的分类模型。

## 一、新增模块与文件概览

为了实现方案，我们对项目结构进行了如下扩展：

```
MathModels/
└── src/
    ├── models/
    │   └── clf/
    │       └── XGBoost/             # (新增) XGBoost分类器模块
    │           ├── build.py         # 核心逻辑：模型训练、评估、保存
    │           └── params.yaml      # 配置文件：数据路径、模型超参等
    │
    └── preprocessing/
        └── signal/                  # (新增) 信号数据预处理模块
            ├── load_bearing_data.py # 任务1：加载.mat，处理多层目录
            ├── feature_extraction.py  # 任务2：信号处理与特征提取
            ├── steps_load_data.yaml   # 流水线配置1：驱动数据加载
            └── steps_feature_extraction.yaml # 流水线配置2：驱动特征提取
```

## 二、全新工作流详解(1,2题)

整个工作流被划分为三个核心阶段，每个阶段都由独立的脚本和配置文件驱动，实现了高度的模块化和可复现性。

### **阶段一：原始数据加载与结构化**

此阶段的目标是将原始、分散、多层级目录下的 `.mat` 格式数据，转换为统一的、结构化的 Parquet 格式，并生成一份总的元数据清单 `manifest.csv`。

  * **执行脚本**: `src/preprocessing/signal/load_bearing_data.py`

      * **作用**:
        1.  **递归搜索**：利用 `os.walk` 遍历多层级目录，查找所有 `.mat` 文件。
        2.  **格式解析**：兼容多种文件名格式（如 `B007_0.mat`, `OR007@3_0.mat`, `Normal_0.mat`），提取故障类型、尺寸、载荷等元信息。
        3.  **数据转换**：读取 `.mat` 文件中的振动信号，将其存储为高效的 Parquet 格式。

  * **配置文件**: `src/preprocessing/signal/steps_load_data.yaml`

      * **作用**: 为 `load_bearing_data.py` 提供输入输出路径。
      * **关键参数**:
          * `source_dir`: **[需要修改]** 指向存放源域数据集的根目录。
          * `target_dir`: **[需要修改]** 指向存放目标域数据集的根目录。
          * `out_dir`: 指定 Parquet 信号文件的输出目录。

  * **生成文件及其作用**:

      * `outputs/data/artifacts/signal_parquet/`: 存放所有单个信号的 **Parquet 文件**。每个文件包含一段纯净的振动信号，是后续特征提取的基础。
      * `outputs/data/artifacts/manifest.csv`: **元数据总清单**。这是承上启下的关键文件，记录了每个信号的来源、标签、对应的Parquet文件路径等，指导下一阶段的处理。

### **阶段二：信号处理与特征提取**

此阶段遵循方案要求，对结构化的信号数据进行标准化处理，并提取多维度特征，最终生成可供机器学习模型使用的特征表。

  * **执行脚本**: `src/preprocessing/signal/feature_extraction.py`

      * **作用**:
        1.  **统一采样率**: 将所有信号重采样至统一频率（如 24kHz）。
        2.  **信号分窗**: 将长信号分割成带有重叠的短窗口。
        3.  **特征提取**: 对每个窗口，提取**时域**、**频域**和**包络谱**等多维度特征。

  * **配置文件**: `src/preprocessing/signal/steps_feature_extraction.yaml`

      * **作用**: 为 `feature_extraction.py` 提供信号处理的核心参数。
      * **关键参数**:
          * `manifest_path`: 输入的 `manifest.csv` 清单路径。
          * `out_path`: 输出的 `features.parquet` 特征表路径。
          * `target_fs: 24000`: 对应方案中的 **“统一采样率”** 要求。
          * `window_size: 4096`: 对应方案中的 **“固定分窗，长度为 ℓ=4096”** 要求。
          * `overlap: 0.5`: 对应方案中的 **“步长为50%重叠”** 要求。

  * **生成文件及其作用**:

      * `outputs/data/artifacts/features.parquet`: **特征总表**。这是整个工作流中最重要的中间产物，它将原始的时序信号转换为了机器学习模型可以直接“阅读”的数值表格，是模型训练的直接输入。

### **阶段三：源域故障诊断（基线模型）**

此阶段我们利用提取出的特征，在源域数据上训练一个强大的基线模型（XGBoost），作为后续迁移学习的对比基础。

  * **执行脚本**: `src/models/clf/XGBoost/build.py`

      * **作用**:
        1.  **数据准备**: 加载特征表，筛选出源域数据，并进行标签编码。
        2.  **防泄漏划分**: 采用 `StratifiedGroupKFold` 策略，确保来自同一原始文件的窗口不会同时出现在训练集和验证集，严格防止数据泄漏。
        3.  **模型训练与评估**: 训练 XGBoost 分类器，并在验证集上计算准确率、F1分数等指标。

  * **配置文件**: `src/models/clf/XGBoost/params.yaml`

      * **作用**: 配置模型训练的方方面面。
      * **关键参数**:
          * `dataset.path`: 输入的 `features.parquet` 特征表路径。
          * `dataset.target`: 指定 `fault_type` 为需要预测的目标标签。
          * `split.n_splits`: 设置交叉验证的折数，对应 `build.py` 中的 `StratifiedGroupKFold`。
          * `model.params`: XGBoost 模型的超参数，对应方案中的 **“XGBoost对照基线”**。

  * **生成文件及其作用**:

      * `outputs/models/source_xgb_baseline.pkl`: **训练好的模型文件**。这是本阶段的核心产出，一个封装了诊断知识的二进制文件，可用于后续对新数据进行预测。
      * `outputs/predictions/.../source_xgb_baseline_preds.csv`: **验证集预测结果**。用于详细分析模型在哪些样本上表现好、哪些表现差。
      * `outputs/reports/source_xgb_baseline_metrics.json`: **评估指标报告**。以JSON格式保存模型的性能指标，便于程序化读取和比较。
      * `outputs/figs/clf/source_xgb_baseline_cm.png`: **混淆矩阵图**。直观地展示模型在不同故障类别上的分类准确情况，是评估模型性能的重要可视化工具。

## 三、如何运行

请严格按照以下顺序执行命令，即可完成从数据处理到模型训练的全过程。

**前置准备：**

1.  确保您的原始数据集已按任意层级结构放入 `data/source_domain` 和 `data/target_domain` 文件夹。
2.  确保 `src/preprocessing/signal/steps_load_data.yaml` 中的 `source_dir` 和 `target_dir` 路径已正确配置。
3.  **（如果之前运行过）请先手动删除 `outputs` 文件夹，以确保结果不被缓存影响。**

**执行命令：**

```bash
# 1. 运行数据加载与结构化
python -m src.pipelines.preprocess_pipeline --config src/preprocessing/signal/steps_load_data.yaml

# 2. 运行信号处理与特征提取
python -m src.pipelines.preprocess_pipeline --config src/preprocessing/signal/steps_feature_extraction.yaml

# 3. 运行源域模型训练
python -m src.pipelines.clf_pipeline --config src/models/clf/XGBoost/params.yaml
```

执行完毕后，所有产出物将自动生成在 `outputs` 文件夹的对应子目录中。

```
```