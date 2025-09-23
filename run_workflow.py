# run_workflow.py
import subprocess
import sys
import os
import shutil

def run_command(command, description):
    """一个辅助函数，用于执行命令行命令并打印清晰的日志信息"""
    print("=" * 80)
    print(f"INFO: Running Step: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print("=" * 80)
    try:
        # 使用 subprocess.run 来执行命令，并设置 check=True
        # check=True 意味着如果命令返回非零退出码（即出错），程序将抛出异常并停止
        subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"SUCCESS: Step '{description}' completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Step '{description}' failed with exit code {e.returncode}.")
        print("Please check the error message above and resolve the issue before running again.")
        # 一旦任何步骤失败，立即退出脚本
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Command '{command[0]}' not found.")
        print("Please ensure Python is correctly installed and in your system's PATH.")
        sys.exit(1)


def main():
    """主函数，按顺序调度和执行整个工作流"""

    # --- 前置准备：清理旧的输出（可选） ---
    # 为了确保每次运行都是一个全新的开始，避免旧文件干扰，我们可以选择性地删除 outputs 文件夹
    if os.path.exists("outputs"):
        print("INFO: Found existing 'outputs' directory. Cleaning it up before starting...")
        try:
            shutil.rmtree("outputs")
            print("INFO: 'outputs' directory has been removed.")
        except OSError as e:
            print(f"ERROR: Could not remove 'outputs' directory: {e}")
            print("Please close any programs that might be using files in this directory and try again.")
            sys.exit(1)

    # 定义Python解释器的路径，确保我们使用的是执行此脚本的同一个Python环境
    python_executable = sys.executable

    # ===================================================================
    # 题目一：数据分析与故障特征提取
    # ===================================================================

    # 步骤 1-1: 加载和结构化原始数据
    # 作用：读取所有 .mat 文件，提取信号和元数据，生成统一的 manifest.csv 清单。
    run_command(
        [python_executable, "-m", "src.pipelines.preprocess_pipeline", "--config",
         "src/preprocessing/signal/steps_load_data.yaml"],
        "Question 1-1: Loading and Structuring Raw Data"
    )

    # 步骤 1-2: 提取用于源域模型训练的混合特征 (24kHz)
    # 作用：基于24kHz采样率，提取时域、频域、包络谱和CWT特征，为训练一个强大的源域模型做准备。
    #      这个模型及其标签编码器是后续迁移学习的基础。
    run_command(
        [python_executable, "-m", "src.pipelines.preprocess_pipeline", "--config",
         "src/preprocessing/signal/steps_feature_extraction.yaml"],
        "Question 1-2: Extracting Mixed Features (24kHz) for Source Model"
    )

    # ===================================================================
    # 题目二：源域故障诊断
    # ===================================================================

    # 步骤 2-1: 训练源域诊断模型 (XGBoost)
    # 作用：在24kHz的混合特征上，使用严格的防泄漏划分方法，训练一个高性能的XGBoost分类器。
    #      核心产出物 source_xgb_baseline.pkl 和 source_xgb_baseline_label_encoder.pkl 是题目三的关键输入。
    run_command(
        [python_executable, "-m", "src.pipelines.clf_pipeline", "--config", "src/models/clf/XGBoost/params.yaml"],
        "Question 2-1: Training Source Domain XGBoost Model"
    )

    # ===================================================================
    # 题目三：迁移诊断 (最终方案：基于原始信号的DANN+MHDCNN)
    # ===================================================================
    # 根据我们的探索历程，直接在原始信号上使用先进的MHDCNN进行端到端迁移，效果优于在手工特征上进行迁移。

    # 步骤 3-1: 执行DANN+MHDCNN迁移，并对目标域进行标定
    # 作用：这是题目三的核心解答。该流水线会：
    #      1. 自动加载原始信号 (基于manifest.csv)。
    #      2. 使用内置的MHDCNN作为特征提取器，DANN进行领域对齐。
    #      3. 训练端到端的迁移模型。
    #      4. 预测目标域标签，并生成最终的可视化t-SNE图。
    run_command(
        [python_executable, "-m", "src.pipelines.transfer_dann_raw_signal_pipeline", "--config", "runs/transfer_dann_mhdcnn_raw.yaml"],
        "Question 3-1: Performing Transfer Diagnosis with DANN+MHDCNN on Raw Signals"
    )

    # ===================================================================
    # 题目四：迁移诊断的可解释性
    # ===================================================================
    # 注意：由于SHAP的DeepExplainer是为DANN这类深度模型设计的，我们需要一个独立的解释性流水线。
    # 我们将对题目二训练的、基于明确物理特征的XGBoost模型进行事后可解释性分析，
    # 这样可以将模型的决策依据与我们提取的物理特征直接关联起来，更具说服力。

    # 步骤 4-1: 生成SHAP图，解释源域模型的决策过程
    # 作用：加载在题目二中训练好的XGBoost模型和24kHz特征集，
    #      应用SHAP框架来分析模型在诊断目标域数据时，主要依赖哪些物理特征（如包络谱峰值、峭度等）。
    #      这直接回答了“模型决策过程”的可解释性问题。
    run_command(
        [python_executable, "-m", "src.pipelines.interpretability_pipeline", "--config", "runs/interpret_tca_shap.yaml"],
        "Question 4-1: Generating SHAP Plots for Post-hoc Interpretability Analysis"
    )

    print("=" * 80)
    print("ALL TASKS COMPLETED SUCCESSFULLY! ")
    print("Please check the 'outputs' directory for all generated results, models, and plots.")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting the entire diagnostic workflow...")
    main()