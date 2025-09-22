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
    # 任务 A：运行主方案 (回答题目 1, 2, 3, 4)
    # ===================================================================

    # A-1. 加载和结构化原始数据
    run_command(
        [python_executable, "-m", "src.pipelines.preprocess_pipeline", "--config",
         "src/preprocessing/signal/steps_load_data.yaml"],
        "A-1: Loading and Structuring Raw Data"
    )

    # A-2. 提取混合特征 (包含CWT)
    run_command(
        [python_executable, "-m", "src.pipelines.preprocess_pipeline", "--config",
         "src/preprocessing/signal/steps_feature_extraction.yaml"],
        "A-2: Extracting Mixed Features for Main Solution"
    )

    # A-3. 训练源域诊断模型 (XGBoost on Mixed Features)
    run_command(
        [python_executable, "-m", "src.pipelines.clf_pipeline", "--config", "src/models/clf/XGBoost/params.yaml"],
        "A-3: Training Source Domain XGBoost Model"
    )

    # A-4. 执行TCA迁移并预测目标域标签
    run_command(
        [python_executable, "-m", "src.pipelines.transfer_pipeline", "--config", "runs/transfer_tca.yaml"],
        "A-4: Performing Transfer Diagnosis with TCA"
    )

    # A-5. 生成SHAP图，解释模型决策
    run_command(
        [python_executable, "-m", "src.pipelines.interpretability_pipeline", "--config", "runs/interpret_shap.yaml"],
        "A-5: Generating SHAP Interpretability Plots"
    )

    # ===================================================================
    # 任务 B：运行对比实验方案
    # ===================================================================

    # B-1. 提取分离式特征
    run_command(
        [python_executable, "-m", "src.pipelines.preprocess_pipeline", "--config",
         "src/preprocessing/signal/steps_feature_extraction_separated.yaml"],
        "B-1: Extracting Separated Features for Comparison"
    )

    # B-2. 运行投票集成对比流水线 (会自动训练三个独立SVM模型并评估)
    run_command(
        [python_executable, "-m", "src.pipelines.ensemble_pipeline", "--config", "runs/run_ensemble_comparison.yaml"],
        "B-2: Running Ensemble Voting Comparison Pipeline"
    )

    print("=" * 80)
    print("🎉 ALL TASKS COMPLETED SUCCESSFULLY! 🎉")
    print("Please check the 'outputs' directory for all generated results, models, and plots.")
    print("=" * 80)


if __name__ == "__main__":
    # 确保在运行前，所有必要的配置文件路径都已正确设置
    print("Starting the entire diagnostic workflow...")
    main()