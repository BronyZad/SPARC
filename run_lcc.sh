#!/bin/bash

# ================= 配置区 =================
# 遍历的 retain_ratio 数组
RATIOS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

# 数据集绝对路径 (带上了 _e)
DATASET="/home/yuan/saber/data/benchmarks/longbench/lcc_e.jsonl"

# 定义结果和日志存放的文件夹
RESULTS_DIR="lcc_results"
LOG_DIR="lcc_logs"

# 如果文件夹不存在则创建它们
mkdir -p $RESULTS_DIR
mkdir -p $LOG_DIR
# ==========================================

echo "🚀 开始执行 LCC 自动化实验 (0.01 -> 0.10)..."
echo "📂 结果将保存在 $RESULTS_DIR/，日志在 $LOG_DIR/"
echo "--------------------------------------------------"

for ratio in "${RATIOS[@]}"; do
    echo "▶️ [$(date +'%T')] 正在运行 retain_ratio = $ratio"
    
    # 1. 运行 Python 脚本，带上 retain_ratio 和 dataset 参数
    python3 -u benchmark_lcc.py \
        --retain_ratio $ratio \
        --dataset $DATASET \
        > "$LOG_DIR/lcc_r${ratio}.log" 2>&1
        
    # 2. 检查命令是否成功运行
    if [ $? -eq 0 ]; then
        # 根据你 python 代码的逻辑拼接出生成的文件名
        # 因为 dataset 是 lcc_e.jsonl，所以 task_name 是 lcc_e
        EXPECTED_FILE="saber_checkpoint_r${ratio}_lcc_e.jsonl"
        
        # 将生成的 jsonl 文件移动到结果文件夹
        if [ -f "$EXPECTED_FILE" ]; then
            mv "$EXPECTED_FILE" "$RESULTS_DIR/"
            echo "✅ 成功！结果已保存至 $RESULTS_DIR/$EXPECTED_FILE"
        else
            echo "⚠️ 运行结束，但没有找到文件 $EXPECTED_FILE，请检查日志！"
        fi
    else
        echo "❌ 运行崩溃，retain_ratio = $ratio 失败！请查看 $LOG_DIR/lcc_r${ratio}.log"
        # 遇到致命错误直接停止后续循环
        exit 1 
    fi
    echo "--------------------------------------------------"
done

echo "🎉 所有 10 组 LCC 实验已全部执行完毕！"