#!/bin/bash

# 1. 定义并创建专门的日志文件夹 (如果不存在会自动创建)
LOG_DIR="/home/yuan/saber/code/lcc_logs"
mkdir -p "$LOG_DIR"

# 定义你要跑的 10 个 retain_ratio
ratios=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

echo "🚀 开始批量执行 LCC 实验 (Retain Ratio: 0.01 -> 0.1)..."

for r in "${ratios[@]}"; do
    echo "====================================================="
    echo "▶️ 正在运行 retain_ratio = $r"
    
    # 2. 将每个比例的日志写到日志文件夹里
    LOG_FILE="$LOG_DIR/lcc_run_r${r}.log"
    
    # 执行 Python 脚本
    python -u /home/yuan/saber/code/benchmark_lcc.py --retain_ratio $r > "$LOG_FILE" 2>&1
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✅ retain_ratio = $r 运行完成！"
    else
        echo "❌ retain_ratio = $r 运行失败，请查看日志 $LOG_FILE"
    fi
done

echo "🎉 所有 LCC 比例测试全部完成！"