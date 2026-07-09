#!/bin/bash

# ================= 配置区 =================
# 遍历的 retain_ratio 数组
RATIOS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

# RepoBench 数据集的绝对路径 (请确保这个路径是正确的)
DATASET="/home/yuan/saber/data/benchmarks/repobeach_python/repobench_cross_file.jsonl"

# 定义结果和日志存放的文件夹
RESULTS_DIR="repo_results"
LOG_DIR="repo_logs"

mkdir -p $RESULTS_DIR
mkdir -p $LOG_DIR
# ==========================================

echo "🚀 开始执行 RepoBench 自动化实验 (0.01 -> 0.10)..."
echo "📂 结果将保存在 $RESULTS_DIR/，日志在 $LOG_DIR/"
echo "--------------------------------------------------"

for ratio in "${RATIOS[@]}"; do
    echo "▶️ [$(date +'%T')] 正在运行 retain_ratio = $ratio"
    
    # 假设你的 Python 脚本名字叫 benchmark_repobench_prefill.py (如果名字不一样请修改这里)
    python3 -u benchmark_repobench_prefill.py \
        --retain_ratio $ratio \
        --dataset $DATASET \
        > "$LOG_DIR/repo_r${ratio}.log" 2>&1
        
    # 检查命令是否成功运行
    if [ $? -eq 0 ]; then
        # 根据你 python 代码 (第84行) 的逻辑，输出文件名如下
        EXPECTED_FILE="saber_checkpoint_r${ratio}_repobench.jsonl"
        
        # 将生成的 jsonl 文件移动到结果文件夹
        if [ -f "$EXPECTED_FILE" ]; then
            mv "$EXPECTED_FILE" "$RESULTS_DIR/"
            echo "✅ 成功！结果已保存至 $RESULTS_DIR/$EXPECTED_FILE"
        else
            echo "⚠️ 运行结束，但没有找到文件 $EXPECTED_FILE，请检查日志！"
        fi
    else
        echo "❌ 运行崩溃，retain_ratio = $ratio 失败！请查看 $LOG_DIR/repo_r${ratio}.log"
        exit 1 
    fi
    echo "--------------------------------------------------"
done

echo "🎉 所有 10 组 RepoBench 实验已全部执行完毕！"