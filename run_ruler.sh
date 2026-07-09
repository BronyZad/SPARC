#!/bin/bash

LOG_FILE="/home/yuan/saber/code/ruler_hard.log"
CODE_DIR="/home/yuan/saber/code"
# 🎯 修正了 RULER 数据集的真实根目录
DATA_DIR="/home/yuan/saber/data/benchmarks/RULER"

DATASETS=(
  "ruler_cwe.jsonl"
  "ruler_fwe.jsonl"
  "ruler_niah_multiquery.jsonl"
  "ruler_niah_multivalue.jsonl"
  "ruler_qa_1.jsonl"
  "ruler_qa_2.jsonl"
  "ruler_test_8k.jsonl"
  "ruler_test_16k.jsonl"
  "ruler_vt.jsonl"
)

# 全部使用 >> 追加，保护历史数据！
echo -e "\n\n=================================================" >> $LOG_FILE
echo " 🚀 开启 RULER Hard Benchmark 评测！当前时间：$(date) " >> $LOG_FILE
echo "=================================================" >> $LOG_FILE

for ds in "${DATASETS[@]}"
do
    # 动态拼接成完整的绝对路径
    FULL_PATH="$DATA_DIR/$ds"
    
    echo "-------------------------------------------------" >> $LOG_FILE
    echo "▶️ 正在评测 RULER 数据集: $ds" >> $LOG_FILE
    echo "   文件路径: $FULL_PATH" >> $LOG_FILE
    echo "-------------------------------------------------" >> $LOG_FILE
    
    # 调用你的评测脚本，带上完整路径，并且绝对使用 >> 追加日志
    python3 -u $CODE_DIR/benchmark_ruler_hard.py --dataset $FULL_PATH >> $LOG_FILE 2>&1
    
done

echo "=================================================" >> $LOG_FILE
echo " 🎉 RULER Hard Benchmark 全部评测完毕！" >> $LOG_FILE
echo "=================================================" >> $LOG_FILE