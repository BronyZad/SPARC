#!/bin/bash
echo "================================================="
echo "开启 LongBench-E (精简 11 项，无代码任务) 连测..."
echo "================================================="

CODE_DIR="/home/yuan/saber/code"
DATA_DIR="/home/yuan/saber/data/benchmarks/longbench"
LOG_FILE="/home/yuan/saber/code/longbench_run.log"

# 🎯 严选：踢除了 lcc_e 和 repobench-p_e，剩下 11 个任务
DATASETS=(
  "2wikimqa_e.jsonl"
  "gov_report_e.jsonl"
  "hotpotqa_e.jsonl"
  "multi_news_e.jsonl"
  "multifieldqa_en_e.jsonl"
  "passage_count_e.jsonl"
  "passage_retrieval_en_e.jsonl"
  "qasper_e.jsonl"
  "samsum_e.jsonl"
  "trec_e.jsonl"
  "triviaqa_e.jsonl"
)

# 💡 写入时间分割线
echo -e "\n\n=================================================" >> $LOG_FILE
echo "   新的 LongBench-E 11 项精简版连测开始！当前时间：$(date)   " >> $LOG_FILE
echo "=================================================" >> $LOG_FILE

for ds in "${DATASETS[@]}"
do
   echo "-------------------------------------------------"
   echo "正在发车: $ds"
   echo "-------------------------------------------------"
   
   # 调用 LongBench 专属做题脚本，并将结果追加到 log 中
   python -u $CODE_DIR/benchmark_longbench.py --dataset $DATA_DIR/$ds >> $LOG_FILE 2>&1
done

echo "================================================="
echo "LongBench-E 11张卷子全部发卷完毕"
echo "================================================="