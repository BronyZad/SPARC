#!/bin/bash
echo "================================================="
echo " 开启 LongBench-E 裁判打分程序 (精简 11 项)"
echo "================================================="

CODE_DIR="/home/yuan/saber/code"
LOG_FILE="/home/yuan/saber/code/longbench_judge.log"

# 🎯 对应你刚刚生成的 11 份成绩单
CHECKPOINTS=(
  "saber_longbench_r0.1_2wikimqa_e.jsonl"
  "saber_longbench_r0.1_gov_report_e.jsonl"
  "saber_longbench_r0.1_hotpotqa_e.jsonl"
  "saber_longbench_r0.1_multi_news_e.jsonl"
  "saber_longbench_r0.1_multifieldqa_en_e.jsonl"
  "saber_longbench_r0.1_passage_count_e.jsonl"
  "saber_longbench_r0.1_passage_retrieval_en_e.jsonl"
  "saber_longbench_r0.1_qasper_e.jsonl"
  "saber_longbench_r0.1_samsum_e.jsonl"
  "saber_longbench_r0.1_trec_e.jsonl"
  "saber_longbench_r0.1_triviaqa_e.jsonl"
)

echo -e "\n\n=================================================" >> $LOG_FILE
echo "  LongBench-E 阅卷开始！当前时间：$(date)   " >> $LOG_FILE
echo "=================================================" >> $LOG_FILE

for ckpt in "${CHECKPOINTS[@]}"
do
   FILE_PATH="$CODE_DIR/$ckpt"
   
   echo "-------------------------------------------------"
   echo "正在阅卷: $ckpt"
   echo "-------------------------------------------------"
   
   if [ -f "$FILE_PATH" ]; then
       # 调用你的裁判脚本
       python3 -u $CODE_DIR/longbench_judge.py --file_path $FILE_PATH >> $LOG_FILE 2>&1
   else
       echo "⚠️ 成绩单未找到: $ckpt ，跳过此科目。请确认做题脚本是否已跑完该项。" >> $LOG_FILE
   fi
done

echo "================================================="
echo " LongBench-E 精简版成绩单全部阅卷完毕！"
echo "================================================="
