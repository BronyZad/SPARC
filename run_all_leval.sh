#!/bin/bash
echo "================================================="
echo "  开启 LEval裁判打分程序..."
echo "================================================="

# 🟢 核心路径定义
CODE_DIR="/home/yuan/saber/code"
LOG_FILE="/home/yuan/saber/code/leval_run.log"

# 这里对应 19 个生成的成绩单文件名（根据你的 retain_ratio 0.10 自动匹配）
CHECKPOINTS=(
  "saber_checkpoint_r0.1_coursera.jsonl"
  "saber_checkpoint_r0.1_gsm100.jsonl"
  "saber_checkpoint_r0.1_quality.jsonl"
  "saber_checkpoint_r0.1_sci_fi.jsonl"
  "saber_checkpoint_r0.1_topic_retrieval_longchat.jsonl"
  "saber_checkpoint_r0.1_tpo.jsonl"
  "saber_checkpoint_r0.1_financial_qa.jsonl"
  "saber_checkpoint_r0.1_gov_report_summ.jsonl"
  "saber_checkpoint_r0.1_legal_contract_qa.jsonl"
  "saber_checkpoint_r0.1_meeting_summ.jsonl"
  "saber_checkpoint_r0.1_multidoc_qa.jsonl"
  "saber_checkpoint_r0.1_narrative_qa.jsonl"
  "saber_checkpoint_r0.1_natural_question.jsonl"
  "saber_checkpoint_r0.1_news_summ.jsonl"
  "saber_checkpoint_r0.1_paper_assistant.jsonl"
  "saber_checkpoint_r0.1_review_summ.jsonl"
  "saber_checkpoint_r0.1_scientific_qa.jsonl"
  "saber_checkpoint_r0.1_tv_show_summ.jsonl"
)

echo -e "\n\n=================================================" >> $LOG_FILE
echo "   LEval打分开始.当前时间：$(date)   " >> $LOG_FILE
echo "=================================================" >> $LOG_FILE

for ckpt in "${CHECKPOINTS[@]}"
do
   FILE_PATH="$CODE_DIR/$ckpt"
   
   echo "-------------------------------------------------"
   echo "正在阅卷: $ckpt"
   echo "-------------------------------------------------"
   
   # 检查该成绩单文件是否存在，存在才跑，不存在就跳过（防止前面有的任务还没跑完导致这里报错）
   if [ -f "$FILE_PATH" ]; then
       # 🟢 依然使用 >> 追加到你指定的 leval_run.log 账本里
       python3 $CODE_DIR/leval_judge.py --file_path $FILE_PATH >> $LOG_FILE 2>&1
   else
       echo "⚠️ 成绩单未找到: $ckpt ，跳过此科目。" >> $LOG_FILE
   fi
done

echo "================================================="
echo "Over"
echo "================================================="