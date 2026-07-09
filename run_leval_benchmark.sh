#!/bin/bash

# 🟢 CONFIGURATION BLOCK: Set your global suite properties here
RETAIN_RATIO="0.45"  # Change to 0.10, 0.20, etc. for sweeps
LOG_FILE="leval_suite_r${RETAIN_RATIO}_$(date +%Y%m%d_%H%M%S).log"

# 🟢 The complete suite of verified datasets
datasets=(
    "Exam/topic_retrieval_longchat.jsonl"
    "Exam/quality.jsonl"
    "Generation/gov_report_summ.jsonl"
    "Generation/meeting_summ.jsonl"
    "Generation/multidoc_qa.jsonl"
    "Generation/legal_contract_qa.jsonl"
    "Generation/scientific_qa.jsonl"
    "Generation/financial_qa.jsonl"
)

echo "🚀 Starting Unified LEval Benchmark Suite [Target Ratio: ${RETAIN_RATIO}]. Logging to $LOG_FILE"

{
    for ds in "${datasets[@]}"; do
        echo "========================================================================"
        echo "TIMESTAMP: $(date)"
        echo "DATASET: $ds"
        echo "RETAIN RATIO: $RETAIN_RATIO"
        echo "========================================================================"
        
        # 1. RUN BENCHMARK GENERATION (Passing the ratio config)
        echo "▶️ Phase 1: Generating outputs..."
        PYTORCH_ALLOC_CONF=expandable_segments:True python -u benchmark_prefill_leval.py \
            --dataset "$ds" \
            --retain_ratio "$RETAIN_RATIO"
        
        if [ $? -ne 0 ]; then
            echo "❌ Benchmark generation failed for $ds. Aborting suite."
            exit 1
        fi
        
        # 2. MATCH THE UNIQUELY IDENTIFIED FILE PATTERN
        ckpt_file="saber_checkpoint_r${RETAIN_RATIO}_${ds##*/}"
        
        # Default task assumptions
        task="summarization"
        grader_type="rule_based"
        
        # Override task type routing
        if [[ "$ds" == *"retrieval"* ]]; then 
            task="retrieval"
            grader_type="rule_based"
        elif [[ "$ds" == *"quality"* ]]; then 
            task="exam"
            grader_type="rule_based"
        elif [[ "$ds" == *"_qa"* ]]; then 
            task="qa"
            grader_type="llm_judge"
        fi
        
        echo "------------------------------------------------------------------------"
        echo "▶️ Phase 2: Grading results for $ds"
        echo "File: $ckpt_file"
        echo "Task Mode: $task | Grader Route: $grader_type"
        echo "------------------------------------------------------------------------"
        
        # 3. EXECUTE EVALUATION
        if [[ "$grader_type" == "llm_judge" ]]; then
            # Triggers Qwen3-8B with Auto-Retry Logic
            python -u leval_judge.py --files "$ckpt_file"
        else
            # Triggers ROUGE-L or Strict Multi-Choice EM
            python -u eval_leval.py --file "$ckpt_file" --task "$task"
        fi
        
        if [ $? -ne 0 ]; then
            echo "❌ Evaluation failed for $ds. Aborting suite."
            exit 1
        fi
        
        echo -e "\n✅ Completed $ds at ratio $RETAIN_RATIO\n"
    done
    
    echo "🎉 Total pipeline run finished for ratio $RETAIN_RATIO!"
} 2>&1 | tee -a "$LOG_FILE"
