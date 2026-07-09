import json
import string
import re
import os
import sys
from collections import Counter

# ==========================================
# STRING NORMALIZATION & METRICS (官方原汁原味逻辑)
# ==========================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
        
    if s is None:
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0.0, 0.0, 0.0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def drqa_exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def substring_exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction) 

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif len(ground_truths) > 0 and isinstance(ground_truths[0], list):
        ground_truths = [gt for gt_list in ground_truths for gt in gt_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
        
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0

# ==========================================
# EVALUATION LOOP (适配 Saber Checkpoint 格式)
# ==========================================

def evaluate_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return
        
    if os.path.getsize(filepath) == 0:
        print(f"[{filepath}]\nSkipping: File is empty (0 bytes).\n")
        return

    # 用于按方法(Method)分别统计分数
    metrics_per_method = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: JSON decode error on line {line_num} of {filepath}")
                continue
                
            # 从咱们的格式里提取 Ground Truth
            ground_truths = data.get("ground_truth", [])
            if not ground_truths:
                continue

            # 遍历当前问题下所有的评测方法 (Baseline, Uniform, Saber, SnapKV 等)
            results = data.get("results", {})
            for method, method_data in results.items():
                if not method_data.get("success"):
                    continue
                    
                prediction = method_data.get("generated_text", "")
                
                if method not in metrics_per_method:
                    metrics_per_method[method] = {"em": 0.0, "sub_em": 0.0, "f1": 0.0, "count": 0}

                # 算分！
                em = metric_max_over_ground_truths(drqa_exact_match_score, prediction, ground_truths)
                sub_em = metric_max_over_ground_truths(substring_exact_match_score, prediction, ground_truths)
                f1 = metric_max_over_ground_truths(lambda p, g: f1_score(p, g)[0], prediction, ground_truths)

                metrics_per_method[method]["em"] += float(em)
                metrics_per_method[method]["sub_em"] += float(sub_em)
                metrics_per_method[method]["f1"] += float(f1)
                metrics_per_method[method]["count"] += 1

    if not metrics_per_method:
        print(f"[{filepath}]\nNo valid prediction/answer pairs found.\n")
        return

    # 打印超级华丽的对比表格
    print("\n" + "=" * 80)
    print(f"📊 HELMET OFFICIAL QA METRICS REPORT")
    print(f"📁 Source: {os.path.basename(filepath)}")
    print("=" * 80)
    print(f"{'Method':<20} | {'Samples':<7} | {'Exact Match':>12} | {'Substring EM':>12} | {'F1 Score':>10}")
    print("-" * 80)
    
    for method, scores in metrics_per_method.items():
        count = scores["count"]
        avg_em = (scores["em"] / count) * 100
        avg_sub_em = (scores["sub_em"] / count) * 100
        avg_f1 = (scores["f1"] / count) * 100
        print(f"{method:<20} | {count:<7} | {avg_em:>11.2f}% | {avg_sub_em:>11.2f}% | {avg_f1:>9.2f}%")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python official_judge.py <file1.jsonl> <file2.jsonl> ...")
        sys.exit(1)
        
    for file_path in sys.argv[1:]:
        evaluate_file(file_path)