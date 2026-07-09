"""
Saber Unified LEval Grader
Routes benchmark outputs to domain-specific evaluation metrics.
Fixes: Standalone article/option protection for multiple-choice datasets (QuALITY).
"""
import json
import re
import string
import collections
import numpy as np
from rouge_score import rouge_scorer
import argparse

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        # 🟢 FIX: Do not strip 'a' if it constitutes the entire multiple-choice answer
        if text.strip() == 'a': 
            return text
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """Strict evaluation for Math, Exam, and Topic Retrieval."""
    norm_gt = normalize_answer(ground_truth)
    norm_pred = normalize_answer(prediction)
    
    if not norm_gt: 
        return 0.0
    
    # 🟢 FIX: For single-character responses, check true token boundaries 
    # to avoid false positives (e.g., matching 'c' in 'conclusive')
    if len(norm_gt) == 1:
        return 1.0 if norm_gt in norm_pred.split() else 0.0
        
    # Default substring matching for standard retrieval/math entries
    return 1.0 if norm_gt in norm_pred else 0.0

def f1_score(prediction, ground_truth):
    """Token-level overlap for QA tasks."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class UnifiedSaberGrader:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.methods = ["Native-Baseline", "Uniform-INT4", "Saber-Q", "Saber-CQ"]

    def evaluate_file(self, filepath, task_type="summarization"):
        print(f"📖 Evaluating {filepath} [Mode: {task_type.upper()}]...")
        
        metrics = {m: [] for m in self.methods}
        count = 0
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                reference = record.get("reference", "")
                if not reference: continue
                
                for method in self.methods:
                    if method in record["results"] and record["results"][method]["success"]:
                        pred = record["results"][method].get("generated_text", "")
                        
                        # 🟢 ROUTER LOGIC
                        if task_type == "summarization":
                            score = self.rouge.score(reference, pred)['rougeL'].fmeasure * 100
                        elif task_type in ["retrieval", "exam"]:
                            score = exact_match_score(pred, reference) * 100
                        elif task_type == "qa":
                            score = f1_score(pred, reference) * 100
                            
                        metrics[method].append(score)
                count += 1

        print(f"\n📊 RESULTS: {filepath.split('/')[-1]} (Evaluated {count} docs)")
        print("=" * 60)
        
        # Determine metric name for pretty-printing
        if task_type == "summarization":
            metric_name = "ROUGE-L"
        elif task_type in ["retrieval", "exam"]:
            metric_name = "Exact Match"
        else:
            metric_name = "F1-Score"
            
        print(f"{'Method':<18} | {metric_name:<10}")
        print("-" * 60)
        
        for method in self.methods:
            avg_score = np.mean(metrics[method]) if metrics[method] else 0
            print(f"{method:<18} | {avg_score:>7.2f}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--task', type=str, choices=['summarization', 'retrieval', 'qa', 'exam'], default='summarization')
    args = parser.parse_args()
    
    grader = UnifiedSaberGrader()
    grader.evaluate_file(args.file, args.task)