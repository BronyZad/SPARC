import json
import glob
import os
from collections import defaultdict

def audit_grading_stability():
    graded_files = glob.glob("graded_*.jsonl")
    if not graded_files:
        print("❌ No graded_*.jsonl files found in the current directory.")
        return

    print("🔍 AUDITING GLOBAL GRADING STABILITY...")
    print("Looking for score spreads >= 5.0 (e.g., model swinging between 0 and 5, or 5 and 10)\n")

    # Metrics trackers
    dataset_stats = {}
    
    for filepath in graded_files:
        dataset_name = filepath.replace("graded_saber_checkpoint_r0.1_", "").replace(".jsonl", "")
        
        total_evals = 0
        divergent_evals = 0
        extreme_divergent_evals = 0 # Swings of 8.0+ (e.g., 0 to 10)
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                scores_dict = record.get("llm_judge_scores", {})
                for method, data in scores_dict.items():
                    # Check if it was fast-pathed or if LLM multi-pass was used
                    sampled_scores = data.get("all_sampled_scores", [])
                    
                    if len(sampled_scores) > 1:
                        total_evals += 1
                        spread = max(sampled_scores) - min(sampled_scores)
                        
                        if spread >= 5.0:
                            divergent_evals += 1
                        if spread >= 8.0:
                            extreme_divergent_evals += 1

        if total_evals > 0:
            dataset_stats[dataset_name] = {
                "total": total_evals,
                "divergent": divergent_evals,
                "extreme": extreme_divergent_evals,
                "instability_rate": (divergent_evals / total_evals) * 100
            }

    # 📊 Print Report Table
    print("=" * 85)
    print(f"{'DATASET':<25} | {'TOTAL LLM PASSES':<18} | {'DIVERGENT (≥5 pt swing)':<25} | {'EXTREME (≥8 pt)':<15}")
    print("-" * 85)
    
    # Sort by highest instability rate
    sorted_stats = sorted(dataset_stats.items(), key=lambda x: x[1]['instability_rate'], reverse=True)
    
    for ds_name, stats in sorted_stats:
        rate = stats['instability_rate']
        div = stats['divergent']
        ext = stats['extreme']
        tot = stats['total']
        
        # Color coding for terminal
        if rate > 20:
            status = "🚨 POOR"
        elif rate > 5:
            status = "⚠️ WARN"
        else:
            status = "✅ STABLE"
            
        print(f"{ds_name:<25} | {tot:<18} | {div:<5} ({rate:>5.1f}%) {status:<8} | {ext:<15}")
        
    print("=" * 85)
    print("\n💡 NOTE: Datasets with 0 'TOTAL LLM PASSES' were successfully resolved 100% via the Regex Fast-Path (No LLM needed).")

if __name__ == "__main__":
    audit_grading_stability()