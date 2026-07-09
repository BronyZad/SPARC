import re
import statistics
import os
from collections import defaultdict

def analyze_regrade_log(log_path="regrade.log"):
    if not os.path.exists(log_path):
        print(f"❌ Error: Could not find '{log_path}' in the current directory.")
        return

    # Store flat records for easy grouping and filtering
    # Structure: {'task': str, 'ratio': float, 'method': str, 'score': float}
    records = []
    
    current_task = None
    current_ratio = None
    current_method = None
    
    print(f"📖 Parsing {log_path} for granular trends...\n")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Detect dataset switch and extract Task Type & Retain Ratio
            # Looks for: "📖 Loading results from saber_checkpoint_r0.15_financial_qa.jsonl..."
            dataset_match = re.search(r'Loading results from (.*?\.jsonl)', line)
            if not dataset_match:
                dataset_match = re.search(r'Evaluation of \d+ documents for (.*?\.jsonl)', line)
                
            if dataset_match:
                filename = dataset_match.group(1)
                # Regex logic: optionally match "_r[NUM]_", then capture the rest as task name
                m = re.search(r'saber_checkpoint_(?:r([\d\.]+)_)?(.*)\.jsonl', filename)
                if m:
                    ratio_str = m.group(1)
                    # If no ratio string is found, default to 0.1
                    current_ratio = float(ratio_str) if ratio_str else 0.10
                    current_task = m.group(2)
                continue

            # 2. Detect the current method being evaluated
            method_match = re.search(r'└─\s+([A-Za-z0-9\-]+)\s+\|\s+🔄 Sampling', line)
            if method_match:
                current_method = method_match.group(1).strip()
                continue
                
            # 3. Detect the final Monte Carlo average score for a document
            final_match = re.search(r'⭐ FINAL AGGREGATED AVERAGE SCORE:\s+([\d\.]+)', line)
            if final_match and current_method and (current_task is not None) and (current_ratio is not None):
                final_score = float(final_match.group(1))
                records.append({
                    'task': current_task,
                    'ratio': current_ratio,
                    'method': current_method,
                    'score': final_score
                })
                
    if not records:
        print("⚠️ No complete evaluation records found. Ensure the log format matches expectations.")
        return

    # Extract unique values for our table axes
    # Enforce a logical column order for methods
    all_methods = ["Native-Baseline", "Uniform-INT4", "Saber-Q", "Saber-CQ"] 
    found_methods = set(r['method'] for r in records)
    methods = [m for m in all_methods if m in found_methods]
    if not methods:
        methods = sorted(list(found_methods)) # Fallback if names changed
        
    ratios = sorted(list(set(r['ratio'] for r in records)))
    tasks = sorted(list(set(r['task'] for r in records)))

    def get_avg(filtered_records):
        if not filtered_records: return None
        return statistics.mean(r['score'] for r in filtered_records)

    # ========================================================================
    # SECTION 1: MACRO TREND BY RETAIN RATIO
    # ========================================================================
    print("="*85)
    print("📈 MACRO TREND: AVERAGE SCORE BY RETAIN RATIO (Across All Tasks)")
    print("="*85)
    
    header = f"{'Retain Ratio':<15}"
    for method in methods:
        header += f"| {method:<15}"
    print(header)
    print("-" * 85)
    
    for ratio in ratios:
        row_str = f"{ratio:<15.2f}"
        for method in methods:
            recs = [r for r in records if r['ratio'] == ratio and r['method'] == method]
            avg = get_avg(recs)
            row_str += f"| {avg:<15.2f}" if avg is not None else f"| {'N/A':<15}"
        print(row_str)

    # ========================================================================
    # SECTION 2: TREND BY TASK TYPE
    # ========================================================================
    print("\n" + "="*85)
    print("📂 GRANULAR TRENDS: BY TASK TYPE")
    print("="*85)
    
    for task in tasks:
        print(f"\n📌 TASK: {task}")
        print("-" * 85)
        print(header)
        print("-" * 85)
        
        for ratio in ratios:
            row_str = f"{ratio:<15.2f}"
            for method in methods:
                recs = [r for r in records if r['task'] == task and r['ratio'] == ratio and r['method'] == method]
                avg = get_avg(recs)
                row_str += f"| {avg:<15.2f}" if avg is not None else f"| {'N/A':<15}"
            print(row_str)

    # ========================================================================
    # SECTION 3: TREND BY METHOD
    # ========================================================================
    print("\n" + "="*85)
    print("🔬 GRANULAR TRENDS: BY METHOD (How compression affects each method)")
    print("="*85)
    
    for method in methods:
        # Native-Baseline doesn't undergo compression, so its scores should theoretically
        # remain flat regardless of the ratio row. We print it anyway as a sanity check.
        print(f"\n🛠️ METHOD: {method}")
        print("-" * 85)
        
        m_header = f"{'Retain Ratio':<15}"
        for task in tasks:
            display_task = task[:15] + ".." if len(task) > 17 else task
            m_header += f"| {display_task:<17}"
        m_header += f"| {'Overall Avg':<12}"
        print(m_header)
        print("-" * 85)
        
        for ratio in ratios:
            row_str = f"{ratio:<15.2f}"
            ratio_recs = []
            
            for task in tasks:
                recs = [r for r in records if r['method'] == method and r['ratio'] == ratio and r['task'] == task]
                avg = get_avg(recs)
                ratio_recs.extend(recs)
                row_str += f"| {avg:<17.2f}" if avg is not None else f"| {'N/A':<17}"
            
            overall_avg = get_avg(ratio_recs)
            row_str += f"| {overall_avg:<12.2f}" if overall_avg is not None else f"| {'N/A':<12}"
            print(row_str)
            
    print("\n" + "="*85)

if __name__ == '__main__':
    analyze_regrade_log()