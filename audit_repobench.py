"""
RepoBench Comprehensive Dataset Auditor
Features: Schema Extraction, Token Length Percentiles (Qwen2.5), and Split Mix Detection.
"""
import json
import numpy as np
import os
from collections import defaultdict
from transformers import AutoTokenizer

def audit_repobench(dataset_path, model_id="Qwen/Qwen2.5-Coder-7B-Instruct"):
    print(f"🔍 Starting Comprehensive Audit for: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ File not found: {dataset_path}")
        return

    print(f"⏳ Loading Tokenizer ({model_id}) to calculate exact token lengths...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"⚠️ Could not load tokenizer. Falling back to approximate word count. Error: {e}")
        tokenizer = None

    # Trackers
    schema_types = defaultdict(set)
    token_lengths = []
    snippet_counts = []
    total_lines = 0

    print("📊 Processing dataset (this may take a minute for 10k+ rows)...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            
            data = json.loads(line)
            total_lines += 1
            
            # 1. Schema Extraction
            for key, value in data.items():
                schema_types[key].add(type(value).__name__)
                
            # 2. Extract Text for Analysis
            context = data.get('context', '')
            question = data.get('prompt', data.get('question', ''))
            full_text = f"{context}\n{question}"
            
            # 3. Token Length Calculation
            if tokenizer:
                # Fast tokenization without creating tensors
                tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                token_lengths.append(len(tokens))
            else:
                # Fallback approximation (words * 1.3 usually roughs out code tokens)
                token_lengths.append(int(len(full_text.split()) * 1.3))

            # 4. Fragmentation Analysis (Detecting _first vs _random)
            # RepoBench usually uses "# Path:" or "// Path:" to split cross-file snippets
            delimiters = full_text.count("# Path:") + full_text.count("// Path:") + full_text.count("# ---")
            snippet_counts.append(delimiters)
            
            if (i + 1) % 2500 == 0:
                print(f"   ... processed {i + 1} rows")

    # ==========================================
    # REPORT GENERATION
    # ==========================================
    print("\n" + "="*60)
    print("📈 REPOBENCH AUDIT REPORT")
    print("="*60)
    print(f"Total Valid Samples: {total_lines:,}")

    # --- SCHEMA ---
    print("\n📄 JSON SCHEMA:")
    for key, types in schema_types.items():
        type_str = " | ".join(types)
        print(f"  └─ '{key}': {type_str}")

    # --- TOKEN DISTRIBUTION ---
    lengths = np.array(token_lengths)
    print("\n📏 TOKEN LENGTH DISTRIBUTION (Context + Prompt):")
    print(f"  └─ Minimum:    {np.min(lengths):,}")
    print(f"  └─ Median:     {np.median(lengths):,.0f}")
    print(f"  └─ Mean:       {np.mean(lengths):,.0f}")
    print(f"  └─ 90th Pctl:  {np.percentile(lengths, 90):,.0f}")
    print(f"  └─ 95th Pctl:  {np.percentile(lengths, 95):,.0f}")
    print(f"  └─ 99th Pctl:  {np.percentile(lengths, 99):,.0f}")
    print(f"  └─ Maximum:    {np.max(lengths):,}")
    
    if np.max(lengths) > 16000:
        print("  ⚠️ WARNING: You have samples exceeding 16k tokens. Ensure your max_seq_len handles this.")

    # --- SPLIT MIX DETECTION ---
    print("\n🧩 DATASET SPLIT ANALYSIS:")
    snippets = np.array(snippet_counts)
    
    # Categorize fragmentation
    cohesive_ratio = np.sum(snippets <= 1) / total_lines
    fragmented_ratio = np.sum(snippets >= 3) / total_lines
    
    print(f"  └─ Cohesive Samples (0-1 cross-files): {cohesive_ratio*100:.1f}%")
    print(f"  └─ Fragmented Samples (3+ cross-files): {fragmented_ratio*100:.1f}%")
    print(f"  └─ Average Snippets per prompt: {np.mean(snippets):.2f}")

    print("\n💡 VERDICT:")
    if cohesive_ratio > 0.20 and fragmented_ratio > 0.20:
        print("  🟢 CONFIRMED MIX: This dataset is highly likely a combination of 'cross_file_first' and 'cross_file_random'.")
        print("     It exhibits a bimodal distribution with a healthy mix of both contiguous and highly fragmented contexts.")
    elif cohesive_ratio > 0.75:
        print("  🔵 LIKELY CROSS_FILE_FIRST: The vast majority of the context windows are cohesive and contiguous.")
    elif fragmented_ratio > 0.75:
        print("  🟠 LIKELY CROSS_FILE_RANDOM: The vast majority of the context windows are stitched together from multiple random snippets.")
    else:
        print("  ⚪ UNCLEAR: The dataset has a moderate level of fragmentation that doesn't strongly lean toward a pure 'first' or 'random' split.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    DATASET_PATH = "../data/benchmarks/repobeach_python/repobench_cross_file.jsonl"
    audit_repobench(DATASET_PATH)