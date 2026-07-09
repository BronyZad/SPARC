import re
import sys
from collections import defaultdict
import numpy as np

def parse_divergence_logs(filepath):
    stats = {
        "saber_u": {
            "fpod": [], "confidence_drop": [], "trigger_tokens": defaultdict(int), "premature_eof": 0,
            "telemetry": {"cosine_sim": [], "max_scale": [], "mean_scale": [], "iou": []}
        },
        "snapkv": {
            "fpod": [], "trigger_tokens": defaultdict(int), "premature_eof": 0
        }
    }
    
    in_matrix = False
    current_sample = 0
    saber_diverged = False
    snapkv_diverged = False
    
    # Existing decode divergence matrix pattern
    row_pattern = re.compile(
        r'\s*(\d+)\s+\|\s+(❌)?\s*\'(.*?)\'\s*(?:\(([\d.]+)%\))?\s*\|\s*\'(.*?)\'\s*(?:\(([\d.]+)%\))?\s*\|\s*\'(.*?)\''
    )
    
    # New prefill telemetry regex patterns
    cos_sim_pattern = re.compile(r'Background INT4 Cosine Similarity:\s*([\d.]+)')
    max_scale_pattern = re.compile(r'Background INT4 Max Scale:\s*([\d.]+)')
    mean_scale_pattern = re.compile(r'Background INT4 Mean Scale:\s*([\d.]+)')
    iou_pattern = re.compile(r'Intersection-Over-Union \(IoU\):\s*([\d.]+)%')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Scrape Prefill Telemetry (Saber-U v2)
            if "Background INT4 Cosine Similarity:" in line:
                match = cos_sim_pattern.search(line)
                if match: stats["saber_u"]["telemetry"]["cosine_sim"].append(float(match.group(1)))
                continue
            elif "Background INT4 Max Scale:" in line:
                match = max_scale_pattern.search(line)
                if match: stats["saber_u"]["telemetry"]["max_scale"].append(float(match.group(1)))
                continue
            elif "Background INT4 Mean Scale:" in line:
                match = mean_scale_pattern.search(line)
                if match: stats["saber_u"]["telemetry"]["mean_scale"].append(float(match.group(1)))
                continue
            elif "Intersection-Over-Union (IoU):" in line:
                match = iou_pattern.search(line)
                if match: stats["saber_u"]["telemetry"]["iou"].append(float(match.group(1)))
                continue

            # 2. Scrape Decode Divergence Matrix
            if "[DIVERGENCE TRACKER] Sample" in line:
                in_matrix = True
                saber_diverged = False
                snapkv_diverged = False
                current_sample += 1
                continue
                
            if in_matrix:
                if line.strip().startswith("---") and saber_diverged:
                    in_matrix = False
                    continue
                    
                match = row_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    nat_tok = match.group(3)
                    nat_prob = float(match.group(4)) / 100.0 if match.group(4) else 0.0
                    sab_tok = match.group(5)
                    sab_prob = float(match.group(6)) / 100.0 if match.group(6) else 0.0
                    snap_tok = match.group(7)
                    
                    # Track Saber-U FPoD
                    if nat_tok != sab_tok and not saber_diverged:
                        stats["saber_u"]["fpod"].append(step)
                        stats["saber_u"]["confidence_drop"].append(nat_prob - sab_prob)
                        stats["saber_u"]["trigger_tokens"][nat_tok] += 1
                        saber_diverged = True
                        
                    # Track SnapKV FPoD
                    if nat_tok != snap_tok and not snapkv_diverged:
                        stats["snapkv"]["fpod"].append(step)
                        stats["snapkv"]["trigger_tokens"][nat_tok] += 1
                        snapkv_diverged = True
                        
                    # Track premature termination
                    if sab_tok == "<|im_end|>" and nat_tok != "<|im_end|>":
                        stats["saber_u"]["premature_eof"] += 1

    # Terminal Output Generation
    print("\n" + "="*80)
    print(f"📊 AGGREGATED TELEMETRY & DIVERGENCE PROFILE (Samples Evaluated: {current_sample})")
    print("="*80)
    
    for method in ["saber_u", "snapkv"]:
        data = stats[method]
        
        print(f"\n🚀 {method.upper()}")
        
        # Print Prefill Telemetry (Only exists for Saber variants)
        if method == "saber_u" and data["telemetry"]["cosine_sim"]:
            avg_cos = np.mean(data["telemetry"]["cosine_sim"])
            avg_max_scale = np.mean(data["telemetry"]["max_scale"])
            avg_mean_scale = np.mean(data["telemetry"]["mean_scale"])
            avg_iou = np.mean(data["telemetry"]["iou"])
            
            print(f"  [Prefill Quantization Health]")
            print(f"  ├─ Avg SnapKV Z-Axis IoU: {avg_iou:.1f}%")
            print(f"  ├─ Background INT4 Cosine Sim: {avg_cos:.4f}")
            print(f"  ├─ Background INT4 Max Scale: {avg_max_scale:.2f}")
            print(f"  └─ Background INT4 Mean Scale: {avg_mean_scale:.4f}")
            print(f"  [Decode Divergence Tracker]")
        
        # Print Decode Metrics
        if data["fpod"]:
            avg_fpod = np.mean(data["fpod"])
            avg_drop = np.mean(data["confidence_drop"]) * 100 if method == "saber_u" else 0.0
            
            # Sort and get top 3 hallucination triggers
            top_triggers = sorted(data["trigger_tokens"].items(), key=lambda item: item[1], reverse=True)[:3]
            trigger_str = ", ".join([f"'{tok}': {count}" for tok, count in top_triggers])
            
            print(f"  ├─ Avg Steps to Failure (FPoD): {avg_fpod:.1f} tokens")
            if method == "saber_u":
                print(f"  ├─ Avg Confidence Drop at Failure: -{avg_drop:.1f}%")
                print(f"  ├─ Premature Terminations (<|im_end|>): {data['premature_eof']}")
            print(f"  └─ Top Trigger Tokens: {trigger_str}")
        else:
            print("  └─ No divergence failures recorded!")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_lcc_log.py <logfile>")
        sys.exit(1)
    parse_divergence_logs(sys.argv[1])
