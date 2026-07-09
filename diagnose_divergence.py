import re
from collections import defaultdict
import os

LOG_FILE = "judge_debug.log"

def analyze_divergence():
    if not os.path.exists(LOG_FILE):
        print(f"❌ Could not find {LOG_FILE}")
        return

    print("🔍 Scanning judge_debug.log for extreme scoring divergence...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Split the log into discrete evaluation blocks
    blocks = log_content.split("▶️ DOC: ")[1:]
    records = defaultdict(list)

    for block in blocks:
        try:
            # Parse header: "doc_0_1234 | METHOD: Native-Baseline | PASS: 1"
            header = block.split('\n')[0]
            parts = header.split(" | ")
            doc_id = parts[0].strip()
            method = parts[1].replace("METHOD:", "").strip()
            pass_num = parts[2].replace("PASS:", "").strip()

            # Extract Score
            score_match = re.search(r'\[\[SCORE:\s*(\d+(?:\.\d+)?)\]\]', block)
            if not score_match: continue
            score = float(score_match.group(1))

            # Extract Reasoning
            think_match = re.search(r'<think>(.*?)</think>', block, re.DOTALL)
            think = think_match.group(1).strip() if think_match else "No <think> block found."

            records[(doc_id, method)].append({
                "pass": pass_num,
                "score": score,
                "reasoning": think
            })
        except Exception as e:
            continue

    found_examples = 0
    for (doc, method), passes in records.items():
        if len(passes) > 1:
            scores = [p["score"] for p in passes]
            score_spread = max(scores) - min(scores)
            
            # 🟢 DIVERGENCE THRESHOLD: Look for a swing of 7.0 or greater (e.g., 0 to 10, or 3 to 10)
            if score_spread >= 7.0:
                print("\n" + "="*80)
                print(f"🚨 DIVERGENCE DETECTED: {doc} | {method}")
                print(f"📈 Scores across passes: {scores}")
                print("="*80)
                
                for p in sorted(passes, key=lambda x: x['pass']):
                    print(f"\n--- PASS {p['pass']} (Score Output: {p['score']}) ---")
                    # Print just the first few and last few lines of reasoning to keep it readable
                    lines = p['reasoning'].split('\n')
                    if len(lines) > 8:
                        print('\n'.join(lines[:4]))
                        print("... [snip] ...")
                        print('\n'.join(lines[-3:]))
                    else:
                        print(p['reasoning'])
                
                found_examples += 1
                if found_examples >= 2: # Limit output to 2 clear examples
                    break

    if found_examples == 0:
        print("✅ No extreme divergence instances (>= 7.0 swing) found in the log.")

if __name__ == "__main__":
    analyze_divergence()