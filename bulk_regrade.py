"""
Saber Bulk LLM-as-a-Judge (Ollama HTTP Self-Consistency Edition)
Automatically finds and grades all QA benchmark sweeps using raw HTTP streams.
Features: Monte Carlo Self-Consistency, retry-on-fail extraction logic, library-bypass 
live token streaming, crash recovery, domain rubrics, and automated progress syncing.
"""
import json
import re
import os
import sys
import glob
import time
import numpy as np
import requests

# 🟢 TARGET OLLAMA ENDPOINT AND ENGINE
OLLAMA_API_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "gpt-oss:20b"

def build_judge_prompt(reference, prediction, dataset_name="general"):
    domain_focus = "Evaluate the general semantic accuracy and factual overlap."
    examples = """
- Score 10: Perfectly captures the core facts with no hallucinations.
- Score 5: Captures the broad topic but misses a critical detail.
- Score 0: Completely hallucinated or contradicts the reference.
"""
    dataset_lower = dataset_name.lower()
    
    if "financial" in dataset_lower:
        domain_focus = "FINANCIAL QA: You must strictly penalize hallucinated numbers, dates, fiscal quarters, or financial metrics. In finance, a wrong number is a catastrophic failure."
        examples = """
- **Score 10:** Captures all numbers and dates perfectly.
- **Score 8:** Omits a minor number but gets the core metric right.
- **Score 4:** Gets the trend right, but misses the exact numbers.
- **Score 1:** Hallucinates a critical date or number. (Fatal error).
- **Score 0:** Completely wrong numbers and trend.
"""
    elif "legal" in dataset_lower:
        domain_focus = "LEGAL CONTRACTS: Focus strictly on obligations, time constraints, and conditions. Missing a condition completely alters legal meaning."
        examples = """
- **Score 10:** Captures all legal constraints perfectly.
- **Score 6:** Captures the action but misses a strict constraint.
- **Score 2:** Captures the topic but hallucinates the obligation.
- **Score 0:** Completely irrelevant or hallucinated.
"""
    elif "scientific" in dataset_lower:
        domain_focus = "SCIENTIFIC QA: Focus strictly on causal relationships, methodology, and experimental results. Ensure the correct mechanism is attributed to the correct result."
        examples = """
- **Score 10:** Perfect attribution of mechanism and result.
- **Score 5:** Correct result, but hallucinates or omits the methodology.
- **Score 0:** Scientifically inaccurate or attributes the wrong result entirely.
"""
    elif "multidoc" in dataset_lower:
        domain_focus = "MULTI-DOCUMENT SYNTHESIS: Focus on how well the answer synthesizes cross-document facts. Penalize models that only answer half the question or fail to link the entities."
        examples = """
- **Score 10:** Successfully links facts across different documents accurately.
- **Score 6:** Gets the facts right from Document A, but ignores the connecting facts from Document B.
- **Score 3:** Mentions the correct entities but hallucinates the relationship between them.
- **Score 0:** Fails to retrieve the correct entities entirely.
"""

    prompt = f"""You are an expert, merciless evaluator grading an AI's performance on a complex QA task.
Your goal is to compare the Predicted Answer to the Ground Truth Reference.

Ground Truth Reference:
{reference}

Predicted Answer:
{prediction}

### SCORING FOCUS
{domain_focus}

### RUBRIC EXAMPLES
{examples}

### OUTPUT FORMAT
You MUST restrict your output to exactly the following format. Keep your thinking block under 5 sentences.

<think>
[Your concise reasoning here]
</think>
[[SCORE: X]]
"""
    return prompt

def extract_score(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
        
    match = re.search(r'\[\[SCORE:\s*(\d+(?:\.\d+)?)\]\]', text)
    if match: return float(match.group(1))
    
    fallback_match = re.search(r'(\d+(?:\.\d+)?)\s*\]*$', text.strip())
    if fallback_match: return float(fallback_match.group(1))

    rescue_match = re.search(r'(?i)(?:score\s*is|score\s*should\s*be|score:)\s*(\d+(?:\.\d+)?)', text)
    if rescue_match: return float(rescue_match.group(1))
        
    return None

def run_bulk_judge():
    filepaths = sorted(glob.glob("saber_checkpoint_*qa.jsonl"))
    
    if not filepaths:
        print("⚠️ No checkpoint files found matching 'saber_checkpoint_*qa.jsonl'")
        return

    print(f"⚖️ Ensuring {JUDGE_MODEL} is active via HTTP daemon...")
    try:
        warmup_payload = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": "warmup"}], "stream": False}
        requests.post(OLLAMA_API_URL, json=warmup_payload, timeout=50)
    except Exception as e:
        print(f"🚨 Failed to connect to Ollama over HTTP. Is the server running? Error: {e}")
        sys.exit(1)
 
    methods = ["Native-Baseline", "Uniform-INT4", "Saber-Q", "Saber-CQ"]
    
    try:
        for filepath in filepaths:
            scores = {m: [] for m in methods}
            dataset_name = filepath.split('/')[-1]
            output_filepath = f"graded_{dataset_name}"
            
            processed_ids = set()
            
            if os.path.exists(output_filepath):
                print(f"\n🔄 Found existing graded file {output_filepath}. Recovering progress...")
                with open(output_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            doc_id = record.get("id")
                            if doc_id: processed_ids.add(doc_id)
                            
                            for method in methods:
                                if "llm_judge_scores" in record and method in record["llm_judge_scores"]:
                                    score = record["llm_judge_scores"][method].get("score")
                                    if score is not None: scores[method].append(score)
                print(f"✅ Recovered {len(processed_ids)} previously graded documents.")

            print(f"\n📖 Loading results from {filepath}...")
            records = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))

            print(f"🧪 Evaluation of {len(records)} documents for {dataset_name}...\n")
            
            with open(output_filepath, "a", encoding="utf-8") as out_file:
                for i, record in enumerate(records):
                    doc_id = record.get("id", f"doc_{i}")
                    reference = record.get("reference", "")
                    
                    if doc_id in processed_ids or not reference: 
                        continue
                        
                    print(f"▶️ Grading Document {i+1}/{len(records)} [{doc_id}]")
                    
                    if "llm_judge_scores" not in record: record["llm_judge_scores"] = {}
                    
                    for method in methods:
                        if method in record.get("results", {}) and record["results"][method].get("success"):
                            prediction = record["results"][method].get("generated_text", "")
                            
                            if not prediction.strip():
                                scores[method].append(0.0)
                                record["llm_judge_scores"][method] = {"score": 0.0, "reason": "Empty prediction"}
                                continue
                            
                            prompt = build_judge_prompt(reference, prediction, dataset_name)
                            
                            # 🟢 SELF-CONSISTENCY SAMPLING SETTINGS
                            num_samples = 3  
                            max_retries = 3
                            sampled_scores = []
                            reasons_log = []
                            
                            print(f"  └─ {method:<16} | 🔄 Sampling {num_samples} grading passes...")
                            
                            for sample_idx in range(num_samples):
                                current_temp = 0.6 
                                score = None
                                response_text = ""
                                gen_time = 0
                                
                                # 🟢 RETRY LOOP PER PASS
                                for attempt in range(max_retries):
                                    start_time = time.time()
                                    response_text = ""
                                    
                                    payload = {
                                        "model": JUDGE_MODEL,
                                        "messages": [
                                            {"role": "system", "content": "You are a precise and fair grading assistant."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        "options": {
                                            "temperature": current_temp,
                                            "num_predict": 1024  
                                        },
                                        "stream": True
                                    }
                                    
                                    try:
                                        attempt_label = f"Pass {sample_idx + 1}/{num_samples}" if attempt == 0 else f"Pass {sample_idx + 1}/{num_samples} (Retry {attempt})"
                                        print(f"    ├─ {attempt_label} | 🟢 Streaming Stream Window:")
                                        print("." * 60)
                                        
                                        res = requests.post(OLLAMA_API_URL, json=payload, stream=True)
                                        res.raise_for_status()
                                        
                                        for line in res.iter_lines():
                                            if line:
                                                chunk = json.loads(line.decode('utf-8'))
                                                token = chunk.get("message", {}).get("content", "")
                                                if token:
                                                    #sys.stdout.write(token)
                                                    #sys.stdout.flush()
                                                    response_text += token
                                                    
                                        print(f"\n{'.'*60}")
                                        gen_time = time.time() - start_time
                                        score = extract_score(response_text)
                                        
                                        if score is not None:
                                            break  # Success: break out of the retry loop
                                        elif attempt < max_retries - 1:
                                            print(f"    ├─ ⚠️ Regex extraction failed. Retrying pass ({attempt + 1}/{max_retries - 1})...")
                                            
                                    except Exception as e:
                                        print(f"\n🚨 HTTP STREAM INTERRUPTION DURING SAMPLING ITERATION: {e}")
                                        sys.exit(1)
                                        
                                # 🟢 PARSING & IGNORING LOGIC
                                if score is not None:
                                    score = max(0.0, min(10.0, score))
                                    sampled_scores.append(score)
                                    
                                    reason_preview = response_text.split("</think>")[-1].strip() if "</think>" in response_text else response_text.strip()
                                    clean_reason = reason_preview.replace("\n", " ").strip()
                                    reasons_log.append(f"[Pass {sample_idx+1} ({score:.1f}/10): {' '.join(clean_reason.split()[:10])}...]")
                                    print(f"    └─ ✅ Tracked Pass {sample_idx + 1} | ⏱️ {gen_time:.1f}s | Parsed Score: {score}/10")
                                else:
                                    # Fail gracefully: do NOT append 0.0. The average will only use successful passes.
                                    print(f"    └─ ❌ Pass {sample_idx + 1} completely failed after {max_retries} attempts. Ignoring this pass.")
                                    reasons_log.append(f"[Pass {sample_idx+1} (IGNORED): Regex mismatch after retries]")
                                    
                            # 🟢 COMPUTE FINAL MONTE CARLO AVERAGE MEAN
                            if sampled_scores:
                                final_mean_score = float(np.mean(sampled_scores))
                                scores[method].append(final_mean_score)
                                record["llm_judge_scores"][method] = {
                                    "score": final_mean_score,
                                    "all_sampled_scores": sampled_scores,
                                    "reason": " || ".join(reasons_log)
                                }
                                print(f"  └─ ⭐ FINAL AGGREGATED AVERAGE SCORE: {final_mean_score:>4.2f} / 10.0\n")
                            else:
                                print(f"  └─ ❌ Critical Error: All sampling attempts completely failed for {method}.\n")
                                record["llm_judge_scores"][method] = {"score": None, "reason": "All execution tracks failed."}

                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    os.fsync(out_file.fileno())
                    processed_ids.add(doc_id)

            print("\n" + "=" * 60)
            print(f"📊 LLM-AS-A-JUDGE RESULTS: {dataset_name}")
            print("=" * 60)
            print(f"{'Method':<18} | {'Avg Score (out of 10)':<20}")
            print("-" * 60)
            
            for method in methods:
                avg_score = np.mean(scores[method]) if scores[method] else 0
                print(f"{method:<18} | {avg_score:>5.2f} / 10.0")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n🛑 [KeyboardInterrupt] Gracefully shutting down...")
        print("✅ Current progress has been safely synced to disk. Run the script again to resume.")
        sys.exit(0)

if __name__ == "__main__":
    run_bulk_judge()