"""
Saber Universal Evaluation Engine (RULER Edition)
Filename: ruler_judge.py
Features: Hybrid Routing (Exact Match for NIAH, LLM for QA), Threaded Concurrency, Checkpointing.
"""
import json
import re
import os
import sys
import argparse
import string
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

# 🟢 CONFIGURATION
OLLAMA_API_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "gpt-oss:20b" 
MAX_CONCURRENT_WORKERS = 4  
MAX_PASSES = 3              
EARLY_EXIT_AGREEMENT = 2    

file_lock = threading.Lock()
shutdown_flag = threading.Event()

def get_domain_rubric(dataset_name):
    return ("LONG-CONTEXT QA: Focus strictly on semantic equivalence. The model must correctly answer the question based on the provided reference. Ignore conversational filler.")

def build_universal_prompt(references, prediction, dataset_name):
    ref_string = " OR ".join(references)
    domain_focus = get_domain_rubric(dataset_name)
    
    return f"""You are a STRICT and CRITICAL expert evaluator. 
Your task is to identify inaccuracies, hallucinations, and unnecessary verbosity.

Ground Truth Reference(s):
{ref_string}

Predicted Answer:
{prediction}

### SCORING FOCUS
{domain_focus}

### CRITICAL SCORING RULES:
1. **Factuality & Hallucination:** If the prediction contains hallucinations, factual errors, or contradictory reasoning, score 0 immediately.
2. **Verbosity Penalty:** A correct answer buried in "garbage" text (conversational filler, unnecessary repetitions, preambles) is NOT a 10. 
   - If the answer is correct but significantly longer than necessary, deduct points heavily.
   - Max score for verbose/padded answers is 5, even if the fact is correct.
3. **Reasoning Requirement:** You must explicitly check for "Fluff/Verbosity" in your thinking block before deciding the score.

### OUTPUT FORMAT
You MUST follow this structure strictly:
<think>
1. Check Factuality: [Does it contain correct info? Any hallucinations?]
2. Check Verbosity: [Is it concise? Any filler?]
3. Final Score Assignment: [Determine score based on Rules]
</think>
[[SCORE: X]]
"""

def extract_score(text):
    if "</think>" in text: 
        text = text.split("</think>")[-1]
        
    patterns = [
        r'\[\[SCORE:\s*(\d+(?:\.\d+)?)\]\]', 
        r'\*\*SCORE:\s*(\d+(?:\.\d+)?)\*\*',
        r'(?i)score\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*10',
        r'(\d+(?:\.\d+)?)\s*\]*$'
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match: 
            val = float(match.group(1))
            if 0.0 <= val <= 10.0: return val
            
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    if numbers:
        val = float(numbers[-1])
        if 0.0 <= val <= 10.0: return val
        
    return None

def check_fast_path(references, prediction):
    """
    严厉版拦截器：
    只有当预测答案非常简短，且完美匹配时，才允许秒判。
    """
    pred_str = prediction.strip()
    if not pred_str: return None
    
    # 核心修改：长度约束
    # 允许预测长度比参考答案多出 20 个字符（留给连接词空间），超出即判定为“废话”，强制送审
    ref_min_len = min([len(r.strip()) for r in references])
    if len(pred_str) > (ref_min_len + 20):
        return None 
        
    # 关键词匹配逻辑（保持不变，但对短答案更严谨）
    pred_upper = f" {pred_str.upper()} "
    for ref in references:
        ref_upper = ref.strip().upper()
        # 处理布尔/简单答案
        if ref_upper in ["YES", "NO", "UNANSWERABLE"] and f" {ref_upper} " in pred_upper:
            if ref_upper == "YES" and " NO " not in pred_upper: return 10.0
            if ref_upper == "NO" and " YES " not in pred_upper: return 10.0
            if ref_upper == "UNANSWERABLE": return 10.0
        # 处理短语匹配
        if len(ref.split()) <= 5 and len(ref) > 2:
            if re.search(rf'\b{re.escape(ref_upper)}\b', pred_upper):
                return 10.0
    return None

def fetch_llm_score(prompt, current_temp=0.4):
    if shutdown_flag.is_set(): return "Shutdown triggered", None
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": "You are a precise grading assistant."}, {"role": "user", "content": prompt}],
        "options": {"temperature": current_temp, "num_predict": 2048},
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=240)
        res.raise_for_status()
        response_data = res.json()
        response_text = response_data.get("message", {}).get("content", "")
        
        thinking_text = response_data.get("message", {}).get("thinking", "")
        if thinking_text:
            response_text = f"<think>\n{thinking_text}\n</think>\n{response_text}"
            
        return response_text, extract_score(response_text)
    except Exception as e:
        return f"API_ERROR: {str(e)}", None

def evaluate_method_for_doc(method, record, references, dataset_name, doc_id, is_debug):
    if shutdown_flag.is_set(): return method, {"score": 0.0, "reason": "Interrupted"}
    
    prediction = record["results"][method].get("cleaned_text", record["results"][method].get("generated_text", ""))
    if not prediction.strip(): return method, {"score": 0.0, "reason": "Empty prediction"}

    # 保留正则秒判机制 (针对极短的回答)
    fast_score = check_fast_path(references, prediction)
    if fast_score is not None: return method, {"score": fast_score, "reason": "⚡ Regex Fast-Path"}

    prompt = build_universal_prompt(references, prediction, dataset_name)
    sampled_scores = []
    reasons_log = []
    
    for sample_idx in range(MAX_PASSES):
        if shutdown_flag.is_set(): break
        temp = 0.2 + (sample_idx * 0.2)
        response_text, score = fetch_llm_score(prompt, temp)
        
        if is_debug:
            with file_lock:
                with open("ruler_judge_debug.log", "a", encoding="utf-8") as df:
                    df.write(f"▶️ DOC: {doc_id} | METHOD: {method} | PASS: {sample_idx+1}\n")
                    df.write(f"{'-'*20} PROMPT {'-'*20}\n{prompt}\n")
                    df.write(f"{'-'*20} RAW RESPONSE {'-'*20}\n{response_text}\n")
                    df.write(f"{'-'*20} PARSED SCORE {'-'*20}\nScore extracted: {score}\n\n")
                    df.write("="*60 + "\n\n")
        
        if score is not None:
            score = max(0.0, min(10.0, score))
            sampled_scores.append(score)
            reasons_log.append(f"[P{sample_idx+1}: {score:.1f}]")
            if len(sampled_scores) >= EARLY_EXIT_AGREEMENT and len(set(sampled_scores)) == 1:
                reasons_log.append("[EARLY EXIT]")
                break
        else:
            err_type = "TIMEOUT/ERR" if "API_ERROR" in response_text else "FORMAT_ERR"
            reasons_log.append(f"[P{sample_idx+1}: FAIL({err_type})]")

    if sampled_scores:
        return method, {"score": float(np.mean(sampled_scores)), "all_sampled_scores": sampled_scores, "reason": " || ".join(reasons_log)}
    return method, {"score": 0.0, "reason": "All execution tracks failed."}

def run_bulk_judge(args):
    if not args.file_path:
        print("⚠️ No checkpoint files provided.")
        return

    print(f"⚖️ Testing connection to {JUDGE_MODEL}...")
    try:
        requests.post(OLLAMA_API_URL, json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}, timeout=5)
    except:
        print(f"🚨 Failed to connect to Ollama. Ensure the server is running.")
        sys.exit(1)
        
    if args.debug:
        print("🐞 Debug mode ON. Detailed LLM traces will be saved to 'ruler_judge_debug.log'.")
        if os.path.exists("ruler_judge_debug.log"): os.remove("ruler_judge_debug.log")
 
    methods = ["Native-Baseline", "Uniform-INT4", "ablation_inverted", "Saber-BIC", "SnapKV"]
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
    
    try:
        for filepath in args.file_path:
            if not os.path.exists(filepath):
                print(f"\n⚠️ File not found: {filepath}. Skipping...")
                continue
                
            dataset_name = os.path.basename(filepath)
            # 移除了所有 LongBench 专属的 skipping 逻辑，RULER 全通吃
                
            output_filepath = f"graded_{dataset_name}"
            processed_ids = set()
            scores = {m: [] for m in methods}
            
            if os.path.exists(output_filepath):
                with open(output_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("id"): processed_ids.add(record["id"])
                            for m in methods:
                                if "llm_judge_scores" in record and m in record["llm_judge_scores"]:
                                    if record["llm_judge_scores"][m].get("score") is not None:
                                        scores[m].append(record["llm_judge_scores"][m]["score"])
                print(f"\n🔄 Recovered {len(processed_ids)} graded docs for {dataset_name}.")

            records = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))

            print(f"\n🧪 Evaluating {len(records)} documents for {dataset_name}...")
            
            with open(output_filepath, "a", encoding="utf-8") as out_file:
                for i, record in enumerate(records):
                    if shutdown_flag.is_set(): break
                    doc_id = record.get("id", f"doc_{i}")
                    
                    # 🚀 RULER 专属参考答案字段提取 (通常在 outputs 里)
                    references = record.get("outputs", record.get("reference", record.get("ground_truth", record.get("answers", []))))
                    if isinstance(references, str): references = [references]
                    
                    if doc_id in processed_ids or not references: continue
                        
                    print(f"▶️ Grading Document {i+1}/{len(records)} [{doc_id}]")
                    if "llm_judge_scores" not in record: record["llm_judge_scores"] = {}
                    
                    active_futures = {}
                    for method in methods:
                        if method in record.get("results", {}) and record["results"][method].get("success"):
                            f = executor.submit(evaluate_method_for_doc, method, record, references, dataset_name, doc_id, args.debug)
                            active_futures[f] = method
                    
                    for future in list(active_futures.keys()):
                        method = active_futures[future]
                        try:
                            method, result_data = future.result()
                            record["llm_judge_scores"][method] = result_data
                            scores[method].append(result_data["score"])
                            
                            icon = "🎯" if "Exact Match" in result_data['reason'] else ("⚡" if "Regex" in result_data['reason'] else "🧠")
                            print(f"  └─ {method:<16} | {icon} Score: {result_data['score']:>4.1f}/10 | Logs: {result_data['reason']}")
                        except Exception as exc:
                            print(f"  └─ {method:<16} | 🚨 Thread Exception: {exc}")

                    with file_lock:
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()
                        os.fsync(out_file.fileno())
                    processed_ids.add(doc_id)

            if not shutdown_flag.is_set():
                print("\n" + "=" * 60)
                print(f"📊 RULER QA LLM-AS-A-JUDGE RESULTS: {dataset_name}")
                print("=" * 60)
                for method in methods:
                    avg_score = np.mean(scores[method]) if scores[method] else 0
                    print(f"{method:<18} | {avg_score:>5.2f} / 10.0")
                print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n🛑 [Ctrl-C Detected] Forcefully stopping execution pool...")
        shutdown_flag.set()
        executor.shutdown(wait=False, cancel_futures=True)
        print("✅ Active progress safely flushed to disk. Re-run script anytime to resume.")
        sys.exit(0)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saber RULER Evaluation Engine")
    parser.add_argument(
        '--file_path', 
        nargs='+', 
        required=True, 
        help="Path(s) to the specific JSONL checkpoint file(s) to grade."
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help="Enable detailed logging to ruler_judge_debug.log"
    )
    args = parser.parse_args()
    
    run_bulk_judge(args)