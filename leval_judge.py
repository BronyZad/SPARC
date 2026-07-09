"""
Saber LEval Evaluation Engine (Ollama Edition)
Features: Context-Aware Fair Grading, Metric-Driven Routing, Aggressive MCQ Fast-Path, 
          Multi-Threaded Concurrency, Crash Recovery, and Granular Execution Flags.
Updates: Integrated Granular 0-10 Spectrum Rubrics to resolve discrete scale paralysis.
"""
import json
import re
import os
import sys
import argparse
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

def get_metric_and_rubric(filename):
    name = filename.lower()
    
    if any(x in name for x in ["gsm100", "quality", "sci_fi", "tpo", "coursera", "codeu", "topic_retrieval"]):
        return "exam", (
            "EXAM / MULTIPLE CHOICE: You are grading a multiple-choice or short-answer exam. "
            "Ignore conversational filler like 'The correct answer is...'. "
            "If the predicted text contains the correct Option Letter or exact Boolean (True/False) intended by the Reference, award a 10. "
            "If the model selected the wrong option letter, award a 0. Do not give partial credit for wrong options."
        )
    elif any(x in name for x in ["_summ", "paper_assistant"]):
        return "rouge", (
            "LONG SUMMARIZATION & NARRATIVE: The Ground Truth is often a short 'logline' or brief teaser, "
            "while the Predicted Answer is a highly detailed, full-length plot summary.\n"
            "1. VAGUE TO SPECIFIC MAPPING: If the Prediction expands vague reference terms (e.g., 'an alien device') "
            "into specific named entities (e.g., 'The Pulse'), this is a PERFECT match. It is NOT a hallucination.\n"
            "2. NO PENALTY FOR EXPANSION: Do not penalize the prediction for naming specific characters or plot points missing from the reference.\n"
            "3. NO FALSE CONTRADICTIONS: In complex narratives, seemingly conflicting states can coexist. Evaluate the core thematic event."
        )
    else:
        if "legal" in name:
            return "f1", (
                "LEGAL QA: Focus on precise obligations, conditional boundaries, and structural constraints. "
                "If the core legal outcome is captured but a minor qualifier is omitted, deduct a few points (e.g., 7-9). "
                "If the legal obligation is reversed or completely hallucinated, score 0-3."
            )
        elif "financial" in name:
            return "f1", (
                "FINANCIAL QA: Erroneous numbers, dates, or scale markers (Millions vs Billions) are catastrophic. "
                "If a number is wrong, score 0-3. If the numbers are right but context is slightly off, score 7-9."
            )
        elif "scientific" in name:
            return "f1", (
                "SCIENTIFIC QA: Verify the precise matching of causal mechanisms and data. "
                "Do not award high points for vague structural hand-waving if the specific mechanism is missing."
            )
        return "f1", (
            "GENERAL LONG-CONTEXT QA: Check for high semantic overlap. The prediction must answer the core query."
        )

def build_universal_prompt(question, reference, prediction, dataset_name, rubric_text):
    return f"""You are an expert, highly consistent benchmark evaluator grading an AI's output on a complex long-context task.

### ORIGINAL QUESTION:
{question}

### GROUND TRUTH REFERENCE:
{reference}

### PREDICTED ANSWER:
{prediction}

### SCORING FOCUS (DOMAIN SPECIFIC)
{rubric_text}

### GRANULAR SCORING SCALE (0 to 10)
You must use the full 0-10 integer scale to reflect nuance and prevent rigid scoring loops.
- Score 10: Flawless. Completely answers the question, aligns perfectly with reference facts, no contradictions.
- Score 8-9: Excellent. Core answer is correct, but contains very minor omissions or slight conversational bloat.
- Score 5-7: Partial. Captures the broad gist of the answer, but misses critical qualifiers or includes unverified (but not necessarily contradictory) tangents.
- Score 2-4: Poor. Misses the main point, relies heavily on hallucinated facts, or only gets a trivial detail correct.
- Score 0-1: Total Failure. Completely wrong, contradicts the reference, or chooses the wrong multiple-choice option.

### FAIRNESS RULES
1. Do NOT penalize the prediction for including extra, highly specific contextual details as long as they answer the question and do not contradict the reference. 
2. Do NOT penalize the prediction for omitting facts found in the Reference if the Original Question did not explicitly ask for them.

### OUTPUT FORMAT
You MUST restrict your output to exactly the following format. Keep your thinking block highly concise and under 4 sentences.

<think>
[Your concise reasoning here]
</think>
[[SCORE: X]]
"""

def extract_score(text):
    if "</think>" in text: 
        text = text.split("</think>")[-1]
    for pattern in [r'\[\[SCORE:\s*(\d+(?:\.\d+)?)\]\]', r'(\d+(?:\.\d+)?)\s*\]*$', r'(?i)score:\s*(\d+(?:\.\d+)?)']:
        match = re.search(pattern, text.strip())
        if match: 
            return float(match.group(1))
    return None

def check_fast_path(reference, prediction, metric_type):
    ref_clean = reference.strip().upper()
    pred_clean = prediction.strip().upper()
    
    if ref_clean == pred_clean:
        return 10.0
        
    if metric_type == "exam":
        if len(ref_clean) <= 3:
            if pred_clean.startswith(ref_clean):
                return 10.0
            if re.search(rf'^{re.escape(ref_clean)}\b', pred_clean):
                return 10.0
                
    return None

def fetch_llm_score(prompt, current_temp=0.4):
    if shutdown_flag.is_set(): 
        return "Shutdown triggered", None
        
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are a precise grading assistant. Be concise in your internal reasoning block. Ensure you always output the final score in the strict format requested."
            }, 
            {"role": "user", "content": prompt}
        ],
        "options": {
            "temperature": current_temp, 
            "num_predict": 2048
        },
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        res.raise_for_status()
        response_data = res.json()
        response_text = response_data.get("message", {}).get("content", "")
        
        thinking_text = response_data.get("message", {}).get("thinking", "")
        if thinking_text:
            response_text = f"<think>\n{thinking_text}\n</think>\n{response_text}"
            
        return response_text, extract_score(response_text)
    except Exception as e:
        return str(e), None

def evaluate_method_for_doc(method, record, reference, dataset_name, metric_type, rubric_text, doc_id, is_debug):
    if shutdown_flag.is_set(): 
        return method, {"score": 0.0, "reason": "Interrupted"}
    
    prediction = record["results"][method].get("generated_text", "")
    if not prediction.strip(): 
        return method, {"score": 0.0, "reason": "Empty prediction"}

    fast_score = check_fast_path(reference, prediction, metric_type)
    if fast_score is not None: 
        return method, {"score": fast_score, "reason": "Regex Fast-Path Exact Match"}

    question = record.get("instruction", "Answer based on the document.")
    prompt = build_universal_prompt(question, reference, prediction, dataset_name, rubric_text)
    
    sampled_scores = []
    reasons_log = []
    
    for sample_idx in range(MAX_PASSES):
        if shutdown_flag.is_set(): 
            break
        temp = 0.1 + (sample_idx * 0.2)
        response_text, score = fetch_llm_score(prompt, temp)
        
        if is_debug:
            with file_lock:
                with open("judge_debug.log", "a", encoding="utf-8") as df:
                    df.write(f"▶️ DOC: {doc_id} | METHOD: {method} | PASS: {sample_idx+1}\n")
                    df.write(f"{'-'*40}\n{response_text}\n{'-'*40}\n\n")
        
        if score is not None:
            score = max(0.0, min(10.0, score))
            sampled_scores.append(score)
            reasons_log.append(f"[P{sample_idx+1}: {score:.1f}]")
            if len(sampled_scores) >= EARLY_EXIT_AGREEMENT and len(set(sampled_scores)) == 1:
                reasons_log.append("[EARLY EXIT]")
                break
        else:
            reasons_log.append(f"[P{sample_idx+1}: FAIL]")

    if sampled_scores:
        return method, {"score": float(np.mean(sampled_scores)), "all_sampled_scores": sampled_scores, "reason": " || ".join(reasons_log)}
    return method, {"score": 0.0, "reason": "All execution tracks failed."}

def run_bulk_judge(args):
    if not args.file_path:
        print("⚠️ No checkpoint files provided.")
        return

    print(f"⚖️ Validating connectivity to Ollama server engine ({JUDGE_MODEL})...")
    try:
        requests.post(OLLAMA_API_URL, json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}, timeout=5)
        print("✅ Ollama connection verified.")
    except Exception as e:
        print(f"🚨 Failed to reach Ollama at {OLLAMA_API_URL}. Check your server logs.\nError: {e}")
        sys.exit(1)
        
    if args.debug:
        print("🐞 Debug mode active. Detailed judge outputs logging to 'judge_debug.log'.")
        if os.path.exists("judge_debug.log"): 
            os.remove("judge_debug.log")
 
    #all_methods = ["Native-Baseline", "Uniform-INT4", "Saber-BIC", "SnapKV"]
    all_methods = ["Native-Baseline", "Uniform-INT4", "ablation_inverted", "Saber-BIC", "SnapKV"]
    methods = [args.method] if args.method else all_methods
    
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
    
    try:
        for filepath in args.file_path:
            if shutdown_flag.is_set(): 
                break
            if not os.path.exists(filepath):
                print(f"\n⚠️ File not found: {filepath}. Skipping...")
                continue
                
            dataset_name = os.path.basename(filepath)
            metric_type, rubric_text = get_metric_and_rubric(dataset_name)
            output_filepath = f"graded_{dataset_name}"
            
            processed_ids = set()
            scores = {m: [] for m in methods}
            
            if os.path.exists(output_filepath):
                with open(output_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("id"): 
                                processed_ids.add(record["id"])
                            for m in methods:
                                if "llm_judge_scores" in record and m in record["llm_judge_scores"]:
                                    if record["llm_judge_scores"][m].get("score") is not None:
                                        scores[m].append(record["llm_judge_scores"][m]["score"])
                print(f"\n🔄 Recovered {len(processed_ids)} graded records for {dataset_name}.")

            records = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): 
                        records.append(json.loads(line))

            print(f"\n🧪 Evaluating {len(records)} entries for {dataset_name} | Metric Classification: [{metric_type.upper()}]...")
            
            with open(output_filepath, "a", encoding="utf-8") as out_file:
                for i, record in enumerate(records):
                    if shutdown_flag.is_set(): 
                        break
                    doc_id = record.get("id", f"doc_{i}")
                    
                    if args.doc_id and doc_id != args.doc_id:
                        continue
                        
                    reference = record.get("reference", "")
                    
                    if doc_id in processed_ids or not reference: 
                        continue
                        
                    print(f"▶️ Grading Entry {i+1}/{len(records)} [{doc_id}] (Len: {record.get('seq_len', 'UNK')})")
                    if "llm_judge_scores" not in record: 
                        record["llm_judge_scores"] = {}
                    
                    active_futures = {}
                    for method in methods:
                        if method in record.get("results", {}) and record["results"][method].get("success"):
                            f = executor.submit(
                                evaluate_method_for_doc, 
                                method, record, reference, dataset_name, 
                                metric_type, rubric_text, doc_id, args.debug
                            )
                            active_futures[f] = method
                    
                    for future in list(active_futures.keys()):
                        method = active_futures[future]
                        try:
                            method, result_data = future.result()
                            record["llm_judge_scores"][method] = result_data
                            scores[method].append(result_data["score"])
                            
                            fast_icon = "⚡" if "Regex" in result_data['reason'] else "🧠"
                            print(f"  └─ {method:<16} | {fast_icon} Score: {result_data['score']:>4.1f}/10 | Logs: {result_data['reason']}")
                        except Exception as exc:
                            print(f"  └─ {method:<16} | 🚨 Worker Thread Fault: {exc}")

                    with file_lock:
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()
                        os.fsync(out_file.fileno())
                    processed_ids.add(doc_id)

            if not shutdown_flag.is_set() and len(processed_ids) > 0:
                print("\n" + "=" * 60)
                print(f"📊 INTERMEDIARY SUMMARY REPORT: {dataset_name}")
                print("=" * 60)
                for method in methods:
                    avg_score = np.mean(scores[method]) if scores[method] else 0
                    print(f"{method:<18} | {avg_score:>5.2f} / 10.0")
                print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n🛑 [Ctrl-C Triggered] Processing interruption trap. Terminating worker pool cleanly...")
        shutdown_flag.set()
        executor.shutdown(wait=False, cancel_futures=True)
        print("✅ Current batch states safely written to disk. Run script again to seamlessly resume.")
        sys.exit(0)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saber LEval Evaluation Judgement Engine")
    parser.add_argument(
        '--file_path', 
        nargs='+', 
        required=True, 
        help="Path(s) to the specific sparc checkpoint file(s) to evaluate."
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help="Saves raw model contextual reasoning lines to judge_debug.log"
    )
    parser.add_argument(
        '--doc_id', 
        type=str, 
        default=None, 
        help="Target a specific document ID to evaluate."
    )
    parser.add_argument(
        '--method', 
        type=str, 
        default=None, 
        #choices=["Native-Baseline", "Uniform-INT4", "Saber-BIC", "SnapKV"],
        choices=["Native-Baseline", "Uniform-INT4", "ablation_inverted", "Saber-BIC", "SnapKV"],
        help="Grade a single compression track."
    )
    args = parser.parse_args()
    
    run_bulk_judge(args)