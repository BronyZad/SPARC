"""
Saber Prefill Server - Universal LongBench
Targets: Global Context Synthesis, Attention Density Retention, and Disaggregated Transmission.
Features: Universal Prompt Routing, On-the-fly TPOT/TTFT Systems Profiling, Fail-Fast Crash Handling,
          and Anti-Degeneration (Repetition Penalty).
"""
import torch
import zmq
import time
import pickle
import argparse
import gc
import json
import os
import hashlib
import sys
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

# Import the Saber Engine
from sparc_core_transport import SaberDisaggregatedEngine

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "../local_models/Qwen3-8B" 
#MODEL_PATH = "../local_models/Qwen3-4B-Instruct-2507"
MAX_SEQ_LEN = 14000
NATIVE_FORWARDS = {}

def reset_zmq_socket(context, old_socket, ip, port):
    if old_socket is not None:
        old_socket.setsockopt(zmq.LINGER, 0)
        old_socket.close()
    new_socket = context.socket(zmq.REQ)
    new_socket.connect(f"tcp://{ip}:{port}")
    return new_socket

def recv_with_timeout(socket, timeout_ms=5000):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    if poller.poll(timeout_ms):
        return socket.recv()
    raise TimeoutError("Decode Server stopped responding (Timeout waiting for ACK).")

def recv_pyobj_with_timeout(socket, timeout_ms=120000):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    if poller.poll(timeout_ms):
        return socket.recv_pyobj()
    raise TimeoutError("Decode Server crashed during generation.")

def purge(model=None):
    if model is not None:
        if hasattr(model, "base_model"):
            model.base_model._past_key_values = None
        if hasattr(model, "model") and hasattr(model.model, "layers") and NATIVE_FORWARDS:
            for i, layer in enumerate(model.model.layers):
                if i in NATIVE_FORWARDS:
                    layer.self_attn.forward = NATIVE_FORWARDS[i]
    gc.collect()
    torch.cuda.empty_cache()

def strip_thinking_tags(text):
    """
    Removes reasoning blocks (e.g., <think>...</think>) from the output.
    Works natively for both reasoning and standard instruct models.
    """
    if not text:
        return ""
        
    # If the model successfully closed the thought block, extract everything after it.
    if '</think>' in text:
        text = text.split('</think>')[-1]
        
    # Clean up any unclosed tags or complete blocks if multiple exist
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '')
    
    return text.strip()

def evaluate_rouge(prediction, expected_answers):
    """
    Evaluates LongBench Summarization using ROUGE-L f-measure.
    Returns a score between 0.0 (total failure) and 1.0 (perfect match).
    """
    if not prediction or not prediction.strip(): 
        return 0.0
    
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    best_score = 0.0
    
    # Compare against all valid ground-truth summaries (usually just 1)
    for ans in expected_answers:
        scores = scorer.score(ans, prediction)
        if scores['rougeL'].fmeasure > best_score:
            best_score = scores['rougeL'].fmeasure
            
    return best_score

def run_longbench(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
    task_name = os.path.basename(dataset_path).replace('.jsonl', '')
    checkpoint_file = f"sparc_longbench_r{retain_ratio}_{task_name}.jsonl"

    print(f"🚀 Loading Model on Dual-GPU Prefill Server (LONGBENCH {task_name.upper()})...")
    print(f"⚙️ Configuration: [Retain Ratio: {retain_ratio:.2f}] | [Max Context: {MAX_SEQ_LEN}]")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="sdpa"
    )
    
    for i, layer in enumerate(model.model.layers):
        NATIVE_FORWARDS[i] = layer.self_attn.forward
    
    engine = SaberDisaggregatedEngine(model, retain_ratio, causal_depth=3)
    context = zmq.Context()
    socket = reset_zmq_socket(context, None, ip, port)

    print(f"📚 Loading LongBench Dataset: {dataset_path}...")
    test_pool = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_pool.append(json.loads(line))
        
    print(f"🔥 Warming up CUDA engine...")
    dummy_ids = torch.ones((1, 100), dtype=torch.long, device=model.device)
    with torch.no_grad(): _ = model(input_ids=dummy_ids, attention_mask=dummy_ids)
    purge(model)

    methods = ["Native-Baseline", "Uniform-INT4", "ablation_inverted", "Saber-BIC", "SnapKV"]
    
    metrics = {m: {
        "total": 0, "rougeL": [], 
        "payload": [], "prefill": [], "ttft": [], "tpot": []
    } for m in methods}
    
    processed_ids = set()

    if os.path.exists(checkpoint_file):
        print(f"\n📂 Found existing checkpoint '{checkpoint_file}'. Loading progress...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                processed_ids.add(record.get("id"))
                
                for method, data in record["results"].items():
                    if data.get("success"):
                        metrics[method]["total"] += 1
                        metrics[method]["rougeL"].append(data.get("rouge_score", 0.0))
                        metrics[method]["payload"].append(data.get("payload_mb", 0.0))
                        metrics[method]["prefill"].append(data.get("prefill_ms", 0.0))
                        metrics[method]["ttft"].append(data.get("ttft_ms", 0.0))
                        metrics[method]["tpot"].append(data.get("tpot_ms", 0.0))
                        
        print(f"✅ Resuming after {len(processed_ids)} previously evaluated samples.")

    print(f"\n🧪 STARTING {task_name.upper()} BENCHMARK | Target Valid Samples: {'All' if num_samples <= 0 else num_samples}")
    print("=" * 100)

    valid_samples_count = 0

    try:
        with open(checkpoint_file, "a", encoding="utf-8") as ckpt_file:
            for i, test_data in enumerate(test_pool):
                
                if num_samples > 0 and valid_samples_count >= num_samples:
                    print(f"\n✅ Reached target of {num_samples} valid samples. Stopping benchmark.")
                    break

                haystack_context = test_data.get('context', '')
                expected_answers = test_data.get('answers', []) 
                doc_id = test_data.get('_id', f"lb_{i}")

                if not expected_answers or not haystack_context:
                    continue
                
                # 🟢 DYNAMIC PROMPT ROUTING
                question = test_data.get('input', '')

                if "multi_news" in task_name or "samsum" in task_name or "gov_report" in task_name:
                    system_instruction = "You are an expert editor. Write a comprehensive summary of the provided text."
                    user_prompt = f"Text:\n{haystack_context}\n\nPlease provide a unified summary."
                
                elif "repobench" in task_name or "lcc" in task_name:
                    system_instruction = "You are an expert software engineer. Complete the code strictly based on the repository context."
                    user_prompt = f"Repository Context:\n{haystack_context}\n\nPlease complete the following code strictly based on the context above. Output ONLY the completed code without any markdown or explanations.\n\nCode to complete:\n{question}"
                
                elif "passage_retrieval" in task_name or "trec" in task_name or "lsht" in task_name:
                    system_instruction = "You are a precise information retrieval assistant."
                    user_prompt = f"Context paragraphs:\n{haystack_context}\n\nBased on the paragraphs above, which paragraph contains the following information?\nQuery: {question}\n\nPlease answer with ONLY the exact paragraph name (e.g., 'Paragraph 6')."

                else:
                    system_instruction = "You are an expert AI assistant. Answer the user's question based strictly on the provided context."
                    user_prompt = f"Context:\n{haystack_context}\n\nQuestion: {question}\n\nPlease provide the direct answer based on the context."

                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ]
                
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # ====================================================================
                # 💊 猛药2：强制保底 Chat Template，防止 Tokenizer 没加上 Assistant 前缀
                # ====================================================================
                if "<|im_start|>assistant" not in formatted_prompt and "assistant\n" not in formatted_prompt:
                    # 适配不同模型的特殊 token，这里以最常见的 ChatML 格式为保底
                    formatted_prompt += "<|im_start|>assistant\n"
                # ====================================================================

                if doc_id in processed_ids: 
                    valid_samples_count += 1
                    continue
                
                input_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
                actual_seq_len = input_ids.shape[1]
                
                if actual_seq_len > MAX_SEQ_LEN:
                    print(f"\n▶️ Skipping Snippet {i+1} (Length {actual_seq_len} exceeds {MAX_SEQ_LEN} limit)")
                    continue
                    
                valid_samples_count += 1
                input_ids = input_ids.expand(batch_size, -1).to(model.device)

                print(f"\n▶️ Evaluating Sample {valid_samples_count}/{num_samples} (Length: {actual_seq_len} tokens) ...")

                record = {
                    "id": doc_id,
                    "task": task_name,
                    "seq_len": actual_seq_len,
                    "ground_truth": expected_answers,
                    "results": {}
                }

                critical_failure = False

                for method in methods:
                    generator = None
                    chunk = None
                    internal_method = "baseline" if method == "Native-Baseline" else method.lower().replace("-", "_")
                    
                    payload_bytes = 0
                    prefill_ms = 0.0
                    
                    try:
                        t_start_envelope = time.perf_counter()
                        
                        # ====================================================================
                        # 💊 猛药1：加入温度和重复惩罚，强制打破“复读机”死循环
                        # ====================================================================
                        envelope = pickle.dumps({
                            "method": method, 
                            "input_ids": input_ids[:, -1:].cpu(), 
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.1,            # 极低温度保证确定性
                            "repetition_penalty": 1.05     # 核心：禁止无限复读同一个词
                        })
                        # ====================================================================
                        
                        payload_bytes += len(envelope)
                        socket.send(envelope)
                        recv_with_timeout(socket, 5000) 
                        
                        envelope_rtt_ms = (time.perf_counter() - t_start_envelope) * 1000
                        
                        generator = engine.prefill_and_stream(input_ids[:, :-1], method=internal_method)

                        reply = None
                        for chunk in generator:
                            if chunk["type"] == "metadata":
                                prefill_ms += chunk.get("stats", {}).get("prefill_time_ms", 0.0)
                                prefill_ms += chunk.get("stats", {}).get("routing_time_ms", 0.0)
                            elif chunk["type"] == "done":
                                prefill_ms += chunk.get("quant_time_ms", 0.0)
                                
                            chunk_bytes = pickle.dumps(chunk)
                            payload_bytes += len(chunk_bytes) 
                            
                            socket.send(chunk_bytes)
                            if chunk["type"] != "done":
                                recv_with_timeout(socket, 5000) 
                            else:
                                reply = recv_pyobj_with_timeout(socket, 120000) 
                        
                        if reply and reply.get("status") == "success":
                            raw_generated_text = reply.get('text', '')
                            cleaned_text = strip_thinking_tags(raw_generated_text)
                            rouge_score = evaluate_rouge(cleaned_text, expected_answers)
                            payload_mb = payload_bytes / (1024 * 1024)
                            
                            decode_reported_ttft = reply.get("ttft_ms", 0.0)
                            true_ttft_ms = envelope_rtt_ms + decode_reported_ttft
                            tpot_ms = reply.get("tpot_ms", 0.0)
                            
                            metrics[method]["total"] += 1
                            metrics[method]["rougeL"].append(rouge_score)
                            metrics[method]["payload"].append(payload_mb)
                            metrics[method]["prefill"].append(prefill_ms)
                            metrics[method]["ttft"].append(true_ttft_ms)
                            metrics[method]["tpot"].append(tpot_ms)
                            
                            record["results"][method] = {
                                "success": True,
                                "generated_text": raw_generated_text,
                                "cleaned_text": cleaned_text,
                                "rouge_score": rouge_score,
                                "payload_mb": payload_mb,
                                "prefill_ms": prefill_ms,
                                "ttft_ms": true_ttft_ms,
                                "tpot_ms": tpot_ms
                            }
                            
                            display_pred = (cleaned_text[:50].replace('\n', ' ') + '...') if len(cleaned_text) > 50 else cleaned_text.replace('\n', ' ')
                            print(f"  └─ {method:<16} | 📝 ROUGE-L: {rouge_score:.4f} | Payload: {payload_mb:.1f}MB | TTFT: {true_ttft_ms:.1f}ms | Pred: {display_pred}")
                        else:
                            error_reason = reply.get("message", "UNKNOWN") if reply else "NO_REPLY"
                            print(f"  └─ {method:<16} | ⚠️ DECODE SERVER ERROR: {error_reason}")
                            critical_failure = True
                            break

                    except TimeoutError as e:
                        print(f"  └─ {method:<16} | 🚨 CRITICAL REMOTE CRASH: {e}")
                        critical_failure = True
                        break
                        
                    except Exception as e:
                        print(f"  └─ {method:<16} | ❌ LOCAL CRASH: {str(e)[:50]}")
                        critical_failure = True
                        break
                        
                    finally:
                        del generator, chunk
                        purge(model)

                if critical_failure:
                    print("\n🛑 Halting benchmark to prevent corrupted metrics. The last document was NOT saved.")
                    sys.exit(1)

                ckpt_file.write(json.dumps(record) + "\n")
                ckpt_file.flush()
                os.fsync(ckpt_file.fileno())

    except KeyboardInterrupt:
        print("\n\n🛑 [KeyboardInterrupt] Gracefully shutting down...")
        sys.exit(0)

    total_evaluated = metrics[methods[0]]["total"] if methods else 0
    print("\n" + "=" * 115)
    print(f"📊 {task_name.upper()} ACCURACY & SYSTEMS REPORT (Samples Evaluated: {total_evaluated})")
    print("=" * 115)
    
    header = f"{'Method':<16} | {'ROUGE-L':>7} | {'Payload(MB)':>11} | {'Prefill(ms)':>11} | {'Avg TTFT(ms)':>12} | {'P95 TTFT(ms)':>12} | {'Avg TPOT(ms)':>12} | {'P95 TPOT(ms)':>12}"
    print(header)
    print("-" * 115)

    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        
        avg_rouge = np.mean(m["rougeL"]) * 100 
        
        avg_payload = np.mean(m["payload"])
        avg_prefill = np.mean(m["prefill"])
        avg_ttft = np.mean(m["ttft"])
        p95_ttft = np.percentile(m["ttft"], 95) if m["ttft"] else 0.0
        avg_tpot = np.mean(m["tpot"])
        p95_tpot = np.percentile(m["tpot"], 95) if m["tpot"] else 0.0
        
        row = f"{method:<16} | {avg_rouge:>7.2f} | {avg_payload:>11.1f} | {avg_prefill:>11.1f} | {avg_ttft:>12.1f} | {p95_ttft:>12.1f} | {avg_tpot:>12.1f} | {p95_tpot:>12.1f}"
        print(row)
    
    print("=" * 115)
    print(f"📝 Output saved to {checkpoint_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67")
    parser.add_argument('--port', type=str, default="5555")
    parser.add_argument('--retain_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=4096) 
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="../data/benchmarks/longbench/multi_news.jsonl") 
    args = parser.parse_args()
    
    run_longbench(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)