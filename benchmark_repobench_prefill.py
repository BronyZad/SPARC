"""
Saber Prefill Server - RepoBench Code Completion
Targets: Cross-file next-line prediction.
Features: On-the-fly Exact Match (EM) & Edit Similarity (ES), Payload/TTFT Systems Profiling, Fail-Fast Crash Handling.
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
import re
import numpy as np
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🟢 Clean Separation: Import the Saber Engine from the transport module
from saber_core_transport import SaberDisaggregatedEngine

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "../local_models/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LEN = 18000

# 🟢 NATIVE FORWARD CACHE
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

def evaluate_repobench(prediction, ground_truth):
    """
    Strips instruct-tuning artifacts, extracts the first generated line of code, 
    and computes Exact Match (EM) and Edit Similarity (ES).
    """
    # 1. Clean the prediction using Regex
    chat_patterns = [
        r"^(Here is the.*?code:|The next line.*?is:|Answer:)",
        r"^(Sure, .*?\n)",
        r"^(```python\n|```\n|```)"
    ]
    pred_text = prediction
    for pattern in chat_patterns:
        pred_text = re.sub(pattern, "", pred_text, flags=re.IGNORECASE|re.DOTALL).lstrip()
    
    pred_text = pred_text.split("```")[0]
    
    # 2. Extract lines for evaluation
    pred_lines = [line for line in pred_text.split('\n') if line.strip()]
    pred_line = pred_lines[0].strip() if pred_lines else ""
    truth_line = ground_truth.strip()

    # 3. Calculate Scores
    em_score = 1.0 if pred_line == truth_line else 0.0
    es_score = SequenceMatcher(None, pred_line, truth_line).ratio()

    return em_score, es_score, pred_line

def run_repobench(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
    checkpoint_file = f"saber_checkpoint_r{retain_ratio}_repobench.jsonl"

    print(f"🚀 Loading Model on Dual-GPU Prefill Server (RepoBench Mode)...")
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

    print(f"📚 Loading RepoBench Dataset: {dataset_path}...")
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
    
    # Tracking structures for systems profiling metrics
    metrics = {m: {
        "total": 0, "em": [], "es": [], 
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
                        metrics[method]["em"].append(data.get("em_score", 0.0))
                        metrics[method]["es"].append(data.get("es_score", 0.0))
                        metrics[method]["payload"].append(data.get("payload_mb", 0.0))
                        metrics[method]["prefill"].append(data.get("prefill_ms", 0.0))
                        metrics[method]["ttft"].append(data.get("ttft_ms", 0.0))
                        metrics[method]["tpot"].append(data.get("tpot_ms", 0.0))
                        
        print(f"✅ Resuming after {len(processed_ids)} previously evaluated lines of code.")

    print(f"\n🧪 STARTING REPOBENCH | Target Valid Samples: {'All' if num_samples <= 0 else num_samples}")
    print("=" * 100)

    valid_samples_count = 0

    try:
        with open(checkpoint_file, "a", encoding="utf-8") as ckpt_file:
            for i, test_data in enumerate(test_pool):
                
                if num_samples > 0 and valid_samples_count >= num_samples:
                    print(f"\n✅ Reached target of {num_samples} valid samples. Stopping benchmark.")
                    break

                context_str = test_data.get('context', '')
                question_str = test_data.get('question', '')

                system_instruction = (
                    "You are a pure code autocomplete engine. Your sole task is to provide the EXACT next line of code. "
                    "CRITICAL RULES:\n"
                    "1. DO NOT output any English text, conversational filler, or explanations.\n"
                    "2. DO NOT wrap the code in markdown blocks (e.g., no ```python).\n"
                    "3. Output ONLY the raw syntax for the next line and immediately stop."
                )

                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"# Cross-file context:\n{context_str}\n\n# Current file to complete:\n{question_str}"}
                ]

                prompt_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                ground_truth = test_data.get('answer', '')
                if not ground_truth: continue
                
                hash_str = hashlib.md5((prompt_text[-100:] + ground_truth).encode('utf-8')).hexdigest()[:8]
                doc_id = test_data.get("id", f"repo_{i}_{hash_str}")
                
                if doc_id in processed_ids: 
                    valid_samples_count += 1
                    continue
                
                input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
                actual_seq_len = input_ids.shape[1]
                
                if actual_seq_len > MAX_SEQ_LEN:
                    print(f"\n▶️ Skipping Snippet {i+1} (Length {actual_seq_len} exceeds {MAX_SEQ_LEN} limit)")
                    continue
                    
                valid_samples_count += 1
                input_ids = input_ids.expand(batch_size, -1).to(model.device)

                print(f"\n▶️ Evaluating Snippet {i+1}/{len(test_pool)} (Length: {actual_seq_len} tokens) | Progress: {valid_samples_count}/{num_samples if num_samples > 0 else 'All'}...")

                record = {
                    "id": doc_id,
                    "seq_len": actual_seq_len,
                    "ground_truth": ground_truth,
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
                        # ⏱️ START LOCAL TIMER: Track the initial envelope metadata handshake
                        t_start_envelope = time.perf_counter()
                        
                        envelope = pickle.dumps({"method": method, "input_ids": input_ids[:, -1:].cpu(), "max_new_tokens": max_new_tokens})
                        payload_bytes += len(envelope) 
                        socket.send(envelope)
                        recv_with_timeout(socket, 5000) 
                        
                        # Calculate envelope handshake RTT overhead
                        envelope_rtt_ms = (time.perf_counter() - t_start_envelope) * 1000
                        
                        # Fire the GPU prefill processing and begin streaming out chunks
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
                            generated_text = reply.get('text', '')
                            em_score, es_score, pred_line = evaluate_repobench(generated_text, ground_truth)
                            
                            payload_mb = payload_bytes / (1024 * 1024)
                            
                            # ⏱️ EXTRACT ISOLATED METRIC: Extract drift-free TTFT from Decode Server
                            decode_reported_ttft = reply.get("ttft_ms", 0.0)
                            
                            # True Disaggregated TTFT = Initial Handshake Overhead + Remote Stream-to-First-Token time
                            true_ttft_ms = envelope_rtt_ms + decode_reported_ttft
                            
                            tpot_ms = reply.get("tpot_ms", 0.0)
                            
                            metrics[method]["total"] += 1
                            metrics[method]["em"].append(em_score)
                            metrics[method]["es"].append(es_score)
                            metrics[method]["payload"].append(payload_mb)
                            metrics[method]["prefill"].append(prefill_ms)
                            metrics[method]["ttft"].append(true_ttft_ms)  # 🟢 Storing the true unified calculation
                            metrics[method]["tpot"].append(tpot_ms)
                            
                            record["results"][method] = {
                                "success": True,
                                "generated_text": generated_text,
                                "extracted_pred": pred_line,
                                "em_score": em_score,
                                "es_score": es_score,
                                "payload_mb": payload_mb,
                                "prefill_ms": prefill_ms,
                                "ttft_ms": true_ttft_ms,
                                "tpot_ms": tpot_ms
                            }
                            
                            icon = "🟢" if em_score == 1.0 else ("🟡" if es_score > 0.8 else "🔴")
                            display_pred = (pred_line[:30] + '...') if len(pred_line) > 30 else pred_line
                            print(f"  └─ {method:<16} | {icon} EM: {int(em_score)} | ES: {es_score:.2f} | Payload: {payload_mb:.1f}MB | TTFT: {true_ttft_ms:.1f}ms")
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
                    print("Please check/restart the decode server and re-run the script. It will safely resume.")
                    sys.exit(1)

                ckpt_file.write(json.dumps(record) + "\n")
                ckpt_file.flush()
                os.fsync(ckpt_file.fileno())

    except KeyboardInterrupt:
        print("\n\n🛑 [KeyboardInterrupt] Gracefully shutting down...")
        print("✅ Current progress has been safely synced to disk. You can safely resume later.")
        sys.exit(0)

    # Comprehensive Systems Report Block
    total_evaluated = metrics[methods[0]]["total"] if methods else 0
    print("\n" + "=" * 120)
    print(f"📊 REPOBENCH ACCURACY & SYSTEMS REPORT (Samples Evaluated: {total_evaluated})")
    print("=" * 120)
    
    header = f"{'Method':<16} | {'EM (%)':>6} | {'ES (%)':>6} | {'Payload(MB)':>11} | {'Prefill(ms)':>11} | {'Avg TTFT(ms)':>12} | {'P95 TTFT(ms)':>12} | {'Avg TPOT(ms)':>12} | {'P95 TPOT(ms)':>12}"
    print(header)
    print("-" * 120)

    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        
        # Accuracy Metrics
        avg_em = np.mean(m["em"]) * 100 
        avg_es = np.mean(m["es"]) * 100
        
        # System Metrics
        avg_payload = np.mean(m["payload"])
        avg_prefill = np.mean(m["prefill"])
        avg_ttft = np.mean(m["ttft"])
        p95_ttft = np.percentile(m["ttft"], 95) if m["ttft"] else 0.0
        avg_tpot = np.mean(m["tpot"])
        p95_tpot = np.percentile(m["tpot"], 95) if m["tpot"] else 0.0
        
        row = f"{method:<16} | {avg_em:>6.2f} | {avg_es:>6.2f} | {avg_payload:>11.1f} | {avg_prefill:>11.1f} | {avg_ttft:>12.1f} | {p95_ttft:>12.1f} | {avg_tpot:>12.1f} | {p95_tpot:>12.1f}"
        print(row)
    
    print("=" * 120)
    print(f"📝 Output saved to {checkpoint_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67")
    parser.add_argument('--port', type=str, default="5555")
    parser.add_argument('--retain_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=50) 
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="../data/benchmarks/repobeach_python/repobench_cross_file.jsonl")
    args = parser.parse_args()
    
    run_repobench(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)