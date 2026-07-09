"""
Saber Prefill Server - RULER Benchmark (AAAI Final Evaluation)
Targets: Information Retrieval, Aggregation, and Systems Telemetry.
Updates: Fixed prefill telemetry to accurately capture routing/quantization overhead.
         Added robust JSONL schema parsing for 8k/16k aggregated sets.
"""
import torch
import zmq
import time
import pickle
import argparse
import gc
import json
import os
import sys
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the Saber Engine
from saber_core_transport import SaberDisaggregatedEngine

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "../local_models/Qwen3-4B-Instruct-2507" 
MAX_SEQ_LEN = 18000
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

def evaluate_ruler(prediction, expected_answers):
    """
    Calculates RECALL score (Partial Credit).
    """
    def normalize_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text) 
        return ' '.join(text.split())       

    pred_clean = normalize_text(prediction)
    pred_words = pred_clean.split()
    
    target_list = []
    for ans in expected_answers:
        target_list.extend(normalize_text(ans).split())
        
    target_count = len(target_list)
    if target_count == 0: return 0.0

    allowed_window = target_count + 15
    if len(pred_words) > allowed_window:
        pred_clean = ' '.join(pred_words[:allowed_window])
    
    found_count = 0
    for ans_clean in target_list:
        pattern = r'\b' + re.escape(ans_clean) + r'\b'
        if re.search(pattern, pred_clean):
            found_count += 1
            
    return float(found_count) / float(target_count)

def run_ruler(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
    task_name = os.path.basename(dataset_path).replace('.jsonl', '')
    checkpoint_file = f"saber_checkpoint_r{retain_ratio}_{task_name}.jsonl"

    print(f"🚀 Loading Model on Dual-GPU Prefill Server (RULER {task_name.upper()} Mode)...")
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

    print(f"📚 Loading RULER Dataset: {dataset_path}...")
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
    # Initialize robust metrics tracker
    metrics = {m: {
        "total": 0, "em": [], 
        "payload_mb": [], "prefill": [], "ttft": [], "tpot": []
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
                        metrics[method]["payload_mb"].append(data.get("payload_mb", 0.0))
                        metrics[method]["prefill"].append(data.get("prefill_ms", 0.0))
                        metrics[method]["ttft"].append(data.get("ttft_ms", 0.0))
                        metrics[method]["tpot"].append(data.get("tpot_ms", 0.0))
                        
        print(f"✅ Resuming after {len(processed_ids)} previously evaluated samples.")

    print(f"\n🧪 STARTING {task_name.upper()} BENCHMARK | Target Valid Samples: {'All' if num_samples <= 0 else num_samples}")
    print("=" * 115)

    valid_samples_count = 0

    try:
        with open(checkpoint_file, "a", encoding="utf-8") as ckpt_file:
            for i, test_data in enumerate(test_pool):
                
                if num_samples > 0 and valid_samples_count >= num_samples:
                    print(f"\n✅ Reached target of {num_samples} valid samples. Stopping benchmark.")
                    break

                # 🟢 RULER JSONL SCHEMA FIX
                if 'context' in test_data:
                    context_text = test_data.get('context', '')
                    question_text = test_data.get('question', '')
                    answer_prefix = test_data.get('answer_prefix', '')
                    prompt_text = f"{context_text}\n{question_text}\n{answer_prefix}"
                    expected_answers = test_data.get('answer', [])
                    
                    # Dynamically adjust max_new_tokens if the dataset requests it
                    if 'max_new_tokens' in test_data:
                        max_new_tokens = test_data['max_new_tokens']
                else:
                    # Fallback for older parquet-derived schema
                    prompt_text = test_data.get('input', '')
                    expected_answers = test_data.get('outputs', []) 
                
                if isinstance(expected_answers, str):
                    expected_answers = [expected_answers]

                if not expected_answers or not prompt_text:
                    continue

                # 🟢 RULER PREFIX CONTINUATION FIX
                formatted_prompt = prompt_text
                
                doc_id = f"ruler_{task_name}_row_{test_data.get('index', i)}"
                
                if doc_id in processed_ids: 
                    valid_samples_count += 1
                    continue
                
                input_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
                actual_seq_len = input_ids.shape[1]
                
                if actual_seq_len > MAX_SEQ_LEN:
                    print(f"\n▶️ Skipping Snippet {i+1} (Length {actual_seq_len} exceeds limit)")
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
                    
                    try:
                        total_bytes = 0
                        prefill_ms = 0.0
                        saber_time = 0.0  # 🟢 NEW: Tracks routing and quant overhead
                        
                        envelope = pickle.dumps({"method": method, "input_ids": input_ids[:, -1:].cpu(), "max_new_tokens": max_new_tokens})
                        socket.send(envelope)
                        recv_with_timeout(socket, 5000) 
                        
                        generator = engine.prefill_and_stream(input_ids[:, :-1], method=internal_method)

                        reply = None
                        for chunk in generator:
                            # 🟢 FIXED: Intercept total compute timing
                            if chunk["type"] == "metadata" and "stats" in chunk:
                                prefill_ms = chunk["stats"].get("prefill_time_ms", 0.0)
                                saber_time += chunk["stats"].get("routing_time_ms", 0.0)
                            
                            if chunk["type"] == "done" and "quant_time_ms" in chunk:
                                saber_time += chunk["quant_time_ms"]

                            chunk_bytes = pickle.dumps(chunk)
                            total_bytes += len(chunk_bytes)
                            socket.send(chunk_bytes)
                            
                            if chunk["type"] != "done":
                                recv_with_timeout(socket, 5000) 
                            else:
                                reply = recv_pyobj_with_timeout(socket, 120000) 
                        
                        # 🟢 ADD THE OVERHEAD TO THE FINAL PREFILL METRIC
                        prefill_ms += saber_time
                        
                        if reply and reply.get("status") == "success":
                            generated_text = reply.get('text', '')
                            em_score = evaluate_ruler(generated_text, expected_answers)
                            
                            payload_mb = total_bytes / (1024 * 1024)
                            ttft_ms = reply.get("ttft_ms", 0.0)
                            tpot_ms = reply.get("tpot_ms", 0.0)
                            
                            metrics[method]["total"] += 1
                            metrics[method]["em"].append(em_score)
                            metrics[method]["payload_mb"].append(payload_mb)
                            metrics[method]["prefill"].append(prefill_ms)
                            metrics[method]["ttft"].append(ttft_ms)
                            metrics[method]["tpot"].append(tpot_ms)
                            
                            record["results"][method] = {
                                "success": True,
                                "generated_text": generated_text,
                                "em_score": em_score,
                                "payload_mb": payload_mb,
                                "prefill_ms": prefill_ms,
                                "ttft_ms": ttft_ms,
                                "tpot_ms": tpot_ms
                            }
                            
                            if em_score == 1.0: icon = "🟢"
                            elif em_score > 0: icon = "🟡"
                            else: icon = "🔴"
                            
                            display_pred = (generated_text[:40].replace('\n', ' ') + '...') if len(generated_text) > 40 else generated_text.replace('\n', ' ')
                            print(f"  └─ {method:<16} | {icon} Score: {em_score:.2f} | Payload: {payload_mb:.1f}MB | Pred: {display_pred}")
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
    header = f"{'Method':<16} | {'Recall(%)':<9} | {'Payload(MB)':<11} | {'Prefill(ms)':<11} | {'Avg TTFT(ms)':<12} | {'P95 TTFT(ms)':<12} | {'Avg TPOT(ms)':<12} | {'P95 TPOT(ms)':<12}"
    print(header)
    print("-" * 115)

    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        avg_em = np.mean(m["em"]) * 100 
        avg_payload = np.mean(m["payload_mb"])
        avg_prefill = np.mean(m["prefill"])
        avg_ttft = np.mean(m["ttft"])
        p95_ttft = np.percentile(m["ttft"], 95) if m["ttft"] else 0.0
        avg_tpot = np.mean(m["tpot"])
        p95_tpot = np.percentile(m["tpot"], 95) if m["tpot"] else 0.0
        
        print(f"{method:<16} | {avg_em:>9.2f} | {avg_payload:>11.1f} | {avg_prefill:>11.1f} | {avg_ttft:>12.1f} | {p95_ttft:>12.1f} | {avg_tpot:>12.1f} | {p95_tpot:>12.1f}")
    
    print("=" * 115)
    print(f"📝 Output saved to {checkpoint_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67")
    parser.add_argument('--port', type=str, default="5555")
    parser.add_argument('--retain_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=128) 
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="../data/benchmarks/RULER/ruler_test_8k.jsonl") 
    args = parser.parse_args()
    
    run_ruler(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)