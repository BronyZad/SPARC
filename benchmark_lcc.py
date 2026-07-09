"""
Saber Prefill Server - LongBench (LCC: Long Code Completion)
Targets: Long-Range Syntax Dependency & Edit-Similarity.
Features: On-the-fly Edit Similarity, Payload/TTFT Systems Profiling, Fail-Fast Crash Handling,
          and Logit Divergence Profiling.
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
import numpy as np
import Levenshtein
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the Saber Engine
from saber_core_transport import SaberDisaggregatedEngine

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "../local_models/Qwen2.5-Coder-7B-Instruct" 
MAX_SEQ_LEN = 16000
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

def evaluate_lcc(prediction, expected_answers):
    """
    Evaluates LongBench Code Completion using Edit Similarity.
    1.0 = Exact Match, 0.0 = Completely Wrong.
    """
    # 1. SANITIZE OUTPUT: Remove think tags if the model used them
    pred_clean = prediction
    if "</think>" in pred_clean:
        pred_clean = pred_clean.split("</think>")[-1]
        
    # 2. Remove common markdown wrappers in case instructions are ignored
    pred_clean = pred_clean.replace("```csharp", "").replace("```python", "").replace("```java", "").replace("```", "")
    
    # 3. Extract ONLY the first generated line of code. 
    pred_lines = [line for line in pred_clean.split('\n') if line.strip()]
    pred_first_line = pred_lines[0] if pred_lines else ""
    
    best_sim = 0.0
    for ans in expected_answers:
        ans_clean = ans.strip()
        pred_clean_line = pred_first_line.strip()
        
        pred_trunc = pred_clean.replace('\n', ' ').strip()[:max(1, len(ans_clean))]
        
        for p in [pred_clean_line, pred_trunc]:
            if not p and not ans_clean:
                continue
                
            dist = Levenshtein.distance(p, ans_clean)
            max_len = max(len(p), len(ans_clean), 1)
            sim = 1.0 - (dist / max_len)
            
            if sim > best_sim:
                best_sim = sim
                
    return best_sim

def run_longbench_lcc(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
    task_name = os.path.basename(dataset_path).replace('.jsonl', '')
    checkpoint_file = f"saber_checkpoint_r{retain_ratio}_{task_name}.jsonl"

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

    
    # Tracking structures for full systems profiling metrics
    metrics = {m: {
        "total": 0, "edit_sim": [], 
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
                        metrics[method]["edit_sim"].append(data.get("em_score", 0.0))
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

                # 🟢 LONGBENCH SCHEMA PARSING
                haystack_context = test_data.get('context', '')
                expected_answers = test_data.get('answers', []) 
                doc_id = test_data.get('_id', f"lb_lcc_{i}")

                if not expected_answers or not haystack_context:
                    continue

                # 🟢 LCC AUTO-COMPLETE PROMPT
                system_instruction = (
                    "You are a strict code autocomplete engine. Your only task is to provide the EXACT next line of code. "
                    "Rules:\n"
                    "1. Output ONLY the next logical line of code.\n"
                    "2. Do NOT output markdown blocks, backticks, or comments.\n"
                    "3. Do NOT provide explanations or conversational text."
                )
                user_prompt = f"Code Context:\n{haystack_context}\n\nNext line of code:"
                
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ]
                
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
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
                
                # 🟢 NEW: Store token-trace profiles for Divergence Matrix
                sample_profiles = {} 

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
                            generated_text = reply.get('text', '')
                            edit_sim = evaluate_lcc(generated_text, expected_answers)
                            
                            payload_mb = payload_bytes / (1024 * 1024)
                            
                            decode_reported_ttft = reply.get("ttft_ms", 0.0)
                            true_ttft_ms = envelope_rtt_ms + decode_reported_ttft
                            tpot_ms = reply.get("tpot_ms", 0.0)
                            
                            # 🟢 NEW: Capture the divergence profile sent back from the decode server
                            sample_profiles[method] = reply.get("divergence_profile", [])
                            
                            metrics[method]["total"] += 1
                            metrics[method]["edit_sim"].append(edit_sim)
                            metrics[method]["payload"].append(payload_mb)
                            metrics[method]["prefill"].append(prefill_ms)
                            metrics[method]["ttft"].append(true_ttft_ms) 
                            metrics[method]["tpot"].append(tpot_ms)
                            
                            record["results"][method] = {
                                "success": True,
                                "generated_text": generated_text,
                                "em_score": edit_sim,
                                "payload_mb": payload_mb,
                                "prefill_ms": prefill_ms,
                                "ttft_ms": true_ttft_ms,
                                "tpot_ms": tpot_ms
                            }
                            
                            icon = "🟢" if edit_sim > 0.8 else "🟡" if edit_sim > 0.4 else "🔴"
                            
                            print(f"  └─ {method:<16} | {icon} Sim: {edit_sim:.2f} | Payload: {payload_mb:.1f}MB | TTFT: {true_ttft_ms:.1f}ms")
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
                    
                # 🟢 NEW: Print Divergence Tracker Matrix after all methods finish this sample
                native_prof = sample_profiles.get("Native-Baseline", [])
                saber_prof = sample_profiles.get("Saber-U", [])
                snap_prof = sample_profiles.get("SnapKV", [])
                
                if native_prof and saber_prof and snap_prof:
                    print(f"\n  🔎 [DIVERGENCE TRACKER] Sample {valid_samples_count} Output Alignment:")
                    print(f"  {'Step':<5} | {'NATIVE BASELINE (BF16)':<30} | {'SABER-U (INT4 COMPRESSED)':<30} | {'SNAPKV':<15}")
                    print("  " + "-" * 90)

                    max_steps = max(len(native_prof), len(saber_prof), len(snap_prof))
                    for step in range(min(max_steps, 24)): # Check first 24 tokens generated
                        nat_t = native_prof[step]["token"].replace("\n", "\\n")[:20] if step < len(native_prof) else "EOF"
                        nat_p = native_prof[step]["prob"] if step < len(native_prof) else 0.0
                        
                        sab_t = saber_prof[step]["token"].replace("\n", "\\n")[:20] if step < len(saber_prof) else "EOF"
                        sab_p = saber_prof[step]["prob"] if step < len(saber_prof) else 0.0
                        
                        snap_t = snap_prof[step]["token"].replace("\n", "\\n")[:20] if step < len(snap_prof) else "EOF"
                        
                        marker = "❌" if nat_t != sab_t else "  "
                        
                        print(f"  {step:<5} | {marker} '{nat_t}' ({nat_p:>4.1%}) | '{sab_t}' ({sab_p:>4.1%}) | '{snap_t}'")
                    print("  " + "-" * 90)

                ckpt_file.write(json.dumps(record) + "\n")
                ckpt_file.flush()
                os.fsync(ckpt_file.fileno())

    except KeyboardInterrupt:
        print("\n\n🛑 [KeyboardInterrupt] Gracefully shutting down...")
        sys.exit(0)

    # Consolidated Performance & Accuracy Table Printout
    total_evaluated = metrics[methods[0]]["total"] if methods else 0
    print("\n" + "=" * 120)
    print(f"📊 {task_name.upper()} ACCURACY & SYSTEMS REPORT (Samples Evaluated: {total_evaluated})")
    print("=" * 120)
    
    header = f"{'Method':<16} | {'Edit Sim (%)':>12} | {'Payload(MB)':>11} | {'Prefill(ms)':>11} | {'Avg TTFT(ms)':>12} | {'P95 TTFT(ms)':>12} | {'Avg TPOT(ms)':>12} | {'P95 TPOT(ms)':>12}"
    print(header)
    print("-" * 120)

    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        
        # Accuracy Calculation
        avg_sim = np.mean(m["edit_sim"]) * 100 
        
        # System Telemetry Aggregations
        avg_payload = np.mean(m["payload"])
        avg_prefill = np.mean(m["prefill"])
        avg_ttft = np.mean(m["ttft"])
        p95_ttft = np.percentile(m["ttft"], 95) if m["ttft"] else 0.0
        avg_tpot = np.mean(m["tpot"])
        p95_tpot = np.percentile(m["tpot"], 95) if m["tpot"] else 0.0
        
        row = f"{method:<16} | {avg_sim:>12.2f} | {avg_payload:>11.1f} | {avg_prefill:>11.1f} | {avg_ttft:>12.1f} | {p95_ttft:>12.1f} | {avg_tpot:>12.1f} | {p95_tpot:>12.1f}"
        print(row)
    
    print("=" * 120)
    print(f"📝 Output saved to {checkpoint_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67")
    parser.add_argument('--port', type=str, default="5555")
    parser.add_argument('--retain_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--max_new_tokens', type=int, default=64) 
    parser.add_argument('--num_samples', type=int, default=100)
    
    parser.add_argument('--dataset', type=str, default="../data/benchmarks/longbench/lcc.jsonl") 
    args = parser.parse_args()
    
    run_longbench_lcc(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)
