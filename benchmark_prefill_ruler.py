"""
Saber Prefill Server - RULER Benchmark (Needle In A Haystack)
Targets: Extreme Context Retrieval & Entropy Routing Validation.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the Saber Engine
from sparc_core_transport import SaberDisaggregatedEngine

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
    Checks if any of the valid answers (needles) are present in the generation.
    """
    pred_clean = prediction.strip().lower()
    for ans in expected_answers:
        if str(ans).lower() in pred_clean:
            return 1.0  # Exact Match Found!
    return 0.0

def run_ruler(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
    checkpoint_file = f"sparc_checkpoint_r{retain_ratio}_ruler.jsonl"

    print(f"🚀 Loading Model on Dual-GPU Prefill Server (RULER Mode)...")
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

    methods = ["Native-Baseline", "Uniform-INT4", "Saber-Q", "Saber-CQ"]
    metrics = {m: {"total": 0, "em": []} for m in methods}
    
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
                        
        print(f"✅ Resuming after {len(processed_ids)} previously evaluated samples.")

    print(f"\n🧪 STARTING RULER BENCHMARK | Target Valid Samples: {'All' if num_samples <= 0 else num_samples}")
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
                answer_prefix = test_data.get('answer_prefix', '')
                expected_answers = test_data.get('answer', []) # This is a list in RULER
                
                if not expected_answers: continue

                # 🟢 PROMPT FIX: The Ultimate Instruct Hack
                # We apply the chat template, but inject the prefix at the very end to force the number out.
                system_instruction = "You are a highly precise information extraction engine. Extract the exact value requested."
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"{context_str}\n\n{question_str}"}
                ]
                
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_text += f"{answer_prefix} " # Append prefix inside the assistant turn!
                
                hash_str = hashlib.md5((prompt_text[-100:] + str(expected_answers)).encode('utf-8')).hexdigest()[:8]
                doc_id = f"ruler_{i}_{hash_str}"
                
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

                print(f"\n▶️ Evaluating {test_data.get('task', 'NIAH')} {i+1}/{len(test_pool)} (Length: {actual_seq_len} tokens) | Progress: {valid_samples_count}/{num_samples}...")

                record = {
                    "id": doc_id,
                    "task": test_data.get('task', 'unknown'),
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
                        envelope = pickle.dumps({"method": method, "input_ids": input_ids[:, -1:].cpu(), "max_new_tokens": max_new_tokens})
                        socket.send(envelope)
                        recv_with_timeout(socket, 5000) 
                        
                        generator = engine.prefill_and_stream(input_ids[:, :-1], method=internal_method)

                        reply = None
                        for chunk in generator:
                            chunk_bytes = pickle.dumps(chunk)
                            socket.send(chunk_bytes)
                            if chunk["type"] != "done":
                                recv_with_timeout(socket, 5000) 
                            else:
                                reply = recv_pyobj_with_timeout(socket, 120000) 
                        
                        if reply and reply.get("status") == "success":
                            generated_text = reply.get('text', '')
                            em_score = evaluate_ruler(generated_text, expected_answers)
                            
                            metrics[method]["total"] += 1
                            metrics[method]["em"].append(em_score)
                            
                            record["results"][method] = {
                                "success": True,
                                "generated_text": generated_text,
                                "em_score": em_score
                            }
                            
                            icon = "🟢" if em_score == 1.0 else "🔴"
                            display_pred = (generated_text[:30].replace('\n', ' ') + '...') if len(generated_text) > 30 else generated_text.replace('\n', ' ')
                            print(f"  └─ {method:<16} | {icon} EM: {int(em_score)} | Pred: {display_pred}")
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
    print("\n" + "=" * 60)
    print(f"📊 RULER FINAL REPORT (Samples Evaluated: {total_evaluated})")
    print("=" * 60)
    header = f"{'Method':<16} | {'Retrieval Accuracy (EM)':<20}"
    print(header)
    print("-" * 60)

    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        avg_em = np.mean(m["em"]) * 100 
        print(f"{method:<16} | {avg_em:>6.2f}%")
    
    print("=" * 60)
    print(f"📝 Output saved to {checkpoint_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67")
    parser.add_argument('--port', type=str, default="5555")
    parser.add_argument('--retain_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=32) 
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="../data/benchmarks/RULER/ruler_test_16k.jsonl")
    args = parser.parse_args()
    
    run_ruler(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)