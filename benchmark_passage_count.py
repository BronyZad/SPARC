"""
Saber Prefill Server - LongBench (Passage Count)
Targets: Pure Structural Aggregation & Exact Match Strictness.
Optimized for Reasoning/Thinking Models (Qwen3-8B)
"""
import torch
import zmq
import time
import pickle
import argparse
import gc
import json
import os
import re
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the Saber Engine
from saber_core_transport import SaberDisaggregatedEngine

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "../local_models/Qwen3-8B" 
MAX_SEQ_LEN = 12000
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

def recv_pyobj_with_timeout(socket, timeout_ms=240000):
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

def evaluate_passage_count(prediction, expected_answers):
    """
    Extracts the answer by explicitly ignoring the Chain-of-Thought <think> block.
    """
    # 🟢 THINKING MODEL PARSER
    # Isolate the final answer from the reasoning trace
    if "</think>" in prediction:
        # Take everything AFTER the closing think tag
        final_answer = prediction.split("</think>")[-1]
    else:
        # Fallback if the model didn't use tags or we hit the token limit
        final_answer = prediction
    
 
    # Search for the first integer in the FINAL answer block
    match = re.search(r'\d+', final_answer)
    if not match: return 0.0
    
    pred_num = str(int(match.group(0))) # Normalize to strip leading zeros
    
    for ans in expected_answers:
        ans_num = str(int(ans))
        if pred_num == ans_num:
            return 1.0
            
    return 0.0

def run_longbench_count(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
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
                doc_id = test_data.get('_id', f"lb_count_{i}")

                if not expected_answers or not haystack_context:
                    continue

                # 🟢 REASONING MODEL PROMPT
                # No restrictive system prompt. Just the text and a direct question.
                user_prompt = f"Text:\n{haystack_context}\n\nPlease count the exact number of independent paragraphs in the text above. How many are there?"
                
                messages = [
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
                            em_score = evaluate_passage_count(generated_text, expected_answers)
                            
                            metrics[method]["total"] += 1
                            metrics[method]["em"].append(em_score)
                            
                            record["results"][method] = {
                                "success": True,
                                "generated_text": generated_text,
                                "em_score": em_score
                            }
                            
                            icon = "🟢" if em_score == 1.0 else "🔴"
                            # Clean up display text so we can see what the model actually decided
                            display_text = generated_text.split('</think>')[-1].strip() if '</think>' in generated_text else generated_text.strip()
                            display_pred = (display_text[:40].replace('\n', ' ') + '...') if len(display_text) > 40 else display_text.replace('\n', ' ')
                            
                            print(f"  └─ {method:<16} | {icon} Exact Match | Expected: {expected_answers[0]} | Pred: {display_pred}")
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
    print(f"📊 {task_name.upper()} FINAL REPORT (Samples Evaluated: {total_evaluated})")
    print("=" * 60)
    header = f"{'Method':<16} | {'Exact Match Accuracy':<20}"
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
    
    # 🟢 Massive token budget for Chain of Thought
    parser.add_argument('--max_new_tokens', type=int, default=2048) 
    parser.add_argument('--num_samples', type=int, default=100)
    
    parser.add_argument('--dataset', type=str, default="../data/benchmarks/longbench/passage_count.jsonl") 
    args = parser.parse_args()
    
    run_longbench_count(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)