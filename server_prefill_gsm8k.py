"""
Saber Prefill Server (Master) - Unbiased GSM8K Benchmark Suite
"""
import torch
import zmq
import time
import io
import pickle
import argparse
import gc
import json
import os
import random
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparc_core_transport import SaberDisaggregatedEngine

MODEL_PATH = "../local_models/Qwen3-4B-Instruct-2507"
GSM8K_PATH = "../data/benchmarks/gsm8k/test.jsonl"
CHECKPOINT_FILE = "sparc_benchmark_checkpoint.jsonl"

def extract_answer(text: str) -> str:
    if "####" in text:
        ans_block = text.split("####")[1]
        numbers = re.findall(r'-?\d+\.?\d*', ans_block.replace(",", ""))
        return numbers[0] if numbers else ""
    
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    return numbers[-1] if numbers else ""

def purge(model=None):
    if model is not None and hasattr(model, "base_model"):
        model.base_model._past_key_values = None
    gc.collect()
    torch.cuda.empty_cache()

def build_static_few_shot_string(path, target_seq_len, tokenizer):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GSM8K dataset not found at {path}")
    
    with open(path, "r", encoding="utf-8") as f: 
        lines = f.readlines()
        
    random.seed(42)
    random.shuffle(lines)
    all_data = [json.loads(line) for line in lines]
    
    few_shot_pool = all_data[:40] 
    test_pool = all_data[40:] 

    def get_token_count(text):
        msgs = [{"role": "user", "content": text}]
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=False, return_tensors="pt")
        return enc.shape[1] if isinstance(enc, torch.Tensor) else enc["input_ids"].shape[1]

    few_shot_string = ""
    current_tokens = 150 
    
    for data in few_shot_pool:
        addition = f"Example Question: {data['question']}\nExample Answer: Let's think step by step.\n{data['answer']}\n\n"
        pair_tokens = get_token_count(addition)
        
        if current_tokens + pair_tokens > (target_seq_len - 150):
            break
            
        few_shot_string += addition
        current_tokens += pair_tokens
        
    return few_shot_string, test_pool

def run_benchmark(ip, port, batch_size, seq_len, max_new_tokens, num_samples):
    print(f"🚀 Loading Qwen-4B on Prefill Server...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="sdpa"
    )
    
    engine = SaberDisaggregatedEngine(model, retain_ratio=0.10, causal_depth=3)
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{ip}:{port}")
    
    print(f"📚 Building static GSM8K few-shot reference string...")
    few_shot_string, test_pool = build_static_few_shot_string(GSM8K_PATH, seq_len, tokenizer)
    
    if num_samples > 0:
        test_pool = test_pool[:num_samples]
        
    print(f"🔥 Warming up CUDA engine...")
    dummy_ids = torch.ones((1, 100), dtype=torch.long, device=model.device)
    with torch.no_grad(): _ = model(input_ids=dummy_ids, attention_mask=dummy_ids)
    purge(model)

    # 🟢 ADDED UNIFORM-INT4 BASELINE
    methods = ["Native-Baseline", "Uniform-INT4", "Saber-Q", "Saber-CQ"]
    
    metrics = {m: {"correct": 0, "total": 0, "prefill": [], "sparc": [], "payload": [], "ttft": [], "tpot": []} for m in methods}
    processed_questions = set()

    if os.path.exists(CHECKPOINT_FILE):
        print(f"\n📂 Found existing checkpoint '{CHECKPOINT_FILE}'. Loading previous results...")
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                processed_questions.add(record["question"])
                
                for method, data in record["results"].items():
                    if data["success"]:
                        metrics[method]["total"] += 1
                        if data["is_correct"]: metrics[method]["correct"] += 1
                        metrics[method]["prefill"].append(data["prefill_time"])
                        metrics[method]["sparc"].append(data["sparc_time"])
                        metrics[method]["payload"].append(data["payload_mb"])
                        metrics[method]["ttft"].append(data["ttft"])
                        if data["tpot"] > 0: metrics[method]["tpot"].append(data["tpot"])
                        
        print(f"✅ Resuming after {len(processed_questions)} previously completed questions.")

    print(f"\n🧪 STARTING UNBIASED GSM8K BENCHMARK | Total Pool: {len(test_pool)}")
    print("=" * 80)

    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as ckpt_file:
        for i, test_data in enumerate(test_pool):
            target_q = test_data['question']
            
            if target_q in processed_questions:
                continue

            print(f"\n▶️ Processing Question {i+1}/{len(test_pool)}...")
            gold_num = extract_answer(test_data['answer'])

            prompt_text = (
                f"Please solve the following math problem:\n\n"
                f"--- TARGET PROBLEM ---\n{target_q}\n----------------------\n\n"
                f"To help you, here are several reference examples of similar problems being solved:\n\n"
                f"{few_shot_string}"
                f"Now, please provide your step-by-step solution to the TARGET PROBLEM located at the top."
            )

            current_messages = [
                {"role": "system", "content": "You are a highly accurate math tutor. You must think step-by-step and end your answer with #### [Final Number]."},
                {"role": "user", "content": prompt_text}
            ]
            
            encodings = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True, return_tensors="pt")
            input_ids = encodings if isinstance(encodings, torch.Tensor) else encodings["input_ids"]
            input_ids = input_ids.expand(batch_size, -1).to(model.device)

            question_record = {
                "question": target_q,
                "gold_answer": gold_num,
                "results": {}
            }

            for method in methods:
                internal_method = "baseline" if method == "Native-Baseline" else method.lower().replace("-", "_")
                try:
                    payload_mb = 0.0
                    prefill_time = 0.0
                    sparc_time = 0.0
                    
                    # 1. INITIALIZE ENVELOPE
                    envelope = pickle.dumps({"method": method, "input_ids": input_ids[:, -1:].cpu(), "max_new_tokens": max_new_tokens})
                    socket.send(envelope)
                    socket.recv() # Wait for initial ACK
                    
                    t0_net = time.perf_counter()

                    # 2. RUN FORWARD PASS & GENERATE STREAM
                    generator = engine.prefill_and_stream(input_ids[:, :-1], method=internal_method)

                    # 3. STREAM CHUNKS LAYER-BY-LAYER
                    reply = None
                    for chunk in generator:
                        if chunk["type"] == "metadata" and "stats" in chunk:
                            prefill_time = chunk["stats"]["prefill_time_ms"]
                            sparc_time += chunk["stats"].get("routing_time_ms", 0.0)
                        if chunk["type"] == "done" and "quant_time_ms" in chunk:
                            sparc_time += chunk["quant_time_ms"]
                            
                        chunk_bytes = pickle.dumps(chunk)
                        payload_mb += len(chunk_bytes) / 1024 / 1024
                        socket.send(chunk_bytes)
                        
                        if chunk["type"] != "done":
                            socket.recv() # ACK layer
                        else:
                            reply = socket.recv_pyobj() # Get final answer
                            
                    net_total_time = (time.perf_counter() - t0_net) * 1000
                    
                    if reply and reply.get("status") == "success":
                        decode_time = reply.get('decode_ms', 0.0)
                        ttft = net_total_time - decode_time
                        tpot = reply.get('tpot_ms', 0.0)
                        
                        pred_num = extract_answer(reply.get('text', ''))
                        is_correct = (pred_num == gold_num) and bool(pred_num)
                        
                        metrics[method]["total"] += 1
                        if is_correct: metrics[method]["correct"] += 1
                        metrics[method]["prefill"].append(prefill_time)
                        metrics[method]["sparc"].append(sparc_time)
                        metrics[method]["payload"].append(payload_mb)
                        metrics[method]["ttft"].append(ttft)
                        if tpot > 0: metrics[method]["tpot"].append(tpot)
                        
                        question_record["results"][method] = {
                            "success": True,
                            "is_correct": is_correct,
                            "prefill_time": prefill_time,
                            "sparc_time": sparc_time,
                            "payload_mb": payload_mb,
                            "ttft": ttft,
                            "tpot": tpot
                        }
                        
                        correct_str = "✅" if is_correct else "❌"
                        print(f"  └─ {method:<16} | TTFT: {ttft:>6.1f}ms | Payload: {payload_mb:>5.1f}MB | {correct_str}")
                    else:
                        print(f"  └─ {method:<16} | ⚠️ DECODE SERVER FAILED")
                        question_record["results"][method] = {"success": False, "error": "DECODE FAILED"}
                        
                except RuntimeError as e:
                    error_msg = "OOM CRASH" if "CUDA out of memory" in str(e) else str(e)
                    print(f"  └─ {method:<16} | ❌ {error_msg}")
                    question_record["results"][method] = {"success": False, "error": error_msg}
                finally:
                    purge(model)

            ckpt_file.write(json.dumps(question_record) + "\n")
            ckpt_file.flush()
            os.fsync(ckpt_file.fileno())

    # =========================================================================
    # FINAL STATISTICAL REPORT
    # =========================================================================
    print("\n" + "=" * 115)
    print(f"📊 UNBIASED SABER INFERENCE BENCHMARK REPORT (Total Samples: {len(test_pool)})")
    print("=" * 115)
    header = f"{'Method':<16} | {'Acc (%)':<8} | {'Payload(MB)':<12} | {'Prefill(ms)':<12} | {'Avg TTFT(ms)':<13} | {'P95 TTFT(ms)':<13} | {'Avg TPOT(ms)':<13} | {'P95 TPOT(ms)':<13}"
    print(header)
    print("-" * 115)

    for method in methods:
        m = metrics[method]
        total = max(1, m["total"])
        
        acc = (m["correct"] / total) * 100
        avg_payload = np.mean(m["payload"]) if m["payload"] else 0
        avg_prefill = np.mean(m["prefill"]) + np.mean(m["sparc"]) if m["prefill"] else 0
        
        avg_ttft = np.mean(m["ttft"]) if m["ttft"] else 0
        p95_ttft = np.percentile(m["ttft"], 95) if m["ttft"] else 0
        
        avg_tpot = np.mean(m["tpot"]) if m["tpot"] else 0
        p95_tpot = np.percentile(m["tpot"], 95) if m["tpot"] else 0
        
        print(f"{method:<16} | {acc:>7.1f}% | {avg_payload:>11.1f} | {avg_prefill:>11.1f} | {avg_ttft:>12.1f} | {p95_ttft:>12.1f} | {avg_tpot:>12.1f} | {p95_tpot:>12.1f}")
    
    print("=" * 115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67", help="IP of Decode Server")
    parser.add_argument('--port', type=str, default="5555", help="Port of Decode Server")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--seq_len', type=int, default=2048, help="Prompt length")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Tokens to generate")
    parser.add_argument('--num_samples', type=int, default=-1, help="Number of questions to test (-1 for all)")
    args = parser.parse_args()
    
    run_benchmark(args.ip, args.port, args.batch_size, args.seq_len, args.max_new_tokens, args.num_samples)
