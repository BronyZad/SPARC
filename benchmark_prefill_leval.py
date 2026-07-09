"""
Saber Prefill Server (Master) - LEval Benchmark
Features: Dual-GPU Extended Context, Uniform-INT4 Baseline, Dynamic Prompt Routing.
Updates: Parametric logging, unique output names, Deterministic Crash Recovery, and MCQ Formatting.
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
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from saber_core_transport import SaberDisaggregatedEngine

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "../local_models/Qwen3-4B-Instruct-2507"
MAX_SEQ_LEN = 17000  # 🟢 Increased to 25k for Dual RTX 3090s

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
    raise TimeoutError("Decode Server crashed during generation (Timeout waiting for final text).")

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

def load_leval_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LEval dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def build_prompt(transcript, instruction, tokenizer, task_type="summarization"):
    """
    Dynamically routes the prompt structure based on the LEval task type.
    Crucially, it uses the Chat Template to force the model into Instruction-Following mode.
    """
    if task_type == "summarization":
        sys_msg = "You are an expert summarization assistant. Provide a comprehensive summary based strictly on the provided document."
        user_msg = f"Document:\n{transcript}\n\nTask:\n{instruction}"
        
    elif task_type == "retrieval":
        sys_msg = "You are a precise legal and technical extraction assistant. Extract the specific information requested from the document. Do not hallucinate."
        user_msg = f"Document:\n{transcript}\n\nTask:\n{instruction}"
        
    elif task_type == "qa" or task_type == "exam":
        # 🟢 CRITICAL FIX FOR MCQ AND EXAMS
        sys_msg = "You are an expert answering questions based on the provided text. If the question is multiple choice, output ONLY the correct letter(s) (e.g., 'A', 'C', 'BD'). Do not provide explanations."
        user_msg = f"Document:\n{transcript}\n\nQuestion:\n{instruction}\n\nAnswer:"
        
    else: 
        sys_msg = "You are a helpful assistant. Answer the question based on the provided document."
        user_msg = f"Document:\n{transcript}\n\nQuestion:\n{instruction}"

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]
    
    # Let Qwen's native tokenizer handle the <|im_start|> tags
    encodings = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = encodings if isinstance(encodings, torch.Tensor) else encodings["input_ids"]
    return input_ids

def run_leval_benchmark(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path):
    leval_full_path = os.path.join("../data/benchmarks/LEval", dataset_path)
    
    # Encode the retain_ratio directly into the filename to prevent collision during sweeps
    checkpoint_file = f"saber_checkpoint_r{retain_ratio}_{dataset_path.split('/')[-1]}"

    print(f"🚀 Loading Qwen-4B on Dual-GPU Prefill Server...")
    print(f"⚙️ Configuration: [Saber Retain Ratio: {retain_ratio:.2f}] | [Max Context: {MAX_SEQ_LEN} tokens]")
    
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
    
    # 🟢 DYNAMIC TASK ROUTING FIX
    task_type = "summarization"
    lower_path = dataset_path.lower()
    if "retrieval" in lower_path: 
        task_type = "retrieval"
    elif "exam" in lower_path or "_qa" in lower_path or "codeu" in lower_path or "coursera" in lower_path or "quality" in lower_path: 
        task_type = "qa"

    print(f"📚 Loading LEval dataset: {dataset_path} [Task: {task_type.upper()}]...")
    test_pool = load_leval_dataset(leval_full_path)
        
    print(f"🔥 Warming up CUDA engine...")
    dummy_ids = torch.ones((1, 100), dtype=torch.long, device=model.device)
    with torch.no_grad(): _ = model(input_ids=dummy_ids, attention_mask=dummy_ids)
    purge(model)

    methods = ["Native-Baseline", "Uniform-INT4", "ablation_inverted", "Saber-BIC", "SnapKV"]
    metrics = {m: {"total": 0, "prefill": [], "payload": [], "ttft": [], "tpot": []} for m in methods}
    
    processed_ids = set()
    processed_indices = set()

    if os.path.exists(checkpoint_file):
        print(f"\n📂 Found existing checkpoint '{checkpoint_file}'. Loading previous results...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                rec_id = record.get("id", "")
                processed_ids.add(rec_id)
                
                if str(rec_id).startswith("doc_"):
                    parts = str(rec_id).split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        processed_indices.add(int(parts[1]))
                
                for method, data in record["results"].items():
                    if data["success"]:
                        metrics[method]["total"] += 1
                        metrics[method]["prefill"].append(data["prefill_time"] + data.get("saber_time", 0))
                        metrics[method]["payload"].append(data["payload_mb"])
                        metrics[method]["ttft"].append(data["ttft"])
                        if data["tpot"] > 0: metrics[method]["tpot"].append(data["tpot"])
                        
        print(f"✅ Resuming after {len(processed_ids)} previously completed documents.")

    print(f"\n🧪 STARTING LEVAL BENCHMARK | Target Valid Samples: {'All' if num_samples <= 0 else num_samples}")
    print("=" * 100)

    valid_samples_count = 0

    with open(checkpoint_file, "a", encoding="utf-8") as ckpt_file:
        for i, test_data in enumerate(test_pool):
            
            if num_samples > 0 and valid_samples_count >= num_samples:
                print(f"\n✅ Reached target of {num_samples} valid samples. Stopping benchmark.")
                break

            transcript = test_data.get("input", "")
            instructions = test_data.get("instructions", [])
            outputs = test_data.get("outputs", [])
            
            if not instructions: continue
            
            instruction = instructions[0]
            reference = outputs[0] if outputs else ""
            
            instruction_hash = hashlib.md5(str(instruction).encode('utf-8')).hexdigest()[:8]
            doc_id = test_data.get("id", f"doc_{i}_{instruction_hash}")
            
            if doc_id in processed_ids or i in processed_indices: 
                valid_samples_count += 1
                continue
            
            input_ids = build_prompt(transcript, instruction, tokenizer, task_type)
            actual_seq_len = input_ids.shape[1]
            
            if actual_seq_len > MAX_SEQ_LEN:
                print(f"\n▶️ Skipping Document {i+1} (Length {actual_seq_len} exceeds {MAX_SEQ_LEN} token limit)")
                continue
                
            valid_samples_count += 1
            input_ids = input_ids.expand(batch_size, -1).to(model.device)

            print(f"\n▶️ Processing Document {i+1}/{len(test_pool)} (Length: {actual_seq_len} tokens) | Progress: {valid_samples_count}/{num_samples if num_samples > 0 else 'All'}...")

            record = {
                "id": doc_id,
                "seq_len": actual_seq_len,
                "reference": reference,
                "instruction": instruction,   # 🟢 ADD THIS LINE
                "results": {}
            }

            for method in methods:
                generator = None
                chunk = None
                
                internal_method = "baseline" if method == "Native-Baseline" else method.lower().replace("-", "_")
                try:
                    payload_mb = 0.0
                    prefill_time = 0.0
                    saber_time = 0.0
                    
                    envelope = pickle.dumps({"method": method, "input_ids": input_ids[:, -1:].cpu(), "max_new_tokens": max_new_tokens})
                    socket.send(envelope)
                    recv_with_timeout(socket, 5000) 
                    
                    t0_net = time.perf_counter()

                    generator = engine.prefill_and_stream(input_ids[:, :-1], method=internal_method)

                    reply = None
                    for chunk in generator:
                        if chunk["type"] == "metadata" and "stats" in chunk:
                            prefill_time = chunk["stats"]["prefill_time_ms"]
                            saber_time += chunk["stats"].get("routing_time_ms", 0.0)
                        if chunk["type"] == "done" and "quant_time_ms" in chunk:
                            saber_time += chunk["quant_time_ms"]
                            
                        chunk_bytes = pickle.dumps(chunk)
                        payload_mb += len(chunk_bytes) / 1024 / 1024
                        socket.send(chunk_bytes)
                        
                        if chunk["type"] != "done":
                            recv_with_timeout(socket, 5000) 
                        else:
                            reply = recv_pyobj_with_timeout(socket, 120000) 
                    
                    net_total_time = (time.perf_counter() - t0_net) * 1000
                    
                    if reply and reply.get("status") == "success":
                        decode_time = reply.get('decode_ms', 0.0)
                        ttft = net_total_time - decode_time
                        tpot = reply.get('tpot_ms', 0.0)
                        generated_text = reply.get('text', '')
                        
                        metrics[method]["total"] += 1
                        metrics[method]["prefill"].append(prefill_time + saber_time)
                        metrics[method]["payload"].append(payload_mb)
                        metrics[method]["ttft"].append(ttft)
                        if tpot > 0: metrics[method]["tpot"].append(tpot)
                        
                        record["results"][method] = {
                            "success": True,
                            "prefill_time": prefill_time,
                            "saber_time": saber_time,
                            "payload_mb": payload_mb,
                            "ttft": ttft,
                            "tpot": tpot,
                            "generated_text": generated_text
                        }
                        
                        display_pred = (generated_text[:40].replace('\n', ' ') + '...') if len(generated_text) > 40 else generated_text.replace('\n', ' ')
                        print(f"  └─ {method:<16} | TTFT: {ttft:>6.1f}ms | TPOT: {tpot:>5.1f}ms | Payload: {payload_mb:>5.1f}MB | Pred: {display_pred}")
                    else:
                        error_reason = reply.get("message", "UNKNOWN") if reply else "NO_REPLY"
                        print(f"  └─ {method:<16} | ⚠️ DECODE SERVER ERROR: {error_reason}")
                        record["results"][method] = {"success": False, "error": error_reason}
                        socket = reset_zmq_socket(context, socket, ip, port)

                except TimeoutError as e:
                    print(f"  └─ {method:<16} | ❌ REMOTE CRASH: {e}")
                    record["results"][method] = {"success": False, "error": str(e)}
                    socket = reset_zmq_socket(context, socket, ip, port)
                    del e
                    
                except RuntimeError as e:
                    error_msg = "OOM CRASH" if "CUDA out of memory" in str(e) else str(e)
                    print(f"  └─ {method:<16} | ❌ LOCAL CRASH: {error_msg}")
                    record["results"][method] = {"success": False, "error": error_msg}
                    socket = reset_zmq_socket(context, socket, ip, port)
                    del e
                    
                except Exception as e:
                    print(f"  └─ {method:<16} | ❌ UNEXPECTED ERROR: {e}")
                    record["results"][method] = {"success": False, "error": str(e)}
                    socket = reset_zmq_socket(context, socket, ip, port)
                    del e
                    
                finally:
                    del generator, chunk
                    purge(model)

            ckpt_file.write(json.dumps(record) + "\n")
            ckpt_file.flush()
            os.fsync(ckpt_file.fileno())

    total_evaluated = metrics[methods[0]]["total"] if methods else 0
    print("\n" + "=" * 100)
    print(f"📊 LEVAL SYSTEMS BENCHMARK REPORT (Total Valid Samples Evaluated: {total_evaluated})")
    print("=" * 100)
    header = f"{'Method':<16} | {'Payload(MB)':<12} | {'Prefill(ms)':<12} | {'Avg TTFT(ms)':<13} | {'Avg TPOT(ms)':<13}"
    print(header)
    print("-" * 100)

    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        avg_payload = np.mean(m["payload"])
        avg_prefill = np.mean(m["prefill"])
        avg_ttft = np.mean(m["ttft"])
        avg_tpot = np.mean(m["tpot"]) if m["tpot"] else 0
        print(f"{method:<16} | {avg_payload:>11.1f} | {avg_prefill:>11.1f} | {avg_ttft:>12.1f} | {avg_tpot:>12.1f}")
    
    print("=" * 100)
    print(f"📝 Output saved to {checkpoint_file}. Ready for offline LLM-as-a-judge evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67", help="IP of Decode Server")
    parser.add_argument('--port', type=str, default="5555", help="Port of Decode Server")
    parser.add_argument('--retain_ratio', type=float, default=0.10, help="Ratio of tokens to retain (0-1)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Tokens to generate for summary")
    parser.add_argument('--num_samples', type=int, default=-1, help="Number of valid questions to test (-1 for all)")
    parser.add_argument('--dataset', type=str, default="Exam/coursera.jsonl", help="Relative path to LEval dataset")
    args = parser.parse_args()
    
    run_leval_benchmark(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset)