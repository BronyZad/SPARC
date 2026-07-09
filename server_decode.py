"""
Saber Decode Server (Worker) - Instrumented Version
Uses Pipelined Streaming to seamlessly receive and reconstruct the mixed-precision KV cache.
Features real-time layer unpacking, accurate TPOT/TTFT tracking, and Logit Divergence Profiling.
"""
import torch
import zmq
import time
import pickle
import gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache

from saber_core_transport import SaberDisaggregatedEngine

MODEL_PATH = "../local_models/Qwen2.5-Coder-7B-Instruct"
PORT = "5555"

class TTFTTracker(LogitsProcessor):
    def __init__(self, start_time):
        self.start_time = start_time
        self.ttft = 0.0
        self.is_first = True

    def __call__(self, input_ids, scores):
        if self.is_first:
            self.ttft = (time.perf_counter() - self.start_time) * 1000
            self.is_first = False
        return scores

# 🟢 NEW: Collects token-by-token string mapping and confidence scores from raw logits
class DivergenceTelemetryProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_track=48):
        self.tokenizer = tokenizer
        self.max_track = max_track
        self.profile = []

    def __call__(self, input_ids, scores):
        if len(self.profile) < self.max_track:
            with torch.no_grad():
                # scores shape: [batch, vocab_size]
                probs = F.softmax(scores[0], dim=-1)
                top_prob, top_idx = torch.max(probs, dim=-1)
                
                token_str = self.tokenizer.decode([top_idx.item()], skip_special_tokens=False)
                self.profile.append({
                    "token": token_str,
                    "prob": float(top_prob.item())
                })
        return scores

def purge():
    gc.collect()
    torch.cuda.empty_cache()

def start_server():
    print(f"🚀 Loading Qwen-Coder on Decode Server...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="sdpa"
    )
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"🎧 Decode Worker listening on port {PORT}...")

    while True:
        envelope_bytes = socket.recv()
        envelope = pickle.loads(envelope_bytes)
        
        if "method" not in envelope:
            print("⚠️ Out of sync: Expected Envelope. Sending error.")
            socket.send_pyobj({"status": "error", "message": "Expected stream Envelope."})
            continue
            
        method = envelope["method"]
        input_ids = envelope["input_ids"].to(model.device)
        max_new = envelope["max_new_tokens"]

        socket.send(b"ACK") 
        print(f"\n📦 [{method}] Receiving cache stream...")

        out = None
        cache = None

        try:
            t0_load = time.perf_counter()
            cache = DynamicCache(config=model.config)
            seq_len = 0
            payload_mb = 0.0

            while True:
                chunk_bytes = socket.recv()
                chunk = pickle.loads(chunk_bytes)
                
                if "method" in chunk:
                    print(f"⚠️ Prefill server aborted previous stream. Resetting for [{chunk['method']}]")
                    del cache
                    purge()
                    
                    cache = DynamicCache(config=model.config)
                    seq_len = 0
                    payload_mb = 0.0
                    t0_load = time.perf_counter()
                    
                    method = chunk["method"]
                    input_ids = chunk["input_ids"].to(model.device)
                    max_new = chunk["max_new_tokens"]
                    
                    socket.send(b"ACK")
                    continue
                
                payload_mb += len(chunk_bytes) / 1024 / 1024
                
                if chunk.get("type") == "metadata":
                    if "seq_len" in chunk:
                        seq_len = chunk["seq_len"]
                    socket.send(b"ACK")
                    
                elif chunk.get("type") == "layer":
                    idx = chunk["layer_idx"]
                    layer = cache.layers[idx]
                    layer.is_initialized = True
                    
                    if method == "Native-Baseline":
                        k = chunk["k"].to(model.device).contiguous()
                        v = chunk["v"].to(model.device).contiguous()
                        layer.keys = k
                        layer.values = v
                        seq_len = k.shape[-2]
                        del k, v
                    else:
                        k_full, v_full = SaberDisaggregatedEngine.reconstruct_cache_layer(chunk, model.device)
                        layer.keys, layer.values = k_full, v_full
                        del k_full, v_full

                    if hasattr(layer, "cumulative_length"): 
                        layer.cumulative_length = seq_len
                        
                    socket.send(b"ACK")
                    
                elif chunk.get("type") == "done":
                    break

            load_time = (time.perf_counter() - t0_load) * 1000
            print(f"🔓 Stream Complete | Payload: {payload_mb:.1f} MB | Restored Tokens: {seq_len}")

            t0_gen = time.perf_counter()
            new_attention_mask = torch.ones((input_ids.shape[0], seq_len + 1), dtype=torch.long, device=model.device)
            print("Start to generate...")
            
            ttft_tracker = TTFTTracker(start_time=t0_load)
            # 🟢 NEW: Instantiate logit tracker for this generation run
            divergence_tracker = DivergenceTelemetryProcessor(tokenizer, max_track=48)
            
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=new_attention_mask,
                    past_key_values=cache,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False, # Enforces strict greedy tracking matches scores index
                    use_cache=True,
                    logits_processor=LogitsProcessorList([ttft_tracker, divergence_tracker])
                )
            print("Generation done.")
            decode_time = (time.perf_counter() - t0_gen) * 1000
            
            num_generated = out.shape[1] - input_ids.shape[1]
            tpot = decode_time / num_generated if num_generated > 0 else 0.0
            ttft_ms = ttft_tracker.ttft
            
            generated_ids = out[:, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True) 
            
            print(f"✅ Success | TTFT: {ttft_ms:.1f}ms | TPOT: {tpot:.1f}ms")

            reply = {
                "status": "success",
                "decode_ms": decode_time,
                "ttft_ms": ttft_ms,          
                "num_tokens": num_generated, 
                "tpot_ms": tpot,             
                "text": generated_text,
                "divergence_profile": divergence_tracker.profile # 🟢 NEW: Shipped over ZMQ
            }
            
        except Exception as e:
            print(f"⚠️ Decode Error: {e}")
            reply = {"status": "error", "message": str(e)}

        socket.send_pyobj(reply)
        
        if 'cache' in locals(): del cache
        if 'out' in locals(): del out
        if 'new_attention_mask' in locals(): del new_attention_mask
        purge()

if __name__ == "__main__":
    start_server()