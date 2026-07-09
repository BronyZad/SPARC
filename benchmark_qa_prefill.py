"""
Sparc Prefill Server - General QA Retrieval (Ultimate Deadlock-Proof Edition)
Features: 
- Dynamic Monkey Patch for Head-Step OOM prevention 
- Sparc-BIC INT4 payload with CUDA Spinlock Protection
- SnapKV Empty Shape Bug Fixed
- Extreme 10-Min Timeout for Swapping
- Print Flush enforced to bypass buffer freezing
"""
import torch
import torch.nn.functional as F
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
import string
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparc_core_transport import SparcDisaggregatedEngine

# ==============================================================================
# 🟢 终极魔法注入 (Monkey Patching 2.0 + Chunked Prefill + Mask Fix)
# ==============================================================================
def patched_prefill_and_stream(self, input_ids, method="sparc_bic", chunk_size=8196):
    seq_len = input_ids.shape[1]
    num_layers_total = len(self.model.model.layers)
    target_telemetry_budget = int(seq_len * self.retain_ratio * num_layers_total)
    
    torch.cuda.synchronize()
    start_prefill = time.perf_counter()

    def chunked_forward(model, input_ids, chunk_size):
        seq_len = input_ids.shape[1]
        past_key_values = None
        last_out = None
        for i in range(0, seq_len, chunk_size):
            chunk = input_ids[:, i : min(i + chunk_size, seq_len)]
            out = model(chunk, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            last_out = out
            del out
            torch.cuda.empty_cache()
        return last_out

    if method in ["baseline", "uniform_int4"]:
        with torch.inference_mode():
            out = chunked_forward(self.model.model, input_ids, chunk_size)
            prefill_cache = out.past_key_values
        
        prefill_time_ms = (time.perf_counter() - start_prefill) * 1000
        stats = {"prefill_time_ms": prefill_time_ms, "routing_time_ms": 0.0, "compression_ratio": 1.0}
        layers = prefill_cache.layers if hasattr(prefill_cache, "layers") else prefill_cache
        
        if method == "baseline":
            yield {"type": "metadata", "seq_len": seq_len, "num_layers": len(layers), "stats": stats}
            torch.cuda.synchronize()
            start_quant = time.perf_counter()
            for i, layer in enumerate(layers):
                k = layer.keys if hasattr(layer, "keys") else layer[0]
                v = layer.values if hasattr(layer, "values") else layer[1]
                yield {"type": "layer", "layer_idx": i, "k": k.cpu(), "v": v.cpu()}
                del k, v
                torch.cuda.empty_cache()
            yield {"type": "done", "quant_time_ms": (time.perf_counter() - start_quant) * 1000}
            return
        layer_variance_stats = {"min": 0, "max": 0, "std": 0.0, "mean": 0.0, "topology": []}
        routing_time_ms = 0.0
    else:
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        num_layers = len(self.model.model.layers)
        global_device = input_ids.device
        layer_flat_mass = torch.zeros((num_layers, seq_len), dtype=torch.float32, device=global_device)
        layer_value_norms = torch.zeros((num_layers, seq_len), dtype=torch.float32, device=global_device)
            
        def patch_attention(module, layer_idx):
            original_forward = module.forward
            def shadow_forward(*args, **kwargs):
                original_sdpa = F.scaled_dot_product_attention
                def sdpa_interceptor(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **sdpa_kwargs):
                    with torch.no_grad():
                        q_orig, k_orig, v_orig = q[0], k[0], v[0]
                        layer_device = q_orig.device
                        compute_device = torch.device(f"cuda:{(layer_device.index + 1) % torch.cuda.device_count()}") if torch.cuda.device_count() > 1 else layer_device
                            
                        q_comp = q_orig.to(compute_device, non_blocking=True)
                        k_comp = k_orig.to(compute_device, non_blocking=True)
                        if q_comp.shape[0] != k_comp.shape[0]:
                            k_comp = k_comp.repeat_interleave(q_comp.shape[0] // k_comp.shape[0], dim=0)
                            
                        scale = 1.0 / np.sqrt(q_comp.shape[-1])
                        q_len, k_len = q_comp.shape[1], k_comp.shape[1]
                        num_heads = q_comp.shape[0]
                        
                        chunk_sum_total = torch.zeros((q_len, k_len), dtype=torch.float32, device=compute_device)
                        head_step = 8
                        
                        for h in range(0, num_heads, head_step):
                            end_h = min(h + head_step, num_heads)
                            attn = torch.matmul(q_comp[h:end_h], k_comp[h:end_h].transpose(-2, -1)) * scale
                            causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool, device=compute_device), diagonal=k_len - q_len)
                            attn.masked_fill_(~causal_mask, float('-inf'))
                            attn_probs = F.softmax(attn, dim=-1)
                            chunk_sum_total.add_(attn_probs.sum(dim=0))
                            del attn, causal_mask, attn_probs
                            
                        obs_window = min(32, q_len)
                        layer_flat_mass[layer_idx, :k_len].add_(chunk_sum_total[-obs_window:, :].sum(dim=0).to(global_device))
                        v_norm_tokens = torch.norm(v_orig.float(), p=2, dim=-1).mean(dim=0)
                        layer_value_norms[layer_idx, :v_norm_tokens.shape[0]].add_(v_norm_tokens.to(global_device))
                        del q_comp, k_comp, chunk_sum_total
                        
                    return original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **sdpa_kwargs)
                F.scaled_dot_product_attention = sdpa_interceptor
                output = original_forward(*args, **kwargs)
                F.scaled_dot_product_attention = original_sdpa 
                return output
            module.forward = shadow_forward
            return original_forward 

        original_forwards = {i: patch_attention(layer.self_attn, i) for i, layer in enumerate(self.model.model.layers)}
        
        with torch.inference_mode():
            out = chunked_forward(self.model.model, input_ids, chunk_size)
            prefill_cache = out.past_key_values
            
        for i, layer in enumerate(self.model.model.layers): 
            layer.self_attn.forward = original_forwards[i]

        prefill_time_ms = (time.perf_counter() - start_prefill) * 1000
        torch.cuda.synchronize()
        start_routing = time.perf_counter()
        total_global_budget = int(seq_len * self.retain_ratio * num_layers)
        
        if method == "snapkv":
            iso_ratio = 0.625 * self.retain_ratio + 0.375
            total_iso_budget = int(seq_len * iso_ratio * num_layers)
            layer_budgets = torch.full((num_layers,), total_iso_budget // num_layers, dtype=torch.long, device=global_device)
            target_telemetry_budget = total_iso_budget
        elif method in ["sparc_bic", "sparc_cf", "ablation_inverted"]:
            layer_budgets = torch.full((num_layers,), total_global_budget // num_layers, dtype=torch.long, device=global_device)
            target_telemetry_budget = total_global_budget

        layer_variance_stats = {"min": 0, "max": 0, "std": 0.0, "mean": 0.0, "topology": []}
        lb_cpu = layer_budgets.float().cpu()
        layer_variance_stats['min'] = lb_cpu.min().item()
        layer_variance_stats['max'] = lb_cpu.max().item()
        layer_variance_stats['std'] = lb_cpu.std().item() if num_layers > 1 else 0.0
        layer_variance_stats['mean'] = lb_cpu.mean().item()
        torch.cuda.synchronize()
        routing_time_ms = (time.perf_counter() - start_routing) * 1000

    stats = {"prefill_time_ms": prefill_time_ms, "routing_time_ms": routing_time_ms}
    # 这就是你看到 Decode 端打印 Receiving cache stream 的原因！
    yield {"type": "metadata", "seq_len": seq_len, "stats": stats}

    torch.cuda.synchronize()
    start_quant = time.perf_counter()
    layers = prefill_cache.layers if hasattr(prefill_cache, "layers") else prefill_cache
    debug_stats = {
        "first_256": 0, "last_256": 0, "middle": 0, "total_retained": 0, "iou_sum": 0.0, 
        "target_budget": target_telemetry_budget, "kept_v_norm_sum": 0.0, "packed_v_norm_sum": 0.0,
        "quant_cosine_sim_sum": 0.0, "bg_scale_max_sum": 0.0, "bg_scale_mean_sum": 0.0
    }
                
    for layer_idx, layer in enumerate(layers):
        # 🟢 如果卡在某一层，这句话会立刻暴露！
        if layer_idx == 0 or layer_idx % 5 == 0:
            print(f"      [System-Probe] Processing Layer {layer_idx}/{len(layers)} quant/pack...", flush=True)

        k = layer.keys if hasattr(layer, "keys") else layer[0]
        v = layer.values if hasattr(layer, "values") else layer[1]
        layer_device = k.device
        compute_device = torch.device(f"cuda:{(layer_device.index + 1) % torch.cuda.device_count()}") if torch.cuda.device_count() > 1 else layer_device
            
        mask = torch.zeros(seq_len, dtype=torch.bool, device=compute_device)
        
        if method != "uniform_int4":
            current_budget = layer_budgets[layer_idx].item()
            sink_len = min(4, seq_len)
            local_len = min(32, seq_len)
            heavy_hitter_budget = max(0, current_budget - sink_len - local_len)
            mask[:sink_len] = True
            mask[-local_len:] = True
            
            if heavy_hitter_budget > 0:
                if method in ["sparc_bic", "sparc_cf", "ablation_inverted"]:
                    k_outlier = k.abs().max(dim=-1)[0].amax(dim=(0, 1)).to(compute_device)
                    v_outlier = v.abs().max(dim=-1)[0].amax(dim=(0, 1)).to(compute_device)
                    outlier_score = k_outlier + v_outlier
                    flat_mass = layer_flat_mass[layer_idx].to(compute_device, non_blocking=True)                        
                    hybrid_score = flat_mass * outlier_score
                    hybrid_score[:sink_len] = -float('inf')
                    hybrid_score[-local_len:] = -float('inf') 
                    indices = torch.topk(hybrid_score, heavy_hitter_budget).indices.to(compute_device)
                    mask[indices] = True
                    
                elif method == "snapkv":
                    flat_mass = layer_flat_mass[layer_idx].clone()
                    flat_mass[:sink_len] = -float('inf')
                    flat_mass[-local_len:] = -float('inf') 
                    indices = torch.topk(flat_mass, heavy_hitter_budget).indices.to(compute_device)
                    mask[indices] = True

            if method != "snapkv":
                v_norms_layer = layer_value_norms[layer_idx].to(compute_device, non_blocking=True)
                kept_vn = v_norms_layer[mask].mean().item() if mask.any() else 0.0
                packed_vn = v_norms_layer[~mask].mean().item() if (~mask).any() else 0.0
                debug_stats["kept_v_norm_sum"] += kept_vn
                debug_stats["packed_v_norm_sum"] += packed_vn

                snapkv_budget = int(seq_len * (0.75 * self.retain_ratio + 0.25))
                snap_mask = torch.zeros(seq_len, dtype=torch.bool, device=compute_device)
                snap_mask[:min(4, seq_len)] = True
                snap_mask[-min(32, seq_len):] = True
                snap_heavy = max(0, snapkv_budget - 4 - 32)
                if snap_heavy > 0:
                    snap_flat = layer_flat_mass[layer_idx].clone()
                    snap_flat[:min(4, seq_len)] = -float('inf')
                    snap_flat[-min(32, seq_len):] = -float('inf')
                    snap_idx = torch.topk(snap_flat, snap_heavy).indices.to(compute_device)
                    snap_mask[snap_idx] = True
                iou = (mask & snap_mask).sum().item() / max(1, snap_mask.sum().item())
                debug_stats["iou_sum"] += iou
        
        if method != "uniform_int4":
            debug_stats["total_retained"] += mask.sum().item()
            if seq_len >= 512:
                debug_stats["first_256"] += mask[:256].sum().item()
                debug_stats["last_256"] += mask[-256:].sum().item()
                debug_stats["middle"] += mask[256:-256].sum().item()
            else:
                debug_stats["last_256"] += mask.sum().item()
        
        k_comp = k.to(compute_device, non_blocking=True)
        v_comp = v.to(compute_device, non_blocking=True)
        
        if hasattr(prefill_cache, 'key_cache'):
            prefill_cache.key_cache[layer_idx] = torch.empty(0, device=layer_device)
            prefill_cache.value_cache[layer_idx] = torch.empty(0, device=layer_device)
        del k, v
        
        k_bf16 = k_comp[:, :, mask, :]
        v_bf16 = v_comp[:, :, mask, :]
        
        k_quant_type = 'int4' 
        v_quant_type = 'int4' 
        payload_extras = {}   
        
        if method == "snapkv":
            k_bg_bytes, v_bg_bytes = b"", b""
            kb_scale, vb_scale = torch.empty(0), torch.empty(0)
            kb_shape = (k_comp.shape[0], k_comp.shape[1], 0, k_comp.shape[3])
            vb_shape = kb_shape
        else:
            k_bg = k_comp[:, :, ~mask, :]
            v_bg = v_comp[:, :, ~mask, :]
            
            # 🟢 终极防死锁保护：强制 CUDA 同步，防止算子队列拥堵导致 Kernel Hang
            torch.cuda.synchronize()
            
            if method == "uniform_int4":
                k_bg_bytes, kb_scale, kb_shape = self._gpu_pack_to_int4(k_bg)
                v_bg_bytes, vb_scale, vb_shape = self._gpu_pack_to_int4(v_bg)
                
            elif method == "sparc_cf":
                batch_sz, heads, seq_bg, head_dim = k_bg.shape
                num_outliers = max(1, head_dim // 32)
                channel_max = k_bg.abs().amax(dim=(0, 2))
                outlier_indices = torch.topk(channel_max, num_outliers, dim=-1).indices
                outlier_idx_exp = outlier_indices.unsqueeze(0).unsqueeze(2).expand(batch_sz, heads, seq_bg, num_outliers)
                k_bg_outliers = torch.gather(k_bg, dim=3, index=outlier_idx_exp)
                core_mask = torch.ones((heads, head_dim), dtype=torch.bool, device=compute_device)
                core_mask.scatter_(1, outlier_indices, False)
                core_indices = torch.nonzero(core_mask)[:, 1].view(heads, head_dim - num_outliers)
                core_idx_exp = core_indices.unsqueeze(0).unsqueeze(2).expand(batch_sz, heads, seq_bg, head_dim - num_outliers)
                k_bg_core = torch.gather(k_bg, dim=3, index=core_idx_exp)
                k_bg_bytes, kb_scale, kb_shape = self._gpu_pack_to_int4(k_bg_core)
                k_quant_type = 'sparc_cf'
                payload_extras['k_bg_outliers_bf16'] = k_bg_outliers.cpu()
                payload_extras['k_outlier_indices'] = outlier_indices.cpu()
                payload_extras['k_core_indices'] = core_indices.cpu()
                v_bg_bytes, vb_scale, vb_shape = self._gpu_pack_to_int4(v_bg)

            elif method == "ablation_inverted":
                k_bg_bytes, kb_scale, kb_shape = self._gpu_pack_to_int4(k_bg)
                v_bg_bytes, vb_scale, vb_shape = self._gpu_pack_to_int4(v_bg)

            else: # Sparc-BIC
                # 如果这个非规则形状 k_bg 让算子崩溃了，下面的 synchronize() 会立刻报错！
                k_bg_bytes, kb_scale, kb_shape = self._gpu_pack_to_int4(k_bg)
                v_bg_bytes, vb_scale, vb_shape = self._gpu_pack_to_int4(v_bg)
            
            torch.cuda.synchronize()

            if k_bg.numel() > 0 and layer_idx % max(1, len(layers)//4) == 0: 
                debug_stats["bg_scale_max_sum"] += kb_scale.max().item()
                debug_stats["bg_scale_mean_sum"] += kb_scale.mean().item()

            del k_bg, v_bg
        
        payload = {
            'type': 'layer', 'layer_idx': layer_idx,
            'keep_mask': mask.cpu(),
            'k_bf16': k_bf16.cpu(), 'k_bg_bytes': k_bg_bytes, 'kb_scale': kb_scale, 'kb_shape': kb_shape,
            'k_quant_type': k_quant_type,
            'v_bf16': v_bf16.cpu(), 'v_bg_bytes': v_bg_bytes, 'vb_scale': vb_scale, 'vb_shape': vb_shape,
            'v_quant_type': v_quant_type,
        }
        payload.update(payload_extras)
        yield payload
        
        del k_comp, v_comp, k_bf16, v_bf16, k_bg_bytes, v_bg_bytes, payload, mask
        if 'indices' in locals(): del indices
        if 'payload_extras' in locals(): del payload_extras
        torch.cuda.empty_cache()
        
    if method in ["sparc_bic", "sparc_cf", "ablation_inverted", "uniform_int4"] and seq_len >= 512:
        num_layers = len(layers)
        chunk_256_tokens = 256 * num_layers
        middle_tokens = (seq_len - 512) * num_layers
        print(f"\n  🔍 [DEBUG {method.upper()}] Advanced Diagnostics:", flush=True)
        if method != "uniform_int4":
            budget_accuracy = (debug_stats['total_retained'] / debug_stats['target_budget']) * 100 if debug_stats['target_budget'] > 0 else 100
            print(f"    ├─ [Budget] Acc: {budget_accuracy:.1f}% | Target: {debug_stats['target_budget']} | Actual: {debug_stats['total_retained']}", flush=True)
            print(f"    ├─ [Layers] Min: {layer_variance_stats['min']:.0f} | Max: {layer_variance_stats['max']:.0f} | StdDev: {layer_variance_stats['std']:.1f}", flush=True)
            print(f"    ├─ [Value-Norm] Kept Mean: {debug_stats['kept_v_norm_sum']/num_layers:.4f} | Packed Mean: {debug_stats['packed_v_norm_sum']/num_layers:.4f}", flush=True)
            print(f"    ├─ [SnapKV] Intersection-Over-Union (IoU): {(debug_stats['iou_sum']/num_layers)*100:.1f}%", flush=True)
            print(f"    ├─ [Spatial] First 256: {debug_stats['first_256']/chunk_256_tokens:.1%} retained", flush=True)
            if middle_tokens > 0: print(f"    ├─ [Spatial] Mid Body:  {debug_stats['middle']/middle_tokens:.1%} retained", flush=True)
            print(f"    └─ [Spatial] Last 256:  {debug_stats['last_256']/chunk_256_tokens:.1%} retained", flush=True)
        samples = max(1, num_layers // max(1, num_layers//4))
        print(f"    ├─ [Telemetry] Background K Max Scale: {debug_stats['bg_scale_max_sum']/samples:.4f}", flush=True)

    torch.cuda.synchronize()
    yield {"type": "done", "quant_time_ms": (time.perf_counter() - start_quant) * 1000}
    return

# 强行注入救命补丁
SparcDisaggregatedEngine.prefill_and_stream = patched_prefill_and_stream
# ==============================================================================

# 🟢 GLOBAL CONFIGURATION
MODEL_PATH = "/home/yuan/sparc/local_models/Qwen3-4B-Instruct-2507" 
MAX_SEQ_LEN = 80000  

NATIVE_FORWARDS = {}

def reset_zmq_socket(context, old_socket, ip, port):
    if old_socket is not None:
        old_socket.setsockopt(zmq.LINGER, 0)
        old_socket.close()
    new_socket = context.socket(zmq.REQ)
    new_socket.connect(f"tcp://{ip}:{port}")
    return new_socket

# 超宽容 Timeout: 防御 32k 下的统一内存置换 (PCIe 慢速搬砖)
def recv_with_timeout(socket, timeout_ms=120000): # 2 分钟 (120 秒)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    if poller.poll(timeout_ms):
        return socket.recv()
    raise TimeoutError("Decode Server stopped responding to layer chunks (Timeout > 2 mins).")

def recv_pyobj_with_timeout(socket, timeout_ms=600000): # 10 分钟 (600 秒)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    if poller.poll(timeout_ms):
        return socket.recv_pyobj()
    raise TimeoutError("Decode Server crashed or stuck swapping VRAM (Timeout > 10 mins).")

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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def evaluate_qa(prediction, ground_truths):
    pred_text = normalize_answer(prediction)
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
        
    em_score = 0.0
    contains_score = 0.0
    
    for truth in ground_truths:
        truth_norm = normalize_answer(truth)
        if not truth_norm: continue
        if truth_norm == pred_text:
            em_score = 1.0
        if truth_norm in pred_text:
            contains_score = 1.0
            
    return em_score, contains_score, str(prediction).strip()

def run_qa(ip, port, retain_ratio, batch_size, max_new_tokens, num_samples, dataset_path, top_k_docs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
    checkpoint_file = os.path.join(script_dir, f"sparc_checkpoint_{dataset_basename}_r{retain_ratio}.jsonl")

    print(f"🚀 Loading Model for Long-Context QA...", flush=True)
    print(f"⚙️ Configuration: [Dataset: {dataset_basename}] | [Retain Ratio: {retain_ratio:.2f}] | [Top-K Docs: {top_k_docs}]", flush=True)
    print(f"📁 Output JSONL will be uniquely saved to: {checkpoint_file}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    
    for i, layer in enumerate(model.model.layers):
        NATIVE_FORWARDS[i] = layer.self_attn.forward
    
    engine = SparcDisaggregatedEngine(model, retain_ratio, causal_depth=3)
    context = zmq.Context()
    socket = reset_zmq_socket(context, None, ip, port)

    print(f"📚 Loading QA Dataset: {dataset_path}...", flush=True)
    test_pool = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        if dataset_path.endswith('.json'):
            test_pool = json.load(f)
        else:
            for line in f:
                if line.strip(): test_pool.append(json.loads(line))
        
    print(f"🔥 Warming up CUDA engine...", flush=True)
    dummy_ids = torch.ones((1, 100), dtype=torch.long, device=model.device)
    with torch.no_grad(): _ = model(input_ids=dummy_ids, attention_mask=dummy_ids)
    purge(model)

    methods = ["Native-Baseline", "Uniform-INT4", "Sparc-BIC", "SnapKV"]
    metrics = {m: {"total": 0, "em": [], "contains": [], "payload": [], "ttft": []} for m in methods}
    processed_ids = set()

    if os.path.exists(checkpoint_file):
        print(f"\n📂 Found existing dataset checkpoint. Loading progress...", flush=True)
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                processed_ids.add(record.get("id"))
                for method, data in record["results"].items():
                    if data.get("success"):
                        metrics[method]["total"] += 1
                        metrics[method]["em"].append(data.get("em_score", 0.0))
                        metrics[method]["contains"].append(data.get("contains_score", 0.0))
                        metrics[method]["payload"].append(data.get("payload_mb", 0.0))
                        metrics[method]["ttft"].append(data.get("ttft_ms", 0.0))
                        
        print(f"✅ Resuming after {len(processed_ids)} previously evaluated samples.", flush=True)

    print(f"\n🧪 STARTING QA BENCHMARK | Target Samples: {'All' if num_samples <= 0 else num_samples}", flush=True)
    print("=" * 100, flush=True)

    valid_samples_count = 0

    try:
        with open(checkpoint_file, "a", encoding="utf-8") as ckpt_file:
            for i, test_data in enumerate(test_pool):
                if num_samples > 0 and valid_samples_count >= num_samples:
                    break

                question_str = test_data.get('question', test_data.get('query', ''))
                
                raw_answers = test_data.get('answers', test_data.get('answer', test_data.get('outputs', [])))
                ground_truth = []
                if isinstance(raw_answers, list):
                    for ans in raw_answers:
                        if isinstance(ans, list): ground_truth.extend(ans)
                        else: ground_truth.append(ans)
                else:
                    ground_truth = [raw_answers]

                context_str = test_data.get('context', test_data.get('input', ''))
                if not context_str:
                    ctxs = test_data.get('ctxs', test_data.get('docs', []))
                    if not ctxs:
                        ctxs.extend(test_data.get('positive_ctxs', []))
                        ctxs.extend(test_data.get('hard_negative_ctxs', []))
                        ctxs.extend(test_data.get('negative_ctxs', []))
                    
                    ctxs = ctxs[:top_k_docs] 
                    
                    context_pieces = []
                    for doc_idx, c in enumerate(ctxs):
                        title = c.get('title', '')
                        text = c.get('text', '')
                        context_pieces.append(f"Document [{doc_idx+1}]: {title}\n{text}")
                    
                    context_str = "\n\n".join(context_pieces)

                if not context_str or not ground_truth: continue

                system_instruction = (
                    "You are an expert reading comprehension assistant. "
                    "Read the provided context carefully and answer the question based strictly on the context. "
                    "Keep your answer as concise as possible. Do not include any conversational filler."
                )

                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"# Context:\n{context_str}\n\n# Question:\n{question_str}"}
                ]

                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                doc_id = test_data.get("id", f"qa_{i}_{hashlib.md5(question_str.encode()).hexdigest()[:8]}")
                
                if doc_id in processed_ids: 
                    valid_samples_count += 1
                    continue
                
                input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
                actual_seq_len = input_ids.shape[1]
                
                if actual_seq_len > MAX_SEQ_LEN: 
                    print(f"\n▶️ Skipping Snippet {i+1} (Length {actual_seq_len} exceeds {MAX_SEQ_LEN} limit)", flush=True)
                    continue
                    
                valid_samples_count += 1
                input_ids = input_ids.expand(batch_size, -1).to(model.device)

                print(f"\n▶️ Evaluating Snippet {i+1}/{len(test_pool)} (Length: {actual_seq_len} tokens) | Progress: {valid_samples_count}/{num_samples}...", flush=True)

                record = {"id": doc_id, "seq_len": actual_seq_len, "ground_truth": ground_truth, "results": {}}
                critical_failure = False

                for method in methods:
                    print(f"\n    🧹 [System] Deep VRAM cleaning before starting {method}...", flush=True)
                    purge(model)
                    torch.cuda.empty_cache()
                    time.sleep(5) 
                    
                    internal_method = "baseline" if method == "Native-Baseline" else method.lower().replace("-", "_")
                    payload_bytes = 0
                    
                    try:
                        t_start_envelope = time.perf_counter()
                        envelope = pickle.dumps({"method": method, "input_ids": input_ids[:, -1:].cpu(), "max_new_tokens": max_new_tokens})
                        payload_bytes += len(envelope) 
                        
                        print(f"    🚀 [System] Waking up Decode Server for {method}...", flush=True)
                        socket.send(envelope)
                        recv_with_timeout(socket, 120000) 
                        
                        envelope_rtt_ms = (time.perf_counter() - t_start_envelope) * 1000
                        
                        print(f"    🧠 [System] Running {method} Forward Pass... (If CPU is 100%, it's packing tensors or swapping RAM)", flush=True)
                        generator = engine.prefill_and_stream(input_ids[:, :-1], method=internal_method)

                        reply = None
                        for chunk_idx, chunk in enumerate(generator):
                            if chunk["type"] == "metadata":
                                print(f"    📦 [System] Forward pass completed! Sending chunk streams...", flush=True)
                            
                            chunk_bytes = pickle.dumps(chunk)
                            payload_bytes += len(chunk_bytes) 
                            socket.send(chunk_bytes)
                            
                            if chunk["type"] != "done":
                                recv_with_timeout(socket, 120000) 
                            else:
                                print(f"    ⏳ [System] All chunks transmitted! Waiting for Decode Server to generate text (Max 10 mins)...", flush=True)
                                reply = recv_pyobj_with_timeout(socket, 600000) 
                        
                        if reply and reply.get("status") == "success":
                            generated_text = reply.get('text', '')
                            em_score, contains_score, pred_val = evaluate_qa(generated_text, ground_truth)
                            
                            payload_mb = payload_bytes / (1024 * 1024)
                            true_ttft_ms = envelope_rtt_ms + reply.get("ttft_ms", 0.0)
                            
                            metrics[method]["total"] += 1
                            metrics[method]["em"].append(em_score)
                            metrics[method]["contains"].append(contains_score)
                            metrics[method]["payload"].append(payload_mb)
                            metrics[method]["ttft"].append(true_ttft_ms)
                            
                            record["results"][method] = {
                                "success": True, "generated_text": generated_text,
                                "em_score": em_score, "contains_score": contains_score,
                                "payload_mb": payload_mb, "ttft_ms": true_ttft_ms
                            }
                            
                            icon = "🟢" if contains_score == 1.0 else "🔴"
                            print(f"  └─ {method:<16} | {icon} EM: {int(em_score)} | Cont: {int(contains_score)} | Payload: {payload_mb:.1f}MB | TTFT: {true_ttft_ms:.1f}ms", flush=True)
                        else:
                            error_reason = reply.get("message", "UNKNOWN") if reply else "NO_REPLY"
                            print(f"  └─ {method:<16} | ⚠️ DECODE SERVER ERROR: {error_reason}", flush=True)
                            critical_failure = True
                            break

                    except TimeoutError as e:
                        print(f"  └─ {method:<16} | 🚨 CRITICAL REMOTE CRASH: {e}", flush=True)
                        critical_failure = True
                        break
                    except Exception as e:
                        print(f"  └─ {method:<16} | 🚨 LOCAL CRASH: {str(e)[:50]}", flush=True)
                        critical_failure = True
                        break
                    finally:
                        if 'generator' in locals(): del generator
                        if 'chunk' in locals(): del chunk
                        purge(model)

                if critical_failure: 
                    print("\n🛑 Halting benchmark to prevent corrupted metrics. The last document was NOT saved.", flush=True)
                    sys.exit(1)
                    
                ckpt_file.write(json.dumps(record) + "\n")
                ckpt_file.flush()
                os.fsync(ckpt_file.fileno())

    except KeyboardInterrupt:
        print("\n\n🛑 [KeyboardInterrupt] Gracefully shutting down...", flush=True)
        sys.exit(0)

    print("\n" + "=" * 100, flush=True)
    print(f"📊 QA TASK ACCURACY & SYSTEMS REPORT", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Method':<16} | {'EM (%)':>6} | {'Cont(%)':>7} | {'Payload(MB)':>11} | {'Avg TTFT(ms)':>12}", flush=True)
    print("-" * 100, flush=True)
    for method in methods:
        m = metrics[method]
        if m["total"] == 0: continue
        print(f"{method:<16} | {np.mean(m['em'])*100:>6.2f} | {np.mean(m['contains'])*100:>7.2f} | {np.mean(m['payload']):>11.1f} | {np.mean(m['ttft']):>12.1f}", flush=True)
    print("=" * 100, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="192.168.31.67")
    parser.add_argument('--port', type=str, default="5555")
    parser.add_argument('--retain_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=150) 
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dataset', type=str, required=True, help="Path to QA jsonl/json dataset")
    parser.add_argument('--top_k_docs', type=int, default=80, help="Number of retrieved documents to keep")
    args = parser.parse_args()
    run_qa(args.ip, args.port, args.retain_ratio, args.batch_size, args.max_new_tokens, args.num_samples, args.dataset, args.top_k_docs)