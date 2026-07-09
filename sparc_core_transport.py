import torch
import torch.nn.functional as F
import numpy as np
import time
from transformers.cache_utils import DynamicCache

class SparcDisaggregatedEngine:
    def __init__(self, model, retain_ratio=0.10, causal_depth=3):
        self.model = model
        self.retain_ratio = retain_ratio
        self.causal_depth = causal_depth

    # =========================================================================
    # INT4 PACKING (For Values, Uniform-INT4, and Sparc-CF Core Channels)
    # =========================================================================
    @staticmethod
    def _gpu_pack_to_int4(tensor_gpu):
        if tensor_gpu.numel() == 0: 
            return b"", torch.tensor([1.0], dtype=torch.bfloat16), tensor_gpu.shape
        
        scale = tensor_gpu.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / 7.0
        q_tensor = torch.round(tensor_gpu / scale).clamp(-8, 7).to(torch.int8)
        q_tensor = (q_tensor + 8).to(torch.uint8).flatten()
        
        if q_tensor.numel() % 2 != 0: 
            q_tensor = torch.cat([q_tensor, torch.zeros(1, dtype=torch.uint8, device=q_tensor.device)])
            
        packed_gpu = (q_tensor[0::2] << 4) | q_tensor[1::2]
        return packed_gpu.cpu().numpy().tobytes(), scale.cpu(), tensor_gpu.shape

    @staticmethod
    def _gpu_unpack_from_int4(byte_data, scale, shape, device):
        if len(byte_data) == 0: 
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        
        aligned_array = np.frombuffer(byte_data, dtype=np.uint8).copy()
        packed_gpu = torch.from_numpy(aligned_array).to(device)
        
        unpacked_gpu = torch.empty((packed_gpu.numel() * 2,), dtype=torch.uint8, device=device)
        unpacked_gpu[0::2] = packed_gpu >> 4
        unpacked_gpu[1::2] = packed_gpu & 0x0F
        
        num_elements = int(np.prod(shape))
        unpacked_gpu = unpacked_gpu[:num_elements]
        
        q_tensor = unpacked_gpu.to(torch.float32) - 8.0
        return (q_tensor.reshape(shape) * scale.to(device)).to(torch.bfloat16)

    # =========================================================================
    # INT8 PACKING (For Sparc-BIC Keys to prevent Softmax Poisoning)
    # =========================================================================
    @staticmethod
    def _gpu_pack_to_int8(tensor_gpu):
        if tensor_gpu.numel() == 0: 
            return b"", torch.tensor([1.0], dtype=torch.bfloat16), tensor_gpu.shape
        
        scale = tensor_gpu.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / 127.0
        q_tensor = torch.round(tensor_gpu / scale).clamp(-128, 127).to(torch.int8)
        
        packed_gpu = q_tensor.flatten()
        return packed_gpu.cpu().numpy().tobytes(), scale.cpu(), tensor_gpu.shape

    @staticmethod
    def _gpu_unpack_from_int8(byte_data, scale, shape, device):
        if len(byte_data) == 0: 
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        
        aligned_array = np.frombuffer(byte_data, dtype=np.int8).copy()
        q_tensor = torch.from_numpy(aligned_array).to(device).to(torch.float32)
        
        return (q_tensor.reshape(shape) * scale.to(device)).to(torch.bfloat16)

    # =========================================================================
    # RECONSTRUCTION ENGINE
    # =========================================================================
    @staticmethod
    def reconstruct_cache_layer(payload, device, dtype=torch.bfloat16):
        keep_mask = payload['keep_mask'].to(device)
        seq_len = keep_mask.shape[0]
        
        k_bf16 = payload['k_bf16'].to(device)
        v_bf16 = payload['v_bf16'].to(device)
        
        batch_size, heads = payload['vb_shape'][0], payload['vb_shape'][1]
        head_dim = v_bf16.shape[3]
            
        k_full = torch.zeros((batch_size, heads, seq_len, head_dim), dtype=dtype, device=device)
        v_full = torch.zeros((batch_size, heads, seq_len, head_dim), dtype=dtype, device=device)
        
        if keep_mask.any():
            k_full[:, :, keep_mask, :] = k_bf16
            v_full[:, :, keep_mask, :] = v_bf16
            
        if len(payload.get('k_bg_bytes', b"")) > 0:
            inv_mask = ~keep_mask
            if inv_mask.any():
                k_quant_type = payload.get('k_quant_type', 'int8')
                v_quant_type = payload.get('v_quant_type', 'int4') 
                
                if k_quant_type == 'int4':
                    k_bg = SparcDisaggregatedEngine._gpu_unpack_from_int4(
                        payload['k_bg_bytes'], payload['kb_scale'], payload['kb_shape'], device)
                        
                elif k_quant_type == 'sparc_cf':
                    k_bg_core = SparcDisaggregatedEngine._gpu_unpack_from_int4(
                        payload['k_bg_bytes'], payload['kb_scale'], payload['kb_shape'], device)
                    
                    k_bg_outliers = payload['k_bg_outliers_bf16'].to(device)
                    outlier_indices = payload['k_outlier_indices'].to(device)
                    core_indices = payload['k_core_indices'].to(device)
                    
                    seq_bg = inv_mask.sum().item()
                    k_bg = torch.zeros((batch_size, heads, seq_bg, head_dim), dtype=dtype, device=device)
                    
                    outlier_idx_exp = outlier_indices.unsqueeze(0).unsqueeze(2).expand(batch_size, heads, seq_bg, -1)
                    core_idx_exp = core_indices.unsqueeze(0).unsqueeze(2).expand(batch_size, heads, seq_bg, -1)
                    
                    k_bg.scatter_(dim=3, index=outlier_idx_exp, src=k_bg_outliers)
                    k_bg.scatter_(dim=3, index=core_idx_exp, src=k_bg_core)
                    
                else:
                    k_bg = SparcDisaggregatedEngine._gpu_unpack_from_int8(
                        payload['k_bg_bytes'], payload['kb_scale'], payload['kb_shape'], device)

                # Dynamic unpacking for precision ablation
                if v_quant_type == 'int8':
                    v_bg = SparcDisaggregatedEngine._gpu_unpack_from_int8(
                        payload['v_bg_bytes'], payload['vb_scale'], payload['vb_shape'], device)
                else:
                    v_bg = SparcDisaggregatedEngine._gpu_unpack_from_int4(
                        payload['v_bg_bytes'], payload['vb_scale'], payload['vb_shape'], device)
                
                k_full[:, :, inv_mask, :] = k_bg
                v_full[:, :, inv_mask, :] = v_bg
                
        return k_full.contiguous(), v_full.contiguous()

    # =========================================================================
    # PREFILL & ROUTING HOOK
    # =========================================================================
    def prefill_and_stream(self, input_ids, method="sparc_bic"):
        seq_len = input_ids.shape[1]
        num_layers_total = len(self.model.model.layers)
        
        target_telemetry_budget = int(seq_len * self.retain_ratio * num_layers_total)
        
        torch.cuda.synchronize()
        start_prefill = time.perf_counter()

        if method in ["baseline", "uniform_int4"]:
            with torch.inference_mode():
                out = self.model.model(input_ids, use_cache=True)
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
                            
                            if torch.cuda.device_count() > 1:
                                compute_device = torch.device(f"cuda:{(layer_device.index + 1) % torch.cuda.device_count()}")
                            else:
                                compute_device = layer_device
                                
                            q_comp = q_orig.to(compute_device, non_blocking=True)
                            k_comp = k_orig.to(compute_device, non_blocking=True)
                            
                            if q_comp.shape[0] != k_comp.shape[0]:
                                k_comp = k_comp.repeat_interleave(q_comp.shape[0] // k_comp.shape[0], dim=0)
                                
                            scale = 1.0 / np.sqrt(q_comp.shape[-1])
                            q_len, k_len = q_comp.shape[1], k_comp.shape[1]
                            num_heads = q_comp.shape[0]
                            
                            chunk_sum_total = torch.zeros((q_len, k_len), dtype=torch.float32, device=compute_device)
                            
                            head_step = max(1, num_heads // 4)
                            for h in range(0, num_heads, head_step):
                                end_h = min(h + head_step, num_heads)
                                
                                attn = torch.matmul(q_comp[h:end_h], k_comp[h:end_h].transpose(-2, -1)) * scale
                                
                                causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool, device=compute_device))
                                attn.masked_fill_(~causal_mask, float('-inf'))
                                
                                attn_probs = F.softmax(attn, dim=-1)
                                chunk_sum_total.add_(attn_probs.sum(dim=0))
                                del attn, causal_mask
                                
                            obs_window = min(32, q_len)
                            layer_flat_mass[layer_idx, :k_len].add_(chunk_sum_total[-obs_window:, :].sum(dim=0).to(global_device))
                                
                            v_norm_tokens = torch.norm(v_orig.float(), p=2, dim=-1).mean(dim=0)
                            layer_value_norms[layer_idx, :v_norm_tokens.shape[0]].add_(v_norm_tokens.to(global_device))
                                
                            del q_comp, k_comp, chunk_sum_total, attn_probs
                            
                        return original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **sdpa_kwargs)
                    F.scaled_dot_product_attention = sdpa_interceptor
                    output = original_forward(*args, **kwargs)
                    F.scaled_dot_product_attention = original_sdpa 
                    return output
                module.forward = shadow_forward
                return original_forward 

            original_forwards = {i: patch_attention(layer.self_attn, i) for i, layer in enumerate(self.model.model.layers)}
            
            with torch.inference_mode():
                out = self.model.model(input_ids, use_cache=True)
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

        stats = {
            "prefill_time_ms": prefill_time_ms,
            "routing_time_ms": routing_time_ms,
        }
        yield {"type": "metadata", "seq_len": seq_len, "stats": stats}

        torch.cuda.synchronize()
        start_quant = time.perf_counter()
        
        layers = prefill_cache.layers if hasattr(prefill_cache, "layers") else prefill_cache
        
        debug_stats = {
            "first_256": 0, "last_256": 0, "middle": 0,
            "total_retained": 0, "iou_sum": 0.0, "target_budget": target_telemetry_budget,
            "kept_v_norm_sum": 0.0, "packed_v_norm_sum": 0.0,
            "quant_cosine_sim_sum": 0.0, "bg_scale_max_sum": 0.0, "bg_scale_mean_sum": 0.0
        }
                    
        for layer_idx, layer in enumerate(layers):
            k = layer.keys if hasattr(layer, "keys") else layer[0]
            v = layer.values if hasattr(layer, "values") else layer[1]
            
            layer_device = k.device
            
            if torch.cuda.device_count() > 1:
                target_idx = (layer_device.index + 1) % torch.cuda.device_count()
                compute_device = torch.device(f"cuda:{target_idx}")
            else:
                compute_device = layer_device
                
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
            
            k_quant_type = 'int8' 
            v_quant_type = 'int4' 
            payload_extras = {}   
            
            if method == "snapkv":
                k_bg_bytes, v_bg_bytes = b"", b""
                kb_scale, vb_scale = torch.empty(0), torch.empty(0)
                kb_shape = (k_comp.shape[0], k_comp.shape[1], (~mask).sum().item(), k_comp.shape[3])
                vb_shape = kb_shape
            else:
                k_bg = k_comp[:, :, ~mask, :]
                v_bg = v_comp[:, :, ~mask, :]
                
                if method == "uniform_int4":
                    k_bg_bytes, kb_scale, kb_shape = self._gpu_pack_to_int4(k_bg)
                    k_quant_type = 'int4'
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
                    k_quant_type = 'int4'
                    v_bg_bytes, vb_scale, vb_shape = self._gpu_pack_to_int8(v_bg)
                    v_quant_type = 'int8'

                else:
                    k_bg_bytes, kb_scale, kb_shape = self._gpu_pack_to_int8(k_bg)
                    v_bg_bytes, vb_scale, vb_shape = self._gpu_pack_to_int4(v_bg)
                
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
            
            print(f"\n  🔍 [DEBUG {method.upper()}] Advanced Diagnostics:")
            
            if method != "uniform_int4":
                budget_accuracy = (debug_stats['total_retained'] / debug_stats['target_budget']) * 100 if debug_stats['target_budget'] > 0 else 100
                print(f"     ├─ [Budget] Acc: {budget_accuracy:.1f}% | Target: {debug_stats['target_budget']} | Actual: {debug_stats['total_retained']}")
                print(f"     ├─ [Layers] Min: {layer_variance_stats['min']:.0f} | Max: {layer_variance_stats['max']:.0f} | StdDev: {layer_variance_stats['std']:.1f}")
                print(f"     ├─ [Value-Norm] Kept Mean: {debug_stats['kept_v_norm_sum']/num_layers:.4f} | Packed Mean: {debug_stats['packed_v_norm_sum']/num_layers:.4f}")
                print(f"     ├─ [SnapKV] Intersection-Over-Union (IoU): {(debug_stats['iou_sum']/num_layers)*100:.1f}%")
                print(f"     ├─ [Spatial] First 256: {debug_stats['first_256']/chunk_256_tokens:.1%} retained")
                if middle_tokens > 0: print(f"     ├─ [Spatial] Mid Body:  {debug_stats['middle']/middle_tokens:.1%} retained")
                print(f"     └─ [Spatial] Last 256:  {debug_stats['last_256']/chunk_256_tokens:.1%} retained")
            
            samples = max(1, num_layers // max(1, num_layers//4))
            print(f"     ├─ [Telemetry] Background K Max Scale: {debug_stats['bg_scale_max_sum']/samples:.4f}")

        torch.cuda.synchronize()
        yield {"type": "done", "quant_time_ms": (time.perf_counter() - start_quant) * 1000}
        return