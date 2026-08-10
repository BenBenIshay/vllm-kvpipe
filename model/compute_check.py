#!/usr/bin/env python3
"""Is the 31 ms decode step really bandwidth-bound, or is compute hiding in it?

Breaks a decode step into GEMM + FlashDecoding-attention, and reports
memory time vs compute time for each, on GH200.
"""

BW = 4000e9 * 0.75        # achieved HBM bytes/s
F_BF16 = 495e12 * 0.70    # achieved dense bf16 FLOP/s (H100 SXM, no sparsity)
F_FP8 = 990e12 * 0.70     # achieved dense fp8 FLOP/s
HBM = 96e9
OVH = 4e9


def step(name, L, hq, hkv, hd, params, w_bytes, S, kv_b=2):
    W = params * w_bytes
    C = HBM - W - OVH                  # HBM left for KV
    k = L * 2 * hkv * hd * kv_b        # KV bytes per token
    B = C / (k * S)                    # capacity-limited batch
    kv_bytes = B * S * k

    # ---- memory time
    t_w = W / BW
    t_kv = kv_bytes / BW

    # ---- compute time
    gemm_flops = 2 * params * B
    peak = F_FP8 if w_bytes == 1 else F_BF16
    t_gemm = gemm_flops / peak
    # attention: QK^T + PV = 4 * B * hq * S * hd FLOPs per layer
    attn_flops = 4 * B * hq * S * hd * L
    t_attn = attn_flops / F_BF16

    # ---- arithmetic intensity vs machine balance
    ai_gemm = gemm_flops / W                  # FLOP per weight byte
    ai_attn = attn_flops / kv_bytes           # == hq/hkv for fp16 KV
    bal_gemm = peak / BW
    bal_attn = F_BF16 / BW

    print(f"\n{name}   ctx {S}   capacity batch B = {B:.0f}")
    print(f"  {'':<26}{'memory':>10}{'compute':>10}{'  intensity vs machine balance'}")
    print(f"  {'dense GEMMs (weights)':<26}{t_w*1e3:>9.1f}m{t_gemm*1e3:>9.1f}m"
          f"   {ai_gemm:>6.0f} vs {bal_gemm:>4.0f} FLOP/B  "
          f"({'MEM' if ai_gemm < bal_gemm else 'COMPUTE'}-bound, {bal_gemm/ai_gemm:.1f}x margin)")
    print(f"  {'attention (KV)':<26}{t_kv*1e3:>9.1f}m{t_attn*1e3:>9.1f}m"
          f"   {ai_attn:>6.0f} vs {bal_attn:>4.0f} FLOP/B  "
          f"({'MEM' if ai_attn < bal_attn else 'COMPUTE'}-bound, {bal_attn/ai_attn:.1f}x margin)")
    tm, tc = (t_w + t_kv) * 1e3, (t_gemm + t_attn) * 1e3
    print(f"  {'TOTAL':<26}{tm:>9.1f}m{tc:>9.1f}m   -> step = max(...) = {max(tm,tc):.1f} ms"
          f"   (compute is {100*tc/tm:.0f}% of memory time)")


# Llama-3.1-70B: 80 layers, 64 q heads, 8 kv heads, head_dim 128
step("Llama-3.1-70B fp8 weights", 80, 64, 8, 128, 70e9, 1, 8192)
step("Llama-3.1-70B fp8 weights", 80, 64, 8, 128, 70e9, 1, 4096)
# Llama-3.1-8B: 32 layers, 32 q heads, 8 kv heads
step("Llama-3.1-8B  fp16 weights", 32, 32, 8, 128, 8e9, 2, 8192)
step("Llama-3.1-8B  fp16 weights", 32, 32, 8, 128, 8e9, 2, 4096)
# Qwen2.5-32B: 64 layers, 40 q heads, 8 kv heads
step("Qwen2.5-32B   fp16 weights", 64, 40, 8, 128, 32e9, 2, 8192)

print("\nNote: decode-attention arithmetic intensity = (q heads / kv heads) FLOP/byte")
print("for fp16 KV -- i.e. the GQA group size, independent of batch and context.")
