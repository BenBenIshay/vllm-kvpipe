#!/usr/bin/env python3
"""Where does 'ride-along' batch expansion actually pay?

Baseline: decode batch is capped by HBM KV capacity. We compute, per
(model, context), the capacity-limited batch B_max, the resulting median ITL,
then how many host-KV requests are *hiddable* inside that ITL at GH200 C2C
bandwidth, and what that buys in batch / throughput / utilization.

Model of a decode step (memory-time, with an achieved-BW efficiency factor),
plus a compute roof for the dense GEMMs.
"""

GiB = 1024**3


# ---------------------------------------------------------------- hardware
class HW:
    def __init__(self, name, hbm_gb, bw_gbs, x_eff_gbs, flops_dense):
        self.name = name
        self.hbm = hbm_gb * 1e9
        self.bw = bw_gbs * 1e9          # HBM bytes/s
        self.x = x_eff_gbs * 1e9        # host->device bytes/s (one direction)
        self.flops = flops_dense        # dense bf16 FLOP/s


GH200 = HW("GH200-96 (H100 HBM3 + C2C)", 96, 4000, 450, 495e12)
GH200_2 = HW("2xGH200 (TP=2, per-GPU view)", 96, 4000, 450, 495e12)
A100 = HW("A100-80 + PCIe4", 80, 2039, 25, 312e12)

EFF_W = 0.75       # eff_w: achieved fraction of peak HBM BW on a WEIGHT read,
                   # measured A100-80 (paper/model/p4_*.json). NOTE this script
                   # charges KV at the same rate; killer_v2.py is the current
                   # model and uses the measured eff_kv(working set) curve.
EFF_FLOPS = 0.70   # achieved fraction of dense peak on decode GEMMs
OVERHEAD = 4e9     # activations / cudagraph / workspace


# ------------------------------------------------------------------ models
class Model:
    def __init__(self, name, layers, kv_heads, head_dim, params_b,
                 w_bytes_per_param=2, tp=1, kv_dtype_bytes=2):
        self.name = name
        self.tp = tp
        self.layers = layers
        # per-GPU KV bytes per token (KV heads shard across TP)
        self.k = layers * 2 * (kv_heads / tp) * head_dim * kv_dtype_bytes
        self.w = params_b * 1e9 * w_bytes_per_param / tp   # per-GPU weights
        self.params_per_gpu = params_b * 1e9 / tp


MODELS = [
    Model("Llama-3.1-8B  fp16",        32,  8, 128,   8),
    Model("Llama-3.1-70B fp8",         80,  8, 128,  70, w_bytes_per_param=1),
    Model("Llama-3.1-70B fp16 TP=2",   80,  8, 128,  70, tp=2),
    Model("Qwen2.5-32B   fp16",        64,  8, 128,  32),
    Model("Llama-3.1-8B  fp16 KVfp8",  32,  8, 128,   8, kv_dtype_bytes=1),
]

CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


# ------------------------------------------------------------------- model
def analyze(m, S, hw, stage=2.0, kwin=4, double_buffer=True):
    """stage: HBM traffic per offloaded req in units of its KV bytes.
         2.0 = stream host->HBM staging, then attention reads it back
         1.0 = zero-copy: attention reads host KV directly over coherent C2C
    kwin: rolling window of layers held in HBM for an offloaded request."""
    u = m.k * S                       # KV bytes for one request at ctx S
    C = hw.hbm - m.w - OVERHEAD       # HBM available for KV
    if C <= 0:
        return None
    f = (kwin * (2 if double_buffer else 1)) / m.layers  # HBM frac per rider

    B_max = C / u                                   # capacity-limited batch
    T0 = (m.w + C) / (hw.bw * EFF_W)               # baseline step time (s)

    # compute roof: decode GEMM arithmetic intensity == batch (FLOP/byte)
    B_star = (hw.flops * EFF_FLOPS) / (hw.bw * EFF_W)

    # self-consistent rider count: riders inflate T, which enlarges hide budget
    denom = 1.0 - hw.x * (stage - f) / (hw.bw * EFF_W)
    if denom <= 0:
        return None                                  # runaway: cannot hide
    T = T0 / denom
    B_off = T * hw.x / u                             # hiddable riders
    B_tot = B_max + B_off * (1 - f)                  # riders displace f each

    tput0 = B_max / T0
    tput = B_tot / T
    return dict(S=S, u=u, B_max=B_max, T0=T0 * 1e3, T=T * 1e3, B_off=B_off,
                B_tot=B_tot, gain=tput / tput0, itl_infl=T / T0,
                B_star=B_star, mfu0=B_max / B_star, mfu=B_tot / B_star)


def table(hw, stage, note):
    print(f"\n{'='*104}")
    print(f"{hw.name}   |   HBM {hw.hbm/1e9:.0f}GB @ {hw.bw/1e9:.0f} GB/s   "
          f"|   host link {hw.x/1e9:.0f} GB/s   |   {note}")
    print("="*104)
    for m in MODELS:
        rows = [analyze(m, S, hw, stage=stage) for S in CONTEXTS]
        if all(r is None for r in rows):
            print(f"\n  {m.name}: weights do not fit ({m.w/1e9:.0f} GB)")
            continue
        b_star = next(r for r in rows if r)["B_star"]
        print(f"\n  {m.name}   weights {m.w/1e9:.0f} GB/GPU, "
              f"KV {m.k/1024:.0f} KiB/tok/GPU, compute-roof batch B*≈{b_star:.0f}")
        print(f"  {'ctx':>7} {'KV/req':>8} {'B_max':>6} {'ITL0':>7} "
              f"{'util0':>6} | {'riders':>7} {'B_tot':>6} {'ITL':>7} "
              f"{'ITL x':>6} {'tput x':>7} {'util':>6}")
        for r in rows:
            if r is None:
                continue
            print(f"  {r['S']:>7} {r['u']/1e9:>7.2f}G {r['B_max']:>6.0f} "
                  f"{r['T0']:>6.1f}m {100*min(r['mfu0'],1):>5.0f}% | "
                  f"{r['B_off']:>7.1f} {r['B_tot']:>6.0f} {r['T']:>6.1f}m "
                  f"{r['itl_infl']:>5.2f}x {r['gain']:>6.2f}x "
                  f"{100*min(r['mfu'],1):>5.0f}%")


if __name__ == "__main__":
    table(GH200, 2.0, "riders staged into HBM (write+read)")
    table(GH200, 1.0, "riders read zero-copy over coherent C2C")
    table(A100, 2.0, "PCIe4 reference")
