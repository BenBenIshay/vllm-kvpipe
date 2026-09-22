# Concept note — GH200 access for evaluating interconnect-bounded KV tiering

Ben Ishay, Bar-Ilan University, 21 September 2026

Request: one NVIDIA GH200 (96 GB HBM3, Grace-Hopper, NVLink-C2C), about 74 GPU-hours.

## Motivation

The workload that dominates LLM serving today is long-context decode. Agentic
tool loops, retrieval-augmented answering, code assistants reading whole files
and multi-turn chat all arrive with prompts of thousands to tens of thousands of
tokens, and all of them produce their answers one token at a time. Prefill is a
single pass over the prompt while decode runs once per output token, so a serving
node spends most of its life decoding, and modern deployments make this explicit
by splitting prefill and decode onto separate nodes. The decode node is left
holding the key-value (KV) cache of every conversation in flight while doing very
little arithmetic per step. The models carrying this traffic are mid-size, in the
tens of billions of parameters, chosen because they fit a single accelerator and
still answer well.

Such a node runs out of memory long before it runs out of compute. A 32B model in
fp16 takes 65 GB of a 96 GB GPU and leaves room for about a dozen concurrent
requests at 8K context, and a 70B in fp8 leaves room for roughly seven, with the
arithmetic units far from saturated in either case. What sets the serving rate is
how many requests the node can hold, not how fast it serves each one.

Host DRAM is the obvious place for the overflow, and recent systems use it.
DirectKV (OSDI 2026) reads host memory directly from the SMs during attention,
and Pie (Berkeley) swaps cache lines in and out. They differ in mechanism but
share a policy, in that every request is treated the same way, either fully
resident or fully offloaded. We argue the policy is where the performance is,
because the resource actually being spent is not memory but bandwidth. A
host-tier request moves its entire KV cache across the CPU-GPU link once per
decoded token, so the number of such requests a node can sustain is fixed by the
interconnect. That number is a count, not a fraction, which means the right
design serves a mixture: a set of resident requests, plus as many host-tier
riders as the link can carry underneath the decode step.

## The idea, as built

We have implemented this in a fork of vLLM. Each request is assigned a tier, and
an admission controller derives how many riders to accept from the measured link
bandwidth and the measured decode step time. A rider's KV cache is prefetched one
layer ahead into a small rolling buffer, so the transfer overlaps the previous
layer's computation instead of blocking it. The controller is closed-loop, taking
each rider as a probe and withdrawing it if the step time gets worse.

## What we have measured, and why it stops at PCIe

A100-80 over PCIe at 25 GB/s, Qwen2.5-32B at 2048 context, decode-node emulation
so no prefill contaminates the step, three repeats with spread under one percent.

| host-tier riders | decode step | added per rider | link traffic |
|---|---|---|---|
| 0 | 53.5 ms | — | 0.00 GB/step |
| 2 | 71.6 ms | 9.1 ms | 1.12 GB/step |
| 4 | 97.6 ms | 11.0 ms | 2.23 GB/step |
| 6 | 146.9 ms | 15.6 ms | 3.38 GB/step |

The cost per rider rises rather than staying constant, and the data fits a kink:
the step is roughly the larger of the compute time and the transfer time, plus a
fixed overhead. Below the knee the transfer hides behind compute, above it the
step time is the transfer. The knee falls at 2.4 riders and 9 to 18 ms of
overhead remains, putting the best attainable gain on PCIe at 0.86 to 0.99 times
the baseline, just below break-even. That is the right answer for a 25 GB/s link,
since this configuration needs about 70 GB/s to break even and PCIe is a factor
of three beneath it.

NVLink-C2C is the first interconnect above that crossover. At around 400 GB/s
measured from CPU to GPU it carries roughly twice the bandwidth the configuration
needs, and the knee should move to five or six riders, close to doubling the
served batch. The overhead term, however, is software. It does not scale with the
link, it is the term that decides whether the approach pays, and it cannot be
extrapolated from PCIe hardware. Measuring it is what this request is for.

## Required setup

One GH200 Superchip with 96 GB of HBM3 and at least 480 GB of Grace host memory,
which is the configuration our projections are parameterised for. A 144 GB part
would serve as well, with the model sized up to match. We need about 74 GPU-hours
of measurement plus up to 8 hours for the aarch64 and SM90a build, in single-GPU
jobs of 2 to 6 hours. The software is our vLLM fork, the models are open-weight
(Qwen2.5-32B, Llama-3.1-70B in fp8, and Llama-3.1-8B as a small control point),
and there is no private data. Every script is written and dry-run on our A100s,
so the allocation is spent measuring rather than developing.

## Expected results

The first and decisive experiment is the rider cost curve on C2C, which tells us
whether the knee reaches five or six riders and whether the fixed overhead
survives a fast link. We expect about 1.4 times the throughput at a matched
latency target. Combining those points with bandwidth-throttled ones gives a
crossover figure spanning roughly 5 to 400 GB/s, which locates the threshold
above which host-memory KV tiering pays at all and lets others judge whether
their hardware qualifies. We then compare three mechanisms on one substrate for
the first time, ours against Pie's swap and against DirectKV, and test our
bounded rider count against a uniform offload fraction. The target venue is
MLSys or EuroSys class, and a negative result on the first experiment still
publishes as the crossover characterisation.

## Access routes

No commercial cloud provider we surveyed currently has GH200 capacity. Either of
two routes would unblock the work. One is a loan or allocation of a single
Grace-Hopper node from a partner site or through NVIDIA academic relations. The
other is your support as Principal Investigator on a EuroHPC Development Access
application to JUPITER at Jülich, for which a full draft requesting 110
node-hours is prepared. Israel is eligible through its Horizon Europe
association, and the call is open continuously with a decision within two weeks
of each monthly cut-off.
