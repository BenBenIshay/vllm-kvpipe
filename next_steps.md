# Next Steps: KV Cache Pipelining

This file captures the recommended next steps for the decode-only KV cache
pipelining feature. It is intended as a working checklist for continued
development and evaluation.

## 1) End-to-end decode validation (H100)

- Run a decode-only workload with:
  - `--enable-kv-pipelining`
  - `--kv-k-window <K>`
  - `--free-original-kv-cache`
- `--kv-transfer-config '{"kv_connector":"LayerwisePipelinedConnector"}'`
  - `--enable-prefix-caching false`
- Verify:
  - Output correctness (tokens match baseline)
  - GPU memory reduction vs baseline
  - Decode latency impact and throughput

## 2) Telemetry & accounting

- Log window activity:
  - number of window binds / evictions per step
  - bytes transferred CPU↔GPU per step
  - average overlap time (copy stream vs compute stream)

## 3) Guardrails & config validation

- Emit warnings if:
  - pipelining enabled without `H100PcieConnector`
  - prefix caching enabled while pipelining enabled
  - prefill occurs after freeing original KV buffers

## 4) Optional allocator integration (deeper memory savings)

- Avoid allocating full per-layer KV buffers when pipelining is enabled.
- Allocate only the K window buffers (per layer shape).
- Ensure safe fallback to prefill or non-pipelined runs.

## 5) Expand tests

- Add a decode-only integration test (smoke):
  - ensure no sync in hot path
  - assert connector metadata is built in decode
- Add regression coverage for `--free-original-kv-cache`.
