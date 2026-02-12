# KV Cache Pipelining (Decode-Only)

This document describes the experimental KV cache pipelining path for vLLM v1.
The goal is to reduce decode-phase memory pressure by keeping only a small
layer window in GPU HBM while staging full KV history in CPU pinned memory.

## Scope

- **Decode only**: prefill is untouched.
- **No model changes**: the pipeline is implemented in the KV connector
  lifecycle and does not modify model code.
- **Asynchronous**: all CPU↔GPU copies use non-blocking transfers and a
  dedicated CUDA stream.

## How it works

1. **CPU staging**: after a layer runs, the connector writes back only the
   updated KV slots to pinned CPU memory.
2. **HBM window**: only K layers are resident in GPU HBM at any time.
3. **Rebind + prefetch**: when a layer re-enters the window, its full KV
   history is prefetched from CPU into its window buffer.
4. **Overlap**: prefetch runs on a separate CUDA stream to overlap with
   compute.

## Configuration

Enable pipelining (decode only) with:

```
--enable-kv-pipelining
--kv-k-window 4
--free-original-kv-cache
--kv-transfer-config '{"kv_connector":"LayerwisePipelinedConnector"}'
--enable-prefix-caching false
```

### Notes

- `--free-original-kv-cache` allows the connector to release the original
  per-layer KV buffers and keep only the K-window buffers in HBM.
- If prefill is scheduled after freeing buffers, vLLM will reinitialize empty
  KV buffers and emit a warning.

## Status

This feature is experimental and intended for research and prototyping.
Expect interface changes as the design evolves.
