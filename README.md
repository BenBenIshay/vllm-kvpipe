# HeteroKV paper

"HeteroKV: Request-Level KV Tiering Across Heterogeneous Memory for LLM Decode".

This branch holds the paper only. The engine (the vLLM fork) and the benchmarks
that produced the measurements are on the `heterokv-code` branch.

| Path | What |
|---|---|
| `v2/` | The current paper. `v2/build.sh` builds `v2/out/main.pdf` |
| `v2/export_noappendix.py` | Overleaf export: `OUTDIR --final --prerender` writes `OUTDIR/hetero-kv.zip` |
| `model/` | Capacity model, projections and simulations behind the figures |
| `data/` | Measurement data and logs the paper cites (`% src:` comments) |
| `tex/`, `tex-precampaign/`, `out-arxiv/`, `*.md` | Earlier drafts and planning notes |

Building needs [tectonic](https://tectonic-typesetting.github.io/). Figure
rasterising and the Overleaf export also need Python with `pymupdf`; set
`PYTHON` to that interpreter (default `python3`).

`% src:` comments in the LaTeX name files such as `vllm/v1/...` or
`benchmarks/...`; those paths are on `heterokv-code`.
