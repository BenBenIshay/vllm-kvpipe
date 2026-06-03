# Container parity: Docker Compose vs Apptainer

| | [`../docker-compose.yml`](../docker-compose.yml) `vllm-research` | [`vllm-research.def`](vllm-research.def) + [`apptainer_research_shell.sh`](apptainer_research_shell.sh) |
|---|--------|------------|
| Image | `vllm/vllm-openai:latest` | Same (via `Bootstrap: docker`) |
| Code mount | `/home/ben/vllm` → `/workspace/vllm` | Same bind (script uses repo root) |
| HF cache | `${HF_HOME}` → `/root/.cache/huggingface` | Same (default host dir `~/.cache/huggingface`) |
| GPUs | `gpus: all` | `--nv` + env in `%environment` |
| Env | `NVIDIA_*`, `VLLM_GPU_MEMORY_UTILIZATION` | Same in `%environment` |
| Keep alive | `tail -f /dev/null` | `%runscript` |
| Secrets | `HF_TOKEN` in compose | **Not** copied — export on host or use cached login |

**Build SIF:** `apptainer build docker/vllm-research.sif docker/vllm-research.def`

**Shell (like compose):** `bash docker/apptainer_research_shell.sh`

Other compose files on this machine (different layouts): `~/ben/docker-compose.yml`, `swiftLLM/container_setup/docker/docker-compose.yml` — not 1:1 with this def; extend binds/env if you need those instead.
