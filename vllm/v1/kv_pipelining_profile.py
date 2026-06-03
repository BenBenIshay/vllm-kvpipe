# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decode hook profilers for KV pipelining (``VLLM_KVPIPE_PROFILE=1``)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def kv_pipelining_profile_enabled() -> bool:
    return os.environ.get("VLLM_KVPIPE_PROFILE", "0") == "1"


def profile_log_every_n() -> int:
    """Log per-forward summary every N decode steps (default 10)."""
    return max(1, int(os.environ.get("VLLM_KVPIPE_PROFILE_EVERY", "10")))


def profile_steady_window() -> int:
    """Rolling window size for steady stall/ITL aggregates (default 32)."""
    return max(1, int(os.environ.get("VLLM_KVPIPE_PROFILE_WINDOW", "32")))


def profile_json_enabled() -> bool:
    return os.environ.get("VLLM_KVPIPE_PROFILE_JSON", "0") == "1"


@dataclass
class ForwardPassProfile:
    """Metrics for a single decode forward (one output token)."""

    wait_calls: int = 0
    wait_skipped_no_ready: int = 0
    wait_gpu_stall_ms: float = 0.0
    wait_host_ms: float = 0.0
    stall_by_layer: dict[str, float] = field(default_factory=dict)
    stall_samples: list[float] = field(default_factory=list)

    save_calls: int = 0
    save_host_ms: float = 0.0
    save_stage_ms: float = 0.0
    save_prefetch_ms: float = 0.0

    start_load_host_ms: float = 0.0
    start_prefetch_issued: int = 0
    start_prefetch_skipped_ready: int = 0
    start_prefetch_skipped_bound: int = 0
    bootstrap_host_ms: float = 0.0
    bootstrap_prefetch_issued: int = 0
    bootstrap_prefetch_skipped_ready: int = 0
    rolling_prefetch_issued: int = 0
    rolling_prefetch_skipped_not_new: int = 0
    wait_save_host_ms: float = 0.0

    _pending_gpu_waits: list = field(default_factory=list)

    def record_wait(
        self,
        layer_name: str,
        *,
        had_ready: bool,
        host_ms: float,
        before_evt=None,
        after_evt=None,
    ) -> None:
        self.wait_calls += 1
        if not had_ready:
            self.wait_skipped_no_ready += 1
        self.wait_host_ms += host_ms
        if had_ready and before_evt is not None and after_evt is not None:
            self._pending_gpu_waits.append((layer_name, before_evt, after_evt))

    def drain_gpu_waits(self, *, sync: bool = False) -> None:
        if sync:
            import torch

            torch.cuda.synchronize()
        pending = self._pending_gpu_waits
        self._pending_gpu_waits = []
        for layer_name, before_evt, after_evt in pending:
            try:
                if not after_evt.query():
                    self._pending_gpu_waits.append((layer_name, before_evt, after_evt))
                    continue
                ms = float(before_evt.elapsed_time(after_evt))
                self.wait_gpu_stall_ms += ms
                self.stall_samples.append(ms)
                self.stall_by_layer[layer_name] = (
                    self.stall_by_layer.get(layer_name, 0.0) + ms
                )
            except Exception:
                continue

    @staticmethod
    def _percentile(sorted_samples: list[float], q: float) -> float:
        if not sorted_samples:
            return 0.0
        idx = min(len(sorted_samples) - 1, max(0, int(q * len(sorted_samples)) - 1))
        return sorted_samples[idx]

    def stall_stats(self) -> dict[str, float | int]:
        gs = sorted(self.stall_samples)
        return {
            "n": len(gs),
            "sum_ms": self.wait_gpu_stall_ms,
            "p50_ms": self._percentile(gs, 0.50),
            "p99_ms": self._percentile(gs, 0.99),
            "max_ms": gs[-1] if gs else 0.0,
        }

    def worst_layers(self, top_n: int = 4) -> list[tuple[str, float]]:
        ranked = sorted(
            self.stall_by_layer.items(), key=lambda x: x[1], reverse=True
        )
        return [(name, ms) for name, ms in ranked[:top_n] if ms > 0.01]

    def reset(self) -> None:
        self.wait_calls = 0
        self.wait_skipped_no_ready = 0
        self.wait_gpu_stall_ms = 0.0
        self.wait_host_ms = 0.0
        self.stall_by_layer.clear()
        self.stall_samples.clear()
        self.save_calls = 0
        self.save_host_ms = 0.0
        self.save_stage_ms = 0.0
        self.save_prefetch_ms = 0.0
        self.start_load_host_ms = 0.0
        self.start_prefetch_issued = 0
        self.start_prefetch_skipped_ready = 0
        self.start_prefetch_skipped_bound = 0
        self.bootstrap_host_ms = 0.0
        self.bootstrap_prefetch_issued = 0
        self.bootstrap_prefetch_skipped_ready = 0
        self.rolling_prefetch_issued = 0
        self.rolling_prefetch_skipped_not_new = 0
        self.wait_save_host_ms = 0.0
        self._pending_gpu_waits.clear()


@dataclass
class DecodeHookProfile:
    """Per-forward + rolling-window aggregates for one worker."""

    fwd: ForwardPassProfile = field(default_factory=ForwardPassProfile)
    decode_steps: int = 0
    _stall_sum_window: deque[float] = field(default_factory=lambda: deque(maxlen=32))

    def __post_init__(self) -> None:
        w = profile_steady_window()
        self._stall_sum_window = deque(maxlen=w)

    def begin_decode_forward(self) -> None:
        self.fwd.reset()

    def record_wait_load(
        self,
        layer_name: str,
        *,
        had_ready: bool,
        host_ms: float,
        before_evt=None,
        after_evt=None,
    ) -> None:
        self.fwd.record_wait(
            layer_name,
            had_ready=had_ready,
            host_ms=host_ms,
            before_evt=before_evt,
            after_evt=after_evt,
        )

    def end_decode_forward(self, *, step: int, k_window: int) -> None:
        self.fwd.drain_gpu_waits(sync=True)
        self.decode_steps += 1
        self._stall_sum_window.append(self.fwd.wait_gpu_stall_ms)

        log_every = profile_log_every_n()
        if self.decode_steps <= 3 or self.decode_steps % log_every == 0:
            self._log_forward(step=step, k_window=k_window, tag="FWD")

        if len(self._stall_sum_window) >= min(8, profile_steady_window()):
            if self.decode_steps % log_every == 0:
                self._log_steady_window(step=step, k_window=k_window)

    def _log_forward(self, *, step: int, k_window: int, tag: str) -> None:
        st = self.fwd.stall_stats()
        worst = self.fwd.worst_layers()
        worst_s = ", ".join(f"{n.split('.')[-2]}:{ms:.2f}" for n, ms in worst) or "-"
        logger.info(
            "KVPIPE_%s step=%d K=%d | this_forward: "
            "gpu_stall_sum=%.3f ms (p50=%.4f p99=%.4f max=%.4f n=%d) "
            "worst_layers=[%s] | "
            "wait(no_ready=%d/%d) save_host=%.3f(stage=%.3f pf=%.3f) "
            "start_load=%.3f bootstrap=%.3f rolling_pf=%d | "
            "critical_path_note: stall_sum is sum of 24 layer waits on compute "
            "stream (most~0; tails dominate)",
            tag,
            step,
            k_window,
            st["sum_ms"],
            st["p50_ms"],
            st["p99_ms"],
            st["max_ms"],
            st["n"],
            worst_s,
            self.fwd.wait_skipped_no_ready,
            self.fwd.wait_calls,
            self.fwd.save_host_ms,
            self.fwd.save_stage_ms,
            self.fwd.save_prefetch_ms,
            self.fwd.start_load_host_ms,
            self.fwd.bootstrap_host_ms,
            self.fwd.rolling_prefetch_issued,
        )
        if profile_json_enabled():
            logger.info(
                "KVPIPE_JSON %s",
                json.dumps(
                    {
                        "tag": tag,
                        "step": step,
                        "k_window": k_window,
                        "gpu_stall_sum_ms": st["sum_ms"],
                        "wait_host_ms": self.fwd.wait_host_ms,
                        "wait_calls": self.fwd.wait_calls,
                        "save_host_ms": self.fwd.save_host_ms,
                        "save_stage_ms": self.fwd.save_stage_ms,
                        "save_prefetch_ms": self.fwd.save_prefetch_ms,
                        "start_load_host_ms": self.fwd.start_load_host_ms,
                        "bootstrap_host_ms": self.fwd.bootstrap_host_ms,
                        "wait_save_host_ms": self.fwd.wait_save_host_ms,
                        "rolling_prefetch_issued": self.fwd.rolling_prefetch_issued,
                    }
                ),
            )

    def _log_steady_window(self, *, step: int, k_window: int) -> None:
        vals = list(self._stall_sum_window)
        if not vals:
            return
        sv = sorted(vals)
        n = len(sv)
        mean = sum(sv) / n
        p99_i = min(n - 1, max(0, int(0.99 * n) - 1))
        logger.info(
            "KVPIPE_STEADY step=%d K=%d window=%d | "
            "per_forward gpu_stall_sum: mean=%.3f ms p99=%.3f ms max=%.3f ms "
            "(compare to mean ITL from bench; stall_sum NOT additive with attn)",
            step,
            k_window,
            n,
            mean,
            sv[p99_i],
            sv[-1],
        )
