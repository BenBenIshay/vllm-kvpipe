# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Experimental CPU->GPU transfer-aware admission helpers for the V1 scheduler.

Regime A (decode worker): KV is already staged in CPU DDR before scheduling.
Admission decisions should eventually bound PCI/NVLink-style bandwidth impact.

Default-off: when disabled, all hooks are no-ops from the scheduler's perspective.

Phase 4 policy
--------------
When ``enable_transfer_aware_admission`` is True AND a predictor CSV is
configured AND ``transfer_aware_slo_p99_itl_ms > 0``, ``should_admit_from_waiting``
refuses admitting requests whose predicted post-admission p99 ITL would
exceed the SLO budget. The predictor is built from the feature-ON K sweep
data produced by ``benchmarks/kv_pipelining_bench/feature_on_k_sweep.py``.

The hook self-degrades to admit-all when the predictor cannot be loaded
(missing CSV, no rows for the runtime model, etc.) so a misconfiguration
never breaks serving.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


class TransferAwareAdmissionController:
    """Tracks lightweight admission telemetry; policy admits / refuses based on
    predicted post-admission p99 ITL when configured."""

    __slots__ = (
        "scheduler_config",
        "_events_this_step",
        "_schedule_step_index",
        "_predictor",
        "_predictor_loaded",
        "_predictor_load_error",
        "_req_tier",
    )

    def __init__(self, scheduler_config: "SchedulerConfig") -> None:
        self.scheduler_config = scheduler_config
        self._events_this_step: dict[str, int] = {}
        self._schedule_step_index = 0
        # Predictor is lazy-loaded on first admission decision so that turning
        # the feature off avoids any I/O or CSV parsing.
        self._predictor: Optional[object] = None
        self._predictor_loaded: bool = False
        self._predictor_load_error: Optional[str] = None
        self._req_tier: dict[str, str] = {}

    def get_req_tier(self, req_id: str) -> str | None:
        return self._req_tier.get(req_id)

    def release_req_tier(self, req_id: str) -> None:
        self._req_tier.pop(req_id, None)

    def release_finished(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self.release_req_tier(req_id)

    # ------------------------------------------------------- step lifecycle

    def begin_schedule_step(self) -> None:
        """Reset per-step counters at the start of Scheduler.schedule()."""
        self._events_this_step.clear()
        self._schedule_step_index += 1

    def estimate_running_transfer_pressure_bytes_per_step(
        self, running_requests: "list[Request]"
    ) -> float:
        """Rough CPU->GPU KV bytes moved per engine step for running sequences."""
        cfg = self.scheduler_config
        if not cfg.enable_transfer_aware_admission:
            return 0.0
        return len(running_requests) * cfg.transfer_aware_bytes_per_decode_step_per_seq

    # ------------------------------------------------------ admission policy

    def should_admit_from_waiting(
        self,
        *,
        request: "Request",
        running_requests: "list[Request]",
        token_budget: int,
    ) -> tuple[bool, str]:
        """Return whether a WAITING request may proceed past early admission gates.

        Behavior matrix:

          enable=False                                 -> admit, "disabled"
          enable=True,  slo<=0 or no predictor         -> admit, "stub_allow_all"
          enable=True,  predicted p99 <= SLO           -> admit, "ok"
          enable=True,  predicted p99 >  SLO           -> refuse, "predicted_slo_breach"
        """
        cfg = self.scheduler_config
        if not cfg.enable_transfer_aware_admission:
            return True, "disabled"

        if cfg.enable_hybrid_kv_tiers:
            return self._should_admit_hybrid(
                request=request, running_requests=running_requests
            )

        # SLO budget unset -> stub. Preserves the pre-Phase-4 contract.
        slo_ms = float(cfg.transfer_aware_slo_p99_itl_ms or 0.0)
        if slo_ms <= 0:
            return True, "stub_allow_all"

        predictor = self._maybe_load_predictor()
        if predictor is None:
            # Predictor missing or failed to load -> log once, fall back to
            # stub-allow so a misconfigured node never breaks serving.
            return True, "stub_allow_all"

        K_running = len(running_requests)
        K_post_admit = K_running + 1
        input_len = getattr(request, "num_prompt_tokens", None)
        try:
            if hasattr(predictor, "predict_p99_itl_ms") and input_len is not None:
                try:
                    predicted_p99 = float(
                        predictor.predict_p99_itl_ms(
                            K_post_admit, input_len=int(input_len),
                        )
                    )
                except TypeError:
                    predicted_p99 = float(predictor.predict_p99_itl_ms(K_post_admit))
            else:
                predicted_p99 = float(predictor.predict_p99_itl_ms(K_post_admit))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Predictor lookup failed (K=%d): %s", K_post_admit, exc)
            return True, "stub_allow_all"

        if predicted_p99 > slo_ms:
            return False, "predicted_slo_breach"
        return True, "ok"

    def _should_admit_hybrid(
        self,
        *,
        request: "Request",
        running_requests: "list[Request]",
    ) -> tuple[bool, str]:
        from vllm.v1.hybrid_kv_roofline import (
            bytes_per_token,
            effective_pcie_gbps,
            max_offload_batch,
            max_offload_from_env,
        )

        cfg = self.scheduler_config
        hbm_max = int(cfg.hybrid_b_hbm_max or 0)
        hbm_count = sum(
            1 for r in running_requests if self._req_tier.get(r.request_id) == "hbm"
        )
        cpu_count = sum(
            1 for r in running_requests if self._req_tier.get(r.request_id) == "cpu"
        )
        # Requests without tier yet (legacy) count as cpu for safety.
        untagged = sum(
            1 for r in running_requests if r.request_id not in self._req_tier
        )
        cpu_count += untagged

        if hbm_max > 0 and hbm_count < hbm_max:
            self._req_tier[request.request_id] = "hbm"
            return True, "hybrid_hbm"

        itl_ms = float(cfg.hybrid_itl_ms or 0.0)
        if itl_ms <= 0:
            predictor = self._maybe_load_predictor()
            if predictor is not None:
                try:
                    itl_ms = float(predictor.predict_p99_itl_ms(len(running_requests) + 1))
                except Exception:
                    itl_ms = 16.0
            else:
                itl_ms = 16.0

        b_tok = int(cfg.hybrid_b_tok or 0)
        if b_tok <= 0:
            b_tok = bytes_per_token(
                num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2
            )
        x_eff = effective_pcie_gbps(float(cfg.hybrid_pcie_gbps or 0.0))
        seq_tokens = int(cfg.hybrid_seq_tokens or 512)
        off_max = max_offload_batch(seq_tokens, itl_ms, x_eff, b_tok)
        env_cap = max_offload_from_env()
        if env_cap is not None:
            off_max = min(off_max, env_cap)

        if cpu_count < off_max:
            self._req_tier[request.request_id] = "cpu"
            return True, "hybrid_cpu"

        return False, "hybrid_offload_cap"

    # ------------------------------------------------------ telemetry hooks

    def record_skip(self, reason: str) -> None:
        self._events_this_step[reason] = self._events_this_step.get(reason, 0) + 1

    def drain_events(self) -> dict[str, int]:
        """Snapshot admission skip counters for the current schedule() call."""
        return dict(self._events_this_step)

    # ------------------------------------------------ predictor lazy loader

    def _maybe_load_predictor(self) -> Optional[object]:
        if self._predictor_loaded:
            return self._predictor
        self._predictor_loaded = True

        cfg = self.scheduler_config
        csv_path = (cfg.transfer_aware_predictor_csv or "").strip()
        if not csv_path:
            self._predictor_load_error = "no predictor csv configured"
            logger.info(
                "TransferAwareAdmissionController: enable=True but "
                "transfer_aware_predictor_csv is empty; falling back to stub_allow_all."
            )
            return None
        if not os.path.isfile(csv_path):
            self._predictor_load_error = f"csv not found: {csv_path}"
            logger.warning(
                "TransferAwareAdmissionController: predictor CSV not found: %s; "
                "falling back to stub_allow_all.",
                csv_path,
            )
            return None

        try:
            predictor = _load_predictor_from_config(cfg, csv_path)
        except Exception as exc:
            self._predictor_load_error = f"load failed: {exc}"
            logger.warning(
                "TransferAwareAdmissionController: failed to load predictor from %s: %s; "
                "falling back to stub_allow_all.",
                csv_path,
                exc,
            )
            return None

        if predictor is None:
            self._predictor_load_error = "no usable rows for model"
            logger.warning(
                "TransferAwareAdmissionController: predictor CSV %s contains no rows "
                "for model='%s'; falling back to stub_allow_all.",
                csv_path,
                cfg.transfer_aware_predictor_model or "<auto>",
            )
            return None

        self._predictor = predictor
        policy_mode = (cfg.transfer_aware_admission_policy or "combined").strip()
        logger.info(
            "TransferAwareAdmissionController: %s predictor loaded from %s "
            "(model=%s, K range=[%d, %d]). SLO=%.1fms.",
            policy_mode,
            csv_path,
            cfg.transfer_aware_predictor_model or "<auto>",
            getattr(predictor, "k_min", -1),
            getattr(predictor, "k_max", -1),
            cfg.transfer_aware_slo_p99_itl_ms,
        )
        return predictor


def _load_predictor_from_config(cfg: "SchedulerConfig", csv_path: str):
    """Load combined or mechanistic predictor per ``transfer_aware_admission_policy``."""
    import sys as _sys
    from pathlib import Path as _Path

    bench_dir = _Path(__file__).resolve().parents[4] / "benchmarks/kv_pipelining_bench"
    if str(bench_dir) not in _sys.path:
        _sys.path.insert(0, str(bench_dir))
    from admission_predictor import (  # type: ignore
        AdmissionPredictor,
        MechanisticAdmissionPredictor,
    )

    model = cfg.transfer_aware_predictor_model or ""
    k_window = int(cfg.transfer_aware_kv_k_window)
    policy = (cfg.transfer_aware_admission_policy or "combined").strip().lower()
    injection = (cfg.transfer_aware_injection_jsonl or "").strip() or None

    if policy == "mechanistic":
        baseline_json = (cfg.transfer_aware_baseline_compute_json or "").strip()
        if not baseline_json or not os.path.isfile(baseline_json):
            logger.warning(
                "TransferAwareAdmissionController: mechanistic policy requires "
                "transfer_aware_baseline_compute_json; falling back to combined."
            )
            policy = "combined"
        else:
            return MechanisticAdmissionPredictor.from_artifacts(
                feature_on_csv=csv_path,
                model=model,
                baseline_compute_json=baseline_json,
                injection_jsonl=injection,
                k_window=k_window,
            )

    return _load_predictor_from_csv(
        csv_path, model=model, k_window=k_window, injection_jsonl=injection,
    )


# ----------------------------------------------------------- predictor loader
#
# The predictor module lives under ``benchmarks/kv_pipelining_bench/`` (an R&D
# tree, not packaged with vLLM proper) so we import it lazily and via path.


def _load_predictor_from_csv(
    csv_path: str,
    *,
    model: str,
    k_window: int = 4,
    injection_jsonl: str | None = None,
):
    """Load AdmissionPredictor from a feature-ON K sweep CSV.

    If ``model`` is empty, picks the first model name found in the CSV.
    Returns the predictor instance, or None if no usable rows.
    """
    import csv as _csv
    import sys as _sys
    from pathlib import Path as _Path

    bench_dir = _Path(__file__).resolve().parents[4] / "benchmarks/kv_pipelining_bench"
    if str(bench_dir) not in _sys.path:
        _sys.path.insert(0, str(bench_dir))
    from admission_predictor import AdmissionPredictor  # type: ignore

    chosen_model = (model or "").strip()
    if not chosen_model:
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                m = (row.get("model") or "").strip()
                if m:
                    chosen_model = m
                    break
        if not chosen_model:
            return None

    try:
        return AdmissionPredictor.from_csv(
            csv_path,
            model=chosen_model,
            k_window=k_window,
            injection_jsonl=injection_jsonl,
        )
    except (ValueError, FileNotFoundError):
        return None
