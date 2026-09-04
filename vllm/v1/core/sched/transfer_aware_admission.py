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
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig, VllmConfig
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
        # B2 closed-loop ITL guardrail state.
        "_resident_step_ms_ema",
        "_itl_throttled_state",
        # Item 2: request-level tier assignment.
        "_tier_policy",
        "_last_active",
        "_idle_saturation_s",
        "_remaining_saturation",
        "_w_idle",
        # Item 3: migration state.
        "_migrated_at_step",
        "_migration_counts",
        "_migrate_cooldown_steps",
        "_migrate_min_remaining",
        "_max_migrations_per_step",
        # Item 3b: adaptive demotion probe.
        "_probe",
        "_probe_steps",
        "_probe_margin",
        "_demote_suppressed_until",
        "_demote_backoff",
        "_probe_results",
        "_ema_window",
        "_migrate_warmup_steps",
        # KV bytes per token, derived from the model rather than hand-set.
        "_b_tok_cached",
        # B1 hide budget measured from real forward iterations.
        "_resident_only_ms_ema",
        "_resident_only_samples",
        "_measured_budget_enabled",
        "_measured_budget_min_samples",
        "_measured_budget_announced",
    )

    def __init__(self, scheduler_config: "SchedulerConfig",
                 vllm_config: "VllmConfig | None" = None) -> None:
        self.scheduler_config = scheduler_config
        self._b_tok_cached: int = self._resolve_b_tok(vllm_config)
        # B1's hide budget is "how long a forward iteration lasts", which is a
        # MEASURED quantity, not a flag. Sampled only on steps with no host-tier
        # request running: including rider steps would be circular, since a step
        # slowed by transfer would then justify admitting more transfer.
        self._resident_only_ms_ema: float = 0.0
        self._resident_only_samples: int = 0
        self._measured_budget_enabled: bool = (
            os.environ.get("VLLM_KVPIPE_B1_MEASURED", "1") == "1")
        self._measured_budget_min_samples: int = int(
            os.environ.get("VLLM_KVPIPE_B1_MEASURED_MIN", "8") or 8)
        self._measured_budget_announced: bool = False
        self._events_this_step: dict[str, int] = {}
        self._schedule_step_index = 0
        # Predictor is lazy-loaded on first admission decision so that turning
        # the feature off avoids any I/O or CSV parsing.
        self._predictor: Optional[object] = None
        self._predictor_loaded: bool = False
        self._predictor_load_error: Optional[str] = None
        self._req_tier: dict[str, str] = {}
        # B2: EMA of the measured resident decode step time (ms); 0 = no data.
        self._resident_step_ms_ema: float = 0.0
        self._itl_throttled_state: bool = False
        # Item 2: {"arrival", "scored"}. "arrival" reproduces pre-Item-2
        # behaviour exactly and is the ablation baseline for the paper.
        self._tier_policy: str = (
            os.environ.get("VLLM_KVPIPE_TIER_POLICY", "arrival").strip() or "arrival"
        )
        self._last_active: dict[str, float] = {}
        self._idle_saturation_s: float = float(
            os.environ.get("VLLM_KVPIPE_TIER_IDLE_SAT_S", "2.0") or 2.0)
        self._remaining_saturation: int = int(
            os.environ.get("VLLM_KVPIPE_TIER_REMAIN_SAT", "512") or 512)
        self._w_idle: float = float(
            os.environ.get("VLLM_KVPIPE_TIER_W_IDLE", "0.7") or 0.7)
        # Item 3: migration.
        self._migrated_at_step: dict[str, int] = {}
        self._migration_counts: dict[str, int] = {}
        self._migrate_cooldown_steps: int = int(
            os.environ.get("VLLM_KVPIPE_MIGRATE_COOLDOWN", "64") or 64)
        self._migrate_min_remaining: int = int(
            os.environ.get("VLLM_KVPIPE_MIGRATE_MIN_REMAIN", "32") or 32)
        self._max_migrations_per_step: int = int(
            os.environ.get("VLLM_KVPIPE_MIGRATE_MAX_PER_STEP", "1") or 1)
        # Item 3b: adaptive demotion. A demotion is treated as a PROBE whose
        # effect on measured ITL is checked after a settling window; if it made
        # things worse we promote back and back off. This exists because the B1
        # roofline gates on transfer BYTES and cannot see the host path's fixed
        # per-step overhead -- measured on A100, where a byte-legal demotion
        # drove ITL from ~6 to ~24 ms.
        self._probe: dict | None = None
        self._probe_steps: int = int(
            os.environ.get("VLLM_KVPIPE_PROBE_STEPS", "16") or 16)
        self._probe_margin: float = float(
            os.environ.get("VLLM_KVPIPE_PROBE_MARGIN", "0.05") or 0.05)
        self._demote_suppressed_until: int = 0
        self._demote_backoff: int = int(
            os.environ.get("VLLM_KVPIPE_DEMOTE_BACKOFF", "256") or 256)
        self._probe_results: list[tuple[float, float, bool]] = []
        # Recent EMA history. The probe baseline must be a CLEAN steady-state
        # reference: taking the instantaneous EMA at demotion time picked up
        # startup/prefill transients, which made a harmful demotion look like an
        # improvement (measured: baseline 25.5 -> 21.8 "better", while the true
        # steady state was ~6 ms). Using the recent MINIMUM fixes that.
        self._ema_window: list[float] = []
        # Do not migrate on a cold EMA. Without this the first demotion fired at
        # step 2, before any steady state existed, so the probe baseline was a
        # startup transient and a harmful demotion looked beneficial.
        self._migrate_warmup_steps: int = int(
            os.environ.get("VLLM_KVPIPE_MIGRATE_WARMUP", "24") or 24)

    # ------------------------------------------------- KV bytes per token

    def _resolve_b_tok(self, vllm_config: "VllmConfig | None") -> int:
        """KV bytes per token for one request, resolved ONCE at construction.

        Order: model-derived, else the explicit `--hybrid-b-tok`, else a
        last-resort constant. Model-derived wins because the explicit value is
        what was wrong: every harness passed 131072 (a 32-layer model) while
        running Qwen2.5-32B, whose 64 layers make it 262144. B1 was charging
        each host-tier request half its real interconnect cost, which
        over-admits riders by ~2x in exactly the direction that breaks the ITL
        claim -- and does so silently.

        A configured value that DISAGREES with the model is logged as an error,
        not quietly honoured, since it can only be a stale flag.
        """
        from vllm.v1.hybrid_kv_roofline import (
            bytes_per_token,
            bytes_per_token_from_config,
        )

        cfg_b_tok = int(getattr(self.scheduler_config, "hybrid_b_tok", 0) or 0)
        derived = (bytes_per_token_from_config(vllm_config)
                   if vllm_config is not None else None)
        if derived:
            if cfg_b_tok and cfg_b_tok != derived:
                logger.error(
                    "hybrid_b_tok=%d disagrees with the model's actual KV size "
                    "of %d bytes/token; using %d. A too-small value makes B1 "
                    "admit ~%.1fx too many host-tier requests.",
                    cfg_b_tok, derived, derived, derived / max(1, cfg_b_tok))
            else:
                logger.info("hybrid KV bytes/token = %d (model-derived)", derived)
            return derived
        if cfg_b_tok:
            logger.warning(
                "hybrid KV bytes/token = %d (from --hybrid-b-tok; could not be "
                "derived from the model, so it is UNVERIFIED)", cfg_b_tok)
            return cfg_b_tok
        fallback = bytes_per_token(
            num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
        logger.warning(
            "hybrid KV bytes/token falling back to %d (32L/8H/128D/fp16). If "
            "the running model is larger, B1 will over-admit host-tier "
            "requests.", fallback)
        return fallback

    def kv_bytes_per_token(self) -> int:
        return self._b_tok_cached

    def get_req_tier(self, req_id: str) -> str | None:
        return self._req_tier.get(req_id)

    def release_req_tier(self, req_id: str) -> None:
        self._req_tier.pop(req_id, None)
        self._last_active.pop(req_id, None)
        self._migrated_at_step.pop(req_id, None)

    def release_finished(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self.release_req_tier(req_id)

    # ------------------------------------------------------- step lifecycle

    def begin_schedule_step(self) -> None:
        """Reset per-step counters at the start of Scheduler.schedule()."""
        self._events_this_step.clear()
        self._schedule_step_index += 1

    # ------------------------------------------------ B2 closed-loop guardrail

    def observe_step_time(self, step_ms: float, is_decode_only: bool,
                          num_host_tier: int | None = None) -> None:
        """Feed the measured engine step time (≈ resident ITL) back to the
        controller. Only decode-only steps are counted (prefill steps are
        naturally longer). Updates an EMA and the OK/THROTTLED state with
        hysteresis around the SLO.

        `num_host_tier` is how many host-tier requests were running during the
        step. A step with none is a CLEAN measurement of a forward iteration,
        and those are what B1's hide budget is built from.
        """
        if not is_decode_only or step_ms <= 0:
            return
        a = 0.2  # EMA weight
        prev = self._resident_step_ms_ema
        self._resident_step_ms_ema = step_ms if prev <= 0 else (
            a * step_ms + (1 - a) * prev
        )
        if num_host_tier == 0:
            r = self._resident_only_ms_ema
            self._resident_only_ms_ema = step_ms if r <= 0 else (
                a * step_ms + (1 - a) * r)
            self._resident_only_samples += 1
        ema = self._resident_step_ms_ema
        # Kept regardless of whether an SLO is configured: the demotion probe
        # reads this window for its baseline, and it used to be populated only
        # when an SLO happened to be set.
        self._ema_window.append(ema)
        if len(self._ema_window) > 64:
            self._ema_window.pop(0)
        slo = float(self.scheduler_config.transfer_aware_slo_p99_itl_ms or 0.0)
        if slo <= 0:
            self._itl_throttled_state = False
            return
        if self._itl_throttled_state:
            if ema < slo * 0.85:          # low watermark -> resume admitting
                self._itl_throttled_state = False
        elif ema > slo * 0.95:            # high watermark -> stop admitting
            self._itl_throttled_state = True

    # --------------------------------------------- Item 2: tier assignment
    def _host_affinity_threshold(self) -> float:
        raw = os.environ.get("VLLM_KVPIPE_TIER_THRESHOLD", "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
        return 0.5

    def note_active(self, req_id: str, now: float | None = None) -> None:
        """Record that `req_id` produced a token (drives the idle signal)."""
        self._last_active[req_id] = time.monotonic() if now is None else now

    def host_affinity(self, request: "Request", now: float | None = None) -> float:
        """How good a host-tier candidate is this request?  [0, 1], higher=better.

        The analysis behind this matters, because it is *not* what one first
        expects. Moving a request to the host tier frees ``S*k`` bytes of HBM and
        costs ``S*k`` bytes of interconnect **per decoding step**. Both scale
        identically with context length, and the k-window discount ``(1-f)`` is
        independent of ``S`` -- so **context length does not discriminate
        between tiers at all** (the same cancellation as design rule (a), which
        we validated on hardware in P4).

        What does discriminate is TEMPORAL: how many steps a request spends
        *holding* HBM versus *consuming* bandwidth.
          * an idle request (multi-turn / agentic gap) holds HBM but decodes
            nothing -> costs no bandwidth while idle -> ideal host-tier tenant
          * a long-remaining request will hold HBM for many steps -> worth
            displacing
          * a nearly-finished request will release its HBM shortly anyway ->
            migrating it is wasted work
        """
        t = time.monotonic() if now is None else now
        # idle signal: seconds since this request last produced a token
        last = self._last_active.get(request.request_id)
        idle_s = 0.0 if last is None else max(0.0, t - last)
        idle_score = min(1.0, idle_s / max(1e-6, self._idle_saturation_s))

        # remaining-work signal: long-remaining -> better host tenant
        max_toks = 0
        sp = getattr(request, "sampling_params", None)
        if sp is not None:
            max_toks = int(getattr(sp, "max_tokens", 0) or 0)
        done = int(getattr(request, "num_output_tokens", 0) or 0)
        remaining = max(0, max_toks - done) if max_toks > 0 else 0
        rem_score = (
            min(1.0, remaining / float(self._remaining_saturation))
            if remaining > 0 else 0.5      # unknown horizon -> neutral
        )
        return self._w_idle * idle_score + (1.0 - self._w_idle) * rem_score

    # ------------------------------------------------- Item 3: migration
    def plan_migrations(
        self, running_requests: "list[Request]", now: float | None = None
    ) -> list[tuple[str, str]]:
        """Decide tier moves for already-running requests.

        Closes the *gate-new-only* limitation: previously a burst admitted
        everyone before the ITL loop engaged, and a long-running cohort could
        drift over SLO with no recourse but to finish.

        Returns [(req_id, "demote"|"promote")].

        Three guards, all necessary to avoid thrash:
          1. **Hysteresis** — reuses B2's OK/THROTTLED watermarks, so demotion
             starts only above the high watermark and promotion only below the
             low one. A request cannot be demoted and promoted in the same
             neighbourhood of the SLO.
          2. **Cost model** — a migration moves S*k bytes once. Refuse it unless
             the request has enough decoding left to amortise that move.
          3. **Cooldown** — a request that just migrated is ineligible for
             `_migrate_cooldown_steps`, which bounds migrations/second even if
             the ITL signal oscillates.
        """
        if self._tier_policy != "scored":
            return []
        cfg = self.scheduler_config
        if not getattr(cfg, "enable_hybrid_kv_tiers", False):
            return []
        slo = float(cfg.transfer_aware_slo_p99_itl_ms or 0.0)
        ema = self._resident_step_ms_ema
        if slo <= 0 or ema <= 0:
            return []

        t = time.monotonic() if now is None else now
        step = self._schedule_step_index
        out: list[tuple[str, str]] = []
        if step < self._migrate_warmup_steps and self._probe is None:
            return []                      # cold EMA -> no baseline to judge by

        # --- adaptive probe: did the last demotion actually help? ----------
        if self._probe is not None:
            if step - self._probe["step"] < self._probe_steps:
                return []                      # settling; do not stack changes
            base = self._probe["baseline_ema"]
            worse = ema > base * (1.0 + self._probe_margin)
            live = [r for r in running_requests
                    if r.request_id in self._probe["req_ids"]
                    and self._req_tier.get(r.request_id) == "cpu"]
            self._probe_results.append((base, ema, worse))
            self._probe = None
            if worse:
                # REVERT. The model said the bytes fit; the measurement says the
                # step still got slower (fixed per-step overhead the roofline
                # cannot see). Undo, and back off exponentially so we do not
                # re-learn the same thing every few steps.
                self._demote_suppressed_until = step + self._demote_backoff
                self._demote_backoff = min(self._demote_backoff * 2, 100000)
                self.record_skip("probe_demotion_harmful")
                for r in live:
                    out.append((r.request_id, "promote"))
                return out
            self.record_skip("probe_demotion_helpful")

        def eligible(r) -> bool:
            last = self._migrated_at_step.get(r.request_id)
            if last is not None and (step - last) < self._migrate_cooldown_steps:
                return False
            return self._amortises(r)

        if self._itl_throttled_state:
            if step < self._demote_suppressed_until:
                # Learned (by measurement) that demotion hurts in this regime.
                self.record_skip("migrate_demote_suppressed")
                return []
            # Over the SLO: demote the best host-tier candidates from HBM.
            cands = [
                r for r in running_requests
                if self._req_tier.get(r.request_id) == "hbm" and eligible(r)
            ]
            cands.sort(key=lambda r: self.host_affinity(r, now=t), reverse=True)
            for r in cands[: self._max_migrations_per_step]:
                # Only demote if the extra transfer still HIDES. Otherwise
                # demotion trades a fast HBM read for an interconnect read that
                # cannot hide and makes ITL worse -- measured on A100/PCIe,
                # where the burst-recovery loop demoted 8 requests and drove ITL
                # from ~6 to ~24 ms without ever converging.
                if not self._demotion_helps(r, running_requests):
                    self.record_skip("migrate_demote_no_hide_budget")
                    continue
                out.append((r.request_id, "demote"))
            if out:
                # Open a probe: remember the pre-demotion ITL so we can tell
                # whether this actually helped.
                self._probe = {
                    "step": step,
                    # best recent steady state, not the instantaneous EMA
                    "baseline_ema": min(self._ema_window) if self._ema_window
                    else ema,
                    "req_ids": {rid for rid, _ in out},
                }
        elif ema < slo * 0.70:
            # Comfortably under the SLO: pull back the worst host-tier tenants.
            cands = [
                r for r in running_requests
                if self._req_tier.get(r.request_id) == "cpu" and eligible(r)
            ]
            cands.sort(key=lambda r: self.host_affinity(r, now=t))
            for r in cands[: self._max_migrations_per_step]:
                out.append((r.request_id, "promote"))
        return out

    def hide_budget_ms(self) -> float:
        """How long a forward iteration lasts, in ms -- B1's hide budget.

        Prefers the MEASURED resident-only step time over the configured
        `--hybrid-itl-ms`. The configured value was a hand-set constant that
        every harness had to guess, and guessing it high (100000, "capacity
        question, not an SLO question") disables the gate entirely. The
        interconnect budget is a property of the machine and the model, so it
        should be observed, not declared.

        Only steps with NO host-tier request running are sampled, because a
        step already slowed by rider transfer would otherwise enlarge the
        budget that admitted those riders -- a loop that inflates itself.
        Falls back to the configured value until enough clean samples exist.
        """
        cfg_ms = float(self.scheduler_config.hybrid_itl_ms or 0.0)
        if (self._measured_budget_enabled
                and self._resident_only_samples >= self._measured_budget_min_samples
                and self._resident_only_ms_ema > 0):
            if not self._measured_budget_announced:
                self._measured_budget_announced = True
                logger.info(
                    "B1 hide budget now MEASURED: %.2f ms per forward iteration "
                    "(%d clean resident-only steps); configured value was "
                    "%.2f ms", self._resident_only_ms_ema,
                    self._resident_only_samples, cfg_ms)
            return self._resident_only_ms_ema
        if cfg_ms > 0:
            return cfg_ms
        predictor = self._maybe_load_predictor()
        if predictor is not None:
            try:
                return float(predictor.predict_p99_itl_ms(1))
            except Exception:                                  # noqa: BLE001
                pass
        return 16.0

    def hide_budget_source(self) -> str:
        return ("measured" if self._measured_budget_announced else "configured")

    def hide_budget_headroom_bytes(self, running_requests: "list[Request]") -> float:
        """Bytes of per-step transfer still available under the B1 hide budget.

        Demotion is only *beneficial* while the extra transfer it creates can
        hide inside the step. Above that, moving a request to the host tier
        replaces a fast HBM read with an interconnect read that cannot hide,
        which makes ITL WORSE. On PCIe the budget is tiny, so this is usually
        <= 0 and demotion is correctly refused.
        """
        from vllm.v1.hybrid_kv_roofline import effective_pcie_gbps

        cfg = self.scheduler_config
        itl_ms = self.hide_budget_ms()
        b_tok = self._b_tok_cached
        x_eff = effective_pcie_gbps(float(cfg.hybrid_pcie_gbps or 0.0))
        budget = (itl_ms / 1000.0) * x_eff * 1e9 * (1.0 - self._budget_margin())
        used = float(b_tok) * sum(
            int(getattr(r, "num_computed_tokens", 0) or 0)
            for r in running_requests
            if self._req_tier.get(r.request_id) == "cpu"
        )
        return budget - used

    def _demotion_helps(self, request: "Request",
                        running_requests: "list[Request]") -> bool:
        """Would demoting this request actually relieve the step, or add to it?"""
        b_tok = self._b_tok_cached
        need = float(b_tok) * int(getattr(request, "num_computed_tokens", 0) or 0)
        return need <= self.hide_budget_headroom_bytes(running_requests)

    def _amortises(self, request: "Request") -> bool:
        """Cost model: is there enough decoding left to pay for the move?"""
        sp = getattr(request, "sampling_params", None)
        max_toks = int(getattr(sp, "max_tokens", 0) or 0) if sp else 0
        if max_toks <= 0:
            return True            # unknown horizon -> allow, cooldown bounds it
        done = int(getattr(request, "num_output_tokens", 0) or 0)
        return (max_toks - done) >= self._migrate_min_remaining

    def note_migrated(self, req_id: str, new_tier: str) -> None:
        """Record a completed migration (sets the tier and starts the cooldown)."""
        self._req_tier[req_id] = new_tier
        self._migrated_at_step[req_id] = self._schedule_step_index
        self._migration_counts[new_tier] = (
            self._migration_counts.get(new_tier, 0) + 1
        )

    def migration_stats(self) -> dict[str, int]:
        return dict(self._migration_counts)

    def _budget_margin(self) -> float:
        raw = os.environ.get("VLLM_KVPIPE_HYBRID_BUDGET_MARGIN", "").strip()
        if raw:
            try:
                return min(0.9, max(0.0, float(raw)))
            except ValueError:
                pass
        return 0.1

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
            effective_pcie_gbps,
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

        # Item 2: WHICH tier this request should occupy. Default policy is
        # "arrival" (first hbm_max requests get HBM) -- the pre-Item-2
        # behaviour, kept as the ablation baseline. Under "scored" a request
        # with high host-affinity yields its HBM slot even when one is free.
        prefer_host = (
            self._tier_policy == "scored"
            and hbm_max > 0
            and self.host_affinity(request) >= self._host_affinity_threshold()
        )
        if hbm_max > 0 and hbm_count < hbm_max and not prefer_host:
            self._req_tier[request.request_id] = "hbm"
            return True, "hybrid_hbm"

        # B2 closed-loop guardrail: if the measured resident ITL has drifted over
        # the SLO, admit no new CPU-tier requests until it recovers.
        if self._itl_throttled_state:
            return False, "hybrid_itl_throttled"

        # Hide budget: how long one forward iteration lasts. Measured from
        # steps with no rider running when enough of those exist, else the
        # configured value. Never the live EMA -- that already contains the
        # rider transfer, so feeding it back here would be circular.
        budget_ms = self.hide_budget_ms()

        b_tok = self._b_tok_cached
        x_eff = effective_pcie_gbps(float(cfg.hybrid_pcie_gbps or 0.0))

        # B1 seq-length-aware roofline: the per-step CPU->GPU transfer is the
        # cohort re-streaming its FULL history each step, so it scales with each
        # request's ACTUAL sequence length (not a fixed seq_tokens). Admit while
        # the cumulative transfer stays under the hide budget. Re-evaluated on
        # every admission, so the count shrinks as running sequences grow.
        budget_bytes = (budget_ms / 1000.0) * x_eff * 1e9 * (1.0 - self._budget_margin())
        cpu_bytes = float(b_tok) * sum(
            int(getattr(r, "num_computed_tokens", 0) or 0)
            for r in running_requests
            if self._req_tier.get(r.request_id) == "cpu"
        )
        new_bytes = float(b_tok) * int(getattr(request, "num_prompt_tokens", 0) or 0)

        env_cap = max_offload_from_env()
        over_env_cap = env_cap is not None and cpu_count >= env_cap
        if (not over_env_cap) and (cpu_bytes + new_bytes) <= budget_bytes:
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
