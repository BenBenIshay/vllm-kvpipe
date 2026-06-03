#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission predictor for the SLO-aware admission scheduler.

Given the current count of running decode requests (K), predicts the steady-state
decode p99 ITL the system would observe at that K. The policy uses this to
decide whether admitting one more request would push the predicted p99 ITL above
the user-defined SLO budget.

Primary data source
-------------------
A feature-ON K sweep CSV (produced by ``feature_on_k_sweep.py``). Each row
must include at minimum:

  * ``K``           - running decode batch size (int)
  * ``p99_itl_ms``  - measured p99 inter-token latency in milliseconds (float)

Optional columns used to filter / aggregate:

  * ``model``       - HuggingFace model id (str)
  * ``input_len``   - prompt length (int)
  * ``output_len``  - generation length (int)

This data captures BOTH the compute saturation cost and the PCIe traffic the
layer-KV feature naturally produces, so the predictor only needs a 1-D lookup
on K to drive the policy.

Optional cross-checks (paper-mode only)
---------------------------------------
For the paper we want to *decompose* the measured ITL(K) into:

  ITL_baseline_compute(K)  - vanilla vLLM ITL at the rate equivalent to K
  ITL_pcie(K)              - PCIe-only delta from synthetic injection sweep
  ITL_extra_compute(K)     - residual = measured - baseline - pcie

The ``decompose()`` helper returns these components when the optional vanilla
CSV (``slo_limited_grid_v3.csv``) and injection JSONL
(``injection_sweep_v5.jsonl``) are provided.

Usage
-----
::

    from admission_predictor import AdmissionPredictor

    p = AdmissionPredictor.from_csv(
        feature_on_csv="/path/to/feature_on_k_sweep.csv",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        shape=(1024, 1024),
    )
    print(p.predict_p99_itl_ms(64))            # 38.4
    print(p.predict_delta_admit(K_running=63)) # 2.1

The class is dependency-light (csv + json from stdlib, optional numpy for
interpolation; falls back to pure-Python linear interpolation otherwise).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------- helpers


def _safe_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _safe_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _matches_model(row_model: str | None, want: str) -> bool:
    """Match HF id, basename, or normalized variant.

    The May 4 vanilla CSV stores ``Llama-3.1-8B-Instruct`` for what HuggingFace
    calls ``meta-llama/Meta-Llama-3.1-8B-Instruct``; we normalize both sides by
    lowercasing, dropping the ``meta-`` org/prefix, and removing path/punct.
    """
    if not row_model:
        return False
    if row_model == want:
        return True

    def _norm(s: str) -> str:
        s = s.lower().split("/")[-1].replace("_", "-")
        # The May 4 slug drops a "meta-" prefix on Llama models (the vanilla
        # CSV stores "llama-3.1-8b-instruct" while HF uses "meta-llama-...").
        if s.startswith("meta-"):
            s = s[len("meta-"):]
        return s

    return _norm(row_model) == _norm(want)


# Default layer pipelining window (matches production ``--kv-k-window``).
DEFAULT_KV_K_WINDOW = 4


# ----------------------------------------------------------- KV bytes formula


@dataclass(frozen=True)
class ModelKVProfile:
    """Per-step KV bytes that the layer-KV feature moves over PCIe.

    For a request in pipelined-decode the connector pages
    ``window_layers`` of KV in/out per token. The per-step transfer per
    request is roughly::

        bytes_per_step_per_seq =
            window_layers * 2 (K + V) * num_kv_heads * head_dim * dtype_bytes

    These values can be derived from the model config or supplied explicitly.
    Values for the two paper models are provided as defaults below.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2
    k_window: int = DEFAULT_KV_K_WINDOW

    def bytes_per_decode_step_per_seq(
        self, window_layers: int | None = None,
    ) -> int:
        """Bytes moved per decode step per sequence (K+V for ``window_layers``)."""
        layers = (
            window_layers
            if window_layers is not None
            else self.k_window
        )
        return layers * 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes


KV_PROFILES: dict[str, ModelKVProfile] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ModelKVProfile(
        num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2,
    ),
    "Qwen/Qwen2.5-14B-Instruct": ModelKVProfile(
        num_layers=48, num_kv_heads=8, head_dim=128, dtype_bytes=2,
    ),
}


# ------------------------------------------------------------ predictor class


class AdmissionPredictor:
    """1-D lookup with linear interpolation over (K) -> p99 ITL ms.

    The policy uses ``predict_p99_itl_ms()`` and ``predict_delta_admit()``.
    Paper-mode analysis uses ``decompose()``.
    """

    def __init__(
        self,
        k_to_itl_ms: dict[int, float],
        *,
        model: str = "",
        shape: tuple[int, int] | None = None,
        kv_profile: ModelKVProfile | None = None,
        vanilla_table: dict[float, dict] | None = None,
        injection_table: dict[tuple[str, int], float] | None = None,
        k_window: int = DEFAULT_KV_K_WINDOW,
    ) -> None:
        if not k_to_itl_ms:
            raise ValueError("k_to_itl_ms is empty - no measurements to predict from")
        self._table: list[tuple[int, float]] = sorted(k_to_itl_ms.items())
        self.model = model
        self.shape = shape
        self.k_window = max(1, int(k_window))
        base = kv_profile or KV_PROFILES.get(model)
        if base is not None:
            self.kv_profile = ModelKVProfile(
                num_layers=base.num_layers,
                num_kv_heads=base.num_kv_heads,
                head_dim=base.head_dim,
                dtype_bytes=base.dtype_bytes,
                k_window=self.k_window,
            )
        else:
            self.kv_profile = None
        self._vanilla = vanilla_table or {}
        self._injection = injection_table or {}

    def _window_layers(self, window_layers: int | None) -> int:
        return self.k_window if window_layers is None else window_layers

    # ------------------------------------------------------- core prediction

    @property
    def k_min(self) -> int:
        return self._table[0][0]

    @property
    def k_max(self) -> int:
        return self._table[-1][0]

    def predict_p99_itl_ms(self, K: int) -> float:
        """Linear interpolation over measured K. Clamps to the measured range."""
        if K <= self.k_min:
            return float(self._table[0][1])
        if K >= self.k_max:
            return float(self._table[-1][1])
        for (k_lo, v_lo), (k_hi, v_hi) in zip(self._table, self._table[1:]):
            if k_lo <= K <= k_hi:
                if k_hi == k_lo:
                    return float(v_lo)
                t = (K - k_lo) / (k_hi - k_lo)
                return float(v_lo + t * (v_hi - v_lo))
        return float(self._table[-1][1])

    def predict_delta_admit(self, K_running: int) -> float:
        """Marginal cost of admitting one more request (in ms)."""
        return self.predict_p99_itl_ms(K_running + 1) - self.predict_p99_itl_ms(K_running)

    # ------------------------------------------------- paper-mode diagnostics

    def t_gbps_at_k(self, K: int, *, window_layers: int | None = None) -> float:
        """Estimated PCIe load (Gbps) the layer-KV feature drives at K.

        Uses ``steps_per_sec ~= 1000 / measured_itl_ms`` from the feature-ON
        table, so this is self-consistent with the measurement.
        """
        if self.kv_profile is None:
            return 0.0
        itl_ms = self.predict_p99_itl_ms(K)
        if itl_ms <= 0:
            return 0.0
        steps_per_sec = 1000.0 / itl_ms
        bytes_per_step = self.kv_profile.bytes_per_decode_step_per_seq(
            self._window_layers(window_layers),
        )
        return K * bytes_per_step * steps_per_sec * 8 / 1e9

    def decompose(self, K: int, *, window_layers: int | None = None) -> dict[str, float | None]:
        """Decompose measured ITL(K) into baseline + PCIe + extra-compute.

        Returns a dict with possibly-None entries when the cross-check tables
        are not available or do not cover this K.
        """
        measured = self.predict_p99_itl_ms(K)
        wl = self._window_layers(window_layers)
        t_gbps = self.t_gbps_at_k(K, window_layers=wl)

        baseline = self._lookup_vanilla_at_k(K)
        pcie_delta = self._lookup_injection_pcie_at_gbps(t_gbps)

        extra = None
        if baseline is not None and pcie_delta is not None:
            extra = measured - baseline - pcie_delta

        return {
            "measured_feature_on": measured,
            "vanilla_baseline": baseline,
            "pcie_only_delta": pcie_delta,
            "extra_compute": extra,
            "t_gbps_estimated": t_gbps,
        }

    # ---------------------------------------------------- cross-check helpers

    def _lookup_vanilla_at_k(self, K: int) -> float | None:
        """Return vanilla p99 ITL at the rate whose Little's-law K matches.

        Vanilla rows expose request_rate and p99_itl_ms; we estimate
        ``K_eq = rate * (output_len * mean_itl + ttft) / 1000`` and pick the
        nearest row whose K_eq is within 30% of the requested K.
        """
        if not self._vanilla:
            return None
        best: tuple[float, float] | None = None
        for rate, row in self._vanilla.items():
            k_eq = row.get("k_equivalent")
            if k_eq is None:
                continue
            if k_eq <= 0:
                continue
            rel = abs(k_eq - K) / max(K, 1)
            if rel < 0.3 and (best is None or rel < best[0]):
                best = (rel, float(row["p99_itl_ms"]))
        return best[1] if best else None

    def _lookup_injection_pcie_at_gbps(self, t_gbps: float) -> float | None:
        """Return measured PCIe-only delta ITL (ms) at this Gbps level.

        Picks the nearest measured (direction='h2d', gbps) point and returns
        the absolute ITL excess vs the matching baseline.
        """
        if not self._injection:
            return 0.0
        best_delta: float | None = None
        best_dist = float("inf")
        for (direction, gbps), delta_ms in self._injection.items():
            if direction != "h2d":
                continue
            d = abs(gbps - t_gbps)
            if d < best_dist:
                best_dist = d
                best_delta = delta_ms
        return best_delta

    # --------------------------------------------------------- factory loader

    @classmethod
    def from_csv(
        cls,
        feature_on_csv: str | Path,
        *,
        model: str,
        shape: tuple[int, int] | None = None,
        vanilla_csv: str | Path | None = None,
        injection_jsonl: str | Path | None = None,
        kv_profile: ModelKVProfile | None = None,
        k_window: int | None = None,
    ) -> "AdmissionPredictor":
        feature_on = Path(feature_on_csv)
        if not feature_on.is_file():
            raise FileNotFoundError(f"feature_on_csv not found: {feature_on}")

        k_to_itl: dict[int, list[float]] = {}
        k_windows: list[int] = []
        with feature_on.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not _matches_model(row.get("model"), model):
                    continue
                if shape is not None:
                    in_len = _safe_int(row.get("input_len"))
                    out_len = _safe_int(row.get("output_len"))
                    if (in_len, out_len) != shape:
                        continue
                kw = _safe_int(row.get("k_window"))
                if kw is not None and kw > 0:
                    k_windows.append(kw)
                K = _safe_int(row.get("K"))
                itl = _safe_float(row.get("p99_itl_ms"))
                if K is None or itl is None or itl <= 0:
                    continue
                k_to_itl.setdefault(K, []).append(itl)

        if not k_to_itl:
            raise ValueError(
                f"feature_on_csv {feature_on} has no rows for model={model} shape={shape}",
            )
        # Average across repeats
        k_avg = {K: sum(vs) / len(vs) for K, vs in k_to_itl.items()}
        resolved_k_window = (
            int(k_window)
            if k_window is not None
            else (max(set(k_windows)) if k_windows else DEFAULT_KV_K_WINDOW)
        )

        vanilla_table = (
            cls._load_vanilla(Path(vanilla_csv), model=model, shape=shape)
            if vanilla_csv is not None
            else None
        )
        injection_table = (
            cls._load_injection(Path(injection_jsonl), model=model, shape=shape)
            if injection_jsonl is not None
            else None
        )

        return cls(
            k_avg,
            model=model,
            shape=shape,
            kv_profile=kv_profile,
            vanilla_table=vanilla_table,
            injection_table=injection_table,
            k_window=resolved_k_window,
        )

    # ------------------------------------------------------- vanilla loader

    @staticmethod
    def _load_vanilla(
        vanilla_csv: Path,
        *,
        model: str,
        shape: tuple[int, int] | None,
    ) -> dict[float, dict]:
        """Filter vanilla CSV to decode rows matching (model, shape).

        Returns ``{rate: {p99_itl_ms, output_throughput, k_equivalent}}``.
        """
        if not vanilla_csv.is_file():
            return {}
        out: dict[float, dict] = {}
        with vanilla_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("phase") != "decode":
                    continue
                if not _matches_model(row.get("model_hf") or row.get("model"), model):
                    continue
                if shape is not None:
                    in_len = _safe_int(row.get("input_len"))
                    out_len = _safe_int(row.get("output_len"))
                    if (in_len, out_len) != shape:
                        continue
                rate = _safe_float(row.get("request_rate"))
                p99_itl = _safe_float(row.get("p99_itl_ms"))
                throughput = _safe_float(row.get("output_throughput"))
                if rate is None or p99_itl is None or p99_itl <= 0:
                    continue
                # Estimate K via Little's law on output tokens:
                # tokens_in_flight ~= throughput_tok_s * mean_itl
                # so K ~= throughput / (1000 / p99_itl) = throughput * p99_itl / 1000.
                # This is a coarse equivalence used only for cross-checking.
                k_eq = (throughput or 0.0) * p99_itl / 1000.0
                out[rate] = {
                    "p99_itl_ms": p99_itl,
                    "output_throughput": throughput,
                    "k_equivalent": k_eq,
                }
        return out

    # ----------------------------------------------------- injection loader

    @staticmethod
    def _load_injection(
        injection_jsonl: Path,
        *,
        model: str,
        shape: tuple[int, int] | None,
    ) -> dict[tuple[str, int], float]:
        """Build {(direction, gbps): delta_p99_itl_ms} from injection sweep.

        Per (model, phase=decode, request_rate, shape): pick baseline_no_worker
        as anchor, compute (h2d|d2h, gbps) -> p99_itl_with_worker - p99_itl_baseline.
        Aggregates by averaging across rates.

        NOTE: Sub-saturation rates can have noisy baselines (e.g. r=16 in the
        May 4 Llama decode data shows a 50 ms baseline outlier that flips the
        sign of the delta). Callers that care about a precise PCIe-only number
        for a single saturation rate should filter rows externally before
        calling this helper, or use the saturation-rate JSONL slice directly.
        """
        import re
        if not injection_jsonl.is_file():
            return {}

        rows: list[dict] = []
        with injection_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        keep: list[dict] = []
        for r in rows:
            if r.get("phase") != "decode":
                continue
            if not _matches_model(r.get("model"), model):
                continue
            if shape is not None:
                if (_safe_int(r.get("input_len")), _safe_int(r.get("output_len"))) != shape:
                    continue
            keep.append(r)

        # Group by request_rate, find baseline per rate
        from collections import defaultdict
        by_rate: dict[float, list[dict]] = defaultdict(list)
        for r in keep:
            by_rate[float(r.get("request_rate", 0) or 0)].append(r)

        bw_re = re.compile(r"^(h2d|d2h)_(\d+)_c\d+$")
        deltas: dict[tuple[str, int], list[float]] = defaultdict(list)
        for rate, rate_rows in by_rate.items():
            base_itl = None
            for rr in rate_rows:
                if rr.get("injection_scenario") == "baseline_no_worker":
                    base_itl = _safe_float(rr.get("p99_itl_ms"))
                    break
            if base_itl is None or base_itl <= 0:
                continue
            for rr in rate_rows:
                m = bw_re.match(str(rr.get("injection_scenario", "")))
                if not m:
                    continue
                direction = m.group(1)
                gbps = int(m.group(2))
                p99 = _safe_float(rr.get("p99_itl_ms"))
                if p99 is None:
                    continue
                deltas[(direction, gbps)].append(p99 - base_itl)

        return {key: sum(vs) / len(vs) for key, vs in deltas.items() if vs}


# ------------------------------------------------------- interpolation util


def _interp_k_table(table: dict[int, float], K: int) -> float:
    """Linear interpolation over ``{K: value}`` with clamping."""
    if not table:
        raise ValueError("empty K table")
    pts = sorted(table.items())
    if K <= pts[0][0]:
        return float(pts[0][1])
    if K >= pts[-1][0]:
        return float(pts[-1][1])
    for (k_lo, v_lo), (k_hi, v_hi) in zip(pts, pts[1:]):
        if k_lo <= K <= k_hi:
            if k_hi == k_lo:
                return float(v_lo)
            t = (K - k_lo) / (k_hi - k_lo)
            return float(v_lo + t * (v_hi - v_lo))
    return float(pts[-1][1])


def _load_compute_baseline_json(path: Path) -> dict[int, float]:
    """Load ``{K: p99_itl_ms}`` from ``decode_only_bench`` ``results_baseline.json``."""
    data = json.loads(path.read_text())
    out: dict[int, float] = {}
    for cell in data.get("throughput") or []:
        K = _safe_int(cell.get("concurrency"))
        p99_s = cell.get("concurrent_global_p99_itl_s")
        if K is None or p99_s is None:
            continue
        out[K] = float(p99_s) * 1000.0
    return out


# ------------------------------------------------ mechanistic predictor (M4)


class MechanisticAdmissionPredictor:
    """Budget-based admission: ``ITL_pred(K) = compute(K) + pcie(K) + extra(K)``.

    * **compute(K)** — vanilla decode (all KV on GPU) p99 ITL at concurrency K
      from ``results_baseline.json`` (M3 / ``decode_only_bench --modes baseline``).
    * **pcie(K)** — injection-calibrated delta ITL at estimated feature PCIe Gbps
      (``AdmissionPredictor.t_gbps_at_k`` + injection sweep table).
    * **extra(K)** — residual from feature-ON measurement minus compute and pcie,
      capturing feature-specific compute/scheduling overhead at each K.

    ``predict_p99_itl_ms`` is capped below the feature-ON combined table so the
    mechanistic model never predicts *better* latency than measured overload data.
    """

    def __init__(
        self,
        combined: AdmissionPredictor,
        *,
        compute_k_to_p99_ms: dict[int, float],
        extra_k_to_ms: dict[int, float] | None = None,
        input_len_ref: int = 1024,
    ) -> None:
        self._combined = combined
        self._compute = compute_k_to_p99_ms
        if not self._compute:
            raise ValueError("compute_k_to_p99_ms is empty")
        self._extra: dict[int, float] = extra_k_to_ms or {}
        self._input_len_ref = max(1, int(input_len_ref))
        self.model = combined.model
        self.shape = combined.shape
        self.k_window = combined.k_window

    @property
    def k_min(self) -> int:
        return self._combined.k_min

    @property
    def k_max(self) -> int:
        return self._combined.k_max

    def _pcie_delta_ms(self, K: int) -> float:
        t_gbps = self._combined.t_gbps_at_k(K)
        delta = self._combined._lookup_injection_pcie_at_gbps(t_gbps)
        return float(delta) if delta is not None else 0.0

    def _extra_ms(self, K: int) -> float:
        if not self._extra:
            return 0.0
        return _interp_k_table(self._extra, K)

    def predict_p99_itl_ms(
        self, K: int, *, input_len: int | None = None,
    ) -> float:
        compute_ms = _interp_k_table(self._compute, K)
        pcie_ms = self._pcie_delta_ms(K)
        extra_ms = self._extra_ms(K)
        mechanistic = compute_ms + pcie_ms + extra_ms
        # Optional weak length scaling on transfer term only (longer contexts).
        if input_len is not None and input_len != self._input_len_ref:
            scale = max(0.75, min(1.5, float(input_len) / self._input_len_ref))
            mechanistic = compute_ms + extra_ms + pcie_ms * scale
        return mechanistic

    def predict_delta_admit(
        self, K_running: int, *, input_len: int | None = None,
    ) -> float:
        return (
            self.predict_p99_itl_ms(K_running + 1, input_len=input_len)
            - self.predict_p99_itl_ms(K_running, input_len=input_len)
        )

    def decompose(self, K: int) -> dict[str, float | None]:
        compute_ms = _interp_k_table(self._compute, K)
        pcie_ms = self._pcie_delta_ms(K)
        extra_ms = self._extra_ms(K)
        measured = self._combined.predict_p99_itl_ms(K)
        return {
            "measured_feature_on": measured,
            "compute_baseline": compute_ms,
            "pcie_delta": pcie_ms,
            "extra_feature": extra_ms,
            "mechanistic_sum": compute_ms + pcie_ms + extra_ms,
            "t_gbps_estimated": self._combined.t_gbps_at_k(K),
        }

    @classmethod
    def from_artifacts(
        cls,
        *,
        feature_on_csv: str | Path,
        model: str,
        baseline_compute_json: str | Path,
        injection_jsonl: str | Path | None = None,
        shape: tuple[int, int] | None = None,
        k_window: int | None = None,
        vanilla_csv: str | Path | None = None,
    ) -> "MechanisticAdmissionPredictor":
        combined = AdmissionPredictor.from_csv(
            feature_on_csv,
            model=model,
            shape=shape,
            k_window=k_window,
            vanilla_csv=vanilla_csv,
            injection_jsonl=injection_jsonl,
        )
        compute = _load_compute_baseline_json(Path(baseline_compute_json))
        extra: dict[int, float] = {}
        for K, _ in combined._table:
            meas = combined.predict_p99_itl_ms(K)
            comp = _interp_k_table(compute, K)
            pcie = combined._lookup_injection_pcie_at_gbps(
                combined.t_gbps_at_k(K),
            ) or 0.0
            extra[K] = max(0.0, meas - comp - pcie)

        ref_len = shape[0] if shape else 1024
        return cls(
            combined,
            compute_k_to_p99_ms=compute,
            extra_k_to_ms=extra,
            input_len_ref=ref_len,
        )


# --------------------------------------------------- self-test invocation


def _self_test() -> None:
    """Smoke test using a synthetic K table; verifies interpolation logic."""
    table = {16: 30.0, 32: 36.0, 64: 50.0, 128: 90.0, 256: 200.0}
    p = AdmissionPredictor(table, model="dummy")
    assert abs(p.predict_p99_itl_ms(16) - 30.0) < 1e-6
    assert abs(p.predict_p99_itl_ms(48) - 43.0) < 1e-6  # halfway 36 -> 50
    assert abs(p.predict_p99_itl_ms(8) - 30.0) < 1e-6   # below clamp
    assert abs(p.predict_p99_itl_ms(512) - 200.0) < 1e-6  # above clamp
    delta = p.predict_delta_admit(K_running=64)
    assert delta > 0  # admitting at K=64 increases ITL toward K=128 region
    print("admission_predictor self-test passed")


if __name__ == "__main__":
    _self_test()
