# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.transfer_aware_admission import TransferAwareAdmissionController


# ---------------------------------------------------------------- pre-Phase-4


def test_transfer_aware_admission_disabled_is_inert():
    cfg = SchedulerConfig.default_factory()
    assert cfg.enable_transfer_aware_admission is False
    ctrl = TransferAwareAdmissionController(cfg)
    ctrl.begin_schedule_step()
    assert ctrl.estimate_running_transfer_pressure_bytes_per_step([]) == 0.0
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[],
        token_budget=1,
    )
    assert admit and reason == "disabled"


def test_transfer_aware_admission_pressure_estimate():
    cfg = SchedulerConfig.default_factory()
    cfg.enable_transfer_aware_admission = True
    cfg.transfer_aware_bytes_per_decode_step_per_seq = 1024.0
    ctrl = TransferAwareAdmissionController(cfg)
    running = [Mock(), Mock()]
    assert ctrl.estimate_running_transfer_pressure_bytes_per_step(running) == 2048.0


def test_transfer_aware_admission_stub_always_admits_when_enabled():
    cfg = SchedulerConfig.default_factory()
    cfg.enable_transfer_aware_admission = True
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[],
        token_budget=0,
    )
    assert admit and reason == "stub_allow_all"


# -------------------------------------------------------------- Phase 4 policy
#
# A few small, fast tests that drive the policy with a synthetic predictor CSV.


_PREDICTOR_MODEL = "test/synthetic-model"


def _write_predictor_csv(
    tmp_path: Path,
    *,
    rows: list[tuple[int, float]],
    model: str = _PREDICTOR_MODEL,
) -> Path:
    """Write a feature_on_k_sweep-style CSV with (K, p99_itl_ms) rows."""
    p = tmp_path / "predictor.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model", "input_len", "output_len", "K", "k_window",
                "n_repeats", "p99_itl_ms", "mean_itl_ms", "ttft_ms",
                "throughput_tok_s",
            ],
        )
        w.writeheader()
        for K, p99 in rows:
            w.writerow({
                "model": model,
                "input_len": 1024,
                "output_len": 1024,
                "K": K,
                "k_window": 4,
                "n_repeats": 1,
                "p99_itl_ms": p99,
                "mean_itl_ms": p99 / 10.0,
                "ttft_ms": 50.0,
                "throughput_tok_s": K * 5.0,
            })
    return p


def _make_cfg(
    *,
    csv_path: Path | str = "",
    slo_ms: float = 0.0,
    model: str = _PREDICTOR_MODEL,
) -> SchedulerConfig:
    cfg = SchedulerConfig.default_factory()
    cfg.enable_transfer_aware_admission = True
    cfg.transfer_aware_slo_p99_itl_ms = slo_ms
    cfg.transfer_aware_predictor_csv = str(csv_path)
    cfg.transfer_aware_predictor_model = model
    return cfg


def test_policy_admits_when_predicted_below_slo(tmp_path: Path):
    """Predictor says K=11 -> 30 ms. SLO is 100 ms. -> admit."""
    csv_p = _write_predictor_csv(
        tmp_path, rows=[(8, 20.0), (16, 30.0), (32, 40.0), (64, 70.0)],
    )
    cfg = _make_cfg(csv_path=csv_p, slo_ms=100.0)
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 10,
        token_budget=1,
    )
    assert admit
    assert reason == "ok"


def test_policy_refuses_when_predicted_above_slo(tmp_path: Path):
    """Predictor says K=65 -> ~70.5 ms. SLO is 50 ms. -> refuse."""
    csv_p = _write_predictor_csv(
        tmp_path, rows=[(8, 20.0), (16, 30.0), (32, 40.0), (64, 70.0)],
    )
    cfg = _make_cfg(csv_path=csv_p, slo_ms=50.0)
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 64,
        token_budget=1,
    )
    assert not admit
    assert reason == "predicted_slo_breach"


def test_policy_uses_K_plus_one_not_K(tmp_path: Path):
    """Boundary: at K_running=15 the post-admit K is 16, predictor returns 30 ms.

    SLO=29 -> refuse (16 maps to 30 > 29). SLO=31 -> admit.
    """
    csv_p = _write_predictor_csv(
        tmp_path, rows=[(8, 20.0), (16, 30.0), (32, 40.0)],
    )
    for slo, want in [(29.0, False), (31.0, True)]:
        cfg = _make_cfg(csv_path=csv_p, slo_ms=slo)
        ctrl = TransferAwareAdmissionController(cfg)
        admit, _ = ctrl.should_admit_from_waiting(
            request=Mock(),
            running_requests=[Mock()] * 15,
            token_budget=1,
        )
        assert admit is want, f"SLO={slo} expected admit={want} got admit={admit}"


def test_policy_with_slo_zero_falls_back_to_stub(tmp_path: Path):
    csv_p = _write_predictor_csv(tmp_path, rows=[(8, 20.0), (16, 30.0)])
    cfg = _make_cfg(csv_path=csv_p, slo_ms=0.0)  # SLO unset
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 1000,
        token_budget=1,
    )
    assert admit and reason == "stub_allow_all"


def test_policy_with_missing_csv_falls_back_to_stub(tmp_path: Path):
    """Misconfigured node should never fail admissions."""
    cfg = _make_cfg(csv_path=tmp_path / "does_not_exist.csv", slo_ms=20.0)
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 5,
        token_budget=1,
    )
    assert admit and reason == "stub_allow_all"


def test_policy_with_csv_no_matching_model_falls_back_to_stub(tmp_path: Path):
    csv_p = _write_predictor_csv(
        tmp_path, rows=[(8, 20.0), (16, 30.0)], model="some/other-model",
    )
    cfg = _make_cfg(csv_path=csv_p, slo_ms=20.0, model=_PREDICTOR_MODEL)
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 5,
        token_budget=1,
    )
    assert admit and reason == "stub_allow_all"


def test_policy_auto_picks_model_when_unset(tmp_path: Path):
    """Empty predictor_model picks the first model in the CSV."""
    csv_p = _write_predictor_csv(tmp_path, rows=[(8, 20.0), (16, 30.0)])
    cfg = _make_cfg(csv_path=csv_p, slo_ms=29.0, model="")
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 16,
        token_budget=1,
    )
    # K_post_admit=17 interpolates 8->20, 16->30 => extrapolates to ~32 ms.
    # Linear interp (clamped at top): K=17 > 16 -> clamps to 30 ms <= 29? No, 30>29 -> refuse.
    assert not admit and reason == "predicted_slo_breach"


def test_policy_lazy_loads_predictor_only_once(tmp_path: Path):
    """Second admission decision should not re-read the CSV."""
    csv_p = _write_predictor_csv(tmp_path, rows=[(8, 20.0), (16, 30.0)])
    cfg = _make_cfg(csv_path=csv_p, slo_ms=100.0)
    ctrl = TransferAwareAdmissionController(cfg)
    assert not ctrl._predictor_loaded
    ctrl.should_admit_from_waiting(request=Mock(), running_requests=[], token_budget=1)
    assert ctrl._predictor_loaded
    first_predictor = ctrl._predictor
    # Mutate CSV to give different ITL values; controller must not reload.
    csv_p.write_text("model,K,p99_itl_ms\nbogus,1,1\n")
    ctrl.should_admit_from_waiting(request=Mock(), running_requests=[], token_budget=1)
    assert ctrl._predictor is first_predictor


def test_record_skip_counts_predicted_breach(tmp_path: Path):
    """Refusal events get logged via record_skip / drain_events."""
    csv_p = _write_predictor_csv(tmp_path, rows=[(8, 20.0), (16, 30.0), (32, 40.0)])
    cfg = _make_cfg(csv_path=csv_p, slo_ms=25.0)
    ctrl = TransferAwareAdmissionController(cfg)
    ctrl.begin_schedule_step()
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(), running_requests=[Mock()] * 30, token_budget=1,
    )
    assert not admit and reason == "predicted_slo_breach"
    ctrl.record_skip(reason)
    events = ctrl.drain_events()
    assert events.get("predicted_slo_breach") == 1


def test_policy_mechanistic_loads_from_artifacts(tmp_path: Path):
    """Mechanistic policy uses baseline JSON + feature CSV."""
    csv_p = _write_predictor_csv(
        tmp_path, rows=[(8, 20.0), (16, 100.0), (32, 200.0), (64, 400.0)],
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "throughput": [
            {"concurrency": 8, "concurrent_global_p99_itl_s": 0.01},
            {"concurrency": 16, "concurrent_global_p99_itl_s": 0.02},
            {"concurrency": 32, "concurrent_global_p99_itl_s": 0.03},
            {"concurrency": 64, "concurrent_global_p99_itl_s": 0.04},
        ],
    }))
    cfg = _make_cfg(csv_path=csv_p, slo_ms=500.0)
    cfg.transfer_aware_admission_policy = "mechanistic"
    cfg.transfer_aware_baseline_compute_json = str(baseline)
    ctrl = TransferAwareAdmissionController(cfg)
    admit, reason = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * 10,
        token_budget=1,
    )
    assert admit and reason == "ok"
    assert ctrl._predictor is not None
    assert hasattr(ctrl._predictor, "decompose")


@pytest.mark.parametrize("running, want_admit", [
    (5, True),    # K_post=6 -> below clamp -> 20.0 ms (lowest table) -> admit
    (50, False),  # K_post=51 -> 56.6 ms -> refuse
])
def test_policy_parametrized_admit_thresholds(tmp_path: Path, running: int, want_admit: bool):
    csv_p = _write_predictor_csv(
        tmp_path, rows=[(8, 20.0), (16, 30.0), (32, 40.0), (64, 70.0)],
    )
    cfg = _make_cfg(csv_path=csv_p, slo_ms=50.0)
    ctrl = TransferAwareAdmissionController(cfg)
    admit, _ = ctrl.should_admit_from_waiting(
        request=Mock(),
        running_requests=[Mock()] * running,
        token_budget=1,
    )
    assert admit is want_admit
