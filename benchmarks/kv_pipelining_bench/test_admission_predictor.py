"""Unit tests for admission_predictor.

Run with:
    /home/ben/vllm/.venv/bin/python -m pytest -q \\
        benchmarks/kv_pipelining_bench/test_admission_predictor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from admission_predictor import (  # noqa: E402
    KV_PROFILES,
    AdmissionPredictor,
    MechanisticAdmissionPredictor,
    ModelKVProfile,
    _load_compute_baseline_json,
)

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# ---------------------------------------------------------- core lookup tests


def test_clamp_below_min_returns_first_point():
    p = AdmissionPredictor({16: 30.0, 64: 50.0, 256: 200.0})
    assert p.predict_p99_itl_ms(8) == 30.0
    assert p.predict_p99_itl_ms(0) == 30.0
    assert p.predict_p99_itl_ms(-5) == 30.0


def test_clamp_above_max_returns_last_point():
    p = AdmissionPredictor({16: 30.0, 64: 50.0, 256: 200.0})
    assert p.predict_p99_itl_ms(512) == 200.0


def test_exact_lookup():
    p = AdmissionPredictor({16: 30.0, 64: 50.0, 256: 200.0})
    assert p.predict_p99_itl_ms(16) == 30.0
    assert p.predict_p99_itl_ms(64) == 50.0
    assert p.predict_p99_itl_ms(256) == 200.0


def test_linear_interpolation_midpoint():
    p = AdmissionPredictor({16: 30.0, 64: 50.0})
    assert p.predict_p99_itl_ms(40) == pytest.approx(40.0, rel=1e-6)


def test_linear_interpolation_off_center():
    p = AdmissionPredictor({64: 50.0, 128: 90.0})
    assert p.predict_p99_itl_ms(80) == pytest.approx(60.0, rel=1e-6)


def test_delta_admit_is_positive_in_growth_region():
    p = AdmissionPredictor({16: 30.0, 64: 50.0, 256: 200.0})
    assert p.predict_delta_admit(K_running=64) > 0


def test_empty_table_raises():
    with pytest.raises(ValueError):
        AdmissionPredictor({})


def test_single_point_table_returns_constant():
    p = AdmissionPredictor({64: 42.0})
    assert p.predict_p99_itl_ms(1) == 42.0
    assert p.predict_p99_itl_ms(64) == 42.0
    assert p.predict_p99_itl_ms(1024) == 42.0
    assert p.predict_delta_admit(K_running=64) == 0.0


# -------------------------------------------------------------- KV profile tests


def test_llama_kv_bytes_per_step_known_value():
    prof = KV_PROFILES[LLAMA]
    # Default uses k_window=4 (layer pipelining), not full depth.
    assert prof.bytes_per_decode_step_per_seq() == 4 * 2 * 8 * 128 * 2
    assert prof.bytes_per_decode_step_per_seq(window_layers=8) == 8 * 2 * 8 * 128 * 2


def test_t_gbps_at_k_scales_with_k_when_itl_constant():
    """Use a flat ITL table so steps_per_sec stays constant; T_gbps must be linear in K."""
    flat = {16: 50.0, 256: 50.0}
    p = AdmissionPredictor(
        flat, model=LLAMA, kv_profile=KV_PROFILES[LLAMA],
    )
    t1 = p.t_gbps_at_k(16)
    t2 = p.t_gbps_at_k(64)
    assert t2 == pytest.approx(4 * t1, rel=1e-6)


def test_t_gbps_at_k_zero_without_profile():
    p = AdmissionPredictor({16: 50.0}, model="unknown")
    p.kv_profile = None
    assert p.t_gbps_at_k(64) == 0.0


# --------------------------------------------------------- CSV factory tests


def _write_feature_on_csv(tmp: Path, rows: list[dict]) -> Path:
    import csv
    p = tmp / "feature_on.csv"
    fields = list(rows[0].keys())
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def test_from_csv_filters_by_model_and_shape(tmp_path: Path):
    rows = [
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 16, "p99_itl_ms": 30.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 50.0},
        {"model": LLAMA, "input_len": 512, "output_len": 1024, "K": 32, "p99_itl_ms": 999.0},
        {"model": "Qwen/Qwen2.5-14B-Instruct", "input_len": 1024, "output_len": 1024, "K": 32, "p99_itl_ms": 888.0},
    ]
    csv_path = _write_feature_on_csv(tmp_path, rows)
    p = AdmissionPredictor.from_csv(
        csv_path, model=LLAMA, shape=(1024, 1024),
    )
    assert p.k_min == 16 and p.k_max == 64
    assert p.predict_p99_itl_ms(40) == pytest.approx(40.0, rel=1e-6)


def test_from_csv_aggregates_repeats(tmp_path: Path):
    rows = [
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 48.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 52.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 128, "p99_itl_ms": 90.0},
    ]
    csv_path = _write_feature_on_csv(tmp_path, rows)
    p = AdmissionPredictor.from_csv(csv_path, model=LLAMA, shape=(1024, 1024))
    assert p.predict_p99_itl_ms(64) == pytest.approx(50.0, rel=1e-6)


def test_from_csv_raises_when_no_matching_rows(tmp_path: Path):
    rows = [{"model": "other", "input_len": 1, "output_len": 1, "K": 1, "p99_itl_ms": 1.0}]
    csv_path = _write_feature_on_csv(tmp_path, rows)
    with pytest.raises(ValueError):
        AdmissionPredictor.from_csv(csv_path, model=LLAMA)


def test_from_csv_handles_short_model_name(tmp_path: Path):
    """Vanilla rows use 'Llama-3.1-8B-Instruct', injection rows use full HF id."""
    rows = [
        {"model": "Llama-3.1-8B-Instruct", "input_len": 1024, "output_len": 1024, "K": 16, "p99_itl_ms": 30.0},
        {"model": "Llama-3.1-8B-Instruct", "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 50.0},
    ]
    csv_path = _write_feature_on_csv(tmp_path, rows)
    p = AdmissionPredictor.from_csv(csv_path, model=LLAMA, shape=(1024, 1024))
    assert p.predict_p99_itl_ms(64) == pytest.approx(50.0, rel=1e-6)


# ---------------------------------------------------- vanilla cross-check tests


def test_vanilla_loader_filters_decode_and_shape(tmp_path: Path):
    import csv
    rows = [
        # decode, target shape -> kept
        {"model_hf": LLAMA, "phase": "decode", "input_len": 1024, "output_len": 1024,
         "request_rate": 4.0, "p99_itl_ms": 35.0, "output_throughput": 1000.0},
        # prefill -> dropped
        {"model_hf": LLAMA, "phase": "prefill", "input_len": 4096, "output_len": 1,
         "request_rate": 16.0, "p99_itl_ms": 0.0, "output_throughput": 6.0},
        # different shape -> dropped
        {"model_hf": LLAMA, "phase": "decode", "input_len": 512, "output_len": 512,
         "request_rate": 4.0, "p99_itl_ms": 99.0, "output_throughput": 50.0},
    ]
    p_csv = tmp_path / "vanilla.csv"
    with p_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    feat_rows = [
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 16, "p99_itl_ms": 30.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 50.0},
    ]
    feat_csv = _write_feature_on_csv(tmp_path, feat_rows)

    p = AdmissionPredictor.from_csv(
        feat_csv, model=LLAMA, shape=(1024, 1024), vanilla_csv=p_csv,
    )
    assert len(p._vanilla) == 1
    rate, body = next(iter(p._vanilla.items()))
    assert rate == 4.0
    assert body["p99_itl_ms"] == 35.0
    assert body["k_equivalent"] == pytest.approx(1000.0 * 35.0 / 1000.0)


def test_injection_loader_builds_delta_table(tmp_path: Path):
    inj = tmp_path / "inj.jsonl"
    base = {"model": LLAMA, "phase": "decode", "input_len": 1024, "output_len": 1024,
            "request_rate": 16.0, "injection_scenario": "baseline_no_worker",
            "p99_itl_ms": 25.0}
    h2d16 = {**base, "injection_scenario": "h2d_16_c1", "p99_itl_ms": 25.5}
    h2d56 = {**base, "injection_scenario": "h2d_56_c1", "p99_itl_ms": 27.0}
    other_phase = {**base, "phase": "prefill"}
    inj.write_text("\n".join(json.dumps(r) for r in [base, h2d16, h2d56, other_phase]))

    feat_rows = [{"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 16, "p99_itl_ms": 30.0},
                 {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 50.0}]
    feat_csv = _write_feature_on_csv(tmp_path, feat_rows)

    p = AdmissionPredictor.from_csv(
        feat_csv, model=LLAMA, shape=(1024, 1024), injection_jsonl=inj,
    )
    assert p._injection[("h2d", 16)] == pytest.approx(0.5)
    assert p._injection[("h2d", 56)] == pytest.approx(2.0)


def test_decompose_returns_all_keys(tmp_path: Path):
    """Smoke test: decompose() returns a fully-populated dict shape."""
    feat_rows = [
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 16, "p99_itl_ms": 30.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 1024, "K": 64, "p99_itl_ms": 50.0},
    ]
    feat_csv = _write_feature_on_csv(tmp_path, feat_rows)
    p = AdmissionPredictor.from_csv(feat_csv, model=LLAMA, shape=(1024, 1024))
    out = p.decompose(K=32)
    assert set(out.keys()) == {
        "measured_feature_on", "vanilla_baseline", "pcie_only_delta",
        "extra_compute", "t_gbps_estimated",
    }
    assert out["measured_feature_on"] > 0
    assert out["t_gbps_estimated"] > 0


def test_custom_kv_profile_overrides_default():
    custom = ModelKVProfile(
        num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2, k_window=6,
    )
    p = AdmissionPredictor(
        {16: 50.0, 256: 50.0}, model=LLAMA, kv_profile=custom, k_window=6,
    )
    expected_bytes = 6 * 2 * 8 * 128 * 2
    expected_t = 16 * expected_bytes * (1000.0 / 50.0) * 8 / 1e9
    assert p.t_gbps_at_k(16) == pytest.approx(expected_t, rel=1e-6)


def test_load_compute_baseline_json(tmp_path: Path):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({
        "throughput": [
            {"concurrency": 16, "concurrent_global_p99_itl_s": 0.1},
            {"concurrency": 32, "concurrent_global_p99_itl_s": 0.2},
        ],
    }))
    tbl = _load_compute_baseline_json(p)
    assert tbl[16] == pytest.approx(100.0)
    assert tbl[32] == pytest.approx(200.0)


def test_mechanistic_predictor_decomposes(tmp_path: Path):
    feat_rows = [
        {"model": LLAMA, "input_len": 1024, "output_len": 256, "K": 16,
         "k_window": 4, "p99_itl_ms": 100.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 256, "K": 64,
         "k_window": 4, "p99_itl_ms": 200.0},
    ]
    feat_csv = _write_feature_on_csv(tmp_path, feat_rows)
    baseline_json = tmp_path / "baseline.json"
    baseline_json.write_text(json.dumps({
        "throughput": [
            {"concurrency": 16, "concurrent_global_p99_itl_s": 0.02},
            {"concurrency": 64, "concurrent_global_p99_itl_s": 0.04},
        ],
    }))
    inj = tmp_path / "inj.jsonl"
    base = {"model": LLAMA, "phase": "decode", "input_len": 1024, "output_len": 256,
            "request_rate": 16.0, "injection_scenario": "baseline_no_worker",
            "p99_itl_ms": 20.0}
    h2d = {**base, "injection_scenario": "h2d_16_c1", "p99_itl_ms": 25.0}
    inj.write_text("\n".join(json.dumps(r) for r in [base, h2d]))

    m = MechanisticAdmissionPredictor.from_artifacts(
        feature_on_csv=feat_csv,
        model=LLAMA,
        baseline_compute_json=baseline_json,
        injection_jsonl=inj,
        shape=(1024, 256),
    )
    out = m.decompose(16)
    assert out["compute_baseline"] == pytest.approx(20.0)
    assert out["mechanistic_sum"] is not None
    assert m.predict_p99_itl_ms(16) >= out["compute_baseline"]


def test_from_csv_reads_k_window(tmp_path: Path):
    rows = [
        {"model": LLAMA, "input_len": 1024, "output_len": 1024,
         "K": 16, "k_window": 4, "p99_itl_ms": 30.0},
        {"model": LLAMA, "input_len": 1024, "output_len": 1024,
         "K": 64, "k_window": 4, "p99_itl_ms": 50.0},
    ]
    csv_path = _write_feature_on_csv(tmp_path, rows)
    p = AdmissionPredictor.from_csv(csv_path, model=LLAMA, shape=(1024, 1024))
    assert p.k_window == 4
    assert p.kv_profile is not None
    assert p.kv_profile.bytes_per_decode_step_per_seq() == 4 * 2 * 8 * 128 * 2
