#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Correctness suite for the hybrid KV-tiering policy layer.
#
# Per-test environment matters and got this wrong once:
#   * migration_correctness / burst_recovery need VLLM_ENABLE_V1_MULTIPROCESSING=0
#     so the test can reach the in-process scheduler.
#   * hybrid_rungd_real spawns a SERVER subprocess and FAILS if that variable
#     leaks into it. Setting it globally produced a false regression.
# So each test gets exactly the environment it needs, and nothing more.
#
# Usage:  bash benchmarks/kv_pipelining_bench/run_suite.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
P=benchmarks/kv_pipelining_bench
PY=.venv/bin/python
export HF_TOKEN="$(cat hf_token 2>/dev/null | tr -d '\n')"
fails=0
ran=0

run() {  # run <name> <expected-substring> <env-assignments...> -- <cmd...>
  local name="$1" expect="$2"; shift 2
  ran=$((ran+1))
  printf "%-28s " "$name"
  local out
  out=$(env "$@" 2>&1)
  if grep -q "$expect" <<<"$out"; then echo "PASS"; else
    echo "FAIL"; fails=$((fails+1))
    sed -n '$p;/Traceback/,+6p' <<<"$out" | head -8 | sed 's/^/      /'
  fi
}

echo "=== unit (no GPU needed) ==="
run tier_policy_unit  "TIER POLICY UNIT PASS"  $PY $P/tier_policy_unit.py
run migration_unit    "MIGRATION UNIT PASS"    $PY $P/migration_unit.py
# FIRST, because it is seconds and it catches the class of break that
# otherwise costs six failed tests and half an hour of model loads: a typo
# in the pool constructor. T7a read an env var above the line that bound
# _os and took out 6 of 9.
run pool_construct    "POOL CONSTRUCTION PASS" $PY $P/pool_construct_test.py
run controller_unit   "CONTROLLER UNIT PASS"   $PY $P/hybrid_controller_unit.py
# The rider-attend ablation knob is read in flash_attn, not in the pool
# constructor, so pool_construct cannot reach it. This asserts the control flow
# the M19 fusion bound subtracts: noattend drops gather+kernel+scatter, nokernel
# drops only the kernel, and both keep the phase timer. Spawns one child per
# mode, because the mode is resolved once and cached per process.
run attend_ablation   "ATTEND-ABLATION PASS"   $PY $P/attend_ablation_test.py

echo "=== mechanism (bitwise) ==="
run restore_roundtrip "ROUNDTRIP PASS"         $PY $P/restore_roundtrip_test.py
run host_pool         "HOST-POOL PASS"         $PY $P/rungd_host_pool_test.py

echo "=== end-to-end (token-identical) ==="
# NOTE: no VLLM_ENABLE_V1_MULTIPROCESSING here -- this one starts a server.
run rungd_real        "REAL CORRECTNESS PASS"  $PY $P/hybrid_rungd_real.py
run migrate_demote    "CORRECTNESS PASS" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 $PY $P/migration_correctness.py
run migrate_roundtrip "round trip" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 MIGRATE_ROUNDTRIP=1 \
    $PY $P/migration_correctness.py

echo "=== closed loop (in-server) ==="
run burst_recovery    "BURST RECOVERY PASS" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 $PY $P/burst_recovery.py

echo "============================================================"
# COUNT THE TESTS, do not hardcode the total. It read "9/9" when ten were
# registered and "10/10" when eleven were, and that string has been quoted in
# commit messages as if it were evidence of coverage. A total that cannot drift
# away from the number of tests actually run is the minimum that claim deserves.
if [ "$fails" -eq 0 ]; then echo "SUITE PASS ($ran/$ran)"
else echo "SUITE FAIL ($fails of $ran failing)"; fi
exit "$fails"
