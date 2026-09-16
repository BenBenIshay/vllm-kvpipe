#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experiments 4.0 (gate) and 4.1 (grid): SLO-attaining goodput, arm A vs arm C.

WHY THIS EXISTS. The plan rewrote 4.0 on 2026-09-12 so that the gate measures
**SLO-attaining goodput** via `vllm bench serve --goodput tpot:<ms>` -- the same
metric 4.1 reports and the same one Pie's vLLM baseline is measured with. But
nothing in this directory drove the two arms through it. `baseline_integrity.sh`
is the only user of `vllm bench serve` and it deliberately runs *no-flag* arms on
both sides, because its question is "what does our fork cost when off". So the
decisive experiment had no runner, and writing one on a rented machine is
exactly what the plan's own rule forbids: rental time is for measuring.

THE SAVING, AND IT IS THE REASON THIS FILE IS SHAPED THIS WAY. 4.1 is specified
as {3 contexts} x {5 SLOs} x {2 arms}. The SLO is a **client-side threshold on
per-request TPOT**, not a server configuration: `vllm bench serve --save-result`
writes per-request `itls`, `output_lens` and the run `duration`, and goodput at
any TPOT target is a pure function of those three. So the grid needs
{3 contexts} x {2 arms} x one rate sweep, and all five SLO points are derived
offline. Nothing about the measurement is weakened -- the same requests are
scored against five thresholds instead of one -- and roughly two thirds of
4.1's GPU-hours disappear.

Because that saving is only as trustworthy as the recomputation,
`--mode verify-goodput` runs one client WITH `--goodput tpot:X` and compares the
server-side `request_goodput` against the offline number. Run it once per
machine before believing any derived point.

WHAT IT REFUSES TO DO. Three ways this family of harness has produced plausible
wrong answers here, each now a hard check rather than a habit:

  * a load that never binds -- if arm A absorbs the whole offered rate, neither
    arm was under pressure and the comparison is vacuous. Reported as
    INCONCLUSIVE (exit 2), never as a failed gate.
  * an arm C with no riders -- if the tier histogram contains no `cpu` entries,
    the two arms are the same configuration and any delta is noise. Also
    INCONCLUSIVE. (This needs VLLM_KVPIPE_TIER_OCCUPANCY=1, which the arm-C
    server is launched with.)
  * a failed run scored as a slow one -- a server that dies reports 0.000 req/s,
    and averaging that in reads as a catastrophic regression.

Exit codes:  0 GO (gate passed) | 1 NO-GO (gate failed on the numbers) |
             2 INCONCLUSIVE (the run cannot answer the question)

Usage:
    # 4.0 gate: one context, SLO = 1.5*T0, arms A and C, 3 interleaved repeats
    python gate_goodput.py --mode gate --model Qwen/Qwen2.5-32B-Instruct \\
        --ctx 8192 --b-hbm-max 5 --link-gbps 350 --outdir /path/to/runs

    # 4.1 grid: rate sweep per (arm, context); all five SLOs derived offline
    python gate_goodput.py --mode grid --contexts 4096,8192,16384 ...

    # trust check for the offline goodput, once per machine
    python gate_goodput.py --mode verify-goodput --ctx 2048 ...
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

print = functools.partial(print, flush=True)  # noqa: A001

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bench_teardown import terminate_engine  # noqa: E402

MS = 1000.0


# --------------------------------------------------------------------------
# Offline goodput: the same arithmetic vllm/benchmarks/serve.py does
# --------------------------------------------------------------------------
class NoDetail(Exception):
    """The result file carries only aggregates, so nothing can be recomputed.

    `--save-result` alone writes means and percentiles; the per-request arrays
    (`itls`, `output_lens`) exist only under `--save-detailed`. Without them
    every offline quantity is silently empty -- and an empty run scores 0.000
    goodput against another 0.000, which the first version of the verification
    reported as a 0.0000% difference and a PASS. A missing array is a failure,
    never a zero.
    """


def has_detail(result: dict) -> bool:
    return bool(result.get("itls")) and bool(result.get("output_lens"))


def request_tpots(result: dict) -> list[float]:
    """Per-request TPOT in seconds, in `serve.py`'s definition.

    serve.py computes `tpot = (latency - ttft) / (output_len - 1)` and, for a
    request with `output_len <= 1`, *defines* tpot as 0 so that it always counts
    as meeting the SLO. Both details are reproduced here rather than
    approximated, because a definition that differs only in the corner cases is
    the kind of thing that agrees on the dry run and disagrees on the machine
    that matters.

    `latency - ttft` is exactly `sum(itl)`: the client timestamps the first
    chunk as TTFT and appends every subsequent inter-chunk gap to `itl`.
    """
    if not has_detail(result):
        raise NoDetail("result file has no per-request arrays; the client must "
                       "run with --save-detailed")
    itls = result.get("itls") or []
    olens = result.get("output_lens") or []
    out: list[float] = []
    for i, ol in enumerate(olens):
        if ol is None or ol <= 0:
            continue                      # a failed request is not a slow one
        if ol <= 1:
            out.append(0.0)
            continue
        per = itls[i] if i < len(itls) else []
        out.append(float(sum(per)) / float(ol - 1))
    return out


def goodput_at(result: dict, tpot_ms: float) -> dict:
    """SLO-attaining goodput (req/s) recomputed from a saved result file."""
    tpots = request_tpots(result)
    dur = float(result.get("duration") or 0.0)
    slo_s = tpot_ms / MS
    good = sum(1 for t in tpots if t <= slo_s)
    return {
        "tpot_slo_ms": tpot_ms,
        "goodput_req_s": (good / dur) if dur > 0 else float("nan"),
        "good": good,
        "completed": len(tpots),
        "attainment": (good / len(tpots)) if tpots else float("nan"),
    }


def pooled_itls_ms(result: dict) -> list[float]:
    return [v * MS for per in (result.get("itls") or []) for v in per]


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def t0_of(summary: dict, metric: str) -> float:
    """T0 in the units the SLO will be enforced in.

    MEASURED ON THE A100 DRY RUN, and it matters more than it sounds. The plan
    says "record the median ITL at the peak; that median is T0", while goodput
    scores each request on **TPOT** -- `(latency - ttft) / (output_len - 1)`.
    Those are not the same number: 5.85 ms median pooled ITL against 3.39 ms
    median TPOT on the same run, a factor of 1.7.

    The gap is not noise, it is a definition. A streamed chunk can carry more
    than one token, so `len(itl)` under-counts tokens while `sum(itl)` stays
    right; dividing the same total by the true token count gives a smaller
    per-token figure. Deriving the SLO from pooled ITL and then enforcing it on
    TPOT therefore sets every point of the grid ~1.7x looser than intended --
    including `T0 x 1.0`, the point the plan predicts the mixture should LOSE
    at, which would quietly become a target it can pass.

    So T0 defaults to the quantity the gate actually enforces. `--t0-metric
    itl` reproduces the plan's literal wording if that is wanted; the JSON
    records which was used.
    """
    return summary["median_tpot_ms"] if metric == "tpot" else summary["median_itl_ms"]


def safe_summarize(result: dict) -> dict:
    """`summarize`, with a missing-data failure reported as one."""
    try:
        return summarize(result)
    except NoDetail as e:
        return {"error": str(e)}


def summarize(result: dict) -> dict:
    """The shape every downstream verdict reads."""
    itls = pooled_itls_ms(result)
    tpots_ms = [t * MS for t in request_tpots(result)]
    return {
        "completed": int(result.get("completed") or 0),
        "failed": int(result.get("failed") or 0),
        # The achieved concurrency is what arm C's b_hbm_max must sit BELOW,
        # so it is carried through rather than left in the raw file.
        "max_concurrent": float(result.get("max_concurrent_requests") or 0.0),
        "duration_s": float(result.get("duration") or 0.0),
        "request_throughput": float(result.get("request_throughput") or 0.0),
        "output_throughput": float(result.get("output_throughput") or 0.0),
        "median_itl_ms": statistics.median(itls) if itls else float("nan"),
        "p99_itl_ms": pct(itls, 99),
        "median_tpot_ms": statistics.median(tpots_ms) if tpots_ms else float("nan"),
        "p99_tpot_ms": pct(tpots_ms, 99),
    }


# --------------------------------------------------------------------------
# Tier observability: is arm C actually a different configuration?
# --------------------------------------------------------------------------
_CAPACITY = re.compile(
    r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_KVTOK = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_PREEMPT = re.compile(r"Preemptions: (\d+)")


def engine_capacity(server_log: str) -> dict:
    """How many full-length requests HBM actually holds, per the engine itself.

    vLLM prints this at startup -- "GPU KV cache size: 40,400 tokens" and
    "Maximum concurrency for 8,384 tokens per request: 4.82x" -- so the
    capacity the whole experiment is sized against is a fact to be read, not a
    quantity to be inferred from throughput under load. Reading it costs one
    model load we are doing anyway.
    """
    out: dict = {}
    try:
        with open(server_log, errors="replace") as fh:
            for line in fh:
                m = _CAPACITY.search(line)
                if m:
                    out["seq_tokens"] = int(m.group(1).replace(",", ""))
                    out["capacity"] = float(m.group(2))
                m = _KVTOK.search(line)
                if m:
                    out["kv_tokens"] = int(m.group(1).replace(",", ""))
    except OSError:
        pass
    return out


def preemptions(server_log: str) -> int:
    """Cumulative preemptions the engine reported (0 if it never logged any).

    A run with preemptions is not measuring per-token latency, it is measuring
    how often requests were evicted and resumed. Both arms are configured to
    avoid it; this is how we check rather than assume.
    """
    best = 0
    try:
        with open(server_log, errors="replace") as fh:
            for line in fh:
                m = _PREEMPT.search(line)
                if m:
                    best = max(best, int(m.group(1)))
    except OSError:
        pass
    return best


_RUNNING = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs")


def running_batch(server_log: str) -> dict:
    """The engine's own resident batch and queue, from its periodic log line.

    THE CLIENT'S NUMBER IS A DIFFERENT QUANTITY. `vllm bench serve` reports
    `max_concurrent_requests`, which counts requests **in flight** -- accepted
    by the client, including everything queued or preempted inside the engine.
    On the 32B rehearsal that read 28 while the engine was actually running 4-5
    and had the rest waiting. Deriving `b_hbm_max` from 28 set it to 16, far
    above the resident capacity, so arm C had HBM slots for every request it
    could actually run and produced a single rider.

    That is the vacuity trap in a new costume: an arm C that is arm A with
    extra logging. The engine's `Running:` line is the resident batch, which is
    the quantity `b_hbm_max` has to sit below.
    """
    run: list[int] = []
    wait: list[int] = []
    try:
        with open(server_log, errors="replace") as fh:
            for line in fh:
                m = _RUNNING.search(line)
                if m:
                    run.append(int(m.group(1)))
                    wait.append(int(m.group(2)))
    except OSError:
        pass
    busy = [r for r in run if r > 0]
    if not busy:
        return {"samples": 0}
    return {
        "samples": len(busy),
        "median_running": statistics.median(busy),
        "max_running": max(busy),
        "median_waiting": statistics.median(wait) if wait else 0,
        "max_waiting": max(wait) if wait else 0,
    }


_OCC = re.compile(
    r"TIER_OCCUPANCY: n_hbm=(\d+) n_cpu=(\d+) blk_hbm=(\d+) blk_cpu=(\d+) "
    r"free=(-?\d+) total=(-?\d+)")


def tier_stats(server_log: str) -> dict:
    """Parse the scheduler's tier histogram out of a server log.

    Requires VLLM_KVPIPE_TIER_OCCUPANCY=1 on the server, which `launch_server`
    sets for arm C. No samples at all is a different failure from samples with
    no `cpu` entries, and the caller needs to tell them apart: the first means
    the instrumentation was off, the second means no rider was ever created.
    """
    n_hbm: list[int] = []
    n_cpu: list[int] = []
    blk_cpu: list[int] = []
    blk_hbm: list[int] = []
    try:
        with open(server_log, errors="replace") as fh:
            for line in fh:
                m = _OCC.search(line)
                if m:
                    n_hbm.append(int(m.group(1)))
                    n_cpu.append(int(m.group(2)))
                    blk_hbm.append(int(m.group(3)))
                    blk_cpu.append(int(m.group(4)))
    except OSError:
        pass
    tot_blk = sum(blk_cpu) + sum(blk_hbm)
    return {
        "samples": len(n_cpu),
        "max_n_cpu": max(n_cpu) if n_cpu else 0,
        "max_n_hbm": max(n_hbm) if n_hbm else 0,
        "mean_n_cpu": (sum(n_cpu) / len(n_cpu)) if n_cpu else 0.0,
        "cpu_block_share": (sum(blk_cpu) / tot_blk) if tot_blk else 0.0,
    }


# --------------------------------------------------------------------------
# Server / client
# --------------------------------------------------------------------------
def server_cmd(arm: str, a, slo_ms: float | None, max_seqs: int | None = None,
               b_hbm_max: int | None = None) -> tuple[list[str], dict]:
    """Build the server command line and environment for one arm.

    ARM A IS A PLAIN SERVER. Not "hybrid flags with b_hbm_max large" -- a plain
    one, with no hybrid flag parsed at all. That is the arm 4.0b measured
    against stock vLLM (-0.08%), so it is the arm the paper's ratios are
    entitled to be quoted against. Passing the flags and disabling them via a
    knob would make the control our own machinery in a different mode.
    """
    cmd = [
        a.python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", a.model, "--port", str(a.port),
        "--max-model-len", str(a.ctx + a.out_len + 64),
        "--gpu-memory-utilization", str(a.gpu_mem_util),
        "--dtype", a.dtype,
        "--disable-log-requests", "--no-enable-prefix-caching",
    ]
    if a.eager:
        cmd.append("--enforce-eager")
    if a.num_gpu_blocks_override:
        cmd += ["--num-gpu-blocks-override", str(a.num_gpu_blocks_override)]
    # EACH ARM GETS ITS OWN ADMISSION WIDTH, and that is the point rather than
    # a fairness problem. The width is set to what that arm can actually hold
    # without preempting: arm A its HBM capacity, arm C that capacity plus the
    # riders the interconnect can hide. Forcing a common value either caps the
    # mixture below what it can serve -- forbidding the effect under test -- or
    # pushes the baseline into preemption, which inflates its per-token latency
    # by an order of magnitude and corrupts the SLO anchor derived from it.
    width = max_seqs if max_seqs is not None else a.max_num_seqs
    if width:
        cmd += ["--max-num-seqs", str(width)]

    env = dict(os.environ)
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM",
              "VLLM_KVPIPE_TIER_OCCUPANCY", "VLLM_KVPIPE_TIER_POLICY"):
        env.pop(k, None)

    if arm == "C":
        assert slo_ms is not None, "arm C needs the measured SLO"
        cmd += [
            "--enable-transfer-aware-admission",
            "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max",
            str(b_hbm_max if b_hbm_max is not None else a.b_hbm_max),
            "--hybrid-itl-ms", f"{slo_ms:.3f}",
            "--transfer-aware-slo-p99-itl-ms", f"{slo_ms:.3f}",
            "--hybrid-pcie-gbps", str(a.link_gbps),
        ]
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        env["VLLM_KVPIPE_TIER_POLICY"] = a.tier_policy
        # The tier histogram is the sanity check that tells a real arm C from a
        # configuration that silently produced no riders.
        env["VLLM_KVPIPE_TIER_OCCUPANCY"] = "1"
        # Pin the pool to what was checked to fit, rather than letting it size
        # itself from max_num_seqs and be killed by the cgroup.
        if getattr(a, "rungd_hmax", 0):
            env["VLLM_KVPIPE_RUNGD_HMAX"] = str(a.rungd_hmax)
        # The double-buffered prefetch. It is what makes the roofline's premise
        # true -- that a step of duration itl_ms can HIDE itl_ms x X bytes. With
        # it off, the measured step grows by almost exactly the serial transfer
        # time (136 ms of transfer, 167 ms of growth), so riders add latency
        # rather than overlapping with compute.
        if getattr(a, "rungd_async", False):
            env["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"
    return cmd, env


def cgroup_limit() -> int:
    """The smallest memory limit actually binding on this process, v1 or v2.

    Reading `/sys/fs/cgroup/memory.max` is not enough and reads as *no limit*
    here, which is worse than not checking: on nova06 it reported the node's
    876 GB while the job was capped at 120 GB, so the sizing check passed a
    437 GB pinned allocation straight into the OOM killer it existed to
    prevent.

    SLURM places the limit on the JOB cgroup, while the step and task cgroups
    below it are unlimited -- and `/proc/self/cgroup` names the deepest one. So
    resolve the process's own path and walk UP it, taking the smallest real
    limit found. Measured on this cluster: task and step unlimited, job
    128849018880 = 120 GB, which is the true bound.
    """
    best = 0
    try:
        with open("/proc/self/cgroup") as fh:
            entries = [ln.strip().split(":", 2) for ln in fh if ln.strip()]
    except OSError:
        return 0
    for ent in entries:
        if len(ent) != 3:
            continue
        _hid, ctrl, path = ent
        if "memory" in ctrl.split(","):
            base, fname = "/sys/fs/cgroup/memory", "memory.limit_in_bytes"
        elif ctrl == "":                                   # v2 unified
            base, fname = "/sys/fs/cgroup", "memory.max"
        else:
            continue
        parts = [p for p in path.strip("/").split("/") if p]
        for i in range(len(parts), -1, -1):
            try:
                with open(os.path.join(base, *parts[:i], fname)) as fh:
                    raw = fh.read().strip()
            except OSError:
                continue
            if not raw or raw == "max":
                continue
            try:
                val = int(raw)
            except ValueError:
                continue
            # v1 spells "unlimited" as a near-2^63 sentinel.
            if 0 < val < (1 << 62) and (best == 0 or val < best):
                best = val
    return best


def host_pool_check(a) -> dict:
    """Will arm C's PINNED host mirrors fit in host memory?

    They did not, on the 32B rehearsal, and the way they failed is the reason
    this is a pre-launch check rather than a note in a runbook. The pool sizes
    itself as `h_max = max_num_seqs * blocks_per_request + 8` so that any
    admissible request can be fully host-resident. At Qwen-32B/8K that is
    256 x 524 blocks x 4 MB = **536 GB** of pinned memory against a 120 GB
    limit. The cgroup killed the EngineCore, 27 of 32 requests failed, and the
    surviving 5 were scored as a slow arm.

    Nothing in the engine's own logs says "the pool was too big" -- the process
    simply disappears, which is exactly the silent shape RENTAL_REQUIREMENTS §7
    Q4 asks providers about. So compute it before spending a model load, and
    cap `max_num_seqs` to what actually fits rather than discovering the limit
    by being killed.
    """
    sys.path.insert(0, HERE)
    from gh200_preflight import kv_bytes_per_token

    b_tok, geo = kv_bytes_per_token(a.model)
    max_len = a.ctx + a.out_len + 64
    bpr = max(1, -(-max_len // a.block_size))          # ceil
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        avail = 0
    lim = cgroup_limit()
    if lim and (not avail or lim < avail):
        avail = lim

    budget = avail * a.host_pool_frac
    per_req = bpr * a.block_size * b_tok
    fits = int(budget // per_req) if per_req else 0
    seqs = a.max_num_seqs or fits
    need = (seqs * bpr + 8) * a.block_size * b_tok
    return {"b_tok": b_tok, "geometry": geo, "blocks_per_req": bpr,
            "per_req_bytes": per_req, "avail_bytes": avail,
            "budget_bytes": budget, "max_reqs_that_fit": fits,
            "max_num_seqs": seqs, "pinned_bytes": need, "fits": need <= budget}


def wait_ready(port: int, timeout_s: float, proc) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False                      # died during startup
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def wait_vram_free(frac: float = 0.90, timeout_s: float = 300.0) -> None:
    """Wait for the GPU to actually come back, rather than sleeping and hoping.

    A fixed sleep here is how an arm scored 0.000 once: the next server started
    while the previous engine still held the pool, failed to initialise, and the
    verdict averaged the zero in as though it were a slow arm.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            free, total = (int(x) for x in out.splitlines()[0].split(","))
            if free > total * frac:
                return
        except Exception:
            return
        time.sleep(5)


class Server:
    def __init__(self, arm: str, a, slo_ms: float | None, tag: str,
                 max_seqs: int | None = None, b_hbm_max: int | None = None):
        self.arm, self.a, self.tag = arm, a, tag
        self.log = os.path.join(a.outdir, f"server_{tag}.log")
        cmd, env = server_cmd(arm, a, slo_ms, max_seqs=max_seqs,
                              b_hbm_max=b_hbm_max)
        with open(self.log, "w") as fh:
            fh.write("# " + " ".join(cmd) + "\n")
            fh.write("# hybrid env: " + json.dumps(
                {k: v for k, v in env.items() if k.startswith("VLLM_KVPIPE")}) + "\n")
            fh.flush()
            # setsid: own process group, so teardown can signal the GROUP. The
            # API server is only the parent; EngineCore is the child that holds
            # the GPU and the one that survives a parent-only kill.
            self.proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True)
        self.ready = wait_ready(a.port, a.server_timeout_s, self.proc)

    def stop(self) -> None:
        terminate_engine(self.proc)
        wait_vram_free()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def run_client(a, rate: float, tag: str, goodput_ms: float | None = None,
               max_concurrency: int | None = None) -> dict:
    """One `vllm bench serve` run. Returns its result.

    `max_concurrency` holds a fixed number of requests in flight instead of
    offering a Poisson stream. That is what keeps the engine off its
    preemption path: the queue forms in the client, not inside a scheduler
    that has admitted more than KV can hold. Measured cost of getting this
    wrong: median ITL 55 ms (no preemption) against 657 ms (preemption), same
    hardware, same model, same offered rate.
    """
    res_path = os.path.join(a.outdir, f"result_{tag}.json")
    if os.path.exists(res_path):
        os.remove(res_path)
    cmd = [
        a.python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai", "--model", a.model, "--port", str(a.port),
        "--dataset-name", "random",
        "--random-input-len", str(a.ctx), "--random-output-len", str(a.out_len),
        "--num-prompts", str(a.num_prompts),
        "--request-rate", str(rate), "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        # --save-detailed is what writes the per-request arrays. Without it the
        # file holds only means and percentiles, every derived SLO point is
        # computed over nothing, and the result looks like a legitimate zero.
        "--save-result", "--save-detailed", "--result-filename", res_path,
    ]
    if goodput_ms is not None:
        cmd += ["--goodput", f"tpot:{goodput_ms:.3f}"]
    if max_concurrency:
        cmd += ["--max-concurrency", str(max_concurrency)]
    log = os.path.join(a.outdir, f"bench_{tag}.log")
    with open(log, "w") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        try:
            subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=a.client_timeout_s)
        except subprocess.TimeoutExpired:
            # Return empty rather than propagating: the caller holds a live
            # server in a context manager, and an exception here would leave a
            # vLLM engine holding the GPU. That is the shape that drained two
            # nodes on 2026-09-06, and on a rented machine it is worse.
            fh.write(f"\n# CLIENT TIMEOUT after {a.client_timeout_s}s\n")
            return {}
    try:
        with open(res_path) as fh:
            res = json.load(fh)
    except Exception:
        return {}
    res["_offered_rate"] = rate
    res["_tag"] = tag
    return res


# --------------------------------------------------------------------------
# Rate sweep: find the knee, and say whether the load ever bound
# --------------------------------------------------------------------------
def rate_ladder(a) -> list[float]:
    if a.rates != "auto":
        return [float(x) for x in a.rates.split(",") if x]
    out, r = [], a.rate_start
    for _ in range(a.max_rates):
        out.append(round(r, 3))
        r *= a.rate_mult
    return out


def sweep(a, arm: str, slo_ms: float | None, tag_prefix: str) -> dict:
    """Sweep offered rate upward until throughput stops rising.

    The knee is what 4.0 step 1 calls "goodput stops rising": with no SLO fixed
    yet, every completed request counts, so the quantity that plateaus is
    achieved request throughput. T0 -- the median ITL at the peak -- comes from
    that same run, and the SLO is then derived from it.
    """
    runs: list[dict] = []
    with Server(arm, a, slo_ms, f"{tag_prefix}_sweep") as srv:
        if not srv.ready:
            return {"error": f"server for arm {arm} did not become ready",
                    "server_log": srv.log, "runs": []}
        prev = 0.0
        for rate in rate_ladder(a):
            tag = f"{tag_prefix}_r{rate:g}"
            res = run_client(a, rate, tag)
            if not res:
                runs.append({"offered_rate": rate, "error": "no result file"})
                break
            s = safe_summarize(res)
            if "error" in s:
                s["offered_rate"] = rate
                runs.append(s)
                print(f"    rate {rate:6.2f} -> {s['error']}")
                break
            s["offered_rate"] = rate
            s["bound"] = s["request_throughput"] < a.bind_frac * rate
            runs.append(s)
            print(f"    rate {rate:6.2f} -> {s['request_throughput']:6.3f} req/s "
                  f"achieved, median ITL {s['median_itl_ms']:7.2f} ms, "
                  f"p99 {s['p99_itl_ms']:8.2f} ms"
                  f"{'  [BOUND]' if s['bound'] else ''}")
            gain = (s["request_throughput"] - prev) / max(prev, 1e-9)
            if prev > 0 and gain < a.plateau_frac and s["bound"]:
                break
            prev = max(prev, s["request_throughput"])
        tiers = tier_stats(srv.log)
        batch = running_batch(srv.log)
    ok = [r for r in runs if "error" not in r]
    peak = max(ok, key=lambda r: r["request_throughput"]) if ok else None
    if batch.get("samples"):
        print(f"    engine resident batch: median {batch['median_running']:.1f} "
              f"(max {batch['max_running']}), queue median "
              f"{batch['median_waiting']:.1f}")
    return {"runs": runs, "peak": peak, "tiers": tiers, "batch": batch,
            "ever_bound": any(r.get("bound") for r in ok)}


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def mode_verify_goodput(a) -> int:
    """Check the offline goodput against the server-side one. Once per machine.

    Everything the grid saves rests on this equality, so it is measured rather
    than argued. A mismatch does not necessarily mean the recomputation is
    wrong -- if a backend bundles several tokens into one streamed chunk then
    `len(itl)` and `output_len` disagree and TPOT differs slightly -- but it
    does mean the derived SLO points cannot be quoted until it is understood.
    """
    with Server("A", a, None, "verify") as srv:
        if not srv.ready:
            print(f"INCONCLUSIVE: server did not start; see {srv.log}")
            return 2
        probe = run_client(a, a.rate_start, "verify_probe")
        if not probe:
            print("INCONCLUSIVE: probe produced no result")
            return 2
        s = safe_summarize(probe)
        if "error" in s:
            print(f"INCONCLUSIVE: {s['error']}")
            return 2
        # Verify at a threshold that SPLITS the requests. At an SLO nothing
        # meets, or one everything meets, 0==0 and n==n agree for any
        # calculator, correct or not -- the comparison would be free to pass.
        slo_ms = s["median_tpot_ms"] * 1.0
        print(f"  probe: median TPOT {s['median_tpot_ms']:.3f} ms over "
              f"{s['completed']} requests -> verifying at SLO {slo_ms:.3f} ms "
              f"(chosen to split the run roughly in half)")
        res = run_client(a, a.rate_start, "verify_slo", goodput_ms=slo_ms)
    if not res:
        print("INCONCLUSIVE: verification run produced no result")
        return 2
    server_side = res.get("request_goodput")
    if server_side is None:
        print("INCONCLUSIVE: server-side goodput missing from the result file")
        return 2
    try:
        offline = goodput_at(res, slo_ms)
    except NoDetail as e:
        print(f"INCONCLUSIVE: {e}")
        return 2
    completed = int(res.get("completed") or 0)
    if completed == 0:
        print("INCONCLUSIVE: no request completed; there is nothing to verify.")
        return 2
    if offline["completed"] != completed:
        print(f"INCONCLUSIVE: the offline pass saw {offline['completed']} "
              f"requests but the server reported {completed}. The arrays and "
              f"the aggregates disagree; fix that before deriving anything.")
        return 2
    if offline["good"] in (0, completed):
        print(f"INCONCLUSIVE: {offline['good']}/{completed} met the SLO, so "
              f"the check cannot discriminate -- any calculator passes a run "
              f"where every request falls on the same side. Re-run with a "
              f"different --rate-start so the threshold splits the load.")
        return 2
    rel = abs(offline["goodput_req_s"] - server_side) / max(server_side, 1e-9)
    print(f"  server-side  {server_side:.6f} req/s")
    print(f"  offline      {offline['goodput_req_s']:.6f} req/s "
          f"({offline['good']}/{offline['completed']} met the SLO)")
    print(f"  relative difference {rel * 100:.4f}%")
    out = {"slo_ms": slo_ms, "server_side": server_side,
           "offline": offline, "relative_diff": rel}
    with open(os.path.join(a.outdir, "verify_goodput.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    if rel <= a.verify_tol:
        print(f"\nVERIFY PASS: offline goodput reproduces the server's within "
              f"{a.verify_tol * 100:.2f}%. Derived SLO points are trustworthy.")
        return 0
    print(f"\nVERIFY FAIL: {rel * 100:.3f}% apart. Do NOT derive SLO points "
          f"offline until this is understood.")
    return 1


def mode_gate(a) -> int:
    """4.0: one context, SLO = mult x T0, arms A and C, interleaved repeats."""
    print("=" * 62)
    print(f" 4.0 FEASIBILITY GATE   model={a.model} ctx={a.ctx}")
    print(f" threshold: goodput(C) >= {a.gate_ratio:.2f} x goodput(A)")
    print(" protocol: each arm at the concurrency it can hold; riders from "
          "the roofline")
    print("=" * 62)

    hp = host_pool_check(a)
    print(f"\n  host pool: {hp['geometry']['layers']}L x "
          f"{hp['geometry']['kv_heads']}KV x {hp['geometry']['head_dim']}D -> "
          f"{hp['b_tok']} B/token, {hp['per_req_bytes'] / 1e9:.2f} GB per "
          f"request at {a.ctx} ctx")
    print(f"  pinned mirrors would need {hp['pinned_bytes'] / 1e9:.1f} GB of "
          f"{hp['avail_bytes'] / 1e9:.0f} GB available "
          f"({a.host_pool_frac:g} budget = {hp['budget_bytes'] / 1e9:.1f} GB)")
    if not hp["fits"]:
        if a.max_num_seqs:
            print(f"\nINCONCLUSIVE: --max-num-seqs {a.max_num_seqs} needs "
                  f"{hp['pinned_bytes'] / 1e9:.0f} GB pinned; only "
                  f"{hp['max_reqs_that_fit']} requests fit the budget. Lower "
                  f"it, shorten --ctx, or run where there is more host RAM. "
                  f"(Left alone, the cgroup kills the EngineCore mid-run and "
                  f"the arm scores as merely slow.)")
            return 2
        a.max_num_seqs = max(2, hp["max_reqs_that_fit"])
        hp = host_pool_check(a)
        print(f"  -> capping --max-num-seqs at {a.max_num_seqs} so the pool "
              f"fits ({hp['pinned_bytes'] / 1e9:.1f} GB)")
    a.rungd_hmax = a.max_num_seqs * hp["blocks_per_req"] + 8

    # ---------------------------------------------------------------- phase 1
    # Arm A at exactly its HBM capacity: no queue inside the engine, so TPOT
    # is per-token service time and nothing else. T0 comes from HERE, not from
    # a saturated sweep -- taking it at saturation measured queueing and set
    # the SLO anchor ~10x too loose (545 ms against a 39-56 ms roofline).
    print("\n[1/4] arm A at capacity -> T0 (no preemption)")
    with Server("A", a, None, "A_t0", max_seqs=None) as srv:
        if not srv.ready:
            print(f"INCONCLUSIVE: arm A did not start; see {srv.log}")
            return 2
        cap_info = engine_capacity(srv.log)
        capacity = int(cap_info.get("capacity", 0))
        if capacity < 2:
            print(f"\nINCONCLUSIVE: the engine reports capacity "
                  f"{cap_info.get('capacity', 0):.2f} requests at {a.ctx} ctx. "
                  f"Below 2 there is nothing to add riders to. Shorten --ctx "
                  f"or raise --gpu-mem-util.")
            return 2
        print(f"  engine: KV pool {cap_info.get('kv_tokens', 0):,} tokens -> "
              f"capacity {cap_info['capacity']:.2f} requests of "
              f"{cap_info.get('seq_tokens', 0):,} tokens; using {capacity}")
        t0_res = run_client(a, float("inf"), "A_t0", max_concurrency=capacity)
        pre = preemptions(srv.log)
        batch = running_batch(srv.log)
    if not t0_res:
        print("INCONCLUSIVE: the T0 run produced no result")
        return 2
    t0_sum = safe_summarize(t0_res)
    if "error" in t0_sum:
        print(f"INCONCLUSIVE: {t0_sum['error']}")
        return 2
    t0 = t0_of(t0_sum, a.t0_metric)
    slo_ms = a.slo_mult * t0
    print(f"  resident batch {batch.get('median_running', 0):.1f}, "
          f"preemptions {pre}")
    if pre:
        print(f"  WARNING: {pre} preemptions during the T0 run -- T0 now "
              f"includes eviction stalls, not just service time.")
    print(f"  T0 = median {a.t0_metric.upper()} = {t0:.2f} ms "
          f"(pooled ITL median {t0_sum['median_itl_ms']:.2f} ms)")
    print(f"  SLO = {a.slo_mult:g} x T0 = {slo_ms:.2f} ms")

    # ---------------------------------------------------------------- phase 2
    # How many riders can the link HIDE inside that step? This is the feature's
    # own admission model (vllm/v1/hybrid_kv_roofline.py), the same arithmetic
    # B1 applies per admission -- not a fraction chosen by the harness.
    print("\n[2/4] roofline -> how many riders hide inside the SLO")
    from vllm.v1.hybrid_kv_roofline import max_offload_batch

    seq_tokens = a.seq_tokens or (a.ctx + a.out_len)
    # THE HIDE WINDOW IS THE DECODE STEP, NOT THE SLO.
    #
    # A rider's KV has to cross the link inside the step it is needed for, so
    # the window is the DECODE STEP -- measured here as the median pooled ITL.
    # Feeding B1 the SLO instead (derived from TPOT) is what over-admitted:
    # TPOT is a per-request mean and carries prefill interference, so at
    # ctx 2048 it read 108 ms against a 54 ms step -- exactly 2x -- and the
    # 1.5x SLO on top made the budget 3x too large. B1 then admitted 7 riders
    # where the window justifies 2, and the step grew by 167 ms.
    #
    # The roofline and the measurement agree on the step: 38.6-54.6 ms
    # predicted at batch 16 / ctx 2048 on A100, 53.93 ms measured.
    window_ms = {"itl": t0_sum["median_itl_ms"],
                 "tpot": t0_sum["median_tpot_ms"],
                 "slo": slo_ms}[a.hide_window]
    # And only the overlapped fraction of the window is usable. The roofline
    # assumes all of it; measured on this path it is ~23% with the
    # double-buffered prefetch on and ~0% with it off.
    eff_window = window_ms * a.overlap_eff
    riders = (a.riders if a.riders >= 0 else
              max_offload_batch(seq_tokens=seq_tokens, itl_ms=eff_window,
                                x_eff_gbps=a.link_gbps, b_tok=hp["b_tok"]))
    theta = eff_window / 1000.0 * a.link_gbps * 1e9 / hp["b_tok"]
    print(f"  hide window = {a.hide_window.upper()} {window_ms:.1f} ms "
          f"x overlap {a.overlap_eff:g} = {eff_window:.1f} ms")
    print(f"  budget = {eff_window:.1f} ms x {a.link_gbps:g} GB/s / "
          f"{hp['b_tok']} B/tok = {theta:,.0f} token-slots")
    print(f"  at {seq_tokens:,} tokens per rider -> {riders} rider(s)"
          + ("  [OVERRIDDEN by --riders]" if a.riders >= 0 else ""))
    if riders < 1:
        print(f"\nNO-GO, and it is the roofline saying so rather than a "
              f"measurement: at {a.link_gbps:g} GB/s a single {seq_tokens:,}"
              f"-token rider needs "
              f"{seq_tokens * hp['b_tok'] / (a.link_gbps * 1e9) * 1e3:.0f} ms "
              f"of link time per step, against a {slo_ms:.0f} ms budget. No "
              f"rider can hide here, so the mixture has nothing to add. Run a "
              f"shorter context or a faster link.")
        return 1

    # Residents stay FULL and riders are ADDITIONAL. Capping b_hbm_max below
    # capacity makes riders substitutional -- they displace residents instead
    # of extending the batch -- which is the configuration the model says
    # loses, and is not the claim being tested.
    b_hbm = a.b_hbm_max if a.b_hbm_max > 0 else capacity
    width_c = b_hbm + riders
    print(f"  arm C = {b_hbm} resident + {riders} riders = {width_c} concurrent"
          f"  (arm A = {capacity})")

    a.max_num_seqs = width_c
    hp = host_pool_check(a)
    if not hp["fits"]:
        print(f"\nINCONCLUSIVE: {width_c} concurrent needs "
              f"{hp['pinned_bytes'] / 1e9:.0f} GB pinned, over the "
              f"{hp['budget_bytes'] / 1e9:.0f} GB budget.")
        return 2
    a.rungd_hmax = width_c * hp["blocks_per_req"] + 8
    print(f"  pinned pool {hp['pinned_bytes'] / 1e9:.1f} GB "
          f"(hmax {a.rungd_hmax} blocks)")

    # ---------------------------------------------------------------- phase 3
    reps: list[dict] = []
    widths = {"A": capacity, "C": width_c}
    for rep in range(1, a.repeats + 1):
        print(f"\n[3/4] repeat {rep}/{a.repeats} (interleaved A then C)")
        pair: dict = {"repeat": rep}
        for arm in ("A", "C"):
            tag = f"{arm}_rep{rep}"
            with Server(arm, a, slo_ms if arm == "C" else None, tag,
                        max_seqs=widths[arm],
                        b_hbm_max=b_hbm if arm == "C" else None) as srv:
                if not srv.ready:
                    pair[arm] = {"error": "server not ready",
                                 "server_log": srv.log}
                    print(f"    arm {arm}: SERVER DID NOT START ({srv.log})")
                    continue
                # Each arm runs at the concurrency it can actually sustain.
                res = run_client(a, float("inf"), tag,
                                 max_concurrency=widths[arm])
                tiers = tier_stats(srv.log)
                pre = preemptions(srv.log)
                batch = running_batch(srv.log)
            if not res:
                pair[arm] = {"error": "no result"}
                print(f"    arm {arm}: NO RESULT")
                continue
            s = safe_summarize(res)
            if "error" in s:
                pair[arm] = s
                print(f"    arm {arm}: {s['error']}")
                continue
            # AN ARM THAT DID NOT FINISH IS NOT A SLOW ARM. Measured on the
            # 32B rehearsal: arm C's EngineCore died of a host OOM, 27 of 32
            # requests failed, and the 5 that survived were scored as a
            # legitimate 55% slowdown -- a crash reported as a NO-GO on the
            # experiment the whole rental exists to run. Goodput divides by
            # wall time, so a run that dies early looks merely slow rather
            # than broken, and nothing else in the number gives it away.
            if s["failed"] or s["completed"] < a.min_completion_frac * a.num_prompts:
                s["error"] = (
                    f"incomplete: {s['completed']}/{a.num_prompts} completed, "
                    f"{s['failed']} failed -- check {srv.log}")
                pair[arm] = s
                print(f"    arm {arm}: INVALID RUN -- {s['error']}")
                continue
            g = goodput_at(res, slo_ms)
            s.update(g)
            s["tiers"] = tiers
            s["preemptions"] = pre
            s["resident_batch"] = batch.get("median_running", 0.0)
            s["concurrency"] = widths[arm]
            s["server_log"] = srv.log
            pair[arm] = s
            print(f"    arm {arm}: conc {widths[arm]}, goodput "
                  f"{g['goodput_req_s']:.3f} req/s "
                  f"({g['good']}/{g['completed']} met SLO), median TPOT "
                  f"{s['median_tpot_ms']:.0f} ms, p99 {s['p99_tpot_ms']:.0f} ms"
                  + (f", riders {tiers['max_n_cpu']}" if arm == "C" else "")
                  + (f"  [{pre} PREEMPTIONS]" if pre else ""))
        reps.append(pair)

    print("\n[4/4] verdict")
    out = {"mode": "gate", "model": a.model, "ctx": a.ctx, "t0_ms": t0,
           "t0_metric": a.t0_metric, "tail_metric": a.tail_metric,
           "slo_ms": slo_ms, "slo_mult": a.slo_mult,
           "capacity": capacity, "b_hbm_max": b_hbm, "riders_planned": riders,
           "seq_tokens": seq_tokens, "link_gbps": a.link_gbps,
           "b_tok": hp["b_tok"], "widths": widths,
           "gate_ratio": a.gate_ratio, "t0_run": t0_sum, "repeats": reps}
    rc = verdict_gate(a, out)
    out["exit_code"] = rc
    path = os.path.join(a.outdir, a.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return rc


def verdict_gate(a, out: dict) -> int:
    """Decide, and separate 'cannot answer' from 'answered no'."""
    valid = [r for r in out["repeats"]
             if "error" not in r.get("A", {"error": 1})
             and "error" not in r.get("C", {"error": 1})]
    if len(valid) < 2:
        print(f"  INCONCLUSIVE: only {len(valid)} complete repeat(s). A failed "
              f"server is not a slow arm; re-run.")
        return 2

    c_tiers = [r["C"]["tiers"] for r in valid]
    if all(t["samples"] == 0 for t in c_tiers):
        print("  INCONCLUSIVE: no TIER_OCCUPANCY samples in arm C's log. The "
              "instrumentation was off, so 'arm C had riders' is unverified.")
        return 2
    if all(t["max_n_cpu"] == 0 for t in c_tiers):
        print("  INCONCLUSIVE: arm C never created a host-tier request "
              "(max n_cpu = 0 across repeats), so both arms ran the same "
              "configuration. Check b_hbm_max against the achieved "
              "concurrency, and the link constant.")
        return 2

    ratios = [r["C"]["goodput_req_s"] / r["A"]["goodput_req_s"]
              for r in valid if r["A"]["goodput_req_s"] > 0]
    if not ratios:
        print("  INCONCLUSIVE: arm A scored zero goodput; nothing to compare.")
        return 2
    mean_ratio = sum(ratios) / len(ratios)
    worst = min(ratios)
    # THE TAIL IS REPORTED, NOT GATED ON. Ben's objection, and it is correct:
    # goodput ALREADY encodes the SLO -- it counts only requests whose TPOT met
    # the target. Adding "p99 TPOT <= SLO" on top applies the same threshold a
    # second time as a different statistic, and demands that ~99% of requests
    # pass. That is a far stricter bar than "delivers more SLO-meeting requests
    # per second", and it silently replaces the experiment's question.
    #
    # It also grades on distribution SHAPE rather than on service: arm A passes
    # trivially because its TPOTs are tight (median 545, p99 548), while arm C
    # is penalised for spread at 21/24 attainment. Two systems with identical
    # goodput would be graded differently. Attainment is the honest diagnostic
    # and is printed per repeat.
    tail_key = f"p99_{a.tail_metric}_ms"
    p99_ok = all(r["C"][tail_key] <= out["slo_ms"] for r in valid)
    # Is the baseline held to the bar we are holding the mixture to? On the
    # A100 dry run arm A's own p99 pooled ITL was 27.95 ms against an 8.78 ms
    # SLO it met on 64/64 requests -- so the literal condition "p99 ITL(C) <=
    # 1.5 x T0" disqualifies the mixture for missing something the baseline
    # misses by more. A gate only one arm has to clear is not a gate.
    base_tail_ok = all(r["A"][tail_key] <= out["slo_ms"] for r in valid)

    if a.max_num_seqs:
        capped = [r for r in valid
                  if r["C"]["tiers"]["max_n_cpu"] + r["C"]["tiers"]["max_n_hbm"]
                  >= a.max_num_seqs]
        if capped:
            print(f"  WARNING: arm C reached --max-num-seqs {a.max_num_seqs} in "
                  f"{len(capped)}/{len(valid)} repeats, so its concurrency was "
                  f"bounded by the admission width and not by what the riders "
                  f"made room for. Any win measured here is a LOWER bound.")
    for r in valid:
        print(f"  repeat {r['repeat']}: A {r['A']['goodput_req_s']:.3f} -> "
              f"C {r['C']['goodput_req_s']:.3f} req/s  "
              f"({(r['C']['goodput_req_s'] / max(r['A']['goodput_req_s'], 1e-9) - 1) * 100:+.1f}%)"
              f"  riders max {r['C']['tiers']['max_n_cpu']}")
    print(f"\n  mean ratio {mean_ratio:.3f} (worst repeat {worst:.3f}), "
          f"threshold {a.gate_ratio:.2f}")
    print(f"  p99 {a.tail_metric.upper()}(C) within SLO ({out['slo_ms']:.1f} ms): "
          f"{'yes' if p99_ok else 'NO'}   "
          f"[baseline arm A: {'yes' if base_tail_ok else 'NO'}]")
    if not base_tail_ok:
        print("  NOTE: the baseline misses this tail condition too, so it is "
              "not a property the mixture is uniquely failing. Judge the "
              "tail on the ARM-A-RELATIVE comparison, not the absolute bar.")

    if mean_ratio >= a.gate_ratio and worst >= 1.0:
        print("\n  GO: the mixture clears the gate at the grid's most "
              "favourable point.")
        if not p99_ok:
            print("  (Its p99 is above the SLO -- reported, not gated: the "
                  "requests that missed are already excluded from goodput, so "
                  "gating on the tail too would demand ~99% attainment, which "
                  "is a different and much stricter claim.)")
        return 0
    print("\n  NO-GO on the numbers. Diagnose with 4.2b (the rider cost curve) "
          "before spending 4.1's budget: a fixed per-step cost and a "
          "non-overlapping per-byte cost extrapolate differently, and only one "
          "of them is fixable by a faster link.")
    return 1


def mode_grid(a) -> int:
    """4.1: rate sweep per (arm, context); every SLO point derived offline."""
    mults = [float(x) for x in a.slo_mults.split(",") if x]
    contexts = [int(x) for x in a.contexts.split(",") if x]
    print("=" * 62)
    print(f" 4.1 GRID   model={a.model} contexts={contexts}")
    print(f" SLOs derived offline at T0 x {mults} -- no extra GPU time")
    print("=" * 62)

    grid: dict = {"mode": "grid", "model": a.model, "contexts": contexts,
                  "slo_mults": mults, "cells": []}
    for ctx in contexts:
        a.ctx = ctx
        print(f"\n--- context {ctx} ---")
        print("  arm A sweep")
        swa = sweep(a, "A", None, f"A_ctx{ctx}")
        if not swa.get("peak"):
            print(f"  SKIP ctx {ctx}: arm A produced no usable run")
            grid["cells"].append({"ctx": ctx, "error": "arm A sweep failed"})
            continue
        t0 = t0_of(swa["peak"], a.t0_metric)
        # Arm C's B1 budget is a server flag, so it is fixed at the SLO the
        # grid is centred on. The offline sweep over `mults` re-scores the same
        # requests; it does not re-tune the admission budget, and the JSON says
        # so rather than letting a reader assume otherwise.
        centre_slo = a.slo_mult * t0
        print(f"  T0 = {t0:.2f} ms; arm C B1 budget fixed at "
              f"{a.slo_mult:g} x T0 = {centre_slo:.2f} ms")
        print("  arm C sweep")
        swc = sweep(a, "C", centre_slo, f"C_ctx{ctx}")

        cell = {"ctx": ctx, "t0_ms": t0, "centre_slo_ms": centre_slo,
                "arm_A": swa, "arm_C": swc, "points": []}
        for m in mults:
            slo = m * t0
            row = {"slo_mult": m, "slo_ms": slo}
            for arm, sw in (("A", swa), ("C", swc)):
                best = None
                for run_tag in sw["runs"]:
                    if "error" in run_tag:
                        continue
                    res_path = os.path.join(
                        a.outdir,
                        f"result_{arm}_ctx{ctx}_r{run_tag['offered_rate']:g}.json")
                    try:
                        with open(res_path) as fh:
                            res = json.load(fh)
                        g = goodput_at(res, slo)
                    except (OSError, NoDetail):
                        continue
                    if best is None or g["goodput_req_s"] > best["goodput_req_s"]:
                        best = g
                        best["offered_rate"] = run_tag["offered_rate"]
                row[arm] = best
            if row.get("A") and row.get("C") and row["A"]["goodput_req_s"] > 0:
                row["ratio"] = row["C"]["goodput_req_s"] / row["A"]["goodput_req_s"]
            cell["points"].append(row)
            if "ratio" in row:
                print(f"    SLO {m:>4g} x T0 = {slo:8.2f} ms   "
                      f"A {row['A']['goodput_req_s']:.3f}  "
                      f"C {row['C']['goodput_req_s']:.3f}  "
                      f"ratio {row['ratio']:.3f}")
        grid["cells"].append(cell)

    path = os.path.join(a.outdir, a.out)
    with open(path, "w") as fh:
        json.dump(grid, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("gate", "grid", "verify-goodput"),
                   default="gate")
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--python", default=os.path.join(ROOT, ".venv/bin/python"))
    p.add_argument("--ctx", type=int, default=8192,
                   help="prompt tokens (gate/verify)")
    p.add_argument("--contexts", default="4096,8192,16384",
                   help="grid mode: comma-separated prompt lengths")
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--num-gpu-blocks-override", type=int, default=0,
                   help="force a small KV pool so the load binds (dry runs)")
    p.add_argument("--eager", action="store_true",
                   help="--enforce-eager (the hybrid path forces it anyway)")
    p.add_argument("--max-num-seqs", type=int, default=0,
                   help="admission width for BOTH arms. 0 = derive the largest "
                        "value whose pinned host pool fits (see "
                        "host_pool_check); the engine's default of 256 asks "
                        "for 536 GB of pinned memory at 32B/8K")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--host-pool-frac", type=float, default=0.5,
                   help="share of available host RAM the pinned mirrors may use")
    p.add_argument("--min-completion-frac", type=float, default=0.95,
                   help="an arm completing less than this share of its prompts "
                        "is an INVALID run, not a slow one")
    p.add_argument("--b-hbm-max", type=int, default=0,
                   help="arm C: HBM-tier slots. 0 (default) derives it from "
                        "arm A's MEASURED concurrency x --b-hbm-frac; a value "
                        "at or above that ceiling is refused, because arm C "
                        "would then create no riders and silently equal arm A")
    p.add_argument("--b-hbm-frac", type=float, default=1.0,
                   help="unused by the gate: residents stay FULL (b_hbm_max = "
                        "measured capacity) so riders are additional rather "
                        "than substitutional")
    p.add_argument("--hide-window", default="itl", choices=("itl", "tpot", "slo"),
                   help="which measured quantity is the window a rider's KV "
                        "must cross the link inside. Default itl = the decode "
                        "step. 'tpot' carries prefill interference (2x the "
                        "step at ctx 2048) and 'slo' adds the 1.5x on top; "
                        "either over-admits riders that cannot hide.")
    p.add_argument("--overlap-eff", type=float, default=1.0,
                   help="fraction of the window that actually overlaps "
                        "compute. The roofline assumes 1.0; measured ~0.23 "
                        "with the double-buffered prefetch, ~0 without.")
    p.add_argument("--rungd-async", action="store_true",
                   help="arm C: enable the double-buffered prefetch "
                        "(VLLM_KVPIPE_RUNGD_ASYNC=1). Off by default in the "
                        "engine, and it is what the roofline's hide budget "
                        "assumes: without it the step grows by the serial "
                        "transfer time instead of overlapping it")
    p.add_argument("--riders", type=int, default=-1,
                   help="-1 (default) = let the roofline decide how many "
                        "riders hide inside the SLO; a value overrides it")
    p.add_argument("--seq-tokens", type=int, default=0,
                   help="tokens per rider for the roofline; 0 = ctx + out_len")
    p.add_argument("--link-gbps", type=float, default=25.1,
                   help="MEASURED host->device GB/s. On C2C the A100's 25.1 "
                        "makes B1 admit ~36x too few riders; take it from "
                        "transfer_bw_probe.py on the machine under test.")
    p.add_argument("--tier-policy", default="scored", choices=("scored", "arrival"))
    p.add_argument("--rates", default="auto",
                   help="comma-separated offered rates, or 'auto'")
    p.add_argument("--rate-start", type=float, default=1.0)
    p.add_argument("--rate-mult", type=float, default=1.6)
    p.add_argument("--max-rates", type=int, default=6)
    p.add_argument("--plateau-frac", type=float, default=0.03,
                   help="stop the sweep when throughput gains less than this")
    p.add_argument("--bind-frac", type=float, default=0.9,
                   help="achieved < this x offered means the load bound")
    p.add_argument("--slo-mult", type=float, default=1.5,
                   help="gate SLO = this x T0 (plan: 1.5)")
    p.add_argument("--slo-mults", default="1.0,1.2,1.5,2.0,3.0",
                   help="grid: offline SLO points, as multiples of T0")
    p.add_argument("--t0-metric", default="tpot", choices=("tpot", "itl"),
                   help="quantity T0 is measured in. Default tpot, because "
                        "that is what goodput enforces; 'itl' reproduces the "
                        "plan's literal wording and runs ~1.7x looser (see "
                        "t0_of).")
    p.add_argument("--tail-metric", default="tpot", choices=("tpot", "itl"),
                   help="quantity the p99 SLO condition is checked in")
    p.add_argument("--gate-ratio", type=float, default=1.10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--verify-tol", type=float, default=0.005)
    p.add_argument("--port", type=int, default=8677)
    p.add_argument("--server-timeout-s", type=float, default=1800.0)
    p.add_argument("--client-timeout-s", type=float, default=3600.0)
    p.add_argument("--outdir", default="/mnt/beegfs/benishay/paper_runs/gate")
    p.add_argument("--out", default="gate_goodput.json")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    a.hf_cache = os.environ.setdefault(
        "HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    tok = os.path.join(ROOT, "hf_token")
    if os.path.exists(tok) and not os.environ.get("HF_TOKEN"):
        with open(tok) as fh:
            os.environ["HF_TOKEN"] = fh.read().strip()

    if a.mode == "verify-goodput":
        return mode_verify_goodput(a)
    if a.mode == "grid":
        return mode_grid(a)
    return mode_gate(a)


if __name__ == "__main__":
    sys.exit(main())
