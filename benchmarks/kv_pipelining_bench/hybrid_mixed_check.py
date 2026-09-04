import json, os, subprocess, sys, time, signal, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/mnt/beegfs/benishay/vllm-kvpipe")
ROOT="/mnt/beegfs/benishay/vllm-kvpipe"; PY=f"{ROOT}/.venv/bin/python"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"; PROMPT="The quick brown fox jumps over the lazy dog and then"

def launch(port, b_hbm_max, stream):
    env=dict(os.environ); env["HF_HUB_CACHE"]="/mnt/beegfs/benishay/huggingface"
    env["VLLM_KVPIPE_HYBRID_STREAM"]="1" if stream else "0"; env["PYTHONUNBUFFERED"]="1"
    kv={"kv_connector":"PipelinedDecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}
    cmd=[PY,"-m","vllm.entrypoints.openai.api_server","--model",MODEL,"--port",str(port),
         "--max-model-len","1024","--gpu-memory-utilization","0.15","--disable-log-requests",
         "--dtype","auto","--max-num-seqs","8","--enforce-eager","--enable-kv-pipelining","--kv-k-window","4",
         "--kv-transfer-config",json.dumps(kv),"--enable-transfer-aware-admission","--enable-hybrid-kv-tiers",
         "--hybrid-b-hbm-max",str(b_hbm_max),"--hybrid-pcie-gbps","25.1"]
    lf=open(f"/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1/mc_{port}.log","w")
    return subprocess.Popen(cmd,env=env,stdout=lf,stderr=subprocess.STDOUT,preexec_fn=os.setsid)

def ready(port,t=180):
    import urllib.error
    t0=time.time()
    while time.time()-t0<t:
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2); return True
        except Exception: time.sleep(2)
    return False

def one(port,i):
    body=json.dumps({"model":MODEL,"prompt":PROMPT,"max_tokens":24,"temperature":0.0,"stream":False}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as resp: return json.loads(resp.read())["choices"][0]["text"]

def run(port,b_hbm_max,stream,n=8):
    p=launch(port,b_hbm_max,stream)
    try:
        if not ready(port): return None
        with ThreadPoolExecutor(max_workers=n) as ex:
            outs=list(ex.map(lambda i: one(port,i), range(n)))
        return outs
    finally:
        try: os.killpg(os.getpgid(p.pid),signal.SIGTERM)
        except Exception: pass
        time.sleep(3)

print("=== MIXED batch: 8 identical prompts, b_hbm_max=4 (4 hbm + 4 cpu), stream ON ===",flush=True)
mixed=run(9895,4,True,8)
print("=== reference: all resident (b_hbm_max=8, stream off) ===",flush=True)
ref=run(9896,8,False,8)
if mixed is None or ref is None:
    print("INCONCLUSIVE: server did not start"); sys.exit(1)
allsame_mixed = len(set(mixed))==1
mixed_eq_ref = (set(mixed)=={ref[0]}) and len(set(ref))==1
print(f"all 8 mixed outputs identical:   {allsame_mixed}")
print(f"mixed output == resident ref:    {mixed_eq_ref}")
print("sample mixed[0]:", repr(mixed[0][:60]))
if allsame_mixed and mixed_eq_ref:
    print("MIXED PASS: HBM + CPU cohorts coexist correctly in one forward.")
else:
    print("MIXED FAIL:", "cohorts diverge" if not allsame_mixed else "differ from resident ref")
    for i,o in enumerate(mixed): print(f"  [{i}] {o[:50]!r}")
