import json, os, subprocess, time, signal, urllib.request
from concurrent.futures import ThreadPoolExecutor
PY="/mnt/beegfs/benishay/vllm-kvpipe/.venv/bin/python"; MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROMPT="The quick brown fox jumps over the lazy dog and then"
def launch(port, stream, bhm):
    env=dict(os.environ); env["HF_HUB_CACHE"]="/mnt/beegfs/benishay/huggingface"
    env["VLLM_KVPIPE_HYBRID_STREAM"]="1" if stream else "0"
    kv={"kv_connector":"PipelinedDecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}
    cmd=[PY,"-m","vllm.entrypoints.openai.api_server","--model",MODEL,"--port",str(port),
         "--max-model-len","1024","--gpu-memory-utilization","0.15","--disable-log-requests","--dtype","auto",
         "--max-num-seqs","8","--enforce-eager","--no-enable-prefix-caching","--enable-kv-pipelining","--kv-k-window","4",
         "--kv-transfer-config",json.dumps(kv),"--enable-transfer-aware-admission","--enable-hybrid-kv-tiers",
         "--hybrid-b-hbm-max",str(bhm),"--hybrid-pcie-gbps","25.1","--hybrid-b-tok","131072"]
    lf=open(f"/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1/da_{port}.log","w")
    return subprocess.Popen(cmd,env=env,stdout=lf,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
def ready(port,t=180):
    t0=time.time()
    while time.time()-t0<t:
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2); return True
        except Exception: time.sleep(2)
    return False
def one(port):
    b=json.dumps({"model":MODEL,"prompt":PROMPT,"max_tokens":24,"temperature":0.0,"stream":False}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=b,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as resp: return json.loads(resp.read())["choices"][0]["text"]
def run(port,stream,bhm):
    p=launch(port,stream,bhm)
    try:
        if not ready(port): return None
        with ThreadPoolExecutor(max_workers=8) as ex: return list(ex.map(lambda i: one(port), range(8)))
    finally:
        try: os.killpg(os.getpgid(p.pid),signal.SIGTERM)
        except Exception: pass
        time.sleep(3)
r_res=run(9897,False,8)   # all resident (stream off), prefix-cache OFF
r_str=run(9898,True,4)    # mixed streamed (4+4), prefix-cache OFF
print("RESIDENT (stream off): distinct outputs =", len(set(r_res)) if r_res else "ERR")
print("STREAMED (4hbm+4cpu):  distinct outputs =", len(set(r_str)) if r_str else "ERR")
if r_res and len(set(r_res))>1:
    print("=> RESIDENT itself diverges: the 1-vs-7 is a SCHEDULING/batching artifact, NOT a streaming bug.")
elif r_res and r_str and len(set(r_res))==1 and len(set(r_str))==1 and set(r_res)==set(r_str):
    print("=> MIXED STREAMING CORRECT: resident uniform AND streamed==resident.")
elif r_res and len(set(r_res))==1 and r_str and len(set(r_str))>1:
    print("=> REAL multi-request streaming bug: resident uniform but streamed diverges.")
