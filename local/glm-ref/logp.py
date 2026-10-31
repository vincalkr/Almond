#!/usr/bin/env python3
"""logp.py IDS N [RUNS] [CONFIG] - GLM-5.3-Flash's perplexity on the engine over the tokens llama-perplexity scores in
one chunk of N tokens (the second half: N/2+1 .. N-1), for a comparison on the same tokens.

IDS: a file of token ids (int32 binary, .i32, or llama-tokenize --ids output). The engine is started from CONFIG
(default local/glm-local.json) on cards 0 and 1, with LOGP; prints each run's sum, count and PPL.
"""
import ast, json, math, os, struct, subprocess, sys
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(R)
src, n = sys.argv[1], int(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cfg = json.load(open(sys.argv[4] if len(sys.argv) > 4 else "local/glm-local.json"))
raw = open(src, "rb").read()
ids = list(struct.unpack(f"{len(raw)//4}i", raw)) if src.endswith(".i32") else ast.literal_eval(raw.decode().strip().splitlines()[-1])
ids = ids[:n]
env = dict(os.environ, HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES", "0,1"),
           **{k: str(v) for k, v in (cfg.get("env") or {}).items()})
args = list(cfg["args"])
if "--max-context" in args:
    args[args.index("--max-context") + 1] = str(max(int(args[args.index("--max-context") + 1]), n + 64))
p = subprocess.Popen([cfg.get("exe", "build-hip-fused/almond-engine"), "--serve", *args], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=open("local/glm-ref/logp-engine.log", "w"), text=True, env=env)
while True:
    l = p.stdout.readline()
    if not l:
        sys.exit("the engine ended (local/glm-ref/logp-engine.log)")
    if l.startswith("READY"):
        break
csv = ",".join(map(str, ids))
for _ in range(runs):
    p.stdin.write(f"LOGP {n // 2 + 1} {csv}\n")
    p.stdin.flush()
    while True:
        l = p.stdout.readline()
        if l.startswith(("LP", "ERR")) or not l:
            break
    f = l.split()
    print(l.strip(), " PPL %.4f" % math.exp(-float(f[1]) / int(f[2])) if l.startswith("LP") else "", flush=True)
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
