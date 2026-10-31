#!/usr/bin/env python3
"""copytest.py PORT RUNS [PAD_TOKENS] [TEMP]  (env COPYTEST_EXTRA: more request fields, JSON) - copy 40 random sizes back through the chat API, the list after
PAD_TOKENS (about) of filler text (README.md repeated), and count the figures copied wrong."""
import json, os, random, re, sys, urllib.request
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
port, runs = sys.argv[1], int(sys.argv[2])
pad = int(sys.argv[3]) if len(sys.argv) > 3 else 0
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
extra = json.loads(os.environ.get("COPYTEST_EXTRA", "{}"))   # (llama-server: top_k 0, min_p 0, the template's kwargs)
random.seed(7)
nums = [str(random.choice([random.randint(1, 999), random.randint(1000, 99999999)])) + random.choice(["G", "M", "", "K"])
        for _ in range(40)]
text = "\n".join(f"{i+1}. /data/dir{i+1} {n}" for i, n in enumerate(nums))
filler = ""
if pad:
    doc = open(os.path.join(R, "README.md"), encoding="utf-8").read()
    filler = "Un documento di riferimento (non serve per il compito):\n\n" + (doc * (1 + pad * 4 // len(doc)))[:pad * 4] + "\n\n"
q0 = filler + "Ricopia esattamente questa lista, riga per riga, senza commenti:\n\n" + text
tot = 0
for r in range(runs):
    q = (f"(run {r})\n" if pad else "") + q0   # a long prompt read again each run, not resumed from the cache
    b = json.dumps({"messages": [{"role": "user", "content": q}], "reasoning_effort": "low", "max_tokens": 3000,
                    "temperature": temp, "top_p": 0.95, **extra}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", b,
                                                                {"Content-Type": "application/json"}), timeout=1800))
    out = d["choices"][0]["message"].get("content") or ""
    got = dict(re.findall(r"/data/dir(\d+)\s+(\S+)", out))
    bad = [(i + 1, nums[i], got.get(str(i + 1))) for i in range(len(nums)) if got.get(str(i + 1)) != nums[i]]
    tot += len(bad)
    print(f"run {r}: prompt {d['usage']['prompt_tokens']} tokens, {len(bad)} wrong of 40 {bad[:6]}", flush=True)
print("total wrong", tot, "of", runs * 40)
