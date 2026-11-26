"""preserve_thinking A/B on MATH-500 pairs: turn 1 problem A (its answer generated once, reasoning kept), turn 2
problem B graded.  Arm 'off' renders turn 1 without its reasoning (the template's default), 'on' with it.
Usage: preserve_ab.py PORT ARM N OUTDIR   (run 'off' first: it generates and stores the turn-1 answers)."""
import json, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/bench"))
import bench_core as B
port, arm, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
U = f"http://127.0.0.1:{port}/v1/chat/completions"
tasks = [json.loads(l) for l in open(Path(__file__).resolve().parents[2] / "local/bench/data/math500.jsonl")]
P = B.PROMPTS["math"] if hasattr(B, "PROMPTS") else "Solve the following problem. Put the final answer, simplified, inside \\boxed{{}}.\n\n{prompt}"
def ask(msgs):
    body = {"model": "x", "messages": msgs, "max_tokens": 8192, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    m = r["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", r.get("usage", {})
first = out / "turn1.json"
t1 = json.loads(first.read_text()) if first.exists() else {}
res = []
for i in range(n):
    a, b = tasks[2 * i], tasks[2 * i + 1]
    ua = {"role": "user", "content": P.format(prompt=a["prompt"] if "prompt" in a else a["problem"])}
    if str(i) not in t1:
        c, rc, _ = ask([ua]); t1[str(i)] = {"content": c, "reasoning": rc}; first.write_text(json.dumps(t1))
    h = [ua, {"role": "assistant", "content": t1[str(i)]["content"], "reasoning_content": t1[str(i)]["reasoning"]},
         {"role": "user", "content": P.format(prompt=b["prompt"] if "prompt" in b else b["problem"])}]
    t = time.time(); c, rc, u = ask(h)
    ok = B.grade_math(b, c)[0] >= 1.0
    res.append({"i": i, "ok": bool(ok), "prompt_tokens": u.get("prompt_tokens"), "completion": u.get("completion_tokens"), "s": round(time.time() - t, 1)})
    print(arm, i, ok, u.get("prompt_tokens"), u.get("completion_tokens"), flush=True)
(out / f"turn2-{arm}.json").write_text(json.dumps(res))
print(arm, "correct", sum(r["ok"] for r in res), "of", len(res))
