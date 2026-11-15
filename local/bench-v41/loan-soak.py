#!/usr/bin/env python3
"""loan-soak.py PORT TURNS [OFFSET] - one conversation of TURNS turns through the chat API (thinking off, greedy,
200 tokens of answer each): a ~2,500-token text first, then each turn 150-900 more words and a question about them.
Prints, per turn, the new tokens, the timings and the U+FFFD / control characters in the answer (the cache loan's
garbled output showed as those), and the answer's first words.  Text: ALMOND_AB_TEXT or local/glm-ref/long-docs.txt."""
import json, os, sys, time, urllib.request
port, turns = sys.argv[1], int(sys.argv[2])
off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
words = open(os.environ.get('ALMOND_AB_TEXT', 'local/glm-ref/long-docs.txt'), encoding='utf-8', errors='replace').read().split()[off:]
msgs = [{"role": "user", "content": "Read this text and keep it in mind:\n" + " ".join(words[:1100]) + "\nSay in one sentence what it is about."}]
at, bad_total = 1100, 0
sizes = [150, 400, 650, 900, 300, 550, 800, 200, 700, 450]
for t in range(turns):
    b = json.dumps({"messages": msgs, "max_tokens": 200, "temperature": 0, "reasoning_effort": "none"}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", b,
                                                                {"Content-Type": "application/json"}), timeout=1200))
    ans = d["choices"][0]["message"]["content"] or ""
    bad = ans.count("�") + sum(1 for c in ans if ord(c) < 32 and c not in "\n\t\r")
    bad_total += bad
    tm = d.get("timings", {})
    print("turn %2d: %5d new, prompt %6.2f s, %5.1f tok/s, %4.1f s wall, bad chars %d | %s" % (
        t + 1, tm.get("prompt_n", 0), tm.get("prompt_ms", 0) / 1000, tm.get("predicted_per_second", 0), time.time() - t0, bad,
        " ".join(ans.split()[:14])), flush=True)
    n = sizes[t % len(sizes)]
    msgs.append({"role": "assistant", "content": ans})
    msgs.append({"role": "user", "content": "More of the text:\n" + " ".join(words[at:at + n]) + "\nWhat does this part add? Two sentences."})
    at += n
print("bad characters in all answers:", bad_total)
