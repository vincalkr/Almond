#!/usr/bin/env python3
"""two-turns.py PORT WORDS1 WORDS2 [OFFSET] - a conversation of three turns through the chat API (thinking off, 48 tokens
each): the first message WORDS1 words of local/glm-ref/long-docs.txt, the next two WORDS2 more words each.  Prints
the server's timings (the cache loan's measure, docs/SPEED_LEVERS.md)."""
import json, sys, time, urllib.request
port, w1, w2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
off = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # a word offset: another conversation (the disk store keeps the earlier ones)
import os
words = open(os.environ.get('ALMOND_AB_TEXT', 'local/glm-ref/long-docs.txt'), encoding='utf-8', errors='replace').read().split()[off:]   # ALMOND_AB_TEXT: another text
msgs = [{"role": "user", "content": "Read this text and keep it in mind:\n" + " ".join(words[:w1])}]
def ask(msgs, n=48):
    b = json.dumps({"messages": msgs, "max_tokens": n, "temperature": 0, "reasoning_effort": "none"}).encode()
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", b,
                                                                {"Content-Type": "application/json"}), timeout=900))
    return time.time() - t, d
t, d = ask(msgs); print("turn 1 %.2f s" % t, d.get("timings"), flush=True)
msgs.append({"role": "assistant", "content": d["choices"][0]["message"]["content"]})
msgs.append({"role": "user", "content": "Now consider also this:\n" + " ".join(words[w1:w1 + w2]) + "\nSummarize in one line."})
t, d = ask(msgs); print("turn 2 %.2f s" % t, d.get("timings"), flush=True)
msgs.append({"role": "assistant", "content": d["choices"][0]["message"]["content"]})
msgs.append({"role": "user", "content": "And this:\n" + " ".join(words[w1 + w2:w1 + 2 * w2]) + "\nOne line."})
t, d = ask(msgs); print("turn 3 %.2f s" % t, d.get("timings"), flush=True)
