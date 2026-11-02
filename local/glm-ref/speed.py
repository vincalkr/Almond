#!/usr/bin/env python3
"""speed.py PORT [ROUNDS] - decode tok/s through the chat API: thinking low, 600 tokens, three prompts (code,
Italian, English), ROUNDS rounds (the measure of docs/ENGINE_PLAN.md)."""
import json, sys, urllib.request
port, rounds = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
qs = ["Write a Python function that parses a CSV file and returns the rows as dictionaries, with error handling.",
      "Spiegami in dettaglio come funziona la fotosintesi clorofilliana.",
      "Explain the history of the Roman Empire from Augustus to its fall."]
out = []
for r in range(rounds):
    row = []
    for q in qs:
        b = json.dumps({"messages": [{"role": "user", "content": q}], "reasoning_effort": "low", "max_tokens": 600,
                        "temperature": 0}).encode()
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", b,
                                                                    {"Content-Type": "application/json"}), timeout=900))
        row.append(round(d.get("timings", {}).get("predicted_per_second", 0), 1))
    print("round", r, row, flush=True); out += row
print("mean", round(sum(out) / len(out), 1))
