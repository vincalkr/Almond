#!/usr/bin/env python3
"""midturn.py PORT - an agent chain (system prompt, a user turn, tool steps with reasoning) and then a user message
mid-turn; prints how much of each prompt the server reused (a new user message drops the turn's reasoning)."""
import json, os, sys, urllib.request
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
port = sys.argv[1]
doc = open(os.path.join(R, "docs/ds4/V41_EXACT_FAST.md"), encoding="utf-8").read()[:30000]
tools = [{"type": "function", "function": {"name": "bash", "description": "run a command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
import random
msgs = [{"role": "system", "content": f"You are a coding agent (session {random.getrandbits(48):x}).\n\n" + doc},
        {"role": "user", "content": "Elenca i file della cartella e dimmi quanti sono."}]
def ask(tag):
    b = json.dumps({"messages": msgs, "tools": tools, "reasoning_effort": "low", "max_tokens": 64, "temperature": 0}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", b,
                                                                {"Content-Type": "application/json"}), timeout=900))
    u = d["usage"]; c = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"{tag:28} prompt {u['prompt_tokens']:6}  reused {c}  read {u['prompt_tokens'] - (c or 0)}", flush=True)
ask("1 user turn")
for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
    msgs.append({"role": "assistant", "content": "", "reasoning_content": f"Step {k}: I need to look at the folder again, "
                 "listing it with details and counting the entries carefully before answering. " * 6,
                 "tool_calls": [{"id": f"call_{k}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"ls -la part{k}"})}}]})
    msgs.append({"role": "tool", "tool_call_id": f"call_{k}", "content": "\n".join(f"-rw-r--r-- 1 u u {100 + i} file{k}_{i}.txt" for i in range(40))})
    ask(f"{k + 2} tool step {k}")
msgs.append({"role": "user", "content": "contali solo i .txt"})
ask("5 user message mid-turn")
