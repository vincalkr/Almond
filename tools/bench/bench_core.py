#!/usr/bin/env python3
"""almond-bench core: public benchmarks run against an OpenAI-compatible endpoint (the router or one server).

Suites and how they are graded (only arena-hard needs a judge model):
  mmlu-pro    knowledge, 12,032 questions, 10 options      letter of the answer
  gpqa        GPQA Diamond, 198 graduate-level questions    letter (gated: HF_TOKEN)
  math500     MATH-500, competition maths                   \\boxed{} answer, normalized LaTeX
  humaneval+  EvalPlus HumanEval+, 164 functions            the hidden EvalPlus tests run on the code
  mbpp+       EvalPlus MBPP+, 378 functions                 the same
  ifeval      IFEval, 541 prompts with verifiable rules     the 25 rule checkers (ifeval.py)
  bfcl        BFCL v3 function calling, 1,240 cases         structure of the tool calls (bfcl.py)
  arena-hard  Arena-Hard v0.1, 500 hard open prompts        a judge model against GPT-4-0314's answers
KL divergence against a reference build of the same model: kld.py (needs the engines' logit dumps).

Presets: quick = a fixed sample (seed 0) per suite, full = everything. Standard error of a score p over n tasks
is sqrt(p(1-p)/n): +-3.2 points at n=200, +-1.4 at n=1000; two quantizations closer than about twice that are
not told apart by one run.

  .venv/bin/python tools/bench/datasets.py                        # once: download into local/bench/data
  .venv/bin/python tools/bench/bench_core.py --model glm --preset quick
  .venv/bin/python tools/bench/bench_core.py --model qwen27 --suites ifeval,bfcl --preset full --resume

Runs go to local/bench/runs/<time>-<model>-<preset>.json, saved after every task (--resume continues one).
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bfcl  # noqa: E402
import datasets as ds  # noqa: E402
import ifeval  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "local", "bench", "runs")
DATA = os.path.join(ROOT, "local", "bench", "data")

# kind, quick sample size, max_tokens, what it measures
SUITES = {
    "mmlu-pro":   ("choice", 300, 8192, "knowledge"),
    "gpqa":       ("choice", 198, 8192, "knowledge (expert)"),
    "math500":    ("math", 200, 8192, "reasoning / maths"),
    "humaneval+": ("code", 164, 8192, "code"),
    "mbpp+":      ("code", 150, 8192, "code"),
    "ifeval":     ("ifeval", 200, 8192, "instruction following"),
    "bfcl":       ("bfcl", 200, 4096, "tool calling"),
    "arena-hard": ("arena", 50, 8192, "open answers (judge)"),
}
PROMPTS = {
    "choice": "Answer the following multiple-choice question. Think step by step, then finish with the line "
              "'The answer is (X)' where X is the letter of the correct option.\n\n{prompt}",
    "math": "Solve the following problem. Put the final answer, simplified, inside \\boxed{{}}.\n\n{prompt}",
    "humaneval+": "Complete the following Python function. Reply with the whole function (imports included) in "
                  "one ```python code block.\n\n```python\n{prompt}```",
    "mbpp+": "{prompt}\n\nReply with the Python code in one ```python code block.",
}


# ---------------------------------------------------------------- client
def chat(url, model, messages, tools=None, max_tokens=2048, temperature=0.0, api_key=None, timeout=1800, reasoning=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if reasoning:  # none / low / medium / high; unset: the server's default (Almond's: high)
        body["reasoning_effort"] = reasoning
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return msg, {"seconds": time.time() - t0, "completion_tokens": out.get("usage", {}).get("completion_tokens", 0),
                 "truncated": out["choices"][0].get("finish_reason") == "length",
                 "reasoning": msg.get("reasoning_content") or ""}


def loop_score(text):
    """how much of the text's last part is lines it already wrote: near 1 a model repeating itself"""
    lines = [l.strip() for l in text[-6000:].splitlines() if len(l.strip()) > 20]
    return 1 - len(set(lines)) / len(lines) if len(lines) >= 8 else 0.0


def list_models(url, api_key=None):
    req = urllib.request.Request(url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.split("</think>")[-1]
    return text.split("<think>")[0].strip()  # reasoning cut by max_tokens: no answer


# ---------------------------------------------------------------- graders
def grade_choice(task, text):
    m = re.findall(r"answer is\s*\(?\**\s*([A-J])\b", text, re.I) or re.findall(r"[Aa]nswer:\s*\(?\**\s*([A-J])\b", text)
    if not m:
        m = re.findall(r"\b([A-J])\)", text[-200:]) or re.findall(r"\b([A-J])\b", text[-50:])
    got = m[-1].upper() if m else None
    return float(got == task["answer"]), f"got {got}, expected {task['answer']}"


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def norm_math(s):
    """after the MATH reference normalization (Hendrycks et al., math_equivalence.py) plus a few common forms"""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(mathrm|mathbf|textbf)\{([^}]*)\}", r"\2", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("tfrac", "frac"),
                 ("dfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\$", ""), ("$", ""), ("\\%", ""), ("%", ""),
                 ("degrees", ""), ("\\displaystyle", "")):
        s = s.replace(a, b)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)          # "x = 5" -> "5"
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(\d),(?=\d{3}(\D|$))", r"\1", s)    # 1,000 -> 1000
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if s.startswith("0.") is False and re.fullmatch(r"\.\d+", s):
        s = "0" + s
    if s == "0.5":
        s = "\\frac{1}{2}"
    return s


def as_number(s):
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s or "")
    if m:
        return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def grade_math(task, text):
    got, exp = norm_math(last_boxed(text)), norm_math(task["answer"])
    ok = got is not None and (got == exp or (as_number(got) is not None and as_number(exp) is not None
                                             and abs(as_number(got) - as_number(exp)) < 1e-6))
    return float(ok), f"got {got}, expected {exp}"


def extract_code(text):
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def grade_code(task, text):
    code = extract_code(text)
    if task["category"] == "humaneval+" and f"def {task['entry_point']}" not in code:
        code = task["prompt"] + code  # the model wrote only the body
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(code + "\n\n" + task["tests"] + "\n")
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, text=True, timeout=60,
                               env={"PATH": "/usr/bin:/bin", "HOME": d})
        except subprocess.TimeoutExpired:
            return 0.0, "timeout"
    if r.returncode == 0:
        return 1.0, "tests passed"
    return 0.0, (r.stderr.strip().splitlines() or ["failed"])[-1][:160]


def grade_ifeval(task, text):
    strict, loose = ifeval.check(task, text)
    ids = [i.split(":")[1] for i in task["instructions"]]
    bad = [i for i, ok in zip(ids, strict) if not ok]
    return float(all(strict)), (f"strict {sum(strict)}/{len(strict)} loose {sum(loose)}/{len(loose)}"
                                + (f" | failed {','.join(bad)}" if bad else "")), {"inst_strict": strict, "loose": all(loose)}


# ---------------------------------------------------------------- arena-hard judge
JUDGE_SYS = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to "
    "the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is "
    "to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the "
    "prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, "
    "compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate "
    "information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Then consider "
    "creativity and novelty when needed. Finally, identify any missing important information.\n\nAfter providing "
    "your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\".")
VERDICT_B = {"A>>B": 0.0, "A>B": 0.25, "A=B": 0.5, "B=A": 0.5, "B>A": 0.75, "B>>A": 1.0}


def judge_once(task, a, b, opts):
    user = (f"<|User Prompt|>\n{task['prompt']}\n\n<|The Start of Assistant A's Answer|>\n{a}\n"
            f"<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{b}\n"
            "<|The End of Assistant B's Answer|>")
    msg, _ = chat(opts.get("judge_url") or opts["url"], opts["judge_model"],
                  [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
                  max_tokens=opts.get("judge_max_tokens", 4096), api_key=opts.get("api_key"))
    v = re.findall(r"\[\[([AB][<>=]{1,2}[AB])\]\]", strip_think(msg.get("content")))
    return VERDICT_B.get(v[-1]) if v else None


def grade_arena(task, text, opts):
    """win rate against GPT-4-0314 (the Arena-Hard v0.1 baseline), both orders to cancel position bias"""
    if not opts.get("judge_model"):
        return None, "skipped: no judge model"
    s1 = judge_once(task, task["reference"], text, opts)        # the model as B
    s2 = judge_once(task, text, task["reference"], opts)        # the model as A
    got = [x for x in (s1, None if s2 is None else 1 - s2) if x is not None]
    if not got:
        return 0.5, "no verdict parsed"
    return sum(got) / len(got), f"vs gpt-4-0314: as B {s1}, as A {None if s2 is None else 1 - s2}"


# ---------------------------------------------------------------- tasks
def load_suite(name, preset="quick", limit=None):
    p = os.path.join(DATA, f"{name}.jsonl")
    if not os.path.exists(p):
        return []
    tasks = [json.loads(l) for l in open(p)]
    n = limit or (SUITES[name][1] if preset == "quick" else len(tasks))
    if n < len(tasks):
        by = {}
        for t in tasks:
            by.setdefault(t.get("category", ""), []).append(t)
        rng = random.Random(0)
        pick = []  # stratified by category, so a quick sample keeps every subject
        for cat, ts in sorted(by.items()):
            k = max(1, round(n * len(ts) / len(tasks)))
            pick += rng.sample(ts, min(k, len(ts)))
        rng.shuffle(pick)
        tasks = sorted(pick[:n], key=lambda t: t["id"])
    return tasks


def available_suites():
    return [s for s in SUITES if os.path.exists(os.path.join(DATA, f"{s}.jsonl"))]


@dataclass
class Result:
    suite: str
    id: str
    category: str
    score: float | None
    note: str
    answer: str
    seconds: float
    tokens: int
    extra: dict = field(default_factory=dict)


def stderr(scores):
    n = len(scores)
    if n < 2:
        return 0.0
    m = sum(scores) / n
    return math.sqrt(sum((x - m) ** 2 for x in scores) / (n - 1) / n)


@dataclass
class Run:
    model: str
    url: str
    started: str
    preset: str = "quick"
    judge: str | None = None
    reasoning: str | None = None  # the reasoning_effort sent; None: the server's default
    results: list = field(default_factory=list)
    path: str | None = None
    data: dict = field(default_factory=ds.versions)  # suite -> sha256 of its file (datasets.py manifest)

    def summary(self):
        by = {}
        for r in self.results:
            if r.score is not None:
                by.setdefault(r.suite, []).append(r.score)
        cut = {}
        for r in self.results:
            if r.extra.get("truncated"):
                cut[r.suite] = cut.get(r.suite, 0) + 1
        per = {k: {"score": sum(v) / len(v), "se": stderr(v), "n": len(v), "truncated": cut.get(k, 0)} for k, v in by.items()}
        if per:
            per["overall"] = {"score": sum(x["score"] for x in per.values()) / len(per), "se": 0.0,
                              "n": sum(x["n"] for x in per.values())}
        return per

    def save(self):
        os.makedirs(RUNS, exist_ok=True)
        if not self.path:
            self.path = os.path.join(RUNS, f"{self.started.replace(':', '')}-{re.sub(r'[^\w.-]', '_', self.model)}-{self.preset}.json")
        tmp = self.path + ".part"
        json.dump({"model": self.model, "url": self.url, "started": self.started, "preset": self.preset,
                   "judge": self.judge, "reasoning": self.reasoning, "data": self.data, "summary": self.summary(), "results": [asdict(r) for r in self.results]},
                  open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)
        return self.path

    @staticmethod
    def load(path):
        d = json.load(open(path))
        r = Run(d["model"], d["url"], d["started"], d.get("preset", "quick"), d.get("judge"), path=path, data=d.get("data") or {}, reasoning=d.get("reasoning"))
        r.results = [Result(**x) for x in d["results"]]
        return r


def load_runs():
    out = []
    if os.path.isdir(RUNS):
        for fn in sorted(os.listdir(RUNS)):
            if fn.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(RUNS, fn)))
                except (OSError, ValueError):
                    continue
                if not isinstance(next(iter(d.get("summary", {}).values()), {}), dict):
                    continue  # a run from the first proof of concept
                d["path"] = os.path.join(RUNS, fn)
                out.append(d)
    return out


def run_task(suite, task, opts):
    kind, _, max_tok, _ = SUITES[suite]
    max_tok = opts.get("max_tokens") or max_tok
    url, model, key = opts["url"], opts["model"], opts.get("api_key")
    extra = {}
    try:
        if kind == "bfcl":
            msg, st = chat(url, model, task["messages"], tools=bfcl.tools(task), max_tokens=max_tok, api_key=key,
                           reasoning=opts.get("reasoning"))
            calls = []
            for c in msg.get("tool_calls") or []:
                try:
                    calls.append((c["function"]["name"], json.loads(c["function"].get("arguments") or "{}")))
                except json.JSONDecodeError:
                    calls.append((c["function"]["name"], {"__bad_json__": True}))
            score, note = bfcl.grade(task, calls)
            text = json.dumps([{"name": n, "arguments": a} for n, a in calls], ensure_ascii=False) or ""
            text = (strip_think(msg.get("content")) + "\n" + text).strip()
        else:
            tmpl = PROMPTS.get(suite) or PROMPTS.get(kind) or "{prompt}"
            msg, st = chat(url, model, [{"role": "user", "content": tmpl.format(prompt=task["prompt"])}],
                           max_tokens=max_tok, api_key=key, reasoning=opts.get("reasoning"))
            text = strip_think(msg.get("content"))
            if kind == "choice":
                score, note = grade_choice(task, text)
            elif kind == "math":
                score, note = grade_math(task, text)
            elif kind == "code":
                score, note = grade_code(task, text)
            elif kind == "ifeval":
                score, note, extra = grade_ifeval(task, text)
            else:
                score, note = grade_arena(task, text, opts)
    except Exception as ex:  # noqa: BLE001
        return Result(suite, task["id"], task.get("category", suite), 0.0, f"error: {ex}", "", 0, 0)
    if st.get("truncated"):  # the reasoning ran into max_tokens: a 0 here says nothing about the answer
        think = st.get("reasoning") or (msg.get("content") or "")
        loop = loop_score(think)
        note = f"TRUNCATED at max_tokens ({max_tok}){', looping' if loop > 0.5 else ''} | {note}"
        extra.update(truncated=True, loop=round(loop, 2), reasoning_tail=think[-3000:])
    return Result(suite, task["id"], task.get("category", suite), score, note, text, st["seconds"],
                  st["completion_tokens"], extra)


def run_bench(run, suites, opts, limit=None, on_result=None, stop=lambda: False):
    done = {(r.suite, r.id) for r in run.results}
    todo = [(s, t) for s in suites for t in load_suite(s, run.preset, limit) if (s, t["id"]) not in done]
    for i, (s, t) in enumerate(todo):
        if stop():
            break
        r = run_task(s, t, opts)
        run.results.append(r)
        run.save()
        if on_result:
            on_result(i + 1, len(todo), r)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8150/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--suites", default="", help="comma-separated; default every downloaded suite but arena-hard")
    ap.add_argument("--preset", choices=["quick", "full"], default="quick")
    ap.add_argument("--limit", type=int, help="tasks per suite, overriding the preset")
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-url")
    ap.add_argument("--api-key")
    ap.add_argument("--max-tokens", type=int, help="override every suite's own limit")
    ap.add_argument("--reasoning", choices=["none", "low", "medium", "high"],
                    help="the thinking level sent as reasoning_effort (default: the server's, Almond's is high)")
    ap.add_argument("--resume", help="a run file to continue")
    a = ap.parse_args()
    have = available_suites()
    pick = [s for s in a.suites.split(",") if s in have] if a.suites else [s for s in have if s != "arena-hard"]
    opts = {"url": a.url, "model": a.model, "judge_model": a.judge_model, "judge_url": a.judge_url,
            "api_key": a.api_key, "max_tokens": a.max_tokens, "reasoning": a.reasoning}
    run = Run.load(a.resume) if a.resume else Run(a.model, a.url, time.strftime("%Y-%m-%dT%H:%M:%S"), a.preset, a.judge_model, a.reasoning)
    opts["reasoning"] = run.reasoning  # a resumed run keeps its own level

    def show(i, n, r):
        print(f"[{i}/{n}] {r.suite:10} {r.id:28} {'-' if r.score is None else f'{r.score:.2f}':>5}  {r.note[:70]}", flush=True)
    run_bench(run, pick, opts, a.limit, show)
    for k, v in run.summary().items():
        print(f"{k:12} {v['score'] * 100:6.1f} +- {v['se'] * 100:4.1f}  (n={v['n']})")
    print("saved", run.path)


if __name__ == "__main__":
    main()
