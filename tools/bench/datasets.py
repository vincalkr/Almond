#!/usr/bin/env python3
"""Download the public benchmarks and turn them into local/bench/data/<suite>.jsonl (one task per line).

  ./almond-bench fetch                     # the suites that are missing (present ones are left alone)
  ./almond-bench fetch mmlu-pro ifeval     # only these
  ./almond-bench fetch --update            # also re-download a suite whose dataset has a newer revision
  ./almond-bench fetch --list              # the local files against the manifest

Every file's dataset revision (the Hugging Face commit) and sha256 go to local/bench/data/manifest.json, and each
run records the sha256 of the files it used, so two runs are known to have seen the same questions. Every
file is read from the dataset repository at that revision (resolve/<commit>/...), in seconds. MMLU-Pro, MBPP+ and
HumanEval+ are read from their Parquet files with pyarrow (requirements.txt; HumanEval+ also has a test.jsonl, whose
tests compare floats differently, so the Parquet the leaderboard's loader reads is used). Without pyarrow those three
fall back to Hugging Face's datasets-server: the latest revision only, 100 rows a request, rate limited (minutes).
GPQA is gated: accept its terms on huggingface.co/datasets/Idavidrein/gpqa and set HF_TOKEN (or have
~/.cache/huggingface/token), otherwise it is skipped. Each dataset keeps its own licence; the files stay in
local/ (gitignored) and are not redistributed. GPQA's authors ask not to publish its questions in plain text.
"""
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "local", "bench", "data")
HF = "https://huggingface.co/datasets/"


def token():
    t = os.environ.get("HF_TOKEN")
    p = os.path.expanduser("~/.cache/huggingface/token")
    if not t and os.path.exists(p):
        t = open(p).read().strip()
    return t


def get(url, auth=False):
    h = {"User-Agent": "almond-bench"}
    if auth and token():
        h["Authorization"] = f"Bearer {token()}"
    for i in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404) or i == 9:
                raise
            # datasets-server answers 429 when asked too fast
            time.sleep(float(e.headers.get("Retry-After") or min(60, 5 * 2 ** i)))


def rows(dataset, split, config="default"):
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": off, "length": 100})
        d = json.loads(get("https://datasets-server.huggingface.co/rows?" + q))
        out += [r["row"] for r in d["rows"]]
        off += 100
        time.sleep(0.5)
        print(f"\r  {dataset}: {len(out)}/{d['num_rows_total']}", end="", flush=True)
        if off >= d["num_rows_total"]:
            print()
            return out


def jsonl(url):
    return [json.loads(l) for l in get(url).decode().splitlines() if l.strip()]


def load(dataset, rev, path, split="test"):
    """a dataset's file at a fixed revision: .jsonl read as is, .parquet with pyarrow; without pyarrow the
    datasets-server (latest revision only, 100 rows a request, rate limited: minutes for MMLU-Pro)"""
    if path.endswith(".jsonl"):
        return jsonl(HF + f"{dataset}/resolve/{rev}/{path}")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"  {dataset}: pyarrow missing, through the datasets-server (slow)")
        return rows(dataset, split)
    return pq.read_table(io.BytesIO(get(HF + f"{dataset}/resolve/{rev}/{path}"))).to_pylist()


L = "ABCDEFGHIJ"


def mmlu_pro(rev="main"):
    return [{"id": f"mmlu-{r['question_id']}", "category": r["category"],
             "prompt": r["question"] + "\n\n" + "\n".join(f"{L[i]}) {o}" for i, o in enumerate(r["options"])),
             "answer": r["answer"]} for r in load("TIGER-Lab/MMLU-Pro", rev, "data/test-00000-of-00001.parquet")]


def gpqa(rev="main"):
    if not token():
        raise PermissionError("gated: set HF_TOKEN after accepting the terms on huggingface.co/datasets/Idavidrein/gpqa")
    data = list(csv.DictReader(io.StringIO(get(HF + f"Idavidrein/gpqa/resolve/{rev}/gpqa_diamond.csv", auth=True).decode())))
    rng, out = random.Random(0), []
    for i, r in enumerate(data):  # the options shuffled once, with a fixed seed, so the answer letter is fixed
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        out.append({"id": f"gpqa-{i}", "category": r.get("High-level domain", "gpqa"),
                    "prompt": r["Question"].strip() + "\n\n" + "\n".join(f"{L[j]}) {opts[k].strip()}" for j, k in enumerate(order)),
                    "answer": L[order.index(0)]})
    return out


def math500(rev="main"):
    return [{"id": r["unique_id"].replace("test/", "math-").replace(".json", ""), "category": r["subject"],
             "level": r["level"], "prompt": r["problem"], "answer": r["answer"]} for r in load("HuggingFaceH4/MATH-500", rev, "test.jsonl")]


def humaneval(rev="main"):
    return [{"id": r["task_id"], "category": "humaneval+", "prompt": r["prompt"], "entry_point": r["entry_point"],
             "tests": r["test"] + f"\n\ncheck({r['entry_point']})\n"} for r in load("evalplus/humanevalplus", rev, "data/test-00000-of-00001-5973903632b82d40.parquet")]


def mbpp(rev="main"):
    out = []
    for r in load("evalplus/mbppplus", rev, "data/test-00000-of-00001-d5781c9c51e02795.parquet"):
        name = r["code"].split("def ", 1)[1].split("(", 1)[0].strip() if "def " in r["code"] else ""
        out.append({"id": f"mbpp-{r['task_id']}", "category": "mbpp+",
                    "prompt": f"{r['prompt']}\nThe function must be named `{name}`. Example test:\n{r['test_list'][0]}",
                    "tests": "\n".join(r.get("test_imports") or []) + "\n" + r["test"]})
    return out


def ifeval(rev="main"):
    return [{"id": f"ifeval-{r['key']}", "category": "ifeval", "prompt": r["prompt"],
             "instructions": r["instruction_id_list"],
             "kwargs": [{k: v for k, v in (kw or {}).items() if v is not None} for kw in r["kwargs"]]}
            for r in load("google/IFEval", rev, "ifeval_input_data.jsonl", "train")]


BFCL_CATS = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def bfcl(rev="main"):
    base = HF + f"gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/{rev}/"
    out = []
    for cat in BFCL_CATS:
        qs = jsonl(base + f"BFCL_v3_{cat}.json")
        ans = {} if cat == "irrelevance" else {a["id"]: a["ground_truth"] for a in jsonl(base + f"possible_answer/BFCL_v3_{cat}.json")}
        for q in qs:
            out.append({"id": q["id"], "category": cat, "messages": q["question"][0], "functions": q["function"],
                        "ground_truth": ans.get(q["id"], [])})
        print(f"  bfcl {cat}: {len(qs)}")
    return out


def arena_hard(rev="main"):
    base = HF + f"lmarena-ai/arena-hard-auto/resolve/{rev}/data/arena-hard-v0.1/"
    qs = jsonl(base + "question.jsonl")
    ref = {}
    for a in jsonl(base + "model_answer/gpt-4-0314.jsonl"):
        msgs = a.get("messages") or []
        if msgs:
            c = msgs[-1]["content"]
            ref[a["uid"]] = c.get("answer", "") if isinstance(c, dict) else c
        else:  # the older layout
            ref[a.get("question_id", a.get("uid"))] = a["choices"][0]["turns"][0]["content"]
    return [{"id": f"arena-{q['uid'][:10]}", "category": q.get("cluster", "arena"), "prompt": q["prompt"],
             "reference": ref.get(q["uid"], "")} for q in qs if ref.get(q["uid"])]


# suite: (Hugging Face dataset, builder, licence); every file is read at the revision recorded in the manifest
SOURCES = {
    "mmlu-pro":   ("TIGER-Lab/MMLU-Pro", mmlu_pro, "MIT"),
    "gpqa":       ("Idavidrein/gpqa", gpqa, "CC-BY-4.0, gated, do not republish in plain text"),
    "math500":    ("HuggingFaceH4/MATH-500", math500, "MIT (from MATH)"),
    "humaneval+": ("evalplus/humanevalplus", humaneval, "Apache-2.0"),
    "mbpp+":      ("evalplus/mbppplus", mbpp, "Apache-2.0"),
    "ifeval":     ("google/IFEval", ifeval, "Apache-2.0"),
    "bfcl":       ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", bfcl, "Apache-2.0"),
    "arena-hard": ("lmarena-ai/arena-hard-auto", arena_hard, "Apache-2.0; the GPT-4 answers under OpenAI's terms"),
}
MANIFEST = os.path.join(DATA, "manifest.json")


def manifest():
    try:
        return json.load(open(MANIFEST))
    except (OSError, ValueError):
        return {}


def path(name):
    return os.path.join(DATA, f"{name}.jsonl")


def available():
    return {n: os.path.exists(path(n)) for n in SOURCES}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def revision(dataset):
    """the dataset repository's current commit on Hugging Face"""
    return json.loads(get(f"https://huggingface.co/api/datasets/{dataset}", auth=True))["sha"]


def fetch(names=None, update=False, log=print):
    """download the suites that are missing; with update, also those whose dataset has a newer revision.
    Each file's revision and sha256 go to local/bench/data/manifest.json."""
    os.makedirs(DATA, exist_ok=True)
    man = manifest()
    for n in names or list(SOURCES):
        dataset, build, lic = SOURCES[n]
        have = os.path.exists(path(n))
        if have and not update:
            log(f"{n}: present ({(man.get(n) or {}).get('revision', '?')[:12]}), skipped (--update to check for a new revision)")
            continue
        try:
            rev = revision(dataset)
            if have and update and (man.get(n) or {}).get("revision") == rev and (man.get(n) or {}).get("sha256") == sha256(path(n)):
                log(f"{n}: up to date ({rev[:12]})")
                continue
            tasks = build(rev)
        except Exception as ex:  # noqa: BLE001
            log(f"{n}: skipped ({ex})")
            continue
        tmp = path(n) + ".part"
        with open(tmp, "w") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        os.replace(tmp, path(n))
        man[n] = {"dataset": dataset, "revision": rev, "license": lic, "tasks": len(tasks),
                  "sha256": sha256(path(n)), "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(man, open(MANIFEST, "w"), indent=1)
        log(f"{n}: {len(tasks)} tasks, revision {rev[:12]}")


def verify(log=print):
    """the local files against the manifest: changed by hand, or downloaded before the manifest existed"""
    man, ok = manifest(), True
    for n in SOURCES:
        if not os.path.exists(path(n)):
            log(f"{n:11} missing")
            continue
        m = man.get(n)
        if not m:
            log(f"{n:11} present, not in the manifest (fetch {n} --update records it)")
            ok = False
        elif sha256(path(n)) != m["sha256"]:
            log(f"{n:11} CHANGED since it was fetched (fetch {n} --update)")
            ok = False
        else:
            log(f"{n:11} ok  {m['tasks']:>6} tasks  revision {m['revision'][:12]}  {m['fetched'][:10]}  {m['license']}")
    return ok


def versions():
    """{suite: first 12 hex of its file's sha256}, recorded in every run"""
    return {n: m["sha256"][:12] for n, m in manifest().items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", help=", ".join(SOURCES))
    ap.add_argument("--update", action="store_true", help="re-download a suite whose dataset has a newer revision")
    ap.add_argument("--list", action="store_true", help="the local files against the manifest")
    a = ap.parse_args()
    bad = [s for s in a.suites if s not in SOURCES]
    if bad:
        sys.exit(f"unknown suite {', '.join(bad)}: {', '.join(SOURCES)}")
    if a.list:
        sys.exit(0 if verify() else 1)
    fetch(a.suites, a.update)
