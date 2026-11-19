"""BFCL (Berkeley Function Calling Leaderboard v3) through the OpenAI tools API, graded by structure (AST), as the
leaderboard does for its non-live Python categories: the right function names, every required argument present,
no unknown argument, every value among the accepted ones. 'irrelevance': the right answer is no call at all.

Simplified from gorilla's ast_checker: strings are compared after lowercasing and dropping spaces and
punctuation (as its standardize_string), numbers with int/float equivalence, lists and dicts element by element.
"""
import json
import re

TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": None, "integer": "integer",
         "string": "string", "boolean": "boolean", "array": "array", "number": "number", "object": "object"}


def fix_schema(s):
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k == "type":
            t = TYPES.get(v, "string")
            if t:
                out["type"] = t
        elif k == "properties":
            out["properties"] = {n: fix_schema(p) for n, p in v.items()}
        elif k in ("items", "additionalProperties"):
            out[k] = fix_schema(v)
        elif k in ("description", "enum", "required", "default"):
            out[k] = v
    return out


def safe(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def tools(task):
    return [{"type": "function", "function": {"name": safe(f["name"]), "description": f.get("description", ""),
                                              "parameters": fix_schema(f["parameters"])}} for f in task["functions"]]


def std(s):
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def same(got, exp):
    if isinstance(exp, str):
        return isinstance(got, str) and std(got) == std(exp)
    if isinstance(exp, bool) or isinstance(got, bool):
        return got is exp
    if isinstance(exp, (int, float)):
        return isinstance(got, (int, float)) and abs(got - exp) < 1e-9
    if isinstance(exp, list):
        return isinstance(got, (list, tuple)) and len(got) == len(exp) and all(same(g, e) for g, e in zip(got, exp))
    if isinstance(exp, dict):
        return isinstance(got, dict) and set(got) <= set(exp) and all(
            any(same(got[k], o) for o in (exp[k] if isinstance(exp[k], list) else [exp[k]])) for k in got) and all(
            k in got for k, v in exp.items() if not (isinstance(v, list) and "" in v))
    return got == exp


def match_call(call, gt_fn, gt_args, funcs):
    name, args = call
    if name != safe(gt_fn):
        return False, f"name {name} != {gt_fn}"
    spec = next((f for f in funcs if f["name"] == gt_fn), None)
    props = (spec or {}).get("parameters", {}).get("properties", {})
    for k in args:
        if k not in gt_args:
            return False, f"unexpected arg {k}"
    for k, opts in gt_args.items():
        if k not in args:
            if "" in opts or None in opts:
                continue
            return False, f"missing {k}"
        v = args[k]
        if props.get(k, {}).get("type") == "float" and isinstance(v, int):
            v = float(v)
        if not any(same(v, o) for o in opts if o != ""):
            return False, f"{k}={json.dumps(v)[:40]} not accepted"
    return True, "ok"


def grade(task, calls):
    """calls: [(name, args dict)]"""
    if task["category"] == "irrelevance":
        return (1.0, "no call, correct") if not calls else (0.0, f"called {calls[0][0]}")
    gt = task["ground_truth"]
    if len(calls) != len(gt):
        return 0.0, f"{len(calls)} calls, expected {len(gt)}"
    left = list(range(len(calls)))
    for g in gt:  # each expected call matched by a distinct actual one, any order
        (fn, args), = g.items()
        why = "no match"
        for i in left:
            ok, why = match_call(calls[i], fn, args, task["functions"])
            if ok:
                left.remove(i)
                break
        else:
            return 0.0, why
    return 1.0, "ok"
