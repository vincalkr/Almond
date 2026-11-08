#!/usr/bin/env python3
"""launch - starts one model's server or the router: what `./almond serve` does.

  ./almond serve                 DeepSeek-V4.1-Flash-1017 on cards 0 and 1 (card 1 as the expert tier), images on when
                                 the encoder is there
  ./almond serve --gpu 1         one card (0 or 1); --gpu 0,1 two cards, as the default
  ./almond serve --router        every model in local/router.json behind one address (tools/router/almond_router.py):
                                 the request's "model" loads it (stopping the other); /v1/models lists them with their
                                 thinking levels.  The chat: ./almond (starts the router itself when nothing answers)
  ./almond serve --model glm     GLM-5.3-Flash (GSQ-RCO 3.5bit) on Almond's engine (src/glm), the layers split over
                                 both cards: local/glm-local.json, tools/glm/serve/serve_glm.py
  ./almond serve --model qwen    Qwen3.8-Flash-Next (GSQ-RCO IQ3_S) on Almond's engine (src/qwen, --qwen):
                                 local/qwen-local.json, tools/qwen/serve/serve_qwen.py
  ./almond serve --model qwen27  the dense Qwen3.8-27B (GSQ-RCO IQ3_S + its MTP block) on src/qwen35 (--qwen35):
                                 local/qwen27-local.json, tools/qwen/serve/serve_qwen.py
  ./almond serve --ctx 1m        the context window (tokens, or 128k / 256k / 1m), in a copy of the configuration
                                 (ALMOND_CTX the same; the router's models inherit it)

--gpu picks DeepSeek's configuration: local/ds41-local-vision.json for two cards, local/ds41-local-1gpu-vision.json for
one; without the vision encoder in models/vision-1017 (or the image routing bias local/rco-iq2xxs-meta/vl-bias.gguf)
the text-only ones (local/ds41-local.json, local/ds41-local-1gpu.json).  The encoder is loaded on card 0 only for a
request with images and freed after it.  --config FILE uses that file instead.  Other arguments go to the server, for
example:  ./almond serve --port 8151.  ./almond stop stops this folder's servers.  The settings and the measured
numbers: docs/ds4/.

Standard library only: it runs under any python3 and starts the server with the one that has the server's packages.
"""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PT = "/usr/local/sbin/almond-power-tune"
SHIMS = {"ds41": "tools/v41/serve/serve_ds41.py", "glm": "tools/glm/serve/serve_glm.py",
         "qwen": "tools/qwen/serve/serve_qwen.py", "qwen27": "tools/qwen/serve/serve_qwen.py"}
SERVERS = r"tools/(v41/serve/serve_ds41|glm/serve/serve_glm|qwen/serve/serve_qwen|router/almond_router)[.]py"


def say(text: str):
    print(f"almond serve: {text}", file=sys.stderr, flush=True)


def fail(text: str):
    raise SystemExit(f"almond serve: {text}")


def server_python() -> str:
    """The server needs the standard library plus pillow, psutil, regex and jinja2: this repository's own venv, a
    venv/ next to it, or any python3 that has them; ALMOND_PY=/path/to/python forces one."""
    import shutil
    for cand in (os.environ.get("ALMOND_PY", ""), str(ROOT / ".venv/bin/python"), shutil.which("python3") or "",
                 str(ROOT / "venv/bin/python")):
        if cand and os.access(cand, os.X_OK) and subprocess.run(
                [cand, "-c", "import PIL, psutil, regex, jinja2"], capture_output=True).returncode == 0:
            return cand
    fail("no python with pillow, psutil, regex and jinja2 found.\n"
         "  make one:        python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2\n"
         "  or point at one: ALMOND_PY=/path/to/python ./almond serve")


def parse(argv: list) -> tuple:
    """--gpu, --config, --model, --ctx and --router are taken out; the rest go to the server as they are."""
    o = {"gpu": "", "config": "", "model": "ds41", "router": False, "ctx": os.environ.get("ALMOND_CTX", "")}
    rest, i = [], 0
    while i < len(argv):
        a = argv[i]
        key, eq, val = a.partition("=")
        if key in ("--gpu", "--config", "--model", "--ctx"):
            if not eq:
                if i + 1 >= len(argv):
                    fail(f"{a} needs a value")
                i += 1
                val = argv[i]
            o[key[2:]] = val
        elif a == "--router":
            o["router"] = True
        else:
            rest.append(a)
        i += 1
    g = o["gpu"] or "0,1"
    if not all(c.isdigit() for c in g.split(",")):
        fail(f"--gpu takes card numbers: 0, 1 or 0,1 (got '{o['gpu']}')")
    if o["model"] == "deepseek":
        o["model"] = "ds41"
    return o, rest


def pick_config(o: dict) -> str:
    """The configuration run for --model / --gpu when --config does not name one."""
    m = o["model"]
    if m not in SHIMS:
        fail(f"--model takes ds41, glm, qwen or qwen27 (got '{m}')")
    if m != "ds41" and not os.access(ROOT / "build-hip-fused/almond-engine", os.X_OK):
        fail("build-hip-fused/almond-engine is not built (./build.sh builds it)")
    if m not in ("ds41", "qwen27"):   # the dense 27B has no experts
        # the expert profile the engine starts its VRAM pools from and saves what it learns into: its shipped one
        # until this machine has its own
        prof = ROOT / f"local/{m}-expert-profile.bin"
        if not (prof.is_file() and prof.stat().st_size):
            import shutil
            shutil.copy(ROOT / f"tools/{m}/data/expert-profile-{m}.bin", prof)
    if o["config"]:
        return o["config"]
    vision = (ROOT / "build-vision/build/bin/almond-vision").exists()
    if m == "glm":
        # with the image encoder (local/fetch.sh glm-vision) and almond-vision built: images on, loaded per picture
        on = vision and whole(ROOT / "models/glm53-vision/mmproj-GLM-5.3-Flash-F16.gguf")
        return str(ROOT / ("local/glm-local-vision.json" if on else "local/glm-local.json"))
    if m == "qwen27":
        # the dense Qwen3.8-27B on src/qwen35 (--qwen35): one card, text only for now
        return str(ROOT / "local/qwen27-local.json")
    if m == "qwen":
        # with the image encoder (local/fetch.sh qwen-vision): images on, read on the CPU (the engine lends the
        # encoder no VRAM; its cards hold the expert cache)
        on = vision and whole(ROOT / "models/qwen38-iq3s/mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        return str(ROOT / ("local/qwen-local-vision.json" if on else "local/qwen-local.json"))
    o["gpu"] = o["gpu"] or "0,1"
    cards = len(o["gpu"].split(","))
    if cards not in (1, 2):
        fail(f"one or two cards (--gpu 0, --gpu 1, --gpu 0,1), not '{o['gpu']}'")
    base = "ds41-local-1gpu" if cards == 1 else "ds41-local"
    # the vision configurations need the tower and the image routing bias (the pack has none: DS41_VL_BIAS adds
    # local/rco-iq2xxs-meta/vl-bias.gguf, the checkpoint's bias_vl)
    if whole(ROOT / "models/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf") and \
            whole(ROOT / "local/rco-iq2xxs-meta/vl-bias.gguf") and whole(ROOT / f"local/{base}-vision.json"):
        return str(ROOT / f"local/{base}-vision.json")
    say(f"no vision encoder + vision configuration for {base}: text only")
    return str(ROOT / f"local/{base}.json")


def parse_ctx(text: str) -> int:
    """A number of tokens, or with k / m: 128k = 131072, 256k = 262144, 1m = 1048576 (powers of two, as the
    configurations' 262144)."""
    t = str(text).strip().lower()
    mult = {"k": 1 << 10, "m": 1 << 20}.get(t[-1:], 1)
    num = t[:-1] if mult > 1 else t
    try:
        n = float(num) * mult
    except ValueError:
        n = 0
    if n < 1 or n != int(n):
        raise ValueError(f"--ctx takes a number of tokens, or with k / m (128k, 256k, 1m), got '{text}'")
    return int(n)


def with_ctx(cfg: str, ctx: str) -> str:
    """The context window in a copy of the configuration (DeepSeek's --ctx, GLM's and Qwen's --max-context); the
    configuration itself is not changed."""
    try:
        ctx = str(parse_ctx(ctx))
    except ValueError as e:
        fail(str(e))
    c = json.load(open(cfg))
    a = c["args"]
    flag = "--max-context" if "--glm-pack" in a or "--qwen" in a or "--qwen35" in a else "--ctx"
    if flag in a:
        a[a.index(flag) + 1] = ctx
    else:
        a += [flag, ctx]
    dst = ROOT / "local" / f".run-ctx-{Path(cfg).name}"
    dst.write_text(json.dumps(c, indent=2) + "\n")
    return str(dst)


def whole(p) -> bool:
    q = Path(p) if os.path.isabs(p) else ROOT / p
    return q.exists() and not (q.is_file() and q.stat().st_size == 0)


def missing(cfg: str) -> list:
    """The model files the configuration names (-m, --engram, the pack, its tokenizer) that are not here."""
    c = json.load(open(cfg))
    a = c.get("args", [])
    val = lambda f: a[a.index(f) + 1] if f in a else ""
    model, pack = val("-m"), val("--glm-pack")
    qwen = [val(f) for f in ("--pack", "--native", "--ple-gguf", "--mtp") if "--qwen" in a and f in a]
    engram = val("--engram").split(":") if "--engram" in a else []
    env = c.get("env") or {}
    meta = env.get("DS41_META")               # the geometry overlay a foreign pack reads
    mtp = env.get("ALMOND_GLM_MTP_GGUF")      # GLM's draft block (local/glm-mtp-extract.py makes it)
    need = [p for p in [model, pack, *qwen, *engram, c.get("tokenizer"), meta, mtp] if p]
    # An --engram path this box does not have is not fatal when the pack's own Engram GGUF sits beside the pack: the
    # engine falls back to that and prints that it does (ds41_engram.cpp).  Everything else must be there.
    soft = [p for p in engram if not whole(p)]
    sib = (ROOT / model).resolve().parent / "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" if model else None
    fatal = [p for p in need if p not in soft and not whole(p)]
    if soft and not (sib and whole(sib)):
        fatal += soft
    elif soft:
        say(f"{', '.join(soft)} is not here - the engine reads the pack's own Engram table {sib}")
    return fatal


def others_running() -> bool:
    return subprocess.run(["pgrep", "-f", SERVERS], capture_output=True).returncode == 0


def serve(argv: list) -> int:
    o, rest = parse(argv)
    os.chdir(ROOT)
    # ALMOND_HOME must be this repository: the shim imports serve/, tools/ and almond_tokenizer from it
    os.environ["ALMOND_HOME"] = str(ROOT)
    py = server_python()
    say(f"ALMOND_HOME={ROOT}, python {py}")
    if o["router"]:
        # the router starts each model through this module (power-tune with it), so it needs nothing else here
        os.execv(py, [py, str(ROOT / "tools/router/almond_router.py"), *rest])
    if "--describe" in rest:        # the router's listing (the name and thinking levels, nothing loaded)
        os.environ["DS41_POWER_TUNE"] = "0"
    cfg = pick_config(o)
    if o["ctx"]:
        cfg = with_ctx(cfg, o["ctx"])
    say(f"config {cfg}, cards {o['gpu'] or 'as the config says'}" +
        (f", context {parse_ctx(o['ctx'])}" if o["ctx"] else ""))
    gone = missing(cfg)
    if gone:
        fail(f"missing for {cfg}:\n" + "\n".join("  " + p for p in gone) +
             "\nfetch the model first - see the README, '1. Get the model files'.")
    cmd = [py, str(ROOT / SHIMS[o["model"]]), "--engine", "almond", "--port", "8150", "--config", cfg,
           *(["--gpu", o["gpu"]] if o["gpu"] else []), *rest]

    # local/power-tune.sh for as long as the server runs (+11% short answers, +21% at ~25K context, measured
    # 2026-10-21): only when local/install-power-tune.sh has been run once with sudo (a root-owned copy and a sudo
    # rule for it alone); otherwise nothing changes.  DS41_POWER_TUNE=0 leaves the machine's settings alone.
    tune = os.environ.get("DS41_POWER_TUNE", "1") != "0"
    tuned = tune and os.access(PT, os.X_OK) and \
        subprocess.run(["sudo", "-n", PT, "on"], capture_output=True).returncode == 0
    if tuned:
        say("power-tune on (NVMe APST off, CPU C3 off, performance governor); off again when the server ends")
    elif tune and not os.access(PT, os.X_OK):
        say("power-tune not installed (sudo local/install-power-tune.sh switches it on and off with the server)")
    if not tuned:
        os.execv(py, cmd)
    # The server runs in the foreground (Ctrl+C reaches it as before); this process waits and switches power-tune off
    # however the server ends.  No-op handlers keep it alive until then (an ignored signal would be inherited by the
    # server; a handler is reset at exec).
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda *_: None)
    rc = subprocess.call(cmd)
    # Another Almond server may have started meanwhile (a restart, an A/B that stops one and starts the next): it
    # switched power-tune on and still needs it, so it stays on until the last server ends.
    if others_running():
        say("another Almond server is running: power-tune stays on")
    elif subprocess.run(["sudo", "-n", PT, "off"], capture_output=True).returncode == 0:
        say("power-tune off")
    return rc


if __name__ == "__main__":
    sys.exit(serve(sys.argv[1:]))
