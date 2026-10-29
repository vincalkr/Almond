#!/usr/bin/env python3
"""Almond's router: one address for every model this machine has, as llama-server's router mode does.

The models are the ones `local/router.json` lists ({"models": [{"model": "glm"}, {"model": "ds41"}, ...],
"default": "glm"}; each entry's "model" / "config" are what ./run.sh takes).  The router asks each one, without loading
it, for its name, context and thinking levels (`run.sh --describe`), and serves:

- GET /v1/models, /models: every model with its status (llama.cpp's: loaded / loading / unloaded, "failed" with the
  exit code), its context and its thinking levels ("reasoning_levels": [value, label, what it does], and the default)
- POST /models/load, /models/unload {"model": NAME}: llama.cpp's
- everything else goes to the model that is loaded.  A POST whose JSON names another model ("model") loads that one
  first (autoload; `--no-autoload` or ?autoload=false answers 400 instead): the cards hold one model at a time, so the
  one running is stopped once its requests are done.  A request that names no model (or a name the router does not
  know) goes to the loaded one, or loads the default.

Each model runs as its own server (./run.sh, on 127.0.0.1:--child-port), started and stopped by the router; its web
app, API and monitor are forwarded as they are.

  tools/router/almond_router.py --port 8150      (./run.sh --router does this)
"""
import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
LOOPBACK = ("127.0.0.1", "localhost", "::1")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authenticate",
       "proxy-connection", "content-length"}


class Models:
    """The models, the one child server that runs, and the switch between them."""

    def __init__(self, entries: list, default: str | None, child_port: int, api_key: str, start_s: float):
        self.child_port, self.api_key, self.start_s = child_port, api_key, start_s
        self.lock = threading.RLock()           # the switch: one load / unload at a time
        self.cv = threading.Condition()         # requests in flight to the child
        self.active = 0
        self.proc = None
        self.current = None                     # the id the child serves (status "loaded" once it answers)
        self.loading = None
        self.models = {}                        # id -> {"run": [...], "info": describe(), "failed": exit code or None}
        for e in entries:
            run = ["--model", e.get("model", "ds41")] + (["--config", e["config"]] if e.get("config") else [])
            info = describe(run)
            mid = e.get("id") or info["model"]
            self.models[mid] = {"run": run, "info": info, "failed": None}
            print(f"router: {mid} ({' '.join(run)}): context {info.get('max_context')}, thinking "
                  f"{'/'.join(v for v, *_ in info.get('reasoning_levels') or [])}", flush=True)
        self.default = default if default in self.models else next(iter(self.models), None)
        if default and default not in self.models:      # "default" may name a run.sh model ("glm") as well as an id
            self.default = next((m for m, d in self.models.items() if d["run"][1] == default), self.default)

    # ---- the listing
    def status(self, mid: str) -> dict:
        d = self.models[mid]
        if mid == self.current and self.proc is not None and self.proc.poll() is None:
            return {"value": "loaded", "args": ["run.sh", *d["run"]]}
        if mid == self.loading:
            return {"value": "loading", "args": ["run.sh", *d["run"]]}
        if d["failed"] is not None:
            return {"value": "unloaded", "args": ["run.sh", *d["run"]], "failed": True, "exit_code": d["failed"]}
        return {"value": "unloaded"}

    def listing(self) -> dict:
        data = []
        for mid, d in self.models.items():
            i = d["info"]
            data.append({"id": mid, "object": "model", "owned_by": "almond", "created": 0, "status": self.status(mid),
                         "max_context": i.get("max_context"), "default": mid == self.default,
                         "reasoning_levels": i.get("reasoning_levels"), "reasoning_default": i.get("reasoning_default"),
                         "architecture": {"input_modalities": ["text", "image"] if i.get("images") else ["text"],
                                          "output_modalities": ["text"]}})
        return {"object": "list", "data": data}

    # ---- requests in flight (a switch waits for them)
    def enter(self):
        with self.cv:
            self.active += 1

    def leave(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    # ---- load / unload
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def unload(self, mid: str | None = None) -> bool:
        with self.lock:
            if mid is not None and mid != self.current:
                return False
            self._stop()
            return True

    def _stop(self):
        if self.proc is None:
            return
        with self.cv:                           # the requests the running model has in flight finish first
            self.cv.wait_for(lambda: self.active == 0)
        print(f"router: stopping {self.current}", flush=True)
        try:
            os.killpg(self.proc.pid, signal.SIGINT)     # run.sh's group: the server ends its engine, power-tune off
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        self.proc, self.current = None, None

    def ensure(self, mid: str) -> str | None:
        """Load `mid` (stopping the other); None when it runs, else why it could not start."""
        with self.lock:
            if mid == self.current and self.alive():
                return None
            self._stop()
            d = self.models[mid]
            cmd = [str(ROOT / "run.sh"), *d["run"], "--port", str(self.child_port), "--host", "127.0.0.1"]
            if self.api_key:
                cmd += ["--api-key", self.api_key]
            print(f"router: loading {mid}: {' '.join(cmd)}", flush=True)
            self.loading, d["failed"] = mid, None
            self.proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
            self.current = mid
            try:
                t0 = time.time()
                while time.time() - t0 < self.start_s:
                    if self.proc.poll() is not None:
                        d["failed"] = self.proc.returncode
                        self.proc, self.current = None, None
                        return f"{mid} exited with {d['failed']} while loading (its log: the server's output)"
                    try:
                        h = json.loads(fetch(self.child_port, "GET", "/health", api_key=self.api_key)[2] or b"{}")
                        if h.get("loaded"):
                            print(f"router: {mid} loaded in {time.time() - t0:.0f} s", flush=True)
                            return None
                    except (OSError, ValueError, http.client.HTTPException):
                        pass
                    time.sleep(1)
                self._stop()
                d["failed"] = -1
                return f"{mid} did not load in {self.start_s:.0f} s"
            finally:
                self.loading = None


def describe(run: list) -> dict:
    out = subprocess.run([str(ROOT / "run.sh"), *run, "--describe"], cwd=ROOT, capture_output=True, text=True,
                         timeout=120)
    line = (out.stdout.strip().splitlines() or [""])[-1]
    try:
        return json.loads(line)
    except ValueError:
        raise SystemExit(f"router: run.sh {' '.join(run)} --describe failed:\n{out.stdout}{out.stderr}")


def fetch(port: int, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
          api_key: str = "") -> tuple:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = dict(headers or {})
    if api_key:
        h.setdefault("Authorization", "Bearer " + api_key)
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    return r.status, r.getheaders(), r.read()


def make_handler(models: Models, api_keys: list, autoload: bool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if not api_keys:
                return True
            auth = self.headers.get("Authorization", "")
            key = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key in api_keys:
                return True
            self._json(401, {"error": {"message": "invalid API key", "type": "authentication_error"}})
            return False

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_OPTIONS(self):
            self._forward(b"")

        def do_DELETE(self):
            self._forward(self._body())

        def do_GET(self):
            u = urlsplit(self.path)
            q = parse_qs(u.query)
            if u.path in ("/v1/models", "/models"):
                if self._authorized():
                    self._json(200, models.listing())
                return
            if u.path == "/router/health" or (u.path in ("/health", "/api/health") and not models.alive()):
                self._json(200, {"status": "ok", "service": "almond-router", "loaded": models.alive(),
                                 "model": models.current, "models": list(models.models)})
                return
            want = q.get("model", [None])[0]            # llama.cpp's ?model= on GET endpoints
            if not self._route(want, q):
                return
            self._forward(b"")

        def do_POST(self):
            u = urlsplit(self.path)
            q = parse_qs(u.query)
            body = self._body()
            if u.path in ("/models/load", "/models/unload"):
                if not self._authorized():
                    return
                try:
                    mid = json.loads(body or b"{}").get("model")
                except ValueError:
                    mid = None
                if mid not in models.models:
                    self._json(404, {"error": {"message": f"unknown model {mid!r}", "type": "not_found_error"}})
                    return
                if u.path == "/models/unload":
                    self._json(200, {"success": models.unload(mid)})
                    return
                why = models.ensure(mid)
                self._json(200 if why is None else 500, {"success": why is None, **({"error": why} if why else {})})
                return
            want = None
            if body[:1] == b"{":
                try:
                    want = json.loads(body).get("model")
                except (ValueError, AttributeError):
                    want = None
            if not self._route(want, q):
                return
            self._forward(body)

        def _route(self, want, q) -> bool:
            """The child that answers: the named model (loaded first when it is another), else the running one, else
            the default."""
            mid = want if want in models.models else (models.current if models.alive() else models.default)
            if mid is None:
                self._json(503, {"error": {"message": "no models in local/router.json"}})
                return False
            if mid == models.current and models.alive():
                return True
            if not (autoload and q.get("autoload", ["true"])[0] != "false"):
                self._json(400, {"error": {"message": f"model {mid} is not loaded (POST /models/load)"}})
                return False
            if not self._authorized():
                return False
            why = models.ensure(mid)
            if why:
                self._json(503, {"error": {"message": why, "type": "unavailable_error"}})
                return False
            return True

        def _forward(self, body: bytes):
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
            models.enter()
            try:
                c = http.client.HTTPConnection("127.0.0.1", models.child_port, timeout=3600)
                try:
                    c.request(self.command, self.path, body=body or None, headers=headers)
                    r = c.getresponse()
                except OSError as e:
                    self._json(502, {"error": {"message": f"the model's server did not answer: {e}"}})
                    return
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                length = r.getheader("Content-Length")
                if length is not None:
                    self.send_header("Content-Length", length)
                    self.end_headers()
                    self.wfile.write(r.read())
                    return
                # a stream (SSE): passed on as it arrives, chunked
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):     # the client went away: so does the request
                    c.close()
            finally:
                models.leave()

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "local" / "router.json"), help="the models (default local/router.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--child-port", type=int, default=8160, help="where the loaded model's server listens (127.0.0.1)")
    ap.add_argument("--api-key", default=os.environ.get("ALMOND_API_KEY", ""),
                    help="require this key (Authorization: Bearer ... or x-api-key; several: key1,key2)")
    ap.add_argument("--no-autoload", action="store_true", help="a request for a model that is not loaded is a 400")
    ap.add_argument("--load", nargs="?", const="", default=None,
                    help="load a model at the start (no name: the default); without it the first request loads one")
    ap.add_argument("--start-timeout", type=float, default=1200, help="seconds a model may take to load")
    a = ap.parse_args()
    keys = [k.strip() for k in a.api_key.split(",") if k.strip()]
    if a.host not in LOOPBACK and not keys:
        raise SystemExit("router: listening beyond this PC needs --api-key")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    models = Models(cfg.get("models") or [], cfg.get("default"), a.child_port, keys[0] if keys else "",
                    a.start_timeout)
    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            # a client that goes away mid-request (the chat's Ctrl+C) is not an error worth a traceback
            if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

    server = Server((a.host, a.port), make_handler(models, keys, not a.no_autoload))
    server.daemon_threads = True
    if a.load is not None:
        threading.Thread(target=models.ensure, args=(a.load or models.default,), daemon=True).start()

    def bye(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)
    print(f"router: http://{a.host}:{a.port}/v1 - {len(models.models)} models, default {models.default}", flush=True)
    server.serve_forever()
    models.unload()
    print("router: stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
