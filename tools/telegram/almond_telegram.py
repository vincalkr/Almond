#!/usr/bin/env python3
"""Almond on Telegram: a bot that chats with the models behind the router, from the phone.

  ./almond telegram                 (the router must answer, or it is started as ./almond does)

Setup: a bot from @BotFather (its token), then local/telegram.json (kept out of git):

  {"token": "123456:ABC...", "allowed_users": [111222333]}

With "allowed_users" empty the bot answers anyone with their Telegram id and nothing else, so you can read yours and
add it; with ids in it, everyone else is ignored (logged only).  Optional keys: "model" (the one to start with, as
-m), "agent" (true: the agent's tools on from the start), "cwd" (the agent's directory, default the home), "web"
(true: web_search / web_fetch for the agent), "url" (the router, default http://127.0.0.1:8150), "api_key".

Nothing is exposed: the bot asks Telegram for updates (long polling) and talks to the router on 127.0.0.1.

In a chat:
  text                  a message (an answer streams by editing one message); a photo: with its caption, to a
                        -vision model
  /new                  a new conversation (the current one stays in local/chats/)
  /model [NAME]         the models, or switch (the router loads it)
  /agent [on|off]       the agent's tools: files and commands in "cwd", each write, edit and command asks with
                        buttons (Yes / No / Always); off, the model only has schedule (reminders) and its answers
  /schedule [WHEN PROMPT | stop N|all]   as the CLI's: 30m, at 9:00, at 9:00 tomorrow, every 2h; alone, the list
  /stop                 stop the answer
  /status               model, agent, context, scheduled tasks
Messages sent while it answers wait their turn.  Each chat is kept in local/chats/telegram-<chat id>.json, which
`./almond -c telegram-<chat id>` resumes on the terminal too.
"""
import base64
import json
import os
import queue
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
import almond_cli as A  # noqa: E402

ROOT = A.ROOT
CONF = Path(os.environ.get("ALMOND_TELEGRAM_CONF") or ROOT / "local" / "telegram.json")
LIMIT = 4000                                    # Telegram's 4096 per message, with room
EDIT_EVERY = 1.2                                # seconds between the edits of a streaming answer (Telegram's limits)

CHAT_PROMPT = """You are a helpful assistant the user reaches through Telegram, from the phone. Answer in the \
user's language, briefly; plain text (no Markdown tables). Date: {date}.
When the user asks for something later or repeatedly ("remind me in 30 minutes", "every morning at 8 tell me ..."), \
use the schedule tool: the prompt comes back to you as a turn marked [scheduled task N] and your answer reaches the \
user's phone; a reminder: tell the user plainly. Stop a repeating one with schedule_stop when asked."""

TG_NOTE = """
You are reached through Telegram: the user reads you on the phone. Keep answers short and in plain text; the user \
approves each write, edit and command with buttons."""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Bot:
    """The Bot API over HTTPS (stdlib only); ALMOND_TELEGRAM_API points it elsewhere (tests)."""

    def __init__(self, token: str):
        base = os.environ.get("ALMOND_TELEGRAM_API", "https://api.telegram.org")
        self.url, self.file_url = f"{base}/bot{token}/", f"{base}/file/bot{token}/"

    def call(self, method: str, http_timeout=30, **params):
        data = json.dumps({k: v for k, v in params.items() if v is not None}).encode()
        req = urllib.request.Request(self.url + method, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                out = json.loads(e.read())
            except ValueError:
                raise RuntimeError(f"{method}: HTTP {e.code}")
        if not out.get("ok"):
            raise RuntimeError(f"{method}: {out.get('description')}")
        return out.get("result")

    def send(self, chat, text: str, buttons=None, quiet=False) -> int | None:
        """text in as many messages as it takes; the last one's id (the buttons go on it)."""
        mid = None
        parts = split(text or "…")
        for i, p in enumerate(parts):
            markup = {"inline_keyboard": buttons} if buttons and i == len(parts) - 1 else None
            mid = self.call("sendMessage", chat_id=chat, text=p, reply_markup=markup,
                            disable_notification=quiet or None)["message_id"]
        return mid

    def edit(self, chat, mid, text: str, buttons=None):
        try:
            self.call("editMessageText", chat_id=chat, message_id=mid, text=text[:LIMIT] or "…",
                      reply_markup={"inline_keyboard": buttons} if buttons is not None else None)
        except RuntimeError as e:
            if "not modified" not in str(e):
                raise

    def download(self, file_id: str) -> bytes:
        path = self.call("getFile", file_id=file_id)["file_path"]
        with urllib.request.urlopen(self.file_url + path, timeout=60) as r:
            return r.read()


def split(text: str) -> list:
    """Telegram's message size: at paragraph or line ends where it can."""
    out = []
    while len(text) > LIMIT:
        cut = max(text.rfind("\n\n", 0, LIMIT), text.rfind("\n", 0, LIMIT))
        cut = cut if cut > LIMIT // 2 else LIMIT
        out.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return out + [text]


class TgAgent(A.Agent):
    """The CLI's agent with the asking on Telegram: buttons for y / n / always, the questions as messages, the
    task list and the schedule per chat."""

    def __init__(self, chat, yes=False, web=False):
        super().__init__(yes, web)
        self.chat = chat

    def allow(self, name: str, what: str, outside: bool = False) -> bool:
        if not outside and (self.yes or name in self.always):
            return True
        opts = [("Yes", "y"), ("No", "n")] + ([] if outside else [("Always", "a")])
        a = self.chat.ask_buttons(f"{what} - allow?" + (f" (outside {self.root})" if outside else ""), opts)
        if a == "a":
            self.always.add(name)
        return a in ("y", "a")

    def ask_free(self) -> bool:
        return False                            # (free_gpu: the CLI's; the bot never unloads the model)

    def t_ask_user(self, question, options=None):
        opts = [str(o) for o in (options or []) if str(o).strip()][:9]
        a = self.chat.ask_buttons("? " + question, [(o[:60], f"o{i}") for i, o in enumerate(opts)], free_text=True)
        if a and a.startswith("o") and a[1:].isdigit() and int(a[1:]) < len(opts):
            a = opts[int(a[1:])]
        return f"the user answered: {a}" if a else "the user gave no answer"

    def t_todo_write(self, items):
        out = super().t_todo_write(items)
        self.chat.bot.send(self.chat.id, A.todo_text(A.TODO, color=False), quiet=True)
        return out

    def t_schedule(self, when, prompt):
        return self.chat.schedule_tool(when, prompt)

    def t_schedule_stop(self, id=""):
        return self.chat.schedule_stop_tool(id)


class Chat:
    """One Telegram chat: its conversation, model, agent, scheduled tasks, and a worker that takes its messages one
    at a time."""

    def __init__(self, bot: Bot, cid: int, conf: dict, models: list):
        self.bot, self.id, self.conf = bot, cid, conf
        self.api = A.Api(conf.get("url", "http://127.0.0.1:8150"), conf.get("api_key", ""))
        self.messages, self.ctx_used, self.date = [], 0, time.strftime("%Y-%m-%d")
        self.model = pick(models, conf.get("model"))
        self.agent = None
        self.sched = A.Schedule()
        self.sched.notify = lambda: self.inbox.put(("tick", None))
        self.inbox = queue.Queue()
        self.waiting = None                     # (Event, answer box, message id, free_text) while a question waits
        self.stopped = False
        self.file = A.CHATS / f"telegram-{cid}.json"
        self.load()
        if conf.get("agent") and self.agent is None:
            self.set_agent(True)
        threading.Thread(target=self.work, daemon=True).start()

    # ---- state on disk (the CLI's format: ./almond -c telegram-<id> resumes it)
    def load(self):
        try:
            d = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.messages, self.ctx_used = d.get("messages") or [], d.get("ctx_used") or 0
        self.date = d.get("date") or self.date
        if d.get("agent"):
            self.set_agent(True, d.get("cwd"))

    def save(self):
        if not self.messages:
            return
        A.CHATS.mkdir(parents=True, exist_ok=True)
        data = {"model": self.model["id"], "level": None, "sampling": {}, "system": None,
                "cwd": str(self.agent.root if self.agent else self.cwd()), "ctx_used": self.ctx_used,
                "agent": bool(self.agent), "web": bool(self.agent and self.agent.web), "date": self.date,
                "messages": self.messages}
        self.file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def cwd(self) -> Path:
        return Path(os.path.expanduser(self.conf.get("cwd") or "~")).resolve()

    def set_agent(self, on: bool, cwd: str | None = None):
        if on:
            self.agent = TgAgent(self, yes=bool(self.conf.get("yes")), web=bool(self.conf.get("web")))
            self.agent.root = Path(cwd).resolve() if cwd and Path(cwd).is_dir() else self.cwd()
        else:
            self.agent = None

    # ---- the worker
    def work(self):
        while True:
            kind, v = self.inbox.get()
            try:
                if kind == "tick":
                    for t in self.sched.due():
                        rep = f", every {A.fmt_secs(t['every'])}" if t["every"] else ""
                        self.bot.send(self.id, f"⏰ scheduled task {t['id']}{rep}", quiet=True)
                        self.turn(f"[scheduled task {t['id']}{rep}, {time.strftime('%H:%M')}] {t['prompt']}")
                elif kind == "text":
                    if v.startswith("/"):
                        self.command(v)
                    else:
                        self.turn(v)
                elif kind == "photo":
                    self.turn(v[0], images=[v[1]])
            except Exception as e:              # one bad turn must not end the chat's worker
                log(f"chat {self.id}: {e}\n{traceback.format_exc()}")
                try:
                    self.bot.send(self.id, f"error: {e}")
                except Exception:
                    pass

    def ask_buttons(self, text: str, options: list, free_text=False) -> str | None:
        """A question with buttons; the worker waits (an hour at most) for a tap, or a typed answer if free_text."""
        ev, box = threading.Event(), {}
        mid = self.bot.send(self.id, text, buttons=[[{"text": t, "callback_data": d} for t, d in options]]
                            if options else None)
        self.waiting = (ev, box, mid, free_text)
        try:
            ev.wait(3600)
        finally:
            self.waiting = None
        a = box.get("a")
        label = next((t for t, d in options if d == a), a)
        try:
            self.bot.edit(self.id, mid, f"{text}\n→ {label if label else 'no answer'}", buttons=[])
        except RuntimeError:
            pass
        return a

    def answer(self, a: str) -> bool:
        """A tap or a typed answer for the question that waits; False when none waits."""
        w = self.waiting
        if not w:
            return False
        w[1]["a"] = a
        w[0].set()
        return True

    # ---- a turn
    def tools(self) -> list:
        if self.agent:
            return self.agent.tools()
        return [t for t in A.TOOLS if t["function"]["name"] in ("schedule", "schedule_stop")]

    def wire(self) -> list:
        if self.agent:
            text = (A.AGENT_PROMPT.format(os=A.agent_os(), cwd=self.agent.root, date=self.date)
                    + (A.AGENT_WEB_PROMPT if self.agent.web else "") + TG_NOTE)
        else:
            text = CHAT_PROMPT.format(date=self.date)
        return [{"role": "system", "content": text}] + [dict(m) for m in self.messages]

    def turn(self, text: str, images=None):
        content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in images] \
            if images else text
        self.messages.append({"role": "user", "content": content})
        tools = self.agent or TgAgent(self)       # no agent: a bare one only for schedule / schedule_stop
        for _ in range(A.AGENT_STEPS):
            calls = self.reply()
            if calls is None:
                self.messages.pop()
                return
            if not calls or self.stopped:
                break
            results = []
            for c in calls:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except ValueError:
                    args = {}
                brief = str(args.get("command") or args.get("path") or args.get("query") or args.get("url")
                            or args.get("when") or args.get("id") or "")
                if c["name"] not in ("todo_write", "ask_user", "schedule"):   # (they show themselves)
                    self.bot.send(self.id, f"⚙ {c['name']} {brief}"[:LIMIT], quiet=True)
                if not self.agent and c["name"] not in ("schedule", "schedule_stop"):
                    out = f"error: no tool {c['name']} (the agent is off: /agent on)"
                else:
                    out = tools.run(c)
                results.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            self.messages += results
        self.save()

    def reply(self):
        """One answer streamed into one message (edited as it grows); its tool calls, None on an error."""
        body = {"model": self.model["id"], "stream": True, "messages": self.wire(), "tools": self.tools()}
        content, reasoning, calls, end = [], [], {}, None
        mid, last, shown, self.stopped = None, 0.0, "", False
        try:
            self.bot.call("sendChatAction", chat_id=self.id, action="typing")
        except RuntimeError:
            pass
        try:
            for kind, v in self.api.stream(body):
                if self.stopped:
                    try:
                        self.api.live.close()
                    except Exception:
                        pass
                    break
                if kind == "content":
                    content.append(v)
                elif kind == "reasoning":
                    reasoning.append(v)
                elif kind == "tool":
                    c = calls.setdefault(v.get("index", len(calls)), {"id": None, "name": "", "arguments": ""})
                    fn = v.get("function") or {}
                    c["id"] = v.get("id") or c["id"]
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                elif kind == "end":
                    end = v
                now = time.time()
                if now - last >= EDIT_EVERY:
                    text = "".join(content).strip()
                    view = (text[-LIMIT:] if text else ("💭 thinking…" if reasoning else "")) + " ▍"
                    if view.strip(" ▍") and view != shown:
                        if mid is None:
                            mid = self.bot.send(self.id, view)
                        else:
                            self.bot.edit(self.id, mid, view)
                        shown, last = view, now
        except (RuntimeError, OSError, A.http.client.HTTPException) as e:
            self.bot.send(self.id, f"error: {e}")
            return None
        text = "".join(content).strip() + (" [stopped]" if self.stopped else "")
        if text:
            parts = split(text)
            if mid is None:
                self.bot.send(self.id, text)
            else:
                self.bot.edit(self.id, mid, parts[0])
                for p in parts[1:]:
                    self.bot.send(self.id, p)
        elif mid is not None:
            self.bot.edit(self.id, mid, "…")
        msg = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        calls = [dict(c, id=c["id"] or f"call_{i}") for i, c in sorted(calls.items())] if not self.stopped else []
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        self.messages.append(msg)
        if end and end.get("usage"):
            u = end["usage"]
            self.ctx_used = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        return calls

    # ---- schedule (the tools and /schedule)
    def schedule_tool(self, when, prompt):
        try:
            t = self.sched.add_when(when, prompt)
        except ValueError as e:
            return f"error: {e}; nothing scheduled"
        self.bot.send(self.id, sched_line(self.sched, t), quiet=True)
        return (f"scheduled task {t['id']}: first at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t['at']))}"
                + (f", then every {A.fmt_secs(t['every'])}" if t["every"] else "") + "\n" + self.sched.plain())

    def schedule_stop_tool(self, id=""):
        gone = self.sched.stop(str(id).strip()) if str(id).strip() else []
        return (f"stopped: {', '.join(str(x['id']) for x in gone) or 'none'}\n" if str(id).strip() else "") \
            + self.sched.plain()

    # ---- commands
    def command(self, line: str):
        cmd, _, arg = line[1:].partition(" ")
        cmd, arg = cmd.split("@")[0].lower(), arg.strip()
        send = lambda t: self.bot.send(self.id, t)
        if cmd in ("start", "help"):
            send(__doc__.split("In a chat:")[1].split("Messages sent")[0].strip("\n"))
        elif cmd in ("new", "clear"):
            self.save()
            old = self.file
            if old.exists():
                old.rename(old.with_name(f"telegram-{self.id}-{time.strftime('%Y%m%d-%H%M%S')}.json"))
            self.messages, self.ctx_used, self.date = [], 0, time.strftime("%Y-%m-%d")
            A.TODO.clear()
            send("a new conversation")
        elif cmd == "model":
            models = self.api.get("/v1/models").get("data") or []
            if not arg:
                send("\n".join(("● " if m["id"] == self.model["id"] else "  ") + m["id"]
                               + f" ({(m.get('status') or {}).get('value', '?')})" for m in models))
                return
            m = pick(models, arg, strict=True)
            if not m:
                send(f"no model {arg}")
                return
            send(f"loading {m['id']} …")
            t0 = time.time()
            self.api.post("/models/load", {"model": m["id"]})
            self.model = m
            send(f"{m['id']} loaded in {time.time() - t0:.0f} s")
        elif cmd == "agent":
            if arg in ("on", "off"):
                self.set_agent(arg == "on", str(self.agent.root) if self.agent else None)
            send(f"agent {'on in ' + str(self.agent.root) + ' (each write, edit and command asks)' if self.agent else 'off'}")
        elif cmd == "schedule":
            if not arg:
                send(self.sched.plain())
            elif arg.split()[0] == "stop":
                gone = self.sched.stop(arg[4:].strip())
                send(f"stopped: {', '.join(str(x['id']) for x in gone) or 'none'}")
            else:
                try:
                    send(sched_line(self.sched, self.sched.add(arg)))
                except ValueError as e:
                    send(f"{e}\n/schedule 30m PROMPT · at 9:00 PROMPT · at 9:00 tomorrow PROMPT · every 2h PROMPT")
        elif cmd == "stop":
            send("nothing to stop")             # (a running answer is stopped from the poller: see Bridge)
        elif cmd == "status":
            n = sum(1 for m in self.messages if m.get("role") == "user")
            send(f"model {self.model['id']}\nagent {'on in ' + str(self.agent.root) if self.agent else 'off'}\n"
                 f"{n} turns, context {self.ctx_used} / {self.model.get('max_context') or '?'}\n{self.sched.plain()}")
        else:
            send(f"no command /{cmd} (/help)")


def sched_line(s, t) -> str:
    rep = f" · every {A.fmt_secs(t['every'])}" if t["every"] else ""
    return f"{'↻' if t['every'] else '◷'} task {t['id']} · {s.when(t)} (in {A.fmt_in(t['at'] - time.time())}){rep}"


def pick(models: list, name: str | None, strict=False):
    if name:
        n = name.lower()
        hits = [m for m in models if m["id"].lower() == n] or [m for m in models if m["id"].lower().startswith(n)] \
            or [m for m in models if n in m["id"].lower()]
        if len(hits) >= 1 and (len(hits) == 1 or not strict):
            return hits[0]
        if strict:
            return None
    return next((m for m in models if (m.get("status") or {}).get("value") == "loaded"), None) \
        or next((m for m in models if m.get("default")), None) or models[0]


class Bridge:
    def __init__(self, conf: dict):
        self.conf, self.bot = conf, Bot(conf["token"])
        self.allowed = {int(x) for x in conf.get("allowed_users") or []}
        self.chats: dict[int, Chat] = {}
        self.api = A.Api(conf.get("url", "http://127.0.0.1:8150"), conf.get("api_key", ""))

    def chat(self, cid: int) -> Chat:
        if cid not in self.chats:
            self.chats[cid] = Chat(self.bot, cid, self.conf, self.api.get("/v1/models").get("data") or [])
        return self.chats[cid]

    def run(self):
        me = self.bot.call("getMe")
        log(f"@{me.get('username')} ready; allowed users: {sorted(self.allowed) or 'none yet (answers with the id)'}")
        offset = None
        while True:
            try:
                ups = self.bot.call("getUpdates", http_timeout=70, timeout=50, offset=offset,
                                    allowed_updates=["message", "callback_query"])
            except (RuntimeError, OSError) as e:
                log(f"getUpdates: {e}")
                time.sleep(5)
                continue
            for u in ups or []:
                offset = u["update_id"] + 1
                try:
                    self.update(u)
                except Exception as e:
                    log(f"update {u.get('update_id')}: {e}\n{traceback.format_exc()}")

    def update(self, u: dict):
        if "callback_query" in u:
            q = u["callback_query"]
            uid, cid = q["from"]["id"], q["message"]["chat"]["id"]
            try:
                self.bot.call("answerCallbackQuery", callback_query_id=q["id"])
            except RuntimeError:
                pass
            if uid in self.allowed and cid in self.chats:
                self.chats[cid].answer(q.get("data") or "")
            return
        m = u.get("message") or {}
        uid, cid = (m.get("from") or {}).get("id"), (m.get("chat") or {}).get("id")
        if uid is None or cid is None:
            return
        if uid not in self.allowed:
            log(f"ignored a message from {uid} ({(m.get('from') or {}).get('username')})")
            if not self.allowed:                # setup: tell them their id, nothing else
                self.bot.send(cid, f"Your Telegram id is {uid}: add it to \"allowed_users\" in local/telegram.json "
                                   "and start the bot again.")
            return
        c = self.chat(cid)
        text = m.get("text") or m.get("caption") or ""
        if text.split("@")[0].strip() == "/stop":
            c.stopped = True                    # the worker's stream sees it at its next piece
            if c.waiting:
                c.answer("n")
            return
        if c.waiting and c.waiting[3] and text and not text.startswith("/"):
            c.answer(text)                      # a typed answer to ask_user
            return
        if m.get("photo"):
            data = self.bot.download(m["photo"][-1]["file_id"])
            url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
            c.inbox.put(("photo", (text or "What is in this image?", url)))
        elif text:
            if c.waiting and not c.waiting[3]:
                self.bot.send(cid, "(waiting for the buttons above: your message comes after)", quiet=True)
            c.inbox.put(("text", text))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="./almond telegram", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="the model to start with (over local/telegram.json's \"model\")")
    ap.add_argument("--agent", action="store_true", help="the agent's tools on from the start")
    ap.add_argument("--yes", action="store_true",
                    help="with --agent: no buttons before a write, an edit or a command (a path outside cwd still asks)")
    ap.add_argument("--cwd", help="the agent's directory (default: \"cwd\" in the file, else the home)")
    ap.add_argument("--web", action="store_true", help="web_search / web_fetch for the agent")
    a = ap.parse_args()
    try:
        conf = json.loads(CONF.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"almond telegram: {CONF}: {e}\n\n{__doc__}", file=sys.stderr)
        return 1
    for k, v in (("model", a.model), ("cwd", a.cwd)):
        if v:
            conf[k] = v
    for k in ("agent", "yes", "web"):
        if getattr(a, k):
            conf[k] = True
    if not conf.get("token"):
        print(f"almond telegram: no \"token\" in {CONF}", file=sys.stderr)
        return 1
    api = A.Api(conf.get("url", "http://127.0.0.1:8150"), conf.get("api_key", ""))
    if not api.up():
        if conf.get("url", "").split(":")[-1].split("/")[0] not in ("", "8150") and conf.get("url"):
            print(f"almond telegram: nothing answers at {conf['url']}", file=sys.stderr)
            return 1
        A.start_router(api, api.port)
    if conf.get("model"):                       # the chosen model loaded now, not at the first message
        models = api.get("/v1/models").get("data") or []
        m = pick(models, conf["model"], strict=True)
        if not m:
            print(f"almond telegram: no model {conf['model']!r}; there are: " + ", ".join(x["id"] for x in models),
                  file=sys.stderr)
            return 1
        conf["model"] = m["id"]
        if (m.get("status") or {}).get("value") != "loaded":
            log(f"loading {m['id']} ...")
            try:
                api.post("/models/load", {"model": m["id"]})
            except RuntimeError as e:
                print(f"almond telegram: {m['id']} did not load: {e} (local/router.log)", file=sys.stderr)
                return 1
            log(f"{m['id']} loaded")
    A.JOBS.enabled = False                      # background jobs need the terminal's chat: they run in the foreground
    try:
        Bridge(conf).run()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
