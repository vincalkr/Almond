# Vendored from the GLM engine (MIT, serve/loop_guard.py at v1.1.37), unchanged.
"""serve/loop_guard.py - the loop guard: an answer that has fallen into a loop - the same block of tokens again and
again until the token cap (a greedy MiniMax-M3.1 a 2-bit quant answer wrote one number sequence for 12K tokens) - is seen
as it starts and broken.  It works on the token ids the server reads from the engine, so every engine (GLM,
DeepSeek, MiniMax) gets it alike, in the thinking and in the answer.

    LoopDetector   the rolling check: is the tail of what was written exactly periodic - a block of P_MIN..P_MAX
                   tokens repeated back to back REPS_LONG times (REPS_SHORT for a block under SHORT_BELOW tokens),
                   MIN_SPAN tokens at least?  Each token is compared only with the earlier places of the same id
                   (within P_MAX), so a token costs microseconds.
    LoopGuard      one request's guard: the detector, the short tail the stream holds back (at most HOLD_MAX
                   tokens: a block the model copies from earlier in its output streams on as it is written, never
                   held whole), and what each loop does:
                     penalty          the repeats not sent yet are dropped (what was sent stays) and the engine goes
                                      on from there with a short repetition penalty (LOOP_PENALTY over the last
                                      2 x period tokens, at least PENALTY_MIN_N, for the next 2 x period +
                                      PENALTY_EXTRA): the 1st loop in the answer, the 1st and 2nd in the thinking
                     close thinking   the 3rd loop in the thinking: closed with the model's own end-of-thinking token,
                                      it answers (never a silent end while it thinks)
                     stop             the 2nd loop in the answer, or a 4th in all: the answer ends (finish "stop")
                   A loop in the thinking needs more copies (REPS_*_THINK, MIN_SPAN_THINK): reasoning that drafts a
                   plan again is not one.

Until a loop is seen nothing changes: the engine gets the same calls with the same sampling and writes the same
tokens (the stream only holds back up to HOLD_MAX tokens of a tail that repeats while it does).  Legit repetition - a table's rows, an array
literal, a function name over and over, code lines alike - is not an exact block repeated back to back three times
or more, which is what a loop is.

Off: ALMOND_LOOP_GUARD=0, "loop_guard": false in the run config, or "loop_guard": false in a request (true there turns
it on for that request when the server has it off).
"""
from __future__ import annotations

import re
from typing import Callable, NamedTuple

P_MIN = 8               # the shortest block: shorter loops (one token, "0, 0, 0") repeat a multiple of it in 8..15
P_MAX = 2048            # the longest: a loop of whole paragraphs or a plan written again (an IQ2_XXS quant thought a ~1,170-
                        # token plan four times); answers without a loop hold back and cost the same as with 400
SHORT_BELOW = 32        # blocks under this many tokens must repeat REPS_SHORT times, longer ones REPS_LONG times
REPS_SHORT = 4
REPS_LONG = 3
MIN_SPAN = 96           # ... and the copies together cover this many tokens at least
REPS_SHORT_THINK = 5    # in the thinking: more copies (a model re-drafting its plan is not looping)
REPS_LONG_THINK = 4
MIN_SPAN_THINK = 192
HOLD_MAX = 64           # the most the stream holds back (~1-3 s): a copied block streams on; past this tail, a
                        # loop's repeats are already on screen and only the ones not sent yet are dropped
THINK_PENALTIES = 2     # loops in the thinking broken by the penalty before the guard closes the thinking
STOP_AFTER = 4          # loops in all after which the answer ends (in the answer; the thinking is closed first)
LOOP_PENALTY = 1.15     # the temporary repetition penalty after the first loop
PENALTY_MIN_N = 128     # its window (penalty_last_n): 2 x the period, at least this
PENALTY_EXTRA = 128     # it applies to the next 2 x period + this many tokens, then the request's own sampling again
PENALTY_N_MAX = 4096    # (the engines count at most this many: server.PENALTY_WINDOW_MAX)

PENALTY, CLOSE_THINKING, STOP = "penalty", "close_thinking", "stop"


class Loop(NamedTuple):
    """A loop at the tail: `period` tokens a copy; the last `repeats` tokens repeat the `period` before them."""
    period: int
    repeats: int

    @property
    def copies(self) -> float:
        return (self.repeats + self.period) / self.period


class Action(NamedTuple):
    kind: str           # PENALTY, CLOSE_THINKING or STOP
    period: int
    copies: float
    drop: int           # the repeats to drop from the end (the first copy stays)
    where: str          # "thinking" or "answer"
    at: int             # the tokens written when it was seen (before the drop)

    def describe(self) -> str:
        what = {PENALTY: "repeats dropped, a short repetition penalty", CLOSE_THINKING: "thinking closed",
                STOP: "answer ended"}[self.kind]
        return (f"a {self.period}-token block {self.copies:.0f} times in the {self.where} at token {self.at:,}: "
                f"{what}")


class LoopDetector:
    """The tail's exact periodicity, updated a token at a time.  `runs[p]` is how many tokens in a row (to the last
    one) equal the token p before them; p is a loop's period once its run covers the copies it needs (`need[p]`).
    Only the periods where the new token equals the one p back go on (the earlier places of that id within P_MAX),
    every other run ends - so the work a token is the number of times its id occurred in the last P_MAX tokens.
    `hold` is the longest run now: the tokens that could still turn out to be a loop's repeats."""

    def __init__(self, p_min: int = P_MIN, p_max: int = P_MAX, min_span: int = MIN_SPAN,
                 reps_short: int = REPS_SHORT, reps_long: int = REPS_LONG, short_below: int = SHORT_BELOW,
                 reps_short_think: int = REPS_SHORT_THINK, reps_long_think: int = REPS_LONG_THINK,
                 min_span_think: int = MIN_SPAN_THINK):
        self.p_min, self.p_max = p_min, p_max
        self.need = [0] * (p_max + 1)           # the run a period needs in the answer ...
        self.need_think = [0] * (p_max + 1)     # ... and in the thinking (more copies)
        for p in range(p_min, p_max + 1):
            reps = reps_short if p < short_below else reps_long
            self.need[p] = max((reps - 1) * p, min_span - p)
            reps = reps_short_think if p < short_below else reps_long_think
            self.need_think[p] = max((reps - 1) * p, min_span_think - p)
        self.reset()

    def reset(self, tokens=()) -> None:
        """Start again over `tokens` (what is kept after a loop's repeats were dropped): only their last 3 x P_MAX
        count - a run reaches at most 2 x P_MAX before it is a loop.  The runs start from zero: what is kept may end
        in copies that were already sent, and a new loop needs copies of its own."""
        self.n, self.seen, self.runs, self.hold = 0, {}, {}, 0
        tokens = list(tokens)[-3 * self.p_max:]
        for t in tokens:
            self.push(t)                        # (a prefix of what was checked token by token: no loop in it)
        self.runs, self.hold = {}, 0

    def push(self, t: int, thinking: bool = False) -> Loop | None:
        i = self.n
        self.n = i + 1
        q = self.seen.get(t)
        runs, hold, best = {}, 0, 0
        if q is None:
            self.seen[t] = [i]
        else:
            lo, k = i - self.p_max, 0
            while k < len(q) and q[k] < lo:
                k += 1
            if k:
                del q[:k]
            old, need, p_min = self.runs, self.need_think if thinking else self.need, self.p_min
            for j in q:                         # (oldest first: the periods from the longest down)
                p = i - j
                if p < p_min:
                    break
                r = old.get(p, 0) + 1
                runs[p] = r
                if r > hold:
                    hold = r
                if r >= need[p]:
                    best = p                    # the shortest that is a loop: its fundamental period
            q.append(i)
        self.runs, self.hold = runs, hold
        return Loop(best, runs[best]) if best else None


def penalty_window(period: int) -> int:
    return min(PENALTY_N_MAX, max(PENALTY_MIN_N, 2 * period))


class LoopGuard:
    """One request's loop guard (server.Service._sliced runs it): push() every id the engine writes, send only what
    lies before the last `hold` of them (at most HOLD_MAX), and do what an Action says: drop its repeats not sent
    yet, close the thinking (append `think_end`), or end; then resume() with what is kept and sampling() for each
    engine call after it."""

    def __init__(self, think_end: int | None = None, in_think: bool = False, detector: LoopDetector | None = None):
        self.det = detector or LoopDetector()
        self.think_end = think_end
        self.in_think = bool(in_think) and think_end is not None
        self.hits = {"thinking": 0, "answer": 0}
        self.events: list[Action] = []
        self.until = 0                          # the tokens written up to which the temporary penalty applies
        self.window = 0                         # ... and its penalty_last_n

    @property
    def hold(self) -> int:
        """The tail held back now: what could still be a loop's repeats, never more than HOLD_MAX tokens (a whole
        copied block held until it ends was a long silence, then a burst)."""
        return min(self.det.hold, HOLD_MAX)

    def push(self, t: int, written: int | None = None) -> Action | None:
        if self.in_think and t == self.think_end:
            self.in_think = False
        loop = self.det.push(t, self.in_think)
        if loop is None:
            return None
        where = "thinking" if self.in_think else "answer"
        self.hits[where] += 1
        if where == "thinking":                 # (never STOP while it thinks: that ends with no answer at all)
            kind = PENALTY if self.hits[where] <= THINK_PENALTIES else CLOSE_THINKING
        else:
            kind = PENALTY if self.hits[where] == 1 and sum(self.hits.values()) < STOP_AFTER else STOP
        act = Action(kind, loop.period, loop.copies, loop.repeats, where, written if written is not None else 0)
        self.events.append(act)
        return act

    def resume(self, kept: list[int], act: Action) -> None:
        """After the repeats were dropped (and the thinking closed): check on from what is kept, with the temporary
        penalty for the next 2 x period + PENALTY_EXTRA tokens."""
        if act.kind == CLOSE_THINKING:
            self.in_think = False
        self.det.reset(kept)
        if act.kind != STOP:
            self.window = penalty_window(act.period)
            self.until = len(kept) + 2 * act.period + PENALTY_EXTRA

    def sampling(self, base: dict | None, written: int, left: int) -> tuple[dict | None, int]:
        """The sampling for an engine call that starts after `written` tokens, and how many tokens it may write:
        the request's own (and `left`) - unless the temporary penalty runs, which then ends that call where it ends.
        A request's own stronger penalty or longer window stays."""
        if written >= self.until:
            return base, left
        smp = dict(base or {})
        rp = smp.get("repetition_penalty")
        rp = float(rp) if isinstance(rp, (int, float)) and not isinstance(rp, bool) else 1.0
        smp["repetition_penalty"] = max(rp, LOOP_PENALTY)
        pln = smp.get("penalty_last_n")
        if not (isinstance(pln, int) and not isinstance(pln, bool) and (pln == -1 or pln >= self.window)):
            smp["penalty_last_n"] = self.window
        return smp, max(1, min(left, self.until - written))

    def summary(self) -> list[str]:
        return [a.describe() for a in self.events]


# ------------------------------------------------------------------ never inside a tool call
# A tool call is written as text between the family's markers - `<tool_call>` ... `</tool_call>` (GLM, and MiniMax with
# its namespace in front), DeepSeek's `<｜DSML｜ calls>` ... `</｜DSML｜ calls>` - and calls that repeat (three alike in
# one reply: three files read, three commands run) are legitimate: dropping "the repeats" would cut a call in half and
# hand the client broken JSON.  So a loop whose repeated tokens touch a tool call is never cut and never closes the
# thinking; a loop inside one call's own text (a line written again and again) only gets the short repetition
# penalty, once a call.  Outside tool calls nothing changes.
_TOOL_OPEN = re.compile(r"<tool_call>|<｜DSML｜\s*(?:tool_)?calls[^>]*>")
_TOOL_CLOSE = re.compile(r"</tool_call>|</｜DSML｜\s*(?:tool_)?calls\s*>")
_TAIL = 48              # the text kept to find a marker split across tokens


class ToolSafeGuard(LoopGuard):
    """LoopGuard that leaves tool calls alone.  `piece(id)` is the id's text (a server Detokenizer's push): without
    it this is the plain LoopGuard."""

    def __init__(self, think_end: int | None = None, in_think: bool = False, detector: LoopDetector | None = None,
                 piece: Callable[[int], str] | None = None):
        super().__init__(think_end, in_think, detector)
        self.piece = piece
        self.in_tool = False
        self.tool_penalised = False             # (the penalty went to this call's loop already)
        self.call_n = 0                         # the tokens written in the tool call now
        self.since_tool = 1 << 60               # the tokens written since the last one in a tool call
        self.tail = ""

    def _follow(self, t: int) -> bool:
        """Read token t's text for the markers; True when it is in a tool call (or ends one)."""
        was = self.in_tool
        buf = self.tail + (self.piece(t) or "")
        while True:
            m = (_TOOL_CLOSE if self.in_tool else _TOOL_OPEN).search(buf)
            if m is None:
                break
            self.in_tool = not self.in_tool
            if self.in_tool:
                self.tool_penalised, self.call_n = False, 0
            buf = buf[m.end():]
        self.tail = buf[-_TAIL:]
        return was or self.in_tool

    def push(self, t: int, written: int | None = None) -> Action | None:
        if self.piece is None:
            return super().push(t, written)
        inside = self._follow(t)
        self.since_tool = 0 if inside else self.since_tool + 1
        self.call_n += self.in_tool
        act = super().push(t, written)
        if act is None or (not inside and self.since_tool >= act.drop):
            return act                          # (no loop, or one wholly outside the tool calls: as ever)
        self.events.pop()                       # a loop that touches a tool call: not counted, nothing dropped
        self.hits[act.where] -= 1
        if not self.in_tool or self.tool_penalised or self.call_n < act.period + act.drop:
            return None                         # (a loop of whole calls, or one this call began with: nothing)
        self.tool_penalised = True              # (inside one call's text: only the penalty, the tokens stay)
        act = act._replace(kind=PENALTY, drop=0, where="tool call")
        self.events.append(act)
        return act

    def resume(self, kept: list[int], act: Action) -> None:
        self.since_tool = max(0, self.since_tool - act.drop)
        super().resume(kept, act)
