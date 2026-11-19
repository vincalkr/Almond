"""IFEval's 25 verifiable instructions, re-implemented after google-research/instruction_following_eval
(instructions.py), without its dependencies (nltk, langdetect).

Two places differ from the reference: sentences are split with a regular expression instead of nltk's punkt,
and the response language is told by its Unicode script and, within a shared script, stop-word counts,
instead of langdetect. Both only matter for a handful of
the 541 prompts. `check(task, text)` returns (strict, loose) lists of booleans, one per instruction; loose
follows the reference: the instruction passes if any of 8 variants of the response passes (without the first
line, without the last, without both; with the markdown '*' removed).
"""
import json
import re


def words(t):
    return re.findall(r"\b\w+\b", t)


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", t.strip()) if s.strip()]


def cmp(rel, got, n):
    return got < n if rel == "less than" else got >= n


STOP = {
    "en": "the and of to is in that it for you with as are this be on",
    "it": "il la di che e un per non sono una del della con gli le",
    "es": "el la de que y en los se las por un para con una es",
    "fr": "le la de et les des est que une un pour dans qui pas",
    "de": "der die und das ist nicht ich zu den mit sie ein es auf",
    "pt": "de que e o a do da em um para não com uma os se",
    "nl": "de het een en van is dat niet ik zijn op te met voor",
    "ru": "и в не на что я с он как это по но из",
    "hi": "है के में की और का को से एक यह",
    "ar": "في من على أن إلى هذا التي الذي ما مع",
    "ja": "の に は を た が で て と し",
    "ko": "이 는 을 를 에 의 가 고 다 하",
    "zh": "的 是 了 在 我 有 和 就 不 人",
    "fi": "ja on ei se että hän oli ole mutta kun",
    "sw": "na ya wa kwa ni za katika la kuwa",
    "bn": "এবং একটি এই করে হয় থেকে জন্য",
    "th": "ที่ และ การ ใน ของ เป็น ได้",
    "ta": "ஒரு மற்றும் இந்த என்று அது",
    "te": "ఒక మరియు ఈ అని లో",
    "mr": "आणि आहे या की हे",
    "ne": "र छ को मा पनि हो",
    "gu": "અને છે આ કે માં",
    "kn": "ಮತ್ತು ಈ ಒಂದು ಇದು",
    "pa": "ਅਤੇ ਹੈ ਦੇ ਵਿੱਚ ਨੂੰ",
    "ur": "اور ہے کے میں کی",
    "fa": "و در به از که این",
    "vi": "và của là có không những được",
    "bg": "и на е да се за не",
}
STOP = {k: set(v.split()) for k, v in STOP.items()}


# Unicode blocks: a script used by one language of IFEval's settles it; the shared ones (Latin, Cyrillic,
# Arabic, Devanagari) are told apart by stop words among the languages that write with them
SCRIPTS = [
    (0x0C80, 0x0CFF, ["kn"]), (0x0A00, 0x0A7F, ["pa"]), (0x0A80, 0x0AFF, ["gu"]), (0x0B80, 0x0BFF, ["ta"]),
    (0x0C00, 0x0C7F, ["te"]), (0x0980, 0x09FF, ["bn"]), (0x0E00, 0x0E7F, ["th"]), (0xAC00, 0xD7AF, ["ko"]),
    (0x1100, 0x11FF, ["ko"]), (0x3040, 0x30FF, ["ja"]), (0x4E00, 0x9FFF, ["zh", "ja"]),
    (0x0900, 0x097F, ["hi", "mr", "ne"]), (0x0600, 0x06FF, ["ar", "fa", "ur"]), (0x0400, 0x04FF, ["ru", "bg"]),
]


def lang_of(t):
    count = {}
    for ch in t:
        o = ord(ch)
        for lo, hi, langs in SCRIPTS:
            if lo <= o <= hi:
                count[tuple(langs)] = count.get(tuple(langs), 0) + 1
                break
        else:
            if ch.isalpha():
                count[("latin",)] = count.get(("latin",), 0) + 1
    if not count:
        return None
    langs = max(count, key=count.get)
    if ("ja",) in count and langs == ("zh", "ja"):
        return "ja"  # kana beside the ideographs
    if len(langs) == 1 and langs[0] != "latin":
        return langs[0]
    cands = [k for k in STOP if k in langs] if langs != ("latin",) else ["en", "it", "es", "fr", "de", "pt", "nl", "fi", "sw", "vi"]
    low = t.lower()
    toks = set(re.findall(r"\w+", low))
    best, score = None, 0
    for k in cands:
        sc = sum(1 for w in STOP[k] if w in toks)
        if sc > score:
            best, score = k, sc
    return best or (langs[0] if langs != ("latin",) else None)


def verify(iid, kw, r, prompt):
    kind = iid.split(":", 1)[1]
    if kind == "no_comma":
        return "," not in r
    if kind == "number_highlighted_sections":
        hl = [h for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip()]
        hl += [h for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h.removeprefix("**").removesuffix("**").strip()]
        return len(hl) >= kw.get("num_highlights", 1)
    if kind == "number_words":
        return cmp(kw["relation"], len(words(r)), kw["num_words"])
    if kind == "number_placeholders":
        return len(re.findall(r"\[.*?\]", r)) >= kw.get("num_placeholders", 1)
    if kind == "number_bullet_lists":
        n = len(re.findall(r"^\s*\*[^\*].*$", r, re.M)) + len(re.findall(r"^\s*-.*$", r, re.M))
        return n == kw["num_bullets"]
    if kind == "title":
        return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r))
    if kind == "english_lowercase":
        return r == r.lower() and lang_of(r) in ("en", None)
    if kind == "english_capital":
        return r == r.upper() and lang_of(r) in ("en", None)
    if kind == "capital_word_frequency":
        return cmp(kw["capital_relation"], len(re.findall(r"\b[A-Z]+\b", r)), kw["capital_frequency"])
    if kind == "existence":
        return all(re.search(re.escape(k), r, re.I) for k in kw["keywords"])
    if kind == "frequency":
        return cmp(kw["relation"], len(re.findall(re.escape(kw["keyword"]), r, re.I)), kw["frequency"])
    if kind == "forbidden_words":
        return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.I) for w in kw["forbidden_words"])
    if kind == "letter_frequency":
        return cmp(kw["let_relation"], r.lower().count(kw["letter"].lower()), kw["let_frequency"])
    if kind == "number_sentences":
        return cmp(kw["relation"], len(sentences(r)), kw["num_sentences"])
    if kind == "number_paragraphs":
        parts = re.split(r"\s?\*\*\*\s?", r)
        n = len(parts)
        for i, p in enumerate(parts):
            if not p.strip():
                if i in (0, len(parts) - 1):
                    n -= 1
                else:
                    return False
        return n == kw["num_paragraphs"]
    if kind == "nth_paragraph_first_word":
        paras = [p for p in re.split(r"\n\n", r) if p.strip()]
        n, k = kw["num_paragraphs"], kw["nth_paragraph"]
        if len(paras) != n or k > len(paras):
            return False
        first = re.sub(r"[^\w]", "", (paras[k - 1].strip().split() or [""])[0]).lower()
        return first == kw["first_word"].lower()
    if kind == "postscript":
        m = kw["postscript_marker"].strip()
        pat = r"\s*p\.\s?p\.\s?s.*$" if m.lower() == "p.p.s" else r"\s*p\.\s?s\..*$" if m.lower() == "p.s." else r"\s*" + re.escape(m.lower()) + r".*$"
        return bool(re.findall(pat, r.lower(), re.M))
    if kind == "json_format":
        v = r.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON").removeprefix("```")
        v = v.removesuffix("```").strip()
        try:
            json.loads(v)
            return True
        except ValueError:
            return False
    if kind == "multiple_sections":
        sp = kw["section_spliter"].strip()
        return len(re.split(r"\s?" + re.escape(sp) + r"\s?\d+\s?", r)) - 1 >= kw["num_sections"]
    if kind == "constrained_response":
        return any(o in r.strip() for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if kind == "end_checker":
        return r.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())
    if kind == "quotation":
        v = r.strip()
        return len(v) > 1 and v[0] == '"' and v[-1] == '"'
    if kind == "repeat_prompt":
        return r.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())
    if kind == "two_responses":
        parts = r.split("******")
        valid = []
        for i, p in enumerate(parts):
            if not p.strip():
                if i not in (0, len(parts) - 1):
                    return False
            else:
                valid.append(p.strip())
        return len(valid) == 2 and valid[0] != valid[1]
    if kind == "response_language":
        return lang_of(r) == kw["language"]
    raise KeyError(iid)


def variants(r):
    lines = r.split("\n")
    a = "\n".join(lines[1:]).strip()
    b = "\n".join(lines[:-1]).strip()
    c = "\n".join(lines[1:-1]).strip()
    base = [r, a, b, c]
    return base + [v.replace("*", "") for v in base]


def check(task, text):
    strict, loose = [], []
    for iid, kw in zip(task["instructions"], task["kwargs"]):
        try:
            strict.append(bool(text.strip()) and verify(iid, kw, text, task["prompt"]))
            loose.append(any(v.strip() and verify(iid, kw, v, task["prompt"]) for v in variants(text)))
        except KeyError:
            strict.append(False)
            loose.append(False)
    return strict, loose
