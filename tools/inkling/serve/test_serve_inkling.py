#!/usr/bin/env python3
"""serve_inkling.py: the stream rewrite (Inkling's message tokens -> the server's tags, whole and split into pieces of
every size) and the reasoning levels (no GPU or model needed).

  .venv/bin/python tools/inkling/serve/test_serve_inkling.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_inkling as I  # noqa: E402

M, E, TH, TX, C, S = I.MSG, I.END, I.THINK, I.TEXT, I.CALL, I.STOP


def run(text, step):
    r = I.Rewriter()
    return "".join(r.feed(text[i:i + step], False) for i in range(0, len(text), step)) + r.feed("", True)


class Rewrite(unittest.TestCase):
    def check(self, text, want):
        for step in (1, 2, 3, 5, 7, 13, len(text)):
            self.assertEqual(run(text, step), want, f"pieces of {step}")

    def test_think_then_text(self):
        self.check(f"{M}{TH}plan<b>{E}{M}{TX}answer{E}{S}", "plan<b></think>\n\nanswer")

    def test_text_only(self):
        self.check(f"{M}{TX}just this{E}{S}", "</think>\n\njust this")

    def test_call(self):
        self.check(f'{M}{TX}let me look{E}{M}get_weather{C}{{"name":"get_weather","args":{{"city":"Rome","days":2}}}}{E}{S}',
                   "</think>\n\nlet me look<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
                   "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>")

    def test_unfinished_thinking_closes(self):
        self.assertEqual(run(f"{M}{TH}still thinking", 4), "still thinking</think>\n\n")


class Effort(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(I.effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(I.effort_kwargs("off"), {"reasoning_effort": "none"})
        self.assertEqual(I.effort_kwargs(False), {"reasoning_effort": "none"})
        self.assertEqual(I.effort_kwargs("0.5"), {"reasoning_effort": 0.5})
        with self.assertRaises(ValueError):
            I.effort_kwargs("loud")


if __name__ == "__main__":
    unittest.main()
