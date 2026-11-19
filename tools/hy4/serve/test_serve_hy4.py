#!/usr/bin/env python3
"""serve_hy4.py's stream rewrite: Hy4's tags -> the server's, whole and split across pieces (no GPU or model needed).

  .venv/bin/python tools/hy4/serve/test_serve_hy4.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_hy4 as H  # noqa: E402

T = H.HYTK


def run(pieces):
    """The rewrite as the parser sees it: fed piece by piece, then finished."""
    class P:
        pass
    p = P()
    p.hy4_tag_hold = p.hy4_call_hold = ""
    out = "".join(H._rewrite(p, x, False) for x in pieces)
    return out + H._rewrite(p, "", True)


class Rewrite(unittest.TestCase):
    def test_think(self):
        self.assertEqual(run([f"<think{T}>plan</think{T}>answer"]), "<think>plan</think>answer")

    def test_call(self):
        body = (f"<tool_calls{T}><tool_call{T}>get_weather<arg_key{T}>city</arg_key{T}><arg_value{T}>Rome"
                f"</arg_value{T}></tool_call{T}></tool_calls{T}>")
        self.assertEqual(run([body]), "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
                                      "</function>\n</tool_call>")

    def test_split_pieces(self):
        body = (f"ok<tool_calls{T}><tool_call{T}>f<arg_key{T}>a</arg_key{T}><arg_value{T}>1</arg_value{T}>"
                f"<arg_key{T}>b</arg_key{T}><arg_value{T}>x y</arg_value{T}></tool_call{T}></tool_calls{T}>")
        whole = run([body])
        for size in (1, 2, 3, 7):
            self.assertEqual(run([body[i:i + size] for i in range(0, len(body), size)]), whole)
        self.assertIn("<parameter=b>\nx y\n</parameter>", whole)

    def test_plain_text_kept(self):
        self.assertEqual(run(["a < b and c > d", " <tool"]), "a < b and c > d <tool")


class Effort(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(H.effort_kwargs("none"), {"enable_thinking": False})
        self.assertEqual(H.effort_kwargs("medium"), {"reasoning_effort": "high"})
        self.assertEqual(H.effort_kwargs("low"), {"reasoning_effort": "high"})   # not in Tencent's template
        self.assertEqual(H.effort_kwargs("xhigh"), {"reasoning_effort": "max"})
        with self.assertRaises(ValueError):
            H.effort_kwargs("huge")


if __name__ == "__main__":
    unittest.main()
