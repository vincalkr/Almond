"""Chat history normalization before rendering the model's prompt."""
import copy
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service


class EmptyAssistantHistory(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_openai_empty_replies_are_not_rendered(self):
        for content in (None, "", " \n\t", [], [{"type": "text", "text": ""}]):
            for preserve in (True, False):
                with self.subTest(content=content, preserve_thinking=preserve):
                    history = []
                    for _ in range(3):
                        history.extend([{"role": "user", "content": "haz un ls"},
                                        {"role": "assistant", "content": content,
                                         "reasoning_content": "Let me run ls."}])
                    history.append({"role": "user", "content": "haz un ls"})
                    messages, tools, kwargs = openai_to_messages({"messages": history})
                    kwargs["preserve_thinking"] = preserve
                    clean = [m for m in messages if m["role"] == "user"]
                    self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                                     self.service.encode_prompt(clean, tools, kwargs))

    def test_anthropic_thinking_only_replies_are_not_rendered(self):
        messages, tools, kwargs = anthropic_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me run ls."}]},
            {"role": "user", "content": "haz un ls"},
        ]})
        self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                         self.service.encode_prompt([messages[0], messages[2]], tools, kwargs))

    def test_text_and_tool_calls_are_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "text", "text": "Running ls."}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "run", "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "content": "file.txt"},
            {"role": "user", "content": "thanks"},
        ]})
        before = copy.deepcopy(messages)
        prompt = self.template.render(messages, tools=tools, **kwargs)
        self.assertIn("Running ls.", prompt)
        self.assertIn("<function=run>", prompt)
        self.assertIn("file.txt", prompt)
        self.assertEqual(messages, before)

    def test_image_only_assistant_turn_is_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "describe the image"},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "thanks"},
        ]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(messages, tools=tools, **kwargs))

    def test_final_empty_assistant_turn_is_kept(self):
        prompt = self.template.render([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": ""}], add_generation_prompt=False)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_empty_user_and_tool_turns_are_kept(self):
        prompt = self.template.render([{"role": "user", "content": ""},
                                       {"role": "tool", "content": ""},
                                       {"role": "user", "content": "hi"}])
        self.assertEqual(prompt.count("<|im_start|>user\n"), 3)
        self.assertIn("<tool_response>\n\n</tool_response>", prompt)


if __name__ == "__main__":
    unittest.main()


class EmptyTurnsSwitch(unittest.TestCase):
    def test_keep_empty_turns_restores_the_old_prompt(self):
        import os
        from unittest import mock
        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                {"role": "user", "content": "b"}]
        clean = template.render([msgs[0], msgs[2]])
        self.assertEqual(template.render(msgs), clean)
        with mock.patch.dict(os.environ, {"ALMOND_KEEP_EMPTY_TURNS": "1"}):
            self.assertNotEqual(template.render(msgs), clean)


class ForcedCallOpening(unittest.TestCase):
    def test_required_with_one_tool_names_it(self):
        from serve.frontend import forced_call
        one, two = [{"name": "a"}], [{"name": "a"}, {"name": "b"}]
        self.assertTrue(forced_call("required", one).endswith("<function=a>\n"))
        self.assertTrue(forced_call("required", two).endswith("<function="))
        self.assertIsNone(forced_call("auto", one))


class PreserveThinking(unittest.TestCase):
    """preserve_thinking (opt-in): a Qwen3.5-family template that drops earlier reasoning keeps it when asked, so the
    next prompt extends the last one; without the variable it renders as before."""
    SRC = ("{%- set ns = namespace(last_query_index=1) %}{%- for message in messages %}"
           "{%- if message.role == 'assistant' %}{%- if loop.index0 > ns.last_query_index %}"
           "<think>{{ message.reasoning_content }}</think>{%- endif %}{{ message.content }}"
           "{%- else %}{{ message.content }}{%- endif %}{%- endfor %}")

    def make(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = Path(d) / "t.jinja"
        p.write_text(self.SRC)
        return ChatTemplate(p)

    def test_default_drops_and_switch_keeps(self):
        t = self.make()
        msgs = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
                {"role": "user", "content": "Q2"}]
        self.assertNotIn("R1", t.render(msgs))
        t.defaults["preserve_thinking"] = True
        self.assertIn("<think>R1</think>A1", t.render(msgs))
        self.assertNotIn("R1", t.render(msgs, preserve_thinking=False))
