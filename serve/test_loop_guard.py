"""CPU-only: the loop guard in Almond's Service.run (python -m unittest serve.test_loop_guard)."""
import threading
import unittest
from serve.server import Service, HIGH_EFFORT
from serve.test_reasoning_loop_recovery import Tokenizer, RecordingEngine


class LoopGuardInRun(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>system\n' + HIGH_EFFORT +
            '<|im_end|>\n<|im_start|>user\nTask<|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)
        self.loop = 'Recall the verse once more. ' * 40

    def run_with(self, scripts, on=True, max_new=40000):
        engine = RecordingEngine(self.tok, scripts)
        service = Service(engine, self.tok, None)
        service.loop_guard = on
        events = list(service.run(self.ids, True, [], max_new, {'temperature': 1.0}, threading.Event()))
        text = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        return engine, service, events, text

    def test_off_is_one_pass(self):
        engine, service, events, _ = self.run_with([self.loop], on=False, max_new=2000)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(service.loop_guard_counts['detected'], 0)

    def test_penalty_twice_then_the_thinking_closes_and_it_answers(self):
        engine, service, events, text = self.run_with([self.loop] * 5 + ['The answer.'])
        c = service.loop_guard_counts
        self.assertEqual((c['penalised'], c['thinking_closed'], c['stopped']), (2, 1, 0))
        self.assertEqual(engine.sampling_calls[1]['repetition_penalty'], 1.15)   # the short penalty call
        self.assertEqual(engine.sampling_calls[0], {'temperature': 1.0})          # untouched until a loop
        self.assertIn('</think>', self.tok.decode(engine.calls[-1][0]))
        self.assertEqual(text, 'The answer.')
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_no_loop_changes_nothing(self):
        plain = ' '.join(f'step {i} checks a new case.' for i in range(300)) + '\n</think>\nDone.'
        engine, service, events, text = self.run_with([plain])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(service.loop_guard_counts['detected'], 0)
        self.assertEqual(text, 'Done.')


if __name__ == '__main__':
    unittest.main()
