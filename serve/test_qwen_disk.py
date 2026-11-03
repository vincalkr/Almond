"""serve/test_qwen_disk.py - Qwen's conversations on disk (tools/qwen/serve/serve_qwen.py, "conversation_dir"):
a reply is saved, a prompt that continues it is resumed after the engine restarted, not while it holds it in RAM."""
import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_shim():
    spec = importlib.util.spec_from_file_location("serve_qwen_t", ROOT / "tools/qwen/serve/serve_qwen.py")
    m = importlib.util.module_from_spec(spec)
    from serve import server as S
    saved = (S.Service.run, S.Service.__init__, S.REASONING_LEVELS, S.REASONING_DEFAULT)
    argv, sys.argv = sys.argv, ["serve_qwen.py"]
    try:
        spec.loader.exec_module(m)
    finally:                                    # the shim patches the shared server module: the other tests need it as it was
        sys.argv = argv
        S.Service.run, S.Service.__init__, S.REASONING_LEVELS, S.REASONING_DEFAULT = saved
    return m


class FakeEngine:
    def __init__(self):
        self.proc = type("P", (), {"pid": 1})()
        self.calls = []

    def session_file(self, action, path):
        self.calls.append((action, Path(path).name))
        if action == "save":
            Path(path).write_bytes(b"x" * 100)
        return {"tokens": 5, "bytes": 100, "ms": 1.0}


class Svc:
    def __init__(self, m, folder):
        self.engine, self.fifo, self.embeddings = FakeEngine(), threading.Lock(), None
        self.qwen_disk = m.Disk(Path(folder), 10_000)

    def loaded(self):
        return True

    def ensure_loaded(self):
        pass


class QwenDisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_shim()
        cls.m._run_disk_inner = lambda self, *a, **k: iter([("event", None), ("done", {})])

    def turn(self, svc, ids):
        return list(self.m._run_disk(svc, ids, True, None, 10, {}, threading.Event()))

    def test_saved_then_resumed_after_a_restart_only(self):
        with tempfile.TemporaryDirectory() as d:
            svc = Svc(self.m, d)
            self.turn(svc, [1, 2, 3])
            self.assertEqual([c[0] for c in svc.engine.calls], ["save"])
            self.turn(svc, [1, 2, 3, 4, 5])             # the engine holds it in RAM: no restore
            self.assertNotIn("restore", [c[0] for c in svc.engine.calls])
            self.assertEqual(len(list(Path(d).glob("*.bin"))), 1)   # the earlier turn's file went
            svc.engine.calls.clear()
            svc.engine.proc.pid = 2                     # the engine started again
            self.turn(svc, [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual([c[0] for c in svc.engine.calls], ["restore", "save"])

    def test_an_unrelated_prompt_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            svc = Svc(self.m, d)
            self.turn(svc, [1, 2, 3])
            svc.engine.proc.pid, svc.engine.calls = 2, []
            self.turn(svc, [9, 9])
            self.assertEqual([c[0] for c in svc.engine.calls], ["save"])

    def test_the_budget_drops_the_oldest(self):
        with tempfile.TemporaryDirectory() as d:
            svc = Svc(self.m, d)
            svc.qwen_disk.budget = 250
            for k in range(4):
                self.turn(svc, [100 + k])
            self.assertEqual(len(list(Path(d).glob("*.bin"))), 2)


if __name__ == "__main__":
    unittest.main()
