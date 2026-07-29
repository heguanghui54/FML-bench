import inspect
import unittest

from benchmark import runtime_probe


class RuntimeProbePython310Tests(unittest.TestCase):
    def test_generated_profiler_recovers_class_qualified_symbol(self):
        source = inspect.getsource(runtime_probe.prepare_runtime_probe)

        self.assertIn('frame.f_locals.get("self")', source)
        self.assertIn('type(receiver).__name__', source)
        self.assertIn('getattr(frame.f_code, "co_qualname", None)', source)
        self.assertIn("REQUIRED.issubset(SEEN)", source)
        self.assertIn("sys.setprofile(None)", source)
        self.assertIn("target_symbol.rsplit", source)
        self.assertIn("if len(matches) == 1", source)
        self.assertIn("use_deterministic_algorithms(True)", source)
        self.assertIn("runtime_reproducibility.jsonl", source)


if __name__ == "__main__":
    unittest.main()
