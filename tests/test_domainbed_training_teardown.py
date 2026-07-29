import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ml_tasks" / "Generalization_domainbed" / "train_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("domainbed_train_eval_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DomainBedTrainingTeardownTests(unittest.TestCase):
    def test_completed_artifacts_allow_controlled_tail_termination(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import sys,time; "
                        "p=Path(sys.argv[1]); p.mkdir(); "
                        "[(p/n).write_text('ok') for n in "
                        "('done','results.jsonl','model.pkl')]; time.sleep(30)"
                    ),
                    str(output / "results"),
                ],
                start_new_session=True,
            )
            result_dir = output / "results"
            returncode = module.wait_for_training_process(child, str(result_dir), grace_seconds=0.1)
            record = json.loads((result_dir / "teardown_info.json").read_text())

        self.assertEqual(returncode, 0)
        self.assertEqual(record["mode"], "CONTROLLED_TERMINATION_AFTER_COMPLETE_ARTIFACTS")
        self.assertTrue(record["required_artifacts_complete"])
        self.assertIsNotNone(child.poll())


if __name__ == "__main__":
    unittest.main()
