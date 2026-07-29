import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ml_tasks" / "Privacy_privacymeter" / "postprocess.py"


def load_postprocess():
    spec = importlib.util.spec_from_file_location("privacy_postprocess", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class PrivacyMeterPostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_postprocess()

    def _experiment(self, *, val_tpr01=0.02, val_tpr0=0.005, accuracy=0.60393):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        report = root / "report"
        report.mkdir()
        # Seed 42 assigns target index 1 to validation for four target models.
        rows = []
        for index in range(4):
            tpr01 = val_tpr01 if index == 1 else 0.0
            tpr0 = val_tpr0 if index == 1 else 0.0
            rows.append(
                f"Target Model {index}: AUC 0.8000, "
                f"TPR@0.1%FPR of {tpr01:.4f}, TPR@0.0%FPR of {tpr0:.4f}"
            )
            rows.append(f"Test accuracy {accuracy:.5f}")
        (report / "log_time_analysis.log").write_text("\n".join(rows) + "\n")
        return temporary, root

    def test_same_boundary_values_pass_validation_and_test(self):
        for split in ("val", "test"):
            temporary, root = self._experiment()
            with temporary:
                self.module.main(str(root), split)
                payload = json.loads((root / f"{split}_info.json").read_text())
                self.assertFalse(payload["cifar10"]["means"]["constraint_violated"])
                contract = payload["constraint_contract"]
                self.assertTrue(contract["same_thresholds_for_validation_and_test"])
                self.assertAlmostEqual(contract["minimum_test_accuracy"], 0.60393)

    def test_validation_failure_preserves_raw_metrics_and_contract(self):
        temporary, root = self._experiment(val_tpr01=0.0201, accuracy=0.60392)
        with temporary:
            with self.assertRaisesRegex(ValueError, "Invalid results"):
                self.module.main(str(root), "val")
            payload = json.loads((root / "val_info.json").read_text())
            result = payload["cifar10"]
            self.assertTrue(result["means"]["constraint_violated"])
            self.assertIsNone(result["means"]["AUC_gap_mean"])
            self.assertAlmostEqual(result["original_metrics"]["AUC_gap"], 0.3)
            self.assertEqual(
                payload["constraint_contract"]["version"],
                "privacy-privacymeter-confirmatory-v2",
            )


if __name__ == "__main__":
    unittest.main()
