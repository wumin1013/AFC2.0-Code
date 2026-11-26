"""A transient Windows reader lock must not destroy a recoverable job state."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction import common


class AtomicJsonTests(unittest.TestCase):
    def test_transient_sharing_violation_retries_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            common.save_json(path, {"status": "running"})
            replace = common.os.replace
            attempts = []
            def locked_then_free(src, dest):
                attempts.append(1)
                if len(attempts) < 3:
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "running")
                    error = PermissionError("reader holds destination")
                    error.winerror = 32
                    raise error
                replace(src, dest)
            with patch.object(common.os, "replace", side_effect=locked_then_free), patch.object(common.time, "sleep"):
                common.save_json(path, {"status": "cancelled"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "cancelled")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_persistent_failure_preserves_old_state_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            common.save_json(path, {"status": "running"})
            with patch.object(common.os, "replace", side_effect=PermissionError("permanent")) as replace:
                with self.assertRaises(PermissionError):
                    common.save_json(path, {"status": "cancelled"})
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "running")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
