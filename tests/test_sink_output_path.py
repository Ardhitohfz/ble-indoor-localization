import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RASPI_DIR = REPO_ROOT / "raspi4_sink"
if str(RASPI_DIR) not in sys.path:
    sys.path.insert(0, str(RASPI_DIR))

from rpi_sink_parallel import resolve_default_output_dir  # noqa: E402


class TestSinkOutputPath(unittest.TestCase):
    def test_resolve_default_output_dir_prefers_canonical_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            script_dir = root / "raspi4_sink"
            script_dir.mkdir(parents=True, exist_ok=True)
            (root / "ml_pipeline").mkdir(parents=True, exist_ok=True)
            resolved = resolve_default_output_dir(script_dir)
            self.assertEqual(resolved, root / "ml_pipeline" / "data" / "raw")

    def test_resolve_default_output_dir_uses_local_fallback_without_ml_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_dir = Path(tmp_dir) / "raspi4_sink"
            script_dir.mkdir(parents=True, exist_ok=True)
            resolved = resolve_default_output_dir(script_dir)
            self.assertEqual(resolved, script_dir / "data" / "dataset")


if __name__ == "__main__":
    unittest.main()
