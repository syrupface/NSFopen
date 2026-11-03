"""Every example script runs from its own folder, as a user would run it."""

import os
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "NSFopen" / "example"


class TestExamples(unittest.TestCase):
    def test_scripts_run(self):
        scripts = [p for p in sorted(EXAMPLES.glob("*/*.py")) if p.name != "sader.py"]
        self.assertGreaterEqual(len(scripts), 8)
        environment = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
        for script in scripts:
            with self.subTest(script.name):
                done = subprocess.run([sys.executable, script.name], cwd=script.parent, env=environment,
                                      capture_output=True, text=True, timeout=300)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
