"""Keep every lesson runnable and its stated teaching contract visible."""

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).parents[1]
EXAMPLES = tuple(sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")))


class ExampleTests(unittest.TestCase):
    def test_every_numbered_example_runs_without_warnings(self) -> None:
        self.assertEqual(len(EXAMPLES), 11)
        for path in EXAMPLES:
            with self.subTest(example=path.name):
                completed = subprocess.run(
                    [sys.executable, "-W", "error", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Takeaway:", completed.stdout)
                self.assertEqual(completed.stderr, "")

    def test_every_example_declares_a_prediction(self) -> None:
        for path in EXAMPLES:
            with self.subTest(example=path.name):
                self.assertIn("Prediction:", path.read_text())

    def test_optimizer_example_uses_the_observed_terminal_state(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "examples/05_gradient_descent.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("large rate: status=max_steps", completed.stdout)
        self.assertNotIn("status=diverged", completed.stdout)


if __name__ == "__main__":
    unittest.main()
