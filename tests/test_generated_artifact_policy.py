"""Ensure generated binary build outputs are not committed to Git."""
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class GeneratedArtifactPolicyTests(unittest.TestCase):
    def test_no_apk_dex_or_signature_outputs_are_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        )
        tracked = [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
        generated = [
            path for path in tracked
            if path.lower().endswith((".apk", ".dex", ".idsig"))
        ]
        self.assertEqual([], generated, "Generated build artifacts must stay out of Git.")

    def test_bin_outputs_are_ignored(self):
        patterns = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("bin/*.apk", "bin/*.dex", "bin/*.idsig"):
            self.assertIn(pattern, patterns)


if __name__ == "__main__":
    unittest.main()
