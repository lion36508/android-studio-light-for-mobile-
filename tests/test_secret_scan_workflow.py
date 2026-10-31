"""Require a read-only Gitleaks scan on pull requests and main pushes."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "secret_scan.yml"


class SecretScanWorkflowTests(unittest.TestCase):
    def test_secret_scan_runs_on_pull_requests_and_main_pushes(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pull_request:", content)
        self.assertIn("push:", content)
        self.assertIn("branches: [ main ]", content)
        self.assertIn("workflow_dispatch:", content)

    def test_scan_workflow_is_read_only(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("contents: read", content)
        self.assertIn("pull-requests: read", content)
        self.assertNotRegex(content, r"(?m)^\s+(?:contents|pull-requests):\s+write\s*$")

    def test_actions_are_pinned_to_full_commit_shas(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.strip().startswith("uses:"):
                reference = line.strip().split("uses:", 1)[1].strip().split()[0]
                self.assertRegex(reference, r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")

    def test_gitleaks_is_redacting_and_does_not_post_comments_or_artifacts(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e", content)
        self.assertIn('GITLEAKS_VERSION: "8.30.1"', content)
        self.assertIn('GITLEAKS_ENABLE_COMMENTS: "false"', content)
        self.assertIn('GITLEAKS_ENABLE_UPLOAD_ARTIFACT: "false"', content)
        self.assertIn('GITLEAKS_ENABLE_SUMMARY: "true"', content)

    def test_github_token_is_supplied_only_to_the_scan_action(self):
        content = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}", content)
        self.assertEqual(content.count("GITHUB_TOKEN:"), 1)


if __name__ == "__main__":
    unittest.main()
