# Automated secret scanning

OxDroid runs Gitleaks on every pull request targeting `main`, every push to `main`, and on manual workflow dispatch.

## What is scanned

- Pull-request events: commits introduced by that pull request.
- Push events: commits included in that push.
- Manual dispatch: the configured Gitleaks action's default scan behavior for the selected checkout.

The workflow checks out full Git history so the scanner can calculate commit ranges. It uses Gitleaks 8.30.1 and redacts detected secret values in its report. The workflow does not post review comments or upload a separate SARIF artifact. A detected secret or scanner error fails the check.

## Permissions

The workflow uses read-only repository and pull-request permissions. The standard GitHub Actions token is passed only to Gitleaks so it can read pull-request commit metadata. The workflow does not need write access to repository contents, pull requests, issues, or releases.

## What this does not solve

The historical development signing key exposure tracked in issue #14 is a separate cleanup task. A PR/push scanner checks new commits in the event range; it does not rewrite Git history, remove old branch references, or purge cached GitHub views. Treat the old key as compromised and never reuse it. Keep issue #14 open until a coordinated history/ref cleanup is completed and independently verified.

Do not commit secrets to make the scanner test pass. If a real secret is detected, stop the merge, rotate/revoke the credential with its provider, remove it from the current code, and review exposure in Git history. Removing a value from the latest file alone does not invalidate the credential.
