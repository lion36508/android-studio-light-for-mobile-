# Contributing to OxDroid Studio

Thank you for considering a contribution. OxDroid is an early-stage, phone-first Android IDE prototype. Contributions should improve real functionality, reliability, security, or documentation.

## Before you start

1. Read [README.md](README.md) for the current scope and limitations.
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the current build path.
3. Check open issues and pull requests to avoid duplicating work.
4. For large changes, open an issue first and agree on the expected behavior.

## Development workflow

1. Fork the repository.
2. Create a focused branch, for example:
   `git switch -c fix/editor-save-state`
3. Make the smallest coherent change.
4. Run the checks relevant to the files you changed.
5. Update documentation when behavior or setup changes.
6. Commit with a clear message:
   `git commit -m "Fix editor document saving"`
7. Push your branch and open a pull request against `main`.

## Build and validation

The hosted APK build is defined in `.github/workflows/android-apk.yml`. For local build prerequisites and signing configuration, follow [docs/BUILDING.md](docs/BUILDING.md).

For every feature, include evidence appropriate to the change:

- Java changes: compile through the current build pipeline.
- Build-system changes: verify the produced APK, alignment, and signature.
- UI changes: provide screenshots or a reproducible manual test path.
- Runtime changes: document Android version, ABI, permissions, and device tested.
- Security changes: explain the threat addressed and how the fix was checked.

Do not mark a feature complete based only on a visible button or placeholder screen.

## Code and product standards

- Keep status messages truthful. Never show success after a failed or skipped operation.
- Handle errors with actionable messages.
- Avoid blocking the UI thread with long-running work.
- Keep Android storage access compatible with supported Android versions.
- Avoid unnecessary permissions and network access.
- Do not add hard-coded credentials, signing keys, tokens, or personal data.
- Do not commit generated APKs, temporary build directories, logs, or local signing files unless a maintainer explicitly requests a release artifact.
- Preserve the current project constraint: AI intent routing is not active in this version.
- Keep changes scoped and explain any trade-offs.

## Pull request checklist

- [ ] The change solves the stated problem.
- [ ] The implementation is real, not a demo or placeholder presented as complete.
- [ ] Relevant build/test checks pass, or limitations are clearly stated.
- [ ] Documentation and screenshots are updated where useful.
- [ ] No credentials, private keys, or personal data are included.
- [ ] No unrelated files or generated outputs are changed.
- [ ] User-facing success/error states match the actual result.

## Reporting security issues

Do not publish secrets, private keys, or exploitable details in a public issue. Follow [SECURITY.md](SECURITY.md) for responsible reporting.
