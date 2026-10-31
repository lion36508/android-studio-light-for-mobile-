# Security Policy

## Supported versions

Security fixes are currently applied to the active development branch, `main`. This is an early-stage project and does not yet maintain a formal long-term support matrix.

## Reporting a vulnerability

Please do not publish passwords, signing keys, access tokens, private data, or a working exploit in a public issue.

Use GitHub's private vulnerability reporting for this repository if it is enabled. Otherwise, contact the repository maintainer through the GitHub profile and request a private channel before sharing sensitive details.

Include:
- A concise description of the affected component.
- The impact and Android/API levels affected, if known.
- Minimal reproduction steps that do not expose real user data.
- A suggested mitigation, if available.

## Secret and signing-key rules

- Never commit `.jks`, `.keystore`, signing passwords, API tokens, or credentials.
- Keep development and release signing keys outside the repository.
- CI should generate temporary keys for test builds.
- If a key or token is committed, treat it as exposed: rotate or replace it, remove it from the current tree, and assess whether repository history must be cleaned. Deleting the latest copy alone does not remove historical copies.
- Do not paste secrets into issues, pull requests, screenshots, or build logs.

## Historical signing-key exposure

A development signing keystore was previously committed and later removed from the current `main` tree. Treat that historical signing identity as compromised permanently; never reuse it for development or release builds.

Current CI workflows create a fresh temporary signing key under the runner's temporary directory for each build. Those APKs are development/testing artifacts, not a stable production update identity. A production release-signing identity has not been established by this workflow.

Removing a keystore from the current tree does not remove it from earlier Git commits, old branch references, pull-request refs, or cached commit views. The historical cleanup remains outstanding until the repository owner has coordinated a complete history/ref cleanup and verified that no public reference still exposes the old object. Do not claim this cleanup is complete based only on the current default-branch file list.

## Security-minded engineering

- Request only the Android permissions required by a feature.
- Validate file paths and external input.
- Avoid executing untrusted project scripts without explicit user awareness.
- Keep network traffic and credential handling visible and auditable.
- Do not report an operation as successful if it was skipped or failed.
