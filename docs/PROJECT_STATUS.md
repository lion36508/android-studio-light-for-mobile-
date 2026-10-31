# OxDroid Studio: Project Status

**Last reviewed:** 2026-10-30  
**Maturity:** Early prototype  
**Default branch:** `main`

This page records what is present and what still needs implementation. A feature is not complete merely because a button, class name, or bundled file exists.

## Verified in the repository

- Source editing and saving remain available; code execution is intentionally disabled until an OS-enforced sandbox is implemented and verified. See [Code Execution Security](EXECUTION_SECURITY.md).

- Java Android application entry point and editor activity exist.
- Project scaffolding code exists.
- Android manifest declares package `com.bikash.mobileide`, minimum SDK 24, and target SDK 30.
- A GitHub Actions workflow builds the current source with JDK 17 and Android Build Tools 35.0.0.
- The workflow successfully compiled, packaged, aligned, and signed a debug APK in the hosted CI environment.
- The workflow uploaded a temporary artifact named `oxdroid-mobile-ide-debug-apk`.

## Current feature assessment

| Feature | Assessment | Notes |
|---|---|---|
| App shell and dashboard | Present | Needs continued UX and device testing |
| Editor screen | Foundation present | Full IDE editor capabilities remain incomplete |
| File/project scaffolding | Present | Validate generated project outputs with real builds |
| Video-app starter | Starter only | Not a complete production video application |
| CI APK build | Verified | Hosted CI build, debug signing |
| On-device APK build | Not verified as complete | Needs integrated Android-compatible toolchain |
| Kotlin compilation | Not complete | Do not advertise as ready |
| Native C/C++ compilation | Not complete | Requires real ARM64 compiler/linker integration |
| Gradle and dependency resolution | Not complete | Needs real project-level integration |
| Debugger / Logcat | Not complete | Planned |
| Git clone / commit / push | Not complete | Requires a real Git client and secure authentication |

## Priority order

### P0: Security and repository reliability
- Remove signing keys and secrets from tracked files.
- Keep generated artifacts and temporary logs out of source control where possible.
- Keep CI permissions minimal.
- Add checks that prevent accidental secret commits.

### P1: Stable app foundation
- Verify app installation and startup across selected Android versions.
- Test file access and project persistence.
- Make all dashboard actions report real outcomes.
- Add repeatable tests for project scaffolding.

### P2: Editor
- Multi-file tabs and robust save/reopen behavior.
- Syntax highlighting and bracket matching.
- Search and replace.
- Clear diagnostics and editor state recovery.

### P3: Build engine
- Define build-stage interfaces and process lifecycle.
- Integrate and validate Android-compatible resource, Java/Kotlin, DEX, packaging, alignment, and signing stages.
- Test on ARM64 Android devices, not only Linux CI.
- Add cancellation, logs, progress, and error reporting.

### P4: Developer workflow
- Install/run generated APKs.
- Debugging and Logcat.
- Dependency management.
- Secure Git operations.

## Release policy

- CI artifacts are for development/testing and are not production releases.
- A release must have a documented version, source commit, build result, and signing provenance.
- Never publish private keystores or signing passwords.
- Mark a feature as complete only after its behavior is implemented and verified.
