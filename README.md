# OxDroid Studio
### A lightweight, phone-first Android IDE prototype

[![Android APK CI](https://github.com/lion36508/android-studio-light-for-mobile-/actions/workflows/android-apk.yml/badge.svg)](https://github.com/lion36508/android-studio-light-for-mobile-/actions/workflows/android-apk.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Android](https://img.shields.io/badge/Platform-Android-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)

**OxDroid Studio** is an early-stage, open-source Android development environment designed around one idea: make app-development workflows more accessible on a smartphone, without pretending a phone is a desktop workstation.

The project currently provides a Java-based Android application shell, an editor/workspace foundation, project scaffolding, and a repeatable CI pipeline that builds and verifies a debug-signed APK. The longer-term goal is to bring more of the Android development toolchain directly onto Android devices.

> **Project maturity:** early prototype. OxDroid is not yet a full Android Studio replacement. Features marked as planned or in progress below are not represented as finished functionality.

---

## Project at a glance

| Item | Current information |
|---|---|
| Project | OxDroid Studio |
| Repository | `lion36508/android-studio-light-for-mobile-` |
| Application ID | `com.bikash.mobileide` |
| Primary implementation | Java |
| Platform | Android |
| Manifest minimum SDK | API 24 |
| Manifest target SDK | API 30 |
| License | MIT |
| Default branch | `main` |
| Build automation | GitHub Actions |
| Current build output | CI-generated, debug-signed APK |

## What is in the repository today

- **Application shell:** Java-based Android entry point and editor activity.
- **Workspace foundation:** project and file-workspace scaffolding.
- **Project generator:** starter Android project structure generation.
- **Engine modules:** runtime and shell-related helper code that is still being integrated and validated.
- **Video-app starter:** an example starter workflow, not a complete production video platform.
- **Build pipeline:** a CI workflow using JDK 17 and Android Build Tools 35.0.0.
- **Build verification:** APK archive integrity, 4-byte alignment, and signature verification in CI.

### Source layout

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── android-apk.yml
│       └── code_quality_guard.yml
├── assets/
│   ├── bin/
│   ├── lib/
│   ├── linux_env/
│   └── ui/
├── docs/
├── res/
├── src/main/java/com/bikash/mobileide/
│   ├── MainActivity.java
│   ├── EditorActivity.java
│   ├── EngineCore.java
│   ├── OxLinuxEngine.java
│   ├── ProjectScaffold.java
│   └── R.java
├── AndroidManifest.xml
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── build.sh
```

Some binary files under `assets/` are bundled resources or toolchain inputs. Their presence does not by itself mean that the corresponding compiler/runtime is fully integrated into the application.

---

## Build the current source

The repository includes a Bash build script and a GitHub Actions workflow.

### Automated build (recommended)

1. Open the [Android APK workflow](https://github.com/lion36508/android-studio-light-for-mobile-/actions/workflows/android-apk.yml).
2. Select a successful run on the branch or commit you want to test.
3. Download the `oxdroid-mobile-ide-debug-apk` artifact from that run.

The workflow currently uses JDK 17, Android Build Tools 35.0.0, the repository's ECJ JAR, and a temporary CI-only signing key. The resulting APK is for development/testing. It is **not** a production release signed with a permanent release key. Workflow artifacts expire according to the retention period configured in the workflow.

### Local build prerequisites

- Bash
- JDK 17
- Android SDK platform JAR (the script defaults to the repository-root `android.jar`)
- `aapt`, `d8`, `zipalign`, and `apksigner` from Android Build Tools
- `zip`
- The ECJ JAR at `assets/lib/ecj.jar`

See [docs/BUILDING.md](docs/BUILDING.md) for setup and signing instructions.

Never put a personal signing keystore, keystore password, access token, or other secret in this repository.

---

## Current capabilities and limitations

| Area | Status |
|---|---|
| Android application shell | Present |
| Java-based editor/workspace foundation | Present; continued UX and editor work needed |
| Android project scaffolding | Present |
| Video-app starter screen | Present as a starter, not a complete video product |
| CI APK build | Verified on GitHub Actions |
| APK alignment and signature checks | Verified in CI |
| Full on-device Android SDK build engine | Not yet complete |
| Full Kotlin compiler integration | Not yet complete |
| Full native C/C++ toolchain | Not yet complete |
| Production-grade Gradle/dependency integration | Not yet complete |
| Integrated debugger and Logcat workflow | Not yet complete |
| Full Git clone/commit/push client | Not yet complete |

A successful CI build proves that the current source can produce the CI APK in the configured hosted environment. It does not prove that every toolchain component runs locally on an Android phone.

---

## Roadmap

The roadmap is intentionally incremental. Each feature should be implemented, tested, and documented before it is marked complete.

1. **Stabilize the foundation**
   - Keep build status truthful.
   - Add automated source and manifest checks.
   - Remove secrets and generated build outputs from source control where appropriate.
   - Document the actual runtime boundaries.

2. **Editor experience**
   - Reliable multi-file workspace and tabs.
   - Syntax highlighting for Java, XML, JSON, and Kotlin.
   - Search, replace, bracket matching, and editor state recovery.
   - Useful diagnostics without claiming compiler-backed checks that do not exist.

3. **Build system**
   - Integrate a real Android build pipeline.
   - Validate resource compilation, Java compilation, DEX generation, alignment, and signing as separate stages.
   - Add reproducible build reports and actionable errors.
   - Investigate on-device ARM64 compatibility rather than assuming desktop binaries will run on Android.

4. **Developer tools**
   - Build output panel and structured logs.
   - Device installation workflow.
   - Debugging and Logcat integration.
   - Dependency and SDK management.

5. **Git and collaboration**
   - Real Git clone, status, diff, commit, and push support.
   - Secure credential handling.
   - Clear conflict and network error reporting.

## Engineering principles

- **Truthful status:** never display a successful build, clone, compiler, or cleanup result unless the operation actually succeeded.
- **Security first:** secrets and signing keys stay outside version control.
- **Small, verifiable changes:** each feature should have a test or a clearly documented manual verification procedure.
- **Mobile-aware design:** memory, storage, battery, thermal limits, and Android background execution must be considered.
- **No fake feature claims:** a placeholder or starter screen must be labelled as such.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), review the [architecture guide](ARCHITECTURE.md), and choose a scoped task from the [project status and roadmap](docs/PROJECT_STATUS.md).

For bugs, use the [bug report template](https://github.com/lion36508/android-studio-light-for-mobile-/issues/new?template=bug_report.md). For feature proposals, use the [feature request template](https://github.com/lion36508/android-studio-light-for-mobile-/issues/new?template=feature_request.md).

## License

OxDroid Studio is distributed under the [MIT License](LICENSE).

---

**OxDroid Studio is a work in progress.** The goal is ambitious; the implementation will earn each capability through working code, repeatable builds, and honest verification.
