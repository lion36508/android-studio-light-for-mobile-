# OxDroid Studio: Architecture

This document describes the repository as it exists today and separates implemented structure from future design goals.

## 1. Application entry and UI

Package: `com.bikash.mobileide`

| File | Responsibility |
|---|---|
| `MainActivity.java` | Main application entry point, dashboard, navigation, and project entry actions |
| `EditorActivity.java` | Editor/workspace screen and file-oriented UI |
| `ProjectScaffold.java` | Generates the current Android starter-project structure |
| `EngineCore.java` | Shared engine helpers and runtime/build-related utility logic |
| `OxLinuxEngine.java` | Linux/shell integration foundation; capabilities depend on the Android environment and available binaries |
| `R.java` | Resource identifiers generated/maintained for the current lightweight build setup |

The current app is a Java Android application built from a manifest and resource tree. It is not currently a conventional Gradle Android Studio project.

## 2. Build pipeline currently used by CI

The root `build.sh` script and `.github/workflows/android-apk.yml` define the current hosted build path:

1. **Resource processing:** Android `aapt` processes the manifest, resources, assets, and generates Java resource references.
2. **Java compilation:** ECJ compiles the Java source against the configured Android platform JAR.
3. **DEX conversion:** D8 converts compiled class files into DEX bytecode.
4. **APK packaging:** AAPT packages the manifest, resources, and assets; the DEX file is added to the APK.
5. **Alignment:** `zipalign` aligns the APK.
6. **Signing:** `apksigner` signs the APK using a temporary CI key or a locally supplied private key.
7. **Verification:** CI checks ZIP integrity, alignment, APK signature, and package metadata.

The current pipeline uses `aapt`, not a fully integrated AAPT2-based Android Gradle Plugin pipeline. Do not describe it as a full Gradle build.

## 3. CI environment

The Android APK workflow currently provisions:

- JDK 17
- Android Build Tools 35.0.0
- ECJ from `assets/lib/ecj.jar`
- A temporary CI-only signing keystore

The CI key is ephemeral and must never be reused as a production signing identity. The resulting artifact is a debug/testing build.

## 4. Current boundaries

The following are not yet established as complete, end-to-end application features:

- Building arbitrary Android projects entirely on-device.
- A full Kotlin compiler/runtime pipeline.
- A complete ARM64 C/C++ compiler and linker toolchain.
- Gradle wrapper execution and dependency resolution integrated into the app.
- A production debugger and Logcat experience.
- A real Git client with secure authentication and clone/commit/push operations.

Bundled files and UI labels are not proof that a corresponding toolchain is functional. Each capability must be tested at the point where a user invokes it.

## 5. Target architecture

The intended long-term flow is:

```text
Project Workspace
      |
      v
Editor + Diagnostics
      |
      v
Build Orchestrator
      |
      +--> Manifest / Resource Processing
      |
      +--> Java / Kotlin Compilation
      |
      +--> DEX Generation
      |
      +--> APK Packaging
      |
      +--> Alignment + Signing
      |
      v
Artifact Verification
      |
      v
Install / Run / Logs
```

The orchestrator should expose explicit stages, cancellation, progress, exit codes, and captured diagnostics. A stage must not report success unless its output exists and passes the relevant validation.

## 6. Mobile constraints

Any on-device implementation must account for:

- ARM64 ABI compatibility and Android executable restrictions.
- Scoped storage and Android file access permissions.
- Process lifecycle and background execution limits.
- Memory pressure, thermal throttling, and battery use.
- Toolchain storage footprint and safe cleanup.
- Signing-key isolation and secure credential entry.

## 7. Design rules

1. Keep the UI independent from compiler implementation details.
2. Use explicit interfaces for build stages and runtime providers.
3. Keep AI intent routing disabled in the current version; preserve extension points only if needed later.
4. Do not silently substitute a fake evaluator for a compiler.
5. Store generated outputs separately from source files.
6. Keep signing credentials out of Git and CI logs.
7. Document tested Android versions and ABIs for every native component.
