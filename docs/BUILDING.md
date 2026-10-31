# Building OxDroid Studio

This guide documents the repository's current command-line/CI build path. It does not claim that the full toolchain is already integrated into the Android app.

## 1. Prerequisites

- Bash
- JDK 17
- Android SDK Platform 36 (`platforms;android-36`)
- Android SDK Build Tools 36.0.0 (`build-tools;36.0.0`)
- Android Build Tools providing `aapt`, `d8`, `zipalign`, and `apksigner`
- `zip`
- ECJ JAR at `assets/lib/ecj.jar`

When `ANDROID_SDK_ROOT` or `ANDROID_HOME` is set, the script requires and defaults to `platforms/android-36/android.jar`. Without an SDK root, it falls back to the repository's root-level `android.jar`; for an API 36 build, set `ANDROID_JAR` explicitly to Platform 36.

## 2. Prepare the toolchain

Make sure the required executables are available on `PATH`:

```bash
java -version
aapt version
d8 --version
zipalign -h
apksigner version
zip -v
```

The `ecj` command must invoke the bundled compiler JAR. For example, create a local wrapper outside the repository:

```bash
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/ecj" <<'EOF'
#!/usr/bin/env bash
exec java -jar "/absolute/path/to/android-studio-light-for-mobile-/assets/lib/ecj.jar" "$@"
EOF
chmod +x "$HOME/.local/bin/ecj"
export PATH="$HOME/.local/bin:$PATH"
```

Replace the example path with the actual checkout path.

## 3. Configure a private signing key

The build script intentionally requires both `KEYSTORE` and `KEYSTORE_PASS`. Do not use a keystore committed to the repository.

For a local development key, create a keystore outside the repository:

```bash
keytool -genkeypair -v \
  -keystore "$HOME/.android/oxdroid-dev.jks" \
  -alias oxdroid-dev \
  -keyalg RSA -keysize 2048 -validity 3650
```

Set the path and password in your local shell environment. Avoid putting passwords in scripts, source files, command history, screenshots, or logs.

```bash
export KEYSTORE="$HOME/.android/oxdroid-dev.jks"
read -rsp "Keystore password: " KEYSTORE_PASS
echo
export KEYSTORE_PASS
```

The key created above is a local development key. It is not a shared project release key.

## 4. Safe build-output directories

`BUILD_DIR`, `GEN_DIR`, and `OUT_DIR` are canonicalized before cleanup. The build script rejects filesystem/repository roots, source/resource/Git directories, overlapping output paths, unsafe system locations, and path-like `APK_NAME` values. No cleanup command runs until these checks pass.

By default, custom output directories must be inside the repository, the user's home directory, or a temporary directory. For a deliberate external destination, set `ALLOW_EXTERNAL_BUILD_OUTPUTS=1`; the root/source/system and overlap checks still apply. New nested output paths are allowed when their nearest existing parent is available.

## 5. Run the build

From the repository root:

```bash
bash build.sh
```

The default output path is:

```text
bin/OxDroid-Mobile-IDE.apk
```

The script stops if a required tool or signing variable is missing. A successful command should produce the APK at the configured output path.

## 6. GitHub Actions build

The workflow at `.github/workflows/android-apk.yml`:

1. Checks out the selected commit.
2. Sets up JDK 17 and Android Build Tools 35.0.0.
3. Wires AAPT, D8, ZipAlign, ApkSigner, and ECJ.
4. Creates a temporary CI-only signing key.
5. Runs `build.sh`.
6. Verifies archive integrity, alignment, and signature.
7. Uploads `oxdroid-mobile-ide-debug-apk` as a workflow artifact.

Download artifacts from the successful workflow run. The artifact is debug-signed and expires after the retention period in the workflow.

## 7. What a successful CI build means

It means the current repository sources compiled and packaged in the configured GitHub-hosted environment, and the resulting APK passed the configured archive, alignment, and signature checks.

It does **not** establish that:

- The same build tools run on every Android phone.
- Arbitrary Gradle projects can be built by the app.
- Kotlin or C/C++ compilation is fully integrated.
- The APK is production/release-signed.
- The app has passed device installation and runtime tests on every supported Android version.

## 8. Troubleshooting

- **Tool not found:** install the relevant Android Build Tools package and add its directory to `PATH`.
- **Missing Android JAR:** set `ANDROID_JAR` to a valid platform JAR.
- **Missing ECJ command:** create the wrapper described above and ensure it is executable.
- **Keystore variable missing:** set `KEYSTORE` and `KEYSTORE_PASS` in the current shell.
- **Signing fails:** verify the keystore path and password. Never print the password into logs.
- **Build passes in CI but not on device:** the CI environment is Linux x86_64; native Android/ARM64 compatibility requires separate device-side validation.
