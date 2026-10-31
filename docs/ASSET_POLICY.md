# APK asset policy

The build script packages the repository's `assets/` directory into the APK. Keep only assets that are actually read by the application or required by the runtime.

## Platform JAR

- The canonical Android platform JAR stays at the repository root only as a local-build fallback.
- When Android SDK Platform 36 is configured, local and CI builds use `platforms/android-36/android.jar`.
- Do not copy `android.jar` into `assets/`; files there are packaged into the application and are not the compiler classpath.

## Menu artwork

- `assets/ui/blue_neon.png` is used by the top-right menu and settings dialogs.
- `assets/ui/green_neon.png` is used by the drawer menu.
- The former alternate names were byte-identical copies of the green artwork and were not referenced by the application. They have been removed; keep the two referenced files as the canonical images.

## Before adding an image

1. Confirm that source code or a documented runtime feature reads the asset.
2. Check whether the same image bytes already exist in the asset tree.
3. Reuse a canonical asset name rather than copying the same image under multiple names.
4. Add or update tests when an asset becomes part of a screen or runtime contract.
