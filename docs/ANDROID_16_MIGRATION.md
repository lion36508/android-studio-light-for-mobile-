# Android 16 (API 36) migration

## Current configuration

- Minimum supported Android version remains API 24.
- Target SDK is API 36 (Android 16).
- CI compiles against Android SDK Platform 36 and Android Build Tools 36.0.0.
- The manifest continues to deny cleartext traffic globally; the separate Network Security Configuration permits only the local development-daemon host.

## Edge-to-edge layout

Android 15+ enforces edge-to-edge for apps targeting API 35+. OxDroid's dashboard and editor use programmatically built root layouts, so both now apply system-bar insets through `SystemBarInsets`. The helper adds top, bottom, and side insets to the existing root padding; it does not change the child layout structure.

## Required device verification

The CI pipeline proves that the project compiles against Platform 36, produces an APK, and passes archive/alignment/signature checks. It does **not** prove runtime behavior on a physical device or emulator.

Before declaring this migration fully verified, run the following checks on Android 16 (API 36) and Android 15 (API 35):

1. Open the dashboard in portrait and landscape; confirm the app title and menu do not overlap the status bar or display cutout.
2. Confirm the bottom navigation remains above the gesture area and above three-button navigation.
3. Open the editor; confirm the top toolbar, code area, output panel, and symbol row remain reachable.
4. Open the keyboard, type, scroll, and close the keyboard; confirm editor content and controls are not obscured.
5. Rotate the device and reopen the app; confirm the insets are not applied twice.
6. Repeat with large display/font settings and split-screen where available.

Keep the issue open until these device-level checks are recorded. No runtime/device test is claimed by the hosted CI build alone.
