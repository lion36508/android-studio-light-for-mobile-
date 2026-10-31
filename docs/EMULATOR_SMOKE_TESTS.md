# Android emulator smoke tests

The Android APK workflow boots an Android 16 / API 36 emulator after building and verifying the debug APK.

The smoke test:
1. Installs the generated APK.
2. Launches the dashboard and checks the app title and a starter project in the accessibility tree.
3. Opens that project and confirms the editor screen, source tab, and code input field are present.
4. Navigates back to the dashboard and relaunches it.
5. Checks the app process remains alive and its log contains no fatal exception.

This is an automated emulator smoke test, not a substitute for checking the app on physical devices, different display cutouts, gesture navigation, keyboard behavior, large font settings, or manufacturer-specific Android builds. The Android 15/16 manual device checklist remains in `docs/ANDROID_16_MIGRATION.md`.
