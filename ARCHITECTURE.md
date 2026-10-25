# 🏗️ OxDroid Architecture Guide

Welcome to the OxDroid engine room! If you want to contribute, here is how the app is currently structured.

## 1. Frontend (Java UI)
- `MainActivity.java`: Handles the premium dashboard, animations, and intent routing.
- `EditorActivity.java`: The core workspace containing the sidebar file explorer and the coding text area.

## 2. Backend (Shell/Termux Engine)
- `build.sh`: The heart of the compilation pipeline. It uses native Linux/Android binaries to compile apps on-device.
  - **AAPT2**: Compiles and links XML resources.
  - **ECJ (Eclipse Compiler for Java)**: Converts `.java` files into `.class` bytecode.
  - **D8**: Converts `.class` files into Android's native `classes.dex`.
  - **ApkSigner**: Signs the final APK for installation.

## 🚧 The Missing Bridge (Our Main Goal)
Currently, the UI and the Backend are separate. **The ultimate goal of this project is to bridge them.** 
We need to use `java.lang.ProcessBuilder` inside `EditorActivity.java` to trigger `build.sh` dynamically when the user presses the 'RUN' button, capturing the terminal output and displaying it in the Live Logcat UI.
