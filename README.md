# 📱 OxDroid Workspace (Android Studio Lite for Mobile)

![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![Android](https://img.shields.io/badge/Platform-Android-3DDC84?logo=android)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 🛡️ THE IRONCLAD GUARANTEE (हमारा पक्का वादा)
**This project belongs to the developers, the students, and the community.** 
We make a public, unbreakable promise: **OxDroid will NEVER be sold to any corporation, company, or private entity.** It will remain 100% Free and Open-Source Software (FOSS) FOREVER. No hidden paywalls, no forced ads, no premium-only build engines.

---

## ⚠️ PROJECT STATUS: DAY 1 (Early Prototype / Demo)
*Date of Inception: October 2026*

This repository is currently in its **Phase 1 (Infancy)**. We have just laid down the foundational architecture.
**It is not yet a fully functional IDE.** The current APK is a UI/editor prototype; it does not yet contain a complete ARM64 AAPT2/D8/ApkSigner toolchain. The editor now reports this limitation instead of showing a false build-success message.

### Local verification status

The repository build script is fail-fast and requires `aapt`, `ecj`, `d8`, and `apksigner` in `PATH`. Running it without those tools exits with an error and does not claim to have produced an APK. The checked-in APK is an existing prototype artifact; it is not evidence that the current source can build a new APK in this environment.

---

## 💡 The Vision & The Problem
Millions of brilliant minds worldwide want to build Android apps, but they are limited by hardware. Official Android Studio requires heavy RAM and high-end processors. 
**Our Mission:** To democratize app development. If you have a basic Android smartphone, you should be able to write, compile, build, and sign a complete `.apk` file directly from your hands.

---

## ⚙️ How We Will Build It (The Architecture)
We are not running an emulator or a cloud server. Everything will run **Natively on the Android Device (ARM64)**. Here is how the engine will work under the hood:

1. **The Code Editor:** A highly optimized, custom native text editor (using modules similar to Sora/Rose Editor) to handle large Java/Kotlin and XML files without lagging the phone.
2. **The Compiler (ECJ/Javac):** We will integrate Eclipse Compiler for Java (ECJ) to compile `.java` files into `.class` files natively on Android.
3. **Dexing (D8/R8):** Using Google's D8 dexer compiled for ARM to convert `.class` bytecode into `classes.dex` that Android OS understands.
4. **Resource Packaging (AAPT2):** A native binary of AAPT2 to compile and link `res/` folders, `AndroidManifest.xml`, and assets into a flat `.apk` package.
5. **Signing (ApkSigner):** A custom Java-based Keystore manager to align (zipalign) and sign the final APK directly in the device's storage.

---

## 🚀 The Master Roadmap (What We Will Add & When)

### 📌 Phase 1: Foundation (Current Status)
- [x] Initial repository setup and Open Source community guidelines.
- [x] Basic UI structure and File Explorer implementation.
- [x] Background Security Guard for file integrity.
- [x] Defining the architecture and roadmap.

### 📌 Phase 2: The Core Editor 
- [ ] Syntax highlighting for Java, Kotlin, XML, and JSON.
- [ ] Code Auto-completion (IntelliSense) for Android SDK classes.
- [ ] Multi-tab support for seamless file switching.
- [ ] Real-time error detection (Linting) for missing brackets and typos.

### 📌 Phase 3: The Build Engine Integration (The Hardest Part)
- [ ] Compiling AAPT2 and D8 binaries for AArch64 (Android CPU architecture).
- [ ] Setting up the Java Compiler (ECJ) inside the app.
- [ ] Creating the "Build" button that triggers the pipeline: `AAPT2 -> ECJ -> D8 -> ZIP`.

### 📌 Phase 4: Output & Testing
- [ ] Integrated ZipAlign and V2/V3 APK Signing.
- [ ] 1-Click "Install and Run" directly from the IDE.
- [ ] A built-in logcat viewer to debug crashing apps on the phone itself.

### 📌 Phase 5: Advanced Features
- [ ] Git Integration (Clone, Commit, Push directly from OxDroid).
- [ ] Gradle-like dependency management (downloading `.jar` and `.aar` libraries).
- [ ] Dark Mode, Custom Themes, and Font customizations.

---

## 🛠️ Why are all features not here yet?
Creating a mobile IDE means we have to rewrite desktop-level build tools for mobile processors. Phones have aggressive battery-saving modes and limited RAM, which can kill heavy background compiling tasks. We are taking it step-by-step to ensure the IDE is lightweight, battery-friendly, and crash-resistant.

---

## 🤝 Join the Revolution (How to Contribute)
We cannot do this alone. If you are an open-source enthusiast, an Android Engineer, a UI/UX designer, or a C++ developer who knows how to compile binaries for ARM:
1. **Fork** this repository.
2. Check the `CONTRIBUTING.md` file.
3. Pick a task from Phase 2 or Phase 3.
4. Submit a **Pull Request**.

Let's build the ultimate mobile IDE. For the developers, by the developers.
