# Q-Mini AI engine inside OxDroid

## Migration source and snapshot

- Source repository: https://github.com/lion36508/Q-mini-light-0.7M
- Source main commit: 4633fd6bc0be25dae13b4bee0fe4b32aa32dbb37
- Destination repository: https://github.com/lion36508/android-studio-light-for-mobile-
- Destination path: engine/qmini-light/
- Tracked source files copied: 138 text files.
- The migration is a source snapshot; it does not copy Git history, private Drive checkpoints, private datasets or local untracked files.

## What is included

The complete tracked Q-Mini Light project snapshot is preserved under engine/qmini-light/: legacy qmini/ implementation, newer qmini_addsub/ model and OxCore sources, configuration, scripts, tests, docs, Colab notebooks, evaluation fixtures and packaging metadata. The old Q2-v2 and new Add/Sub model tracks remain separate.

## What is deliberately not replaced or deleted

OxDroid Studio remains the root Android IDE project. Its application shell, editor/workspace, Java EngineCore/OxLinuxEngine, manifest/resources, Android platform JAR, ECJ compiler, BusyBox/Linux rootfs assets, security restrictions, build.sh and existing APK workflows remain in their original paths. The Q-Mini subproject is not copied over these files. This migration branch contains no bulk deletion of OxDroid assets.

## Runtime integration status

This is a source/project migration, not a claim of a finished Android AI runtime.

OxCore currently provides a standalone C++20 C ABI and portable native primitives, including Mirror-63 dot/linear operations, causal depthwise convolution, a Q15 diagonal SSM recurrence/cache operator, activation LUT lookup, deterministic Top-1 selection, integer FWHT, Sinkhorn and L2 normalization. The Add/Sub training/reference graph still runs through its existing Python/PyTorch path; the native operators are not yet wired into a complete integer-only model forward and generation path.

The OxDroid APK build remains Java/ECJ/D8 based. This migration does not add JNI integration or claim that the APK loads Q-Mini. Android NDK/clang integration, JNI lifecycle, tokenizer/output decoding, a standalone complete model archive, full integer operator parity, device-level tests and measured performance remain follow-on requirements.

## CI boundary

The separate `qmini-engine-native.yml` workflow:
1. Builds the portable host C++20 OxCore library.
2. Runs the OxCore AddressSanitizer/UndefinedBehaviorSanitizer test binary.
3. Runs OxCore's Python ABI and arithmetic tests.
4. Cross-compiles the C++ library for AArch64.

The existing Android APK CI remains independent. Cross-compilation is not a substitute for running on a real Android device.

## Preservation contract

Do not delete the original Q-Mini repository or its checkpoints as part of this migration. Do not delete OxDroid Studio's existing application code or assets. After human review and successful destination CI, future cleanup should remove only verified duplicates or explicitly rejected assets, using a separate reviewed change.
