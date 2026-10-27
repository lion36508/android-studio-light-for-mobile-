#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ANDROID_JAR="${ANDROID_JAR:-$ROOT_DIR/android.jar}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/obj}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/bin}"
GEN_DIR="${GEN_DIR:-$ROOT_DIR/gen}"
APK_NAME="${APK_NAME:-OxDroid-Mobile-IDE.apk}"

require_tool() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: required build tool '$1' was not found in PATH." >&2
        echo "Install the Android build toolchain or set PATH before running this script." >&2
        exit 127
    }
}

[[ -f "$ANDROID_JAR" ]] || { echo "ERROR: android.jar not found: $ANDROID_JAR" >&2; exit 2; }
for tool in aapt ecj d8 apksigner zip; do require_tool "$tool"; done

rm -rf "$BUILD_DIR" "$GEN_DIR" "$OUT_DIR/dex" "$OUT_DIR/app-unsigned.apk" "$OUT_DIR/app-aligned.apk" "$OUT_DIR/$APK_NAME"
mkdir -p "$BUILD_DIR" "$GEN_DIR" "$OUT_DIR"

echo "[*] Compiling Android resources..."
aapt package -f -m -J "$GEN_DIR" -S res -M AndroidManifest.xml -A assets -I "$ANDROID_JAR"

echo "[*] Compiling Java sources..."
mapfile -t JAVA_SOURCES < <(
    find src/main/java -name '*.java' -type f ! -name 'R.java'
    find "$GEN_DIR" -name '*.java' -type f
)
[[ "${#JAVA_SOURCES[@]}" -gt 0 ]] || { echo "ERROR: no Java sources found." >&2; exit 2; }
ecj -source 8 -target 8 -d "$BUILD_DIR" -classpath "$ANDROID_JAR" "${JAVA_SOURCES[@]}"

echo "[*] Converting bytecode to DEX..."
rm -rf "$OUT_DIR/dex"
mkdir -p "$OUT_DIR/dex"
d8 --release --lib "$ANDROID_JAR" --output "$OUT_DIR/dex" $(find "$BUILD_DIR" -name '*.class' -type f | sort)

echo "[*] Packaging unsigned APK..."
aapt package -f -M AndroidManifest.xml -S res -A assets -I "$ANDROID_JAR" -F "$OUT_DIR/app-unsigned.apk"
(cd "$OUT_DIR/dex" && zip -q -u ../app-unsigned.apk classes.dex)

echo "[*] Signing APK..."
KEYSTORE="${KEYSTORE:-$ROOT_DIR/ide_keystore.jks}"
[[ -f "$KEYSTORE" ]] || { echo "ERROR: signing keystore not found: $KEYSTORE" >&2; exit 2; }
zipalign -f -p 4 "$OUT_DIR/app-unsigned.apk" "$OUT_DIR/app-aligned.apk"
apksigner sign --ks "$KEYSTORE" --ks-pass pass:android --out "$OUT_DIR/$APK_NAME" "$OUT_DIR/app-aligned.apk"

echo "BUILD SUCCESSFUL: $OUT_DIR/$APK_NAME"
