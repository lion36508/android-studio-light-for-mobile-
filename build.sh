#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT_DIR"

SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
DEFAULT_ANDROID_JAR="$ROOT_DIR/android.jar"
if [[ -n "$SDK_ROOT" ]]; then
    DEFAULT_ANDROID_JAR="$SDK_ROOT/platforms/android-36/android.jar"
fi
ANDROID_JAR="${ANDROID_JAR:-$DEFAULT_ANDROID_JAR}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/obj}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/bin}"
GEN_DIR="${GEN_DIR:-$ROOT_DIR/gen}"
APK_NAME="${APK_NAME:-OxDroid-Mobile-IDE.apk}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

if [[ -n "$SDK_ROOT" ]]; then
    [[ -f "$SDK_ROOT/platforms/android-36/android.jar" ]] ||
        fail "Android SDK Platform 36 is required when ANDROID_SDK_ROOT/ANDROID_HOME is set."
fi

require_tool() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: required build tool '$1' was not found in PATH." >&2
        echo "Install the Android build toolchain or set PATH before running this script." >&2
        exit 127
    }
}

normalize_absolute_path() {
    local input="$1"
    local part last_index
    local -a raw_parts=()
    local -a parts=()
    local result="/"

    IFS='/' read -r -a raw_parts <<< "$input"
    for part in "${raw_parts[@]}"; do
        case "$part" in
            ""|".") ;;
            "..")
                if (( ${#parts[@]} > 0 )); then
                    last_index=$((${#parts[@]} - 1))
                    unset "parts[$last_index]"
                fi
                ;;
            *) parts+=("$part") ;;
        esac
    done

    for part in "${parts[@]}"; do
        if [[ "$result" == "/" ]]; then
            result+="$part"
        else
            result+="/$part"
        fi
    done
    printf '%s' "$result"
}

canonicalize_output_dir() {
    local variable_name="$1"
    local requested="$2"
    local absolute parent base suffix="" parent_real resolved

    [[ -n "$requested" ]] || fail "$variable_name must not be empty."
    if [[ "$requested" == /* ]]; then
        absolute="$requested"
    else
        absolute="$ROOT_DIR/$requested"
    fi
    absolute="$(normalize_absolute_path "$absolute")"

    if [[ -L "$absolute" && ! -e "$absolute" ]]; then
        fail "$variable_name points to a dangling symbolic link: $absolute"
    fi

    if [[ -e "$absolute" ]]; then
        [[ -d "$absolute" ]] || fail "$variable_name must refer to a directory: $absolute"
        resolved="$(cd -P -- "$absolute" && pwd -P)" ||
            fail "Could not resolve $variable_name: $absolute"
    else
        parent="$absolute"
        while [[ ! -e "$parent" ]]; do
            [[ ! -L "$parent" ]] ||
                fail "$variable_name contains a dangling symbolic link: $parent"
            [[ "$parent" != "/" ]] || break
            base="$(basename -- "$parent")"
            suffix="/$base$suffix"
            parent="$(dirname -- "$parent")"
        done

        [[ -d "$parent" ]] ||
            fail "The nearest existing parent of $variable_name is not a directory: $parent"
        parent_real="$(cd -P -- "$parent" && pwd -P)" ||
            fail "Could not resolve the parent of $variable_name: $parent"
        if [[ "$parent_real" == "/" ]]; then
            resolved="$suffix"
        else
            resolved="$parent_real$suffix"
        fi
    fi

    printf -v "$variable_name" '%s' "$resolved"
}

is_within_path() {
    local candidate="$1"
    local root="$2"
    [[ "$candidate" == "$root" || "$candidate" == "$root/"* ]]
}

paths_overlap() {
    local first="$1"
    local second="$2"
    [[ "$first" == "$second" || "$first" == "$second/"* || "$second" == "$first/"* ]]
}

validate_output_dir() {
    local label="$1"
    local candidate="$2"
    local protected

    [[ "$candidate" != "/" ]] || fail "$label cannot be the filesystem root."
    [[ "$candidate" != "/tmp" ]] || fail "$label cannot be the temporary-directory root."
    if [[ -n "${TMPDIR:-}" ]]; then
        [[ "$candidate" != "$TMPDIR" ]] || fail "$label cannot be the TMPDIR root."
    fi
    [[ "$candidate" != "$ROOT_DIR" ]] || fail "$label cannot be the repository root."
    [[ "$ROOT_DIR" != "$candidate/"* ]] ||
        fail "$label is an ancestor of the repository and is unsafe: $candidate"
    if [[ -n "${HOME:-}" ]]; then
        [[ "$candidate" != "$HOME" ]] || fail "$label cannot be the home directory."
        [[ "$HOME" != "$candidate/"* ]] ||
            fail "$label is an ancestor of the home directory and is unsafe: $candidate"
    fi

    for protected in \
        "$ROOT_DIR/src" "$ROOT_DIR/res" "$ROOT_DIR/assets" \
        "$ROOT_DIR/.git" "$ROOT_DIR/.github" "$ROOT_DIR/docs" \
        "$ROOT_DIR/issues" "$ROOT_DIR/LaunchKit"
    do
        if paths_overlap "$candidate" "$protected"; then
            fail "$label overlaps a protected source/project directory: $candidate"
        fi
    done

    for protected in /usr /etc /var /opt /system /proc /sys /dev /bin /sbin /lib /lib64; do
        if is_within_path "$candidate" "$protected"; then
            fail "$label points into a protected system directory: $candidate"
        fi
    done

    if ! is_within_path "$candidate" "$ROOT_DIR" &&
       { [[ -z "${HOME:-}" ]] || ! is_within_path "$candidate" "$HOME"; } &&
       ! is_within_path "$candidate" "/tmp" &&
       { [[ -z "${TMPDIR:-}" ]] || ! is_within_path "$candidate" "$TMPDIR"; } &&
       [[ "${ALLOW_EXTERNAL_BUILD_OUTPUTS:-0}" != "1" ]]; then
        fail "$label must be inside the repository, home, or temporary directory. Set ALLOW_EXTERNAL_BUILD_OUTPUTS=1 only for a deliberate external destination."
    fi
}

canonicalize_output_dir BUILD_DIR "$BUILD_DIR"
canonicalize_output_dir GEN_DIR "$GEN_DIR"
canonicalize_output_dir OUT_DIR "$OUT_DIR"

validate_output_dir BUILD_DIR "$BUILD_DIR"
validate_output_dir GEN_DIR "$GEN_DIR"
validate_output_dir OUT_DIR "$OUT_DIR"

paths_overlap "$BUILD_DIR" "$GEN_DIR" &&
    fail "BUILD_DIR and GEN_DIR must be separate, non-overlapping directories."
paths_overlap "$BUILD_DIR" "$OUT_DIR" &&
    fail "BUILD_DIR and OUT_DIR must be separate, non-overlapping directories."
paths_overlap "$GEN_DIR" "$OUT_DIR" &&
    fail "GEN_DIR and OUT_DIR must be separate, non-overlapping directories."

[[ -n "$APK_NAME" && "$APK_NAME" != */* && "$APK_NAME" != "." && "$APK_NAME" != ".." ]] ||
    fail "APK_NAME must be a simple file name, not a path."
case "$APK_NAME" in
    dex|app-unsigned.apk|app-aligned.apk)
        fail "APK_NAME conflicts with a reserved build output name: $APK_NAME"
        ;;
esac

[[ -f "$ANDROID_JAR" ]] || { echo "ERROR: android.jar not found: $ANDROID_JAR" >&2; exit 2; }
for tool in aapt ecj d8 zipalign apksigner zip; do require_tool "$tool"; done

echo "[*] Cleaning validated build output directories..."
rm -rf -- \
    "$BUILD_DIR" \
    "$GEN_DIR" \
    "$OUT_DIR/dex" \
    "$OUT_DIR/app-unsigned.apk" \
    "$OUT_DIR/app-aligned.apk" \
    "$OUT_DIR/$APK_NAME"
mkdir -p -- "$BUILD_DIR" "$GEN_DIR" "$OUT_DIR"

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
mkdir -p -- "$OUT_DIR/dex"
mapfile -t CLASS_FILES < <(find "$BUILD_DIR" -name '*.class' -type f | sort)
[[ "${#CLASS_FILES[@]}" -gt 0 ]] || { echo "ERROR: no compiled class files found." >&2; exit 2; }
d8 --release --lib "$ANDROID_JAR" --output "$OUT_DIR/dex" "${CLASS_FILES[@]}"

echo "[*] Packaging unsigned APK..."
aapt package -f -M AndroidManifest.xml -S res -A assets -I "$ANDROID_JAR" -F "$OUT_DIR/app-unsigned.apk"
(cd "$OUT_DIR/dex" && zip -q -u ../app-unsigned.apk classes.dex)

echo "[*] Signing APK..."
: "${KEYSTORE:?ERROR: set KEYSTORE to a private signing-keystore path; do not use a keystore committed to the repository.}"
: "${KEYSTORE_PASS:?ERROR: set KEYSTORE_PASS in your local environment.}"
[[ -f "$KEYSTORE" ]] || { echo "ERROR: signing keystore not found: $KEYSTORE" >&2; exit 2; }
KEY_PASS="${KEY_PASS:-$KEYSTORE_PASS}"
export KEY_PASS
APKSIGNER_ALIAS_ARGS=()
if [[ -n "${KEY_ALIAS:-}" ]]; then
    APKSIGNER_ALIAS_ARGS+=(--ks-key-alias "$KEY_ALIAS")
fi
zipalign -f -p 4 "$OUT_DIR/app-unsigned.apk" "$OUT_DIR/app-aligned.apk"
apksigner sign --ks "$KEYSTORE" --ks-pass env:KEYSTORE_PASS --key-pass env:KEY_PASS "${APKSIGNER_ALIAS_ARGS[@]}" --out "$OUT_DIR/$APK_NAME" "$OUT_DIR/app-aligned.apk"

echo "BUILD SUCCESSFUL: $OUT_DIR/$APK_NAME"
