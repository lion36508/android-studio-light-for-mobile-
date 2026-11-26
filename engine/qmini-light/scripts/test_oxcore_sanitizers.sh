#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "A C++20 compiler is required for the OxCore sanitizer tests." >&2
  exit 2
fi

"$CXX" \
  -std=c++20 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  -Wall -Wextra -Wpedantic -Werror \
  -I "$ROOT/qmini_addsub/oxcore/include" \
  "$ROOT/qmini_addsub/oxcore/src/oxcore.cpp" \
  "$ROOT/tests/native/oxcore_sanitizer.cpp" \
  -o "$BUILD_DIR/oxcore_sanitizer"

ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
  "$BUILD_DIR/oxcore_sanitizer"
