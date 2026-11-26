#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-c++}"
OUTPUT="${1:-/tmp/libqmini_oxcore.so}"

if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "OxCore requires a C++20 compiler (set CXX to its executable)." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
"$CXX" \
  -std=c++20 -O3 -fPIC -shared -fvisibility=hidden \
  -Wall -Wextra -Wpedantic -Werror \
  -I "$ROOT/qmini_addsub/oxcore/include" \
  "$ROOT/qmini_addsub/oxcore/src/oxcore.cpp" \
  -o "$OUTPUT"

echo "Built standalone OxCore library: $OUTPUT"
