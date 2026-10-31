#!/bin/bash
set -euo pipefail

echo "================================================="
echo "OxDroid security file guard started"
echo "================================================="

# Only the current CI/build output name is allowed as a generated APK.
# No signing keystore is ever allowlisted.
WHITELIST_APK="./bin/OxDroid-Mobile-IDE.apk"

while true; do
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        UNAUTHORIZED="$(find . -path './.git' -prune -o -type f \( -name '*.apk' -o -name '*.keystore' -o -name '*.jks' \) ! -path "$WHITELIST_APK" -print)"

        if [ -n "$UNAUTHORIZED" ]; then
            echo "[$(date '+%T')] ALERT: unexpected build/signing file(s) detected:"
            printf '%s\n' "$UNAUTHORIZED"
            if command -v termux-notification >/dev/null 2>&1; then
                termux-notification --title "OxDroid Security Alert" --content "Unexpected APK/keystore file detected. Check the project files." --priority high 2>/dev/null || true
            fi
        fi

        sleep 60
    else
        sleep 10
    fi
done
