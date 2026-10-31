#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${OX_GUARD_PROJECT_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")" && pwd)}"
STATE_BASE="${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}"
STATE_DIR="${OX_GUARD_STATE_DIR:-$STATE_BASE/oxdroid-guard}"
PID_FILE="$STATE_DIR/guard.pid"
LOG_FILE="$STATE_DIR/guard.log"
INTERVAL="${OX_GUARD_INTERVAL_SECONDS:-60}"
SLEEP_PID=""
WHITELIST_APK="${OX_GUARD_WHITELIST_APK:-./bin/OxDroid-Mobile-IDE.apk}"

usage() {
    cat <<'USAGE'
OxDroid Background Guard (opt-in)

Usage:
  bash ox_background_guard.sh start     Start the guard in the background
  bash ox_background_guard.sh stop      Stop the guard
  bash ox_background_guard.sh status    Show whether the guard is running
  bash ox_background_guard.sh run-once  Scan the project once
  bash ox_background_guard.sh log       Show the latest guard log
  bash ox_background_guard.sh help      Show this help

The guard does not start automatically. Use 'start' only when you want it enabled.
USAGE
}

ensure_state_dir() {
    mkdir -p "$STATE_DIR"
}

read_pid() {
    [[ -r "$PID_FILE" ]] || return 1
    local pid
    IFS= read -r pid < "$PID_FILE" || return 1
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$pid"
}

is_guard_running() {
    local pid cmdline
    pid="$(read_pid)" || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"$(basename "$SCRIPT_PATH") daemon"* ]]
}

scan_once() {
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        echo "Guard scan failed: project directory does not exist: $PROJECT_ROOT" >&2
        return 2
    fi

    local findings
    findings="$(cd "$PROJECT_ROOT" && find . -path './.git' -prune -o -type f \( -name '*.apk' -o -name '*.keystore' -o -name '*.jks' \) ! -path "$WHITELIST_APK" -print)"

    if [[ -n "$findings" ]]; then
        {
            echo "[$(date '+%F %T')] ALERT: unexpected APK/signing file(s) detected:"
            printf '%s\n' "$findings"
        } | tee -a "$LOG_FILE"

        if command -v termux-notification >/dev/null 2>&1; then
            termux-notification --title "OxDroid Security Alert" \
                --content "Unexpected APK or keystore file detected. Check the project files." \
                --priority high 2>/dev/null || true
        fi
        return 1
    fi

    echo "No unexpected APK or keystore files found."
    return 0
}

start_guard() {
    ensure_state_dir
    if is_guard_running; then
        echo "OxDroid guard is already running (PID $(read_pid))."
        return 0
    fi

    rm -f "$PID_FILE"
    nohup bash "$SCRIPT_PATH" daemon >/dev/null 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"

    for _ in 1 2 3 4 5; do
        if is_guard_running; then
            echo "OxDroid guard started (PID $pid)."
            echo "Log: $LOG_FILE"
            return 0
        fi
        sleep 1
    done

    rm -f "$PID_FILE"
    echo "Guard did not start. Check Termux permissions and the script path." >&2
    return 1
}

stop_guard() {
    local pid
    if ! is_guard_running; then
        rm -f "$PID_FILE"
        echo "OxDroid guard is not running."
        return 0
    fi

    pid="$(read_pid)"
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        if ! is_guard_running; then
            rm -f "$PID_FILE"
            echo "OxDroid guard stopped."
            return 0
        fi
        sleep 1
    done

    echo "Guard did not stop within 5 seconds. PID: $pid" >&2
    return 1
}

show_status() {
    if is_guard_running; then
        echo "OxDroid guard: RUNNING (PID $(read_pid), interval ${INTERVAL}s)"
        echo "Log: $LOG_FILE"
    else
        rm -f "$PID_FILE"
        echo "OxDroid guard: STOPPED"
    fi
}

cleanup_daemon() {
    if [[ -n "$SLEEP_PID" ]]; then
        kill "$SLEEP_PID" 2>/dev/null || true
        wait "$SLEEP_PID" 2>/dev/null || true
        SLEEP_PID=""
    fi
    rm -f "$PID_FILE"
}

daemon_main() {
    ensure_state_dir
    if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 10 )); then
        echo "Invalid guard interval. Set OX_GUARD_INTERVAL_SECONDS to at least 10." >&2
        return 2
    fi

    printf '%s\n' "$" > "$PID_FILE"
    trap cleanup_daemon EXIT
    trap 'exit 0' INT TERM

    while true; do
        scan_once >/dev/null || true
        sleep "$INTERVAL" &
        SLEEP_PID=$!
        wait "$SLEEP_PID" || true
        SLEEP_PID=""
    done
}

case "${1:-help}" in
    start) start_guard ;;
    stop) stop_guard ;;
    status) show_status ;;
    run-once) ensure_state_dir; scan_once ;;
    log)
        if [[ -f "$LOG_FILE" ]]; then
            tail -n 40 "$LOG_FILE"
        else
            echo "No guard log exists yet."
        fi
        ;;
    daemon) daemon_main ;;
    help|-h|--help) usage ;;
    *)
        usage >&2
        exit 2
        ;;
esac
