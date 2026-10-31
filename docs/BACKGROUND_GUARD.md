# OxDroid Background Guard

The file guard is **opt-in**. It never starts itself and does not install a boot receiver or an always-on Android service.

## Commands

Run these from the repository directory in Termux:

```bash
bash ox_background_guard.sh status
bash ox_background_guard.sh start
bash ox_background_guard.sh log
bash ox_background_guard.sh stop
bash ox_background_guard.sh run-once
```

- `start` launches a background scan loop.
- `stop` terminates the guard started by this script.
- `status` reports the recorded process state and PID.
- `run-once` scans immediately without starting a background process.
- `log` displays the most recent alerts.

## Resource and permission behavior

- The default scan interval is 60 seconds; set `OX_GUARD_INTERVAL_SECONDS` to a value of at least 10 if a different interval is needed.
- The guard scans local project files only. It does not ping an external host or require network connectivity.
- It alerts for unexpected APK, JKS, and keystore files. The current CI APK output path is the only APK allowlisted; **no signing keystore is ever allowlisted**.
- Termux notifications are optional. If `termux-notification` is not installed or notification permission is unavailable, alerts remain in the log.
- The process runs under the current Android/Termux user; it does not request root access.
- A background process can still consume battery while enabled. Stop it when the scan is not needed. The app does not silently restart it after reboot.

## State files

By default, PID and log files are stored under `~/.local/state/oxdroid-guard/`. You may override the location with `OX_GUARD_STATE_DIR` and the project root with `OX_GUARD_PROJECT_ROOT`.
