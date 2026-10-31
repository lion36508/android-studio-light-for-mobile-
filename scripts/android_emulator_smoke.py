#!/usr/bin/env python3
"""Install OxDroid on an Android emulator and smoke-test core navigation."""
import pathlib
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PACKAGE = "com.bikash.mobileide"
MAIN_ACTIVITY = PACKAGE + "/.MainActivity"
EDITOR_ACTIVITY = PACKAGE + "/.EditorActivity"
UI_XML = "/sdcard/oxdroid-window.xml"


def run_adb(*args, timeout=30):
    result = subprocess.run(
        ["adb", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError("adb {} failed:\n{}".format(" ".join(args), result.stdout))
    return result.stdout.strip()


def wait_for_activity(activity, timeout=35):
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        last = run_adb("shell", "dumpsys", "activity", "activities", timeout=15)
        if activity in last:
            return
        time.sleep(1)
    raise AssertionError("Activity did not become active: {}\n{}".format(activity, last[-2500:]))


def dump_ui():
    run_adb("shell", "uiautomator", "dump", UI_XML, timeout=30)
    raw = run_adb("exec-out", "cat", UI_XML, timeout=15)
    return ET.fromstring(raw)


def node_texts(root):
    values = []
    for node in root.iter():
        text = node.attrib.get("text", "").strip()
        description = node.attrib.get("content-desc", "").strip()
        if text:
            values.append(text)
        if description:
            values.append(description)
    return values


def tap_node(root, expected_text):
    for node in root.iter():
        if node.attrib.get("text", "").strip() == expected_text:
            bounds = node.attrib.get("bounds", "")
            numbers = [int(value) for value in bounds.replace("][", ",").replace("[", "").replace("]", "").split(",")]
            if len(numbers) != 4:
                raise AssertionError("Invalid accessibility bounds for {}: {}".format(expected_text, bounds))
            x = (numbers[0] + numbers[2]) // 2
            y = (numbers[1] + numbers[3]) // 2
            run_adb("shell", "input", "tap", str(x), str(y))
            return
    raise AssertionError("Could not find accessible UI text: {}".format(expected_text))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: android_emulator_smoke.py path/to/apk")

    apk = pathlib.Path(sys.argv[1])
    if not apk.is_file() or apk.stat().st_size == 0:
        raise AssertionError("APK is missing or empty: {}".format(apk))

    print("[1/5] Install APK")
    print(run_adb("install", "-r", str(apk.resolve()), timeout=120))

    print("[2/5] Launch dashboard")
    print(run_adb("shell", "am", "start", "-W", "-n", MAIN_ACTIVITY, timeout=45))
    wait_for_activity(".MainActivity")
    dashboard = dump_ui()
    dashboard_text = node_texts(dashboard)
    if "OxDroid Studio" not in dashboard_text:
        raise AssertionError("Dashboard title is missing from the accessibility tree.")
    if "Androidify" not in dashboard_text:
        raise AssertionError("Expected starter project is not visible in the dashboard.")

    print("[3/5] Open a starter project in the editor")
    tap_node(dashboard, "Androidify")
    wait_for_activity(".EditorActivity")
    editor = dump_ui()
    editor_text = node_texts(editor)
    if not any("MainActivity.java" in value for value in editor_text):
        raise AssertionError("Editor tab/toolbar did not render the expected source file.")
    edit_fields = [
        node for node in editor.iter()
        if node.attrib.get("class") == "android.widget.EditText"
    ]
    if not edit_fields:
        raise AssertionError("Code editor input field is not present in the accessibility tree.")

    print("[4/5] Verify back navigation and relaunch")
    run_adb("shell", "input", "keyevent", "4")
    wait_for_activity(".MainActivity")
    run_adb("shell", "am", "start", "-W", "-n", MAIN_ACTIVITY, timeout=45)
    wait_for_activity(".MainActivity")

    print("[5/5] Check app process for fatal exceptions")
    pid = run_adb("shell", "pidof", PACKAGE)
    if not pid:
        raise AssertionError("OxDroid process exited after navigation.")
    log = run_adb("logcat", "-d", "--pid={}".format(pid), timeout=30)
    if "FATAL EXCEPTION" in log:
        raise AssertionError("OxDroid emitted a fatal exception:\n{}".format(log[-5000:]))

    print("ANDROID EMULATOR SMOKE TEST PASSED (API 36): install, dashboard, editor, back navigation, process health.")


if __name__ == "__main__":
    main()
