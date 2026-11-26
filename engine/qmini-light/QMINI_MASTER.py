#!/usr/bin/env python3
"""Q-Mini project master entry point; model implementation remains in qmini/."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def usage() -> None:
    print("Q-MINI MASTER")
    print("  python QMINI_MASTER.py audit [--config configs/qmini_light_0.7m.json]")
    print("  python QMINI_MASTER.py verify")
    print("  python QMINI_MASTER.py kernel")
    print("  python QMINI_MASTER.py tokenizer ...")
    print("  python QMINI_MASTER.py prepare ...")
    print("  python QMINI_MASTER.py train ...")
    print("  python QMINI_MASTER.py generate ...")
    print("  python QMINI_MASTER.py chat ...")
    print("  python -m qmini.cli eval-lm --help")
    print("  python -m qmini.cli eval-tasks --help")
    print("  python QMINI_MASTER.py verify-export --path MODEL.qmini.npz")
    print("  python QMINI_MASTER.py memory --db PATH add --text TEXT")


def main() -> int:
    # Make every relative config/data path resolve from this repository, even when
    # the script is launched via an absolute path from another working directory.
    os.chdir(ROOT)
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        usage()
        return 0
    command = sys.argv[1]
    if command == "verify":
        checks = [
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "-m", "qmini.native", "--build"],
            [sys.executable, "-m", "qmini.cli", "audit", "--config", "configs/qmini_smoke.json"],
        ]
        for check in checks:
            status = run(check)
            if status:
                return status
        print("Q-Mini verification completed.")
        return 0
    if command == "kernel":
        return run([sys.executable, "-m", "qmini.native", "--build"])
    sys.argv = ["qmini", *sys.argv[1:]]
    from qmini.cli import main as cli_main
    return int(cli_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
