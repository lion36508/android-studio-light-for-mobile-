from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("torch_float32", re.compile(r"\btorch\.float32\b")),
    ("torch_float_alias", re.compile(r"\btorch\.float\b")),
    ("numpy_float32", re.compile(r"\b(?:np|numpy)\.float32\b")),
    ("float_method", re.compile(r"\.float\(\)")),
    ("float32_token", re.compile(r"\bfloat32\b", re.IGNORECASE)),
    ("fp32_token", re.compile(r"\bfp32\b", re.IGNORECASE)),
    ("native_float_pointer", re.compile(r"\bfloat\s*\*")),
)

TEXT_SUFFIXES = {
    ".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".md", ".rst",
    ".json", ".ipynb", ".yml", ".yaml", ".toml", ".txt",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "node_modules", "dist", "build",
}


def classify(path: Path, line: str) -> str:
    """Assign a review bucket; this is triage, not semantic proof."""
    parts = path.parts
    lowered = line.lower()

    if parts and parts[0] == "tests":
        return "test_or_reference"
    if (parts and parts[0] == "docs") or path.suffix.lower() in {".md", ".rst"}:
        return "documentation"
    if path.name == "train.py":
        if any(word in lowered for word in ("optimizer", "master", "checkpoint", "parameter")):
            return "training_parameter_or_optimizer"
        return "training_numeric_or_conversion"
    if path.name == "model.py":
        return "model_numeric_or_parameter"
    if path.name in {"quantization.py", "export.py", "packed_runtime.py", "initial_build.py"}:
        return "quantization_export_or_runtime"
    if path.suffix.lower() in {".cpp", ".cc", ".c", ".h", ".hpp"}:
        return "native_kernel_or_abi"
    if path.suffix.lower() == ".ipynb":
        return "notebook"
    return "other"


def collect_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    findings: list[dict[str, Any]] = []
    scanned_files = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        scanned_files += 1
        for line_number, line in enumerate(source.splitlines(), start=1):
            matches = [name for name, pattern in PATTERNS if pattern.search(line)]
            if not matches:
                continue
            findings.append(
                {
                    "path": relative.as_posix(),
                    "line": line_number,
                    "category": classify(relative, line),
                    "matches": matches,
                    "snippet": " ".join(line.strip().split())[:240],
                    "review_required": True,
                }
            )

    category_counts = Counter(item["category"] for item in findings)
    pattern_counts = Counter(
        pattern for item in findings for pattern in item["matches"]
    )
    return {
        "schema": "qmini-precision-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "files_scanned": scanned_files,
        "finding_count": len(findings),
        "category_counts": dict(sorted(category_counts.items())),
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "findings": findings,
        "limitations": [
            "Static text matching only; it cannot prove whether a tensor is persistent or temporary.",
            "Implicit dtype promotion, default tensor dtypes, third-party kernels and runtime allocations require dynamic inspection.",
            "Every finding requires source-level review before changing numerical precision.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory explicit FP32 and float32 paths in the Q-Mini source tree."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Repository root to scan (defaults to the checked-out repository).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional JSON report path. Parent directories are created.",
    )
    parser.add_argument(
        "--fail-on-category", action="append", default=[],
        help="Exit with status 2 when a category has findings; may be repeated.",
    )
    args = parser.parse_args(argv)

    report = collect_report(args.root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")

    print(
        f"Q-Mini precision audit: {report['files_scanned']} files scanned, "
        f"{report['finding_count']} matched lines"
    )
    for category, count in report["category_counts"].items():
        print(f"  {category}: {count}")

    blocked = [
        category for category in args.fail_on_category
        if report["category_counts"].get(category, 0) > 0
    ]
    if blocked:
        print("Strict audit gate failed: " + ", ".join(blocked))
        return 2
    if args.output:
        print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
