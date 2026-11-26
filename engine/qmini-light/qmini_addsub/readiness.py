from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON report: {target}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {target}.")
    return value


def _load_metrics(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"SFT metrics not found: {target}")
    records = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid SFT metrics at {target}:{line_number}.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"SFT metrics at {target}:{line_number} must be an object.")
        records.append(value)
    if not records:
        raise ValueError("SFT metrics file is empty.")
    return records


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def assess_addsub_readiness(
    metrics: list[dict[str, Any]],
    heldout: dict[str, Any],
    *,
    checkpoint_sha256: str,
    min_validation_improvement: float = 0.01,
    min_test_records: int = 50,
    min_mean_token_f1: float = 0.15,
    min_unique_answer_fraction: float = 0.40,
    max_largest_identical_answer_fraction: float = 0.35,
    max_repeated_4gram_fraction: float = 0.08,
    max_repeated_4gram_count: int = 4,
) -> dict[str, Any]:
    """Fail closed on weak SFT learning or collapsed held-out Add/Sub answers."""
    if not metrics:
        raise ValueError("SFT metrics are empty; no training trajectory can be verified.")
    for name, value in (
        ("min_validation_improvement", min_validation_improvement),
        ("min_mean_token_f1", min_mean_token_f1),
        ("min_unique_answer_fraction", min_unique_answer_fraction),
        ("max_largest_identical_answer_fraction", max_largest_identical_answer_fraction),
        ("max_repeated_4gram_fraction", max_repeated_4gram_fraction),
    ):
        number = _finite(value, name)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if isinstance(min_test_records, bool) or not isinstance(min_test_records, int) or min_test_records < 1:
        raise ValueError("min_test_records must be a positive integer.")
    if isinstance(max_repeated_4gram_count, bool) or not isinstance(max_repeated_4gram_count, int) or max_repeated_4gram_count < 1:
        raise ValueError("max_repeated_4gram_count must be a positive integer.")

    previous_step = 0
    for index, record in enumerate(metrics, 1):
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            raise ValueError("SFT metric steps must be positive and strictly increasing.")
        previous_step = step
        for key in ("train_loss", "valid_response_loss", "learning_rate"):
            if key in record:
                _finite(record[key], f"sft_metrics[{index}].{key}")

    latest = metrics[-1]
    identity_keys = (
        "train_sha256", "valid_sha256", "test_sha256",
        "tokenizer_sha256", "data_manifest_sha256",
    )
    identity = {key: latest.get(key) for key in identity_keys}
    for key, value in identity.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Latest SFT metrics is missing a valid {key}.")
    if len({identity["train_sha256"], identity["valid_sha256"], identity["test_sha256"]}) != 3:
        raise ValueError("SFT train, validation and test fingerprints must all be different.")

    same_run = [
        record for record in metrics
        if all(record.get(key) == value for key, value in identity.items())
        and "valid_response_loss" in record
    ]
    validation_losses = [
        _finite(record["valid_response_loss"], "validation response loss")
        for record in same_run
    ]
    checks: list[dict[str, Any]] = []
    if len(validation_losses) < 2:
        checks.append({
            "name": "sft_validation_learning",
            "passed": False,
            "observed": {"evaluation_points": len(validation_losses)},
            "required": {"evaluation_points": 2, "relative_improvement": min_validation_improvement},
            "detail": "At least two validation response-loss evaluations from the same data/tokenizer run are required.",
        })
    else:
        initial, final = validation_losses[0], validation_losses[-1]
        improvement = (initial - final) / max(abs(initial), 1e-12)
        checks.append({
            "name": "sft_validation_learning",
            "passed": improvement >= min_validation_improvement,
            "observed": {
                "first_validation_response_loss": initial,
                "last_validation_response_loss": final,
                "relative_improvement": improvement,
                "evaluation_points": len(validation_losses),
            },
            "required": {
                "evaluation_points": 2,
                "minimum_relative_improvement": min_validation_improvement,
            },
            "detail": "Assistant-response validation loss must improve; optimizer steps and training loss alone are insufficient.",
        })

    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
    ):
        raise ValueError("Selected checkpoint SHA-256 must be a lowercase 64-character hexadecimal digest.")
    if heldout.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Held-out report checkpoint_sha256 does not match the selected SFT checkpoint.")

    if heldout.get("format") != "qmini-addsub-heldout-eval-v1":
        raise ValueError("Held-out report must use qmini-addsub-heldout-eval-v1.")
    for report_key, identity_key in (
        ("test_sha256", "test_sha256"),
        ("tokenizer_sha256", "tokenizer_sha256"),
        ("dataset_manifest_sha256", "data_manifest_sha256"),
    ):
        if heldout.get(report_key) != identity[identity_key]:
            raise ValueError(f"Held-out report {report_key} does not match the SFT metrics lineage.")

    count = heldout.get("test_records")
    failures = heldout.get("generation_failures")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Held-out report has an invalid test_records count.")
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise ValueError("Held-out report has an invalid generation_failures count.")
    mean_f1 = _finite(heldout.get("mean_token_f1"), "held-out mean_token_f1")
    if not 0.0 <= mean_f1 <= 1.0:
        raise ValueError("Held-out mean_token_f1 must be in [0, 1].")
    mean_prediction_tokens = _finite(
        heldout.get("mean_prediction_tokens"), "held-out mean_prediction_tokens"
    )
    checks.append({
        "name": "heldout_response_quality",
        "passed": (
            count >= min_test_records
            and failures == 0
            and mean_f1 >= min_mean_token_f1
            and mean_prediction_tokens > 0
        ),
        "observed": {
            "test_records": count,
            "generation_failures": failures,
            "mean_token_f1": mean_f1,
            "mean_prediction_tokens": mean_prediction_tokens,
        },
        "required": {
            "minimum_test_records": min_test_records,
            "generation_failures": 0,
            "minimum_mean_token_f1": min_mean_token_f1,
            "mean_prediction_tokens": "> 0",
        },
        "detail": "Held-out answers must generate successfully and meet the configured lexical-overlap floor.",
    })

    diversity = heldout.get("output_diversity")
    if not isinstance(diversity, dict):
        raise ValueError("Held-out report has no output_diversity diagnostics; regenerate it with the current evaluator.")
    examples = diversity.get("examples")
    unique = diversity.get("unique_normalized_answers")
    largest = diversity.get("largest_identical_answer_count")
    empty = diversity.get("empty_answer_count")
    all_identical = diversity.get("all_outputs_identical")
    repeated_4gram_fraction = _finite(
        diversity.get("repeated_4gram_fraction"), "repeated_4gram_fraction"
    )
    max_repeated_4gram_count_observed = diversity.get("max_repeated_4gram_count")
    unique_fraction = _finite(diversity.get("unique_answer_fraction"), "unique_answer_fraction")
    largest_fraction = _finite(
        diversity.get("largest_identical_answer_fraction"),
        "largest_identical_answer_fraction",
    )
    if (
        examples != count
        or isinstance(examples, bool)
        or not isinstance(examples, int)
        or isinstance(unique, bool)
        or not isinstance(unique, int)
        or not 1 <= unique <= examples
        or isinstance(largest, bool)
        or not isinstance(largest, int)
        or not 1 <= largest <= examples
        or isinstance(empty, bool)
        or not isinstance(empty, int)
        or not 0 <= empty <= examples
        or not isinstance(all_identical, bool)
        or isinstance(max_repeated_4gram_count_observed, bool)
        or not isinstance(max_repeated_4gram_count_observed, int)
        or max_repeated_4gram_count_observed < 0
        or not 0.0 <= repeated_4gram_fraction <= 1.0
        or not 0.0 <= unique_fraction <= 1.0
        or not 0.0 <= largest_fraction <= 1.0
        or not math.isclose(unique_fraction, unique / examples, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(largest_fraction, largest / examples, rel_tol=0.0, abs_tol=1e-9)
        or all_identical != (unique == 1)
    ):
        raise ValueError("Held-out output-diversity fields are malformed or inconsistent.")
    checks.append({
        "name": "heldout_repetition_collapse",
        "passed": (
            count >= min_test_records
            and not all_identical
            and empty == 0
            and unique_fraction >= min_unique_answer_fraction
            and largest_fraction <= max_largest_identical_answer_fraction
            and repeated_4gram_fraction <= max_repeated_4gram_fraction
            and max_repeated_4gram_count_observed <= max_repeated_4gram_count
        ),
        "observed": {
            "test_records": count,
            "unique_normalized_answers": unique,
            "unique_answer_fraction": unique_fraction,
            "largest_identical_answer_count": largest,
            "largest_identical_answer_fraction": largest_fraction,
            "empty_answer_count": empty,
            "all_outputs_identical": all_identical,
            "repeated_4gram_fraction": repeated_4gram_fraction,
            "max_repeated_4gram_count": max_repeated_4gram_count_observed,
        },
        "required": {
            "minimum_test_records": min_test_records,
            "minimum_unique_answer_fraction": min_unique_answer_fraction,
            "maximum_largest_identical_answer_fraction": max_largest_identical_answer_fraction,
            "empty_answer_count": 0,
            "all_outputs_identical": False,
            "maximum_repeated_4gram_fraction": max_repeated_4gram_fraction,
            "maximum_repeated_4gram_count": max_repeated_4gram_count,
        },
        "detail": "Constant or heavily repeated answers block readiness even if training loss fell.",
    })

    ready = all(item["passed"] for item in checks)
    return {
        "format": "qmini-addsub-readiness-v1",
        "status": "ready_for_review" if ready else "not_ready",
        "ready": ready,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "limitations": [
            "Passing is not proof of human-level conversation, factuality, reasoning, safety or semantic understanding.",
            "Token F1 is a lexical diagnostic; open-ended English replies require human review.",
            "The minimum thresholds are engineering floors and must be calibrated against a representative, independently reviewed English benchmark.",
            "This gate validates one manifest-bound held-out SFT test split; it does not certify mobile deployment or the complete Add/Sub-only runtime.",
        ],
    }


def _save_json(report: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed held-out readiness gate for Q-Mini Add/Sub SFT."
    )
    parser.add_argument("--metrics", required=True, help="sft_metrics.jsonl")
    parser.add_argument("--checkpoint", required=True, help="Exact SFT checkpoint used for held-out evaluation")
    parser.add_argument("--heldout-summary", required=True, help="heldout_summary.json from qmini_addsub.evaluate")
    parser.add_argument("--output", required=True, help="Readiness JSON output")
    parser.add_argument("--min-validation-improvement", type=float, default=0.01)
    parser.add_argument("--min-test-records", type=int, default=50)
    parser.add_argument("--min-mean-token-f1", type=float, default=0.15)
    parser.add_argument("--min-unique-answer-fraction", type=float, default=0.40)
    parser.add_argument("--max-largest-identical-answer-fraction", type=float, default=0.35)
    parser.add_argument("--max-repeated-4gram-fraction", type=float, default=0.08)
    parser.add_argument("--max-repeated-4gram-count", type=int, default=4)
    args = parser.parse_args(argv)
    report = assess_addsub_readiness(
        _load_metrics(args.metrics),
        _load_json(args.heldout_summary),
        checkpoint_sha256=_file_sha256(args.checkpoint),
        min_validation_improvement=args.min_validation_improvement,
        min_test_records=args.min_test_records,
        min_mean_token_f1=args.min_mean_token_f1,
        min_unique_answer_fraction=args.min_unique_answer_fraction,
        max_largest_identical_answer_fraction=args.max_largest_identical_answer_fraction,
        max_repeated_4gram_fraction=args.max_repeated_4gram_fraction,
        max_repeated_4gram_count=args.max_repeated_4gram_count,
    )
    _save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Q-Mini Add/Sub readiness: {report['status']}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
