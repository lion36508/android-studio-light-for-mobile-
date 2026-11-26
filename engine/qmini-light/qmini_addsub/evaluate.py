from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from .generate import generate_reply, load_addsub_model
from .sft import _load_conversations
from .train import sha256_file


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def lexical_scores(reference: str, prediction: str) -> dict[str, float]:
    """Return transparent lexical metrics; these are not semantic-quality judgments."""
    reference_tokens = _words(reference)
    predicted_tokens = _words(prediction)
    exact = float(reference_tokens == predicted_tokens)
    if not reference_tokens or not predicted_tokens:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        reference_counts = Counter(reference_tokens)
        predicted_counts = Counter(predicted_tokens)
        overlap = sum((reference_counts & predicted_counts).values())
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(reference_tokens)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": exact,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "reference_tokens": float(len(reference_tokens)),
        "prediction_tokens": float(len(predicted_tokens)),
    }


def _answer_diversity(predictions: list[str]) -> dict[str, Any]:
    """Measure repeated whole answers and repeated four-word loops."""
    if not predictions:
        raise ValueError("Output-diversity analysis requires at least one prediction.")
    normalized = [" ".join(_words(value)) for value in predictions]
    counts = Counter(normalized)
    unique_count = len(counts)
    largest_count = max(counts.values())
    total = len(normalized)
    total_ngrams = 0
    repeated_ngrams = 0
    max_repeated_4gram_count = 0
    for prediction in predictions:
        tokens = _words(prediction)
        ngrams = Counter(
            tuple(tokens[index:index + 4])
            for index in range(max(0, len(tokens) - 3))
        )
        total_ngrams += sum(ngrams.values())
        repeated_ngrams += sum(count - 1 for count in ngrams.values() if count > 1)
        if ngrams:
            max_repeated_4gram_count = max(
                max_repeated_4gram_count, max(ngrams.values())
            )
    return {
        "examples": total,
        "unique_normalized_answers": unique_count,
        "unique_answer_fraction": unique_count / total,
        "largest_identical_answer_count": largest_count,
        "largest_identical_answer_fraction": largest_count / total,
        "empty_answer_count": sum(not value for value in normalized),
        "all_outputs_identical": unique_count == 1,
        "repeated_4gram_fraction": (
            repeated_ngrams / total_ngrams if total_ngrams else 0.0
        ),
        "max_repeated_4gram_count": max_repeated_4gram_count,
        "metric_note": (
            "Exact equality after Unicode word normalization plus within-answer "
            "four-word phrase repetition. Paraphrased or semantically repetitive "
            "answers still require human review."
        ),
    }


@torch.inference_mode()
def evaluate_heldout_conversations(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    test_jsonl: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    device_name: str = "cpu",
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Generate answers only from the manifest-declared held-out SFT test split."""
    checkpoint_file = Path(checkpoint_path)
    tokenizer_file = Path(tokenizer_path)
    test_file = Path(test_jsonl)
    manifest_file = Path(dataset_manifest_path)
    output = Path(output_dir)
    for path in (checkpoint_file, tokenizer_file, test_file, manifest_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format") != "qmini-addsub-data-v1":
        raise ValueError("Unsupported Add/Sub evaluation dataset manifest.")
    splits = manifest.get("instruction_splits", {})
    test_entry = splits.get("test", {})
    if test_entry.get("path") != test_file.name or test_entry.get("sha256") != sha256_file(test_file):
        raise ValueError(
            "Evaluation file must exactly match the manifest-declared held-out SFT test split."
        )
    tokenizer_metadata = manifest.get("tokenizer", {})
    tokenizer_sha = sha256_file(tokenizer_file)
    if tokenizer_metadata.get("sha256") != tokenizer_sha:
        raise ValueError("Evaluation tokenizer does not match the curated data manifest.")

    approved_values = manifest.get("approved_instruction_sources")
    if (
        not isinstance(approved_values, list)
        or not approved_values
        or any(not isinstance(value, str) or not value.strip() for value in approved_values)
    ):
        raise ValueError("Evaluation manifest must list approved instruction sources.")
    records = _load_conversations(test_file, set(approved_values))
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    model = load_addsub_model(
        checkpoint_file,
        device_name=device_name,
        tokenizer_path=tokenizer_file,
    )

    rows = []
    scores = []
    for index, record in enumerate(records):
        messages = record["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(f"Test record {index} does not end with an assistant reference.")
        reference = messages[-1]["content"]
        history = messages[:-1]
        generation_error = None
        try:
            prediction = generate_reply(
                model,
                tokenizer,
                history,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                greedy=True,
            )
        except ValueError as exc:
            prediction = ""
            generation_error = str(exc)
        row = {
            "index": index,
            "source": record["source"],
            "prediction": prediction,
            "reference": reference,
            "generation_error": generation_error,
            "scores": lexical_scores(reference, prediction),
        }
        rows.append(row)
        scores.append(row["scores"])

    if not scores:
        raise ValueError("Held-out SFT test set contains no usable conversations.")
    summary = {
        "format": "qmini-addsub-heldout-eval-v1",
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "tokenizer_sha256": tokenizer_sha,
        "dataset_manifest_sha256": sha256_file(manifest_file),
        "test_sha256": sha256_file(test_file),
        "test_records": len(scores),
        "generation_failures": sum(row["generation_error"] is not None for row in rows),
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
        "mean_exact_match": sum(item["exact_match"] for item in scores) / len(scores),
        "mean_token_precision": sum(item["token_precision"] for item in scores) / len(scores),
        "mean_token_recall": sum(item["token_recall"] for item in scores) / len(scores),
        "mean_token_f1": sum(item["token_f1"] for item in scores) / len(scores),
        "mean_reference_tokens": sum(item["reference_tokens"] for item in scores) / len(scores),
        "mean_prediction_tokens": sum(item["prediction_tokens"] for item in scores) / len(scores),
        "output_diversity": _answer_diversity([row["prediction"] for row in rows]),
        "metric_limit": (
            "Exact match and bag-of-words token overlap are lexical indicators only; "
            "they do not evaluate factuality, reasoning or semantic correctness."
        ),
    }

    if output.exists():
        raise FileExistsError(
            "Evaluation output path already exists. Choose a new folder to avoid mixing reports."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        results_path = staging / "heldout_responses.jsonl"
        summary_path = staging / "heldout_summary.json"
        with results_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy-evaluate a trained Add/Sub assistant on its manifest-held-out SFT test split."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    result = evaluate_heldout_conversations(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        test_jsonl=args.test_jsonl,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output,
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
