from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel

from .sft import SYSTEM_TEXT
from .train import sha256_file


SPECIAL_TOKENS = (
    "<unk>", "<pad>", "<bos>", "<eos>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
)
COMMON_CHAT_TERMS = (
    "hello", "Hello", "hi", "sorry", "help", "name", "you", "can",
    "what", "how", "thank", "please", "because", "why", "yes", "no",
    "I", "I'm", "don't", "can't", "the", "is", "are", "and", "with",
    "explain", "count", "colour", "color", "friend", "question", "answer",
)
CHAT_PHRASES = (
    " Hello, how can I help you?",
    " I'm sorry. Can you please explain?",
    " What is your name, and why are you here?",
    " Count from one to ten.",
    " I don't know the answer yet.",
)


def _tokenize_probe(tokenizer: Tokenizer, text: str) -> dict[str, Any]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return {
        "text": text,
        "utf8_bytes": len(text.encode("utf-8")),
        "token_count": len(encoded.ids),
        "tokens_per_byte": (
            len(encoded.ids) / len(text.encode("utf-8"))
            if text.encode("utf-8") else 0.0
        ),
        "token_ids": [int(value) for value in encoded.ids],
        "token_pieces": [tokenizer.id_to_token(int(value)) for value in encoded.ids],
    }


def _read_validation_prefix(path: Path, byte_budget: int) -> str:
    if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget < 1:
        raise ValueError("validation byte budget must be a positive integer.")
    chunks: list[str] = []
    used = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            encoded = line.encode("utf-8")
            remaining = byte_budget - used
            if len(encoded) <= remaining:
                chunks.append(line)
                used += len(encoded)
                if used == byte_budget:
                    break
                continue
            if remaining > 0:
                # Avoid splitting a UTF-8 sequence at the audit budget boundary.
                chunks.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                used = byte_budget
            break
    sample = "".join(chunks)
    if not sample.strip():
        raise ValueError("Validation text sample is empty.")
    return sample


def audit_tokenizer(
    tokenizer: Tokenizer,
    *,
    validation_text: str | None = None,
    validation_sample_bytes: int | None = None,
) -> dict[str, Any]:
    """Audit byte-alphabet coverage, chat-word splitting and optional held-out text."""
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError("Tokenizer vocabulary is empty.")
    vocab_ids = list(vocab.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in vocab_ids):
        raise ValueError("Tokenizer vocabulary contains a non-integer token ID.")
    duplicate_ids = len(vocab_ids) - len(set(vocab_ids))
    if duplicate_ids:
        raise ValueError("Tokenizer vocabulary maps multiple tokens to the same ID.")

    byte_alphabet = set(ByteLevel.alphabet())
    present_byte_tokens = byte_alphabet.intersection(vocab)
    missing_byte_tokens = byte_alphabet - set(vocab)
    non_ascii_tokens = sorted(token for token in vocab if not token.isascii())
    non_ascii_byte_glyphs = sorted(
        token for token in non_ascii_tokens if token in byte_alphabet
    )
    non_ascii_other_tokens = sorted(
        token for token in non_ascii_tokens if token not in byte_alphabet
    )
    special_ids = {
        token: tokenizer.token_to_id(token)
        for token in SPECIAL_TOKENS
    }
    missing_special_tokens = [
        token for token, token_id in special_ids.items() if token_id is None
    ]

    word_probes = []
    for term in COMMON_CHAT_TERMS:
        probe = _tokenize_probe(tokenizer, " " + term)
        word_probes.append({
            "term": term,
            "leading_space_token_count": probe["token_count"],
            "single_token": probe["token_count"] == 1,
            "token_ids": probe["token_ids"],
            "token_pieces": probe["token_pieces"],
        })
    phrase_probes = [_tokenize_probe(tokenizer, phrase) for phrase in CHAT_PHRASES]
    prompt_probe = _tokenize_probe(tokenizer, SYSTEM_TEXT)
    validation_report = None
    if validation_text is not None:
        validation_probe = _tokenize_probe(tokenizer, validation_text)
        validation_report = {
            **validation_probe,
            "sample_strategy": "deterministic UTF-8 prefix, whole lines where possible",
            "sample_bytes_requested": validation_sample_bytes,
        }

    merge_tokens = [
        token for token in vocab
        if token not in byte_alphabet and token not in SPECIAL_TOKENS
    ]
    single_token_count = sum(item["single_token"] for item in word_probes)
    return {
        "format": "qmini-addsub-tokenizer-audit-v1",
        "tokenizer_model": type(tokenizer.model).__name__,
        "vocabulary_size": tokenizer.get_vocab_size(),
        "byte_level_alphabet": {
            "expected_byte_symbols": len(byte_alphabet),
            "present_byte_symbols": len(present_byte_tokens),
            "missing_byte_symbols": sorted(missing_byte_tokens),
            "non_ascii_display_glyphs_from_byte_alphabet": len(non_ascii_byte_glyphs),
            "example_non_ascii_byte_glyphs": non_ascii_byte_glyphs[:32],
            "non_ascii_tokens_outside_byte_alphabet": len(non_ascii_other_tokens),
            "example_other_non_ascii_tokens": non_ascii_other_tokens[:32],
            "interpretation": (
                "ByteLevel uses a reversible visible-Unicode mapping for its 256 byte "
                "symbols. Non-ASCII-looking entries in that base alphabet are not, "
                "by themselves, wasted vocabulary items."
            ),
        },
        "learned_non_byte_non_special_token_count": len(merge_tokens),
        "special_tokens": {
            "ids": special_ids,
            "missing": missing_special_tokens,
            "ids_are_unique": (
                not missing_special_tokens
                and len(set(special_ids.values())) == len(SPECIAL_TOKENS)
            ),
        },
        "common_chat_terms": {
            "term_count": len(word_probes),
            "single_token_count": single_token_count,
            "single_token_fraction": single_token_count / len(word_probes),
            "probes": word_probes,
        },
        "chat_phrase_probes": phrase_probes,
        "system_prompt_probe": prompt_probe,
        "heldout_validation_probe": validation_report,
        "limitations": [
            "Single-token coverage is a lexical diagnostic, not a measure of language quality.",
            "A split word is not automatically a tokenizer defect; context and token frequency matter.",
            "The validation sample is an evaluation-only prefix and is never used to train or modify the tokenizer.",
            "No fixed tokens-per-byte threshold is treated as a universal pass condition.",
        ],
    }


def audit_tokenizer_file(
    tokenizer_path: str | Path,
    *,
    validation_text_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    validation_byte_budget: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    tokenizer_file = Path(tokenizer_path)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(tokenizer_file)
    if (validation_text_path is None) != (dataset_manifest_path is None):
        raise ValueError("Validation text and dataset manifest must be supplied together.")

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    validation_sample = None
    manifest_sha = None
    validation_sha = None
    if validation_text_path is not None:
        validation_file = Path(validation_text_path)
        manifest_file = Path(dataset_manifest_path)
        if not validation_file.is_file() or not manifest_file.is_file():
            raise FileNotFoundError("Validation text and dataset manifest files must exist.")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("format") != "qmini-addsub-data-v1":
            raise ValueError("Unsupported dataset manifest format for tokenizer audit.")
        entry = manifest.get("pretraining", {}).get("splits", {}).get("validation", {})
        if not isinstance(entry, dict) or entry.get("path") != validation_file.name:
            raise ValueError("Validation text is not the manifest-declared pretraining validation split.")
        validation_sha = sha256_file(validation_file)
        if entry.get("sha256") != validation_sha:
            raise ValueError("Validation text does not match the dataset manifest SHA-256.")
        expected_tokenizer_sha = manifest.get("tokenizer", {}).get("sha256")
        actual_tokenizer_sha = sha256_file(tokenizer_file)
        if expected_tokenizer_sha != actual_tokenizer_sha:
            raise ValueError("Tokenizer does not match the selected dataset manifest.")
        validation_sample = _read_validation_prefix(validation_file, validation_byte_budget)
        manifest_sha = sha256_file(manifest_file)

    report = audit_tokenizer(
        tokenizer,
        validation_text=validation_sample,
        validation_sample_bytes=(
            len(validation_sample.encode("utf-8"))
            if validation_sample is not None else None
        ),
    )
    report["tokenizer_sha256"] = sha256_file(tokenizer_file)
    report["dataset_manifest_sha256"] = manifest_sha
    report["validation_text_sha256"] = validation_sha
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Q-Mini Add/Sub ByteLevel BPE vocabulary coverage and held-out token efficiency."
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--validation-text", default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--validation-byte-budget", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = audit_tokenizer_file(
        args.tokenizer,
        validation_text_path=args.validation_text,
        dataset_manifest_path=args.dataset_manifest,
        validation_byte_budget=args.validation_byte_budget,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError("Tokenizer audit output already exists; choose a new path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Temporary audit path already exists: {temporary}")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(encoded, end="")


if __name__ == "__main__":
    main()
