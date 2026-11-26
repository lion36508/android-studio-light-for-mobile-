from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qmini_addsub.data import normalize_training_text, sha256_file


def _normalize_pretraining_file(source: Path, destination: Path) -> dict[str, Any]:
    """Normalize user-selected UTF-8 English text and reject sensitive/low-quality lines."""
    if not source.is_file():
        raise FileNotFoundError(source)
    raw_text = source.read_text(encoding="utf-8")
    raw_text = unicodedata.normalize("NFC", raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned_lines = []
    rejected: dict[str, int] = {}
    for line in raw_text.split("\n"):
        if not line.strip():
            cleaned_lines.append("")
            continue
        cleaned, reason = normalize_training_text(line, min_chars=1, max_chars=32768)
        if reason == "accepted":
            cleaned_lines.append(cleaned)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
            cleaned_lines.append("")
    text = "\n".join(cleaned_lines).strip()
    if not text:
        raise ValueError(f"No usable English text remains after cleaning: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8", newline="\n")
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "rejected_line_counts": dict(sorted(rejected.items())),
    }

def _curate_sft_file(
    source_path: Path,
    destination: Path,
    source_name: str,
    license_name: str,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    records = []
    seen = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source_path}:{line_number}: invalid JSONL") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source_path}:{line_number}: record must be an object.")
            if record.get("human_authored") is not True or record.get("synthetic") is not False:
                raise ValueError(
                    f"{source_path}:{line_number}: every record must explicitly be "
                    "human_authored=true and synthetic=false."
                )
            if record.get("source") not in (None, source_name):
                raise ValueError(
                    f"{source_path}:{line_number}: record source does not match "
                    f"the declared source {source_name!r}."
                )
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{source_path}:{line_number}: messages must contain user/assistant turns.")
            roles = [item.get("role") if isinstance(item, dict) else None for item in messages]
            if roles[0] != "user" or roles[-1] != "assistant":
                raise ValueError(f"{source_path}:{line_number}: conversation must start with user and end with assistant.")
            if any(role not in {"user", "assistant"} for role in roles):
                raise ValueError(f"{source_path}:{line_number}: only user/assistant roles are accepted.")
            if any(roles[index] == roles[index - 1] for index in range(1, len(roles))):
                raise ValueError(f"{source_path}:{line_number}: roles must alternate.")

            clean_messages = []
            for message in messages:
                cleaned, reason = normalize_training_text(message.get("content"))
                if reason != "accepted":
                    raise ValueError(
                        f"{source_path}:{line_number}: message rejected by cleaner ({reason})."
                    )
                clean_messages.append({"role": message["role"], "content": cleaned})
            fingerprint = hashlib.sha256(
                json.dumps(clean_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            cleaned_record = {
                "source": source_name,
                "license": license_name,
                "human_authored": True,
                "synthetic": False,
                "messages": clean_messages,
            }
            records.append(cleaned_record)

    if not records:
        raise ValueError(f"No valid human-authored SFT conversations found in {source_path}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": destination.name,
        "conversations": len(records),
        "sha256": sha256_file(destination),
    }


def import_user_dataset(
    pretrain_train: str | Path,
    pretrain_validation: str | Path,
    pretrain_test: str | Path,
    output_dir: str | Path,
    pretrain_source: str,
    pretrain_license: str,
    confirm_rights_and_human_data: bool = False,
    confirm_english_data: bool = False,
    sft_train: str | Path | None = None,
    sft_validation: str | Path | None = None,
    sft_test: str | Path | None = None,
    sft_source: str | None = None,
    sft_license: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import local English splits transactionally. No downloads are performed."""
    if not confirm_rights_and_human_data:
        raise ValueError(
            "Explicit confirmation of dataset rights and non-synthetic data is required."
        )
    if not confirm_english_data:
        raise ValueError("Explicit confirmation that all selected data is English is required.")
    if not isinstance(pretrain_source, str) or not pretrain_source.strip():
        raise ValueError("Pretraining source name must be a non-empty string.")
    if not isinstance(pretrain_license, str) or not pretrain_license.strip():
        raise ValueError("Pretraining license must be a non-empty string.")

    pretrain_paths = tuple(
        Path(value).expanduser().resolve()
        for value in (pretrain_train, pretrain_validation, pretrain_test)
    )
    if len(set(pretrain_paths)) != 3:
        raise ValueError("Pretraining train/validation/test inputs must be separate files.")

    sft_values = (sft_train, sft_validation, sft_test)
    if any(value is None for value in sft_values) and not all(value is None for value in sft_values):
        raise ValueError("Provide SFT train, validation and test files together, or none.")
    has_sft = sft_train is not None
    if has_sft:
        sft_paths = tuple(Path(value).expanduser().resolve() for value in sft_values)
        if len(set(sft_paths)) != 3:
            raise ValueError("SFT train, validation and test inputs must be separate files.")
        if not isinstance(sft_source, str) or not sft_source.strip():
            raise ValueError("SFT source name must be a non-empty string.")
        if not isinstance(sft_license, str) or not sft_license.strip():
            raise ValueError("SFT license must be a non-empty string.")
    else:
        sft_paths = ()

    output = Path(output_dir).expanduser().resolve()
    source_paths = set(pretrain_paths) | set(sft_paths)
    if any(path == output or output in path.parents for path in source_paths):
        raise ValueError(
            "Dataset output folder cannot contain the input source files; "
            "use a separate output folder to protect the originals."
        )

    manifest_path = output / "dataset_manifest.json"
    managed_names = {
        "dataset_manifest.json",
        "pretrain_train.txt",
        "pretrain_validation.txt",
        "pretrain_test.txt",
        "sft_train.jsonl",
        "sft_validation.jsonl",
        "sft_test.jsonl",
    }
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if not overwrite and output.exists() and any((output / name).exists() for name in managed_names):
        raise FileExistsError(
            f"Dataset output folder {output} already contains prepared files. "
            "Choose a new folder or pass --overwrite after backing up the data."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        pretraining_splits = {}
        for split, source in zip(
            ("train", "validation", "test"),
            pretrain_paths,
        ):
            pretraining_splits[split] = _normalize_pretraining_file(
                source, staging / f"pretrain_{split}.txt"
            )
        normalized_split_hashes = {
            name: entry["sha256"]
            for name, entry in pretraining_splits.items()
        }
        if len(set(normalized_split_hashes.values())) != 3:
            raise ValueError(
                "Pretraining train/validation/test contain identical normalized split files."
            )

        instruction_splits: dict[str, Any] = {}
        instruction_sources: dict[str, Any] = {}
        approved_sources: list[str] = []
        if has_sft:
            train_meta = _curate_sft_file(
                sft_paths[0], staging / "sft_train.jsonl",
                str(sft_source), str(sft_license),
            )
            valid_meta = _curate_sft_file(
                sft_paths[1], staging / "sft_validation.jsonl",
                str(sft_source), str(sft_license),
            )
            test_meta = _curate_sft_file(
                sft_paths[2], staging / "sft_test.jsonl",
                str(sft_source), str(sft_license),
            )

            def conversation_fingerprints(path: Path) -> set[str]:
                return {
                    hashlib.sha256(
                        json.dumps(
                            record["messages"], ensure_ascii=False, sort_keys=True
                        ).encode("utf-8")
                    ).hexdigest()
                    for record in (
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                }

            split_paths = {
                "train": staging / "sft_train.jsonl",
                "validation": staging / "sft_validation.jsonl",
                "test": staging / "sft_test.jsonl",
            }
            fingerprints = {
                name: conversation_fingerprints(path)
                for name, path in split_paths.items()
            }
            split_names = tuple(fingerprints)
            for left_index, left_name in enumerate(split_names):
                for right_name in split_names[left_index + 1:]:
                    overlap = fingerprints[left_name] & fingerprints[right_name]
                    if overlap:
                        raise ValueError(
                            f"SFT {left_name}/{right_name} leakage: "
                            f"{len(overlap)} duplicate conversations."
                        )
            instruction_splits = {
                "train": train_meta,
                "validation": valid_meta,
                "test": test_meta,
            }
            instruction_sources = {
                str(sft_source): {
                    "license": str(sft_license),
                    "user_declared_human_authored": True,
                    "synthetic_records_included": 0,
                }
            }
            approved_sources = [str(sft_source)]

        manifest = {
            "format": "qmini-addsub-data-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "language": "English",
            "synthetic_training_records_included": 0,
            "source_attestation": {
                "rights_and_non_synthetic_confirmation": True,
                "english_language_confirmation": True,
                "pretraining_source": pretrain_source,
                "pretraining_license": pretrain_license,
                "split_integrity": (
                    "User supplied distinct train/validation/test files; document-level "
                    "leakage cannot be inferred from plain text files."
                ),
            },
            "cleaning_policy": {
                "version": "qmini-addsub-user-import-v1",
                "pretraining": (
                    "Unicode NFC, normalized line endings, per-line quality/sensitive "
                    "text rejection and trailing-space cleanup."
                ),
                "instruction": (
                    "Existing Add/Sub text cleaner; explicit human_authored=true and "
                    "synthetic=false required."
                ),
            },
            "pretraining": {
                "repository": pretrain_source,
                "subset": "user-selected local English text files",
                "license": pretrain_license,
                "source_attested_by_user": True,
                "synthetic_records_included": 0,
                "splits": pretraining_splits,
            },
            "instruction_sources": instruction_sources,
            "approved_instruction_sources": approved_sources,
            "instruction_splits": instruction_splits,
        }
        staged_manifest = staging / "dataset_manifest.json"
        temporary_manifest = staging / "dataset_manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(staged_manifest)

        # The manifest is the final commit marker. Consumers reject a folder
        # without it, so a failed replacement cannot be mistaken for a valid dataset.
        output.mkdir(parents=True, exist_ok=True)
        (output / "dataset_manifest.json").unlink(missing_ok=True)
        staged_names = {path.name for path in staging.iterdir() if path.is_file()}
        for name in sorted(staged_names - {"dataset_manifest.json"}):
            (staging / name).replace(output / name)
        if overwrite and not has_sft:
            for name in ("sft_train.jsonl", "sft_validation.jsonl", "sft_test.jsonl"):
                (output / name).unlink(missing_ok=True)
        staged_manifest.replace(output / "dataset_manifest.json")
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import user-selected local English datasets for Q-Mini Add/Sub v1; no downloads are performed."
    )
    parser.add_argument("--pretrain-train", required=True)
    parser.add_argument("--pretrain-validation", required=True)
    parser.add_argument("--pretrain-test", required=True)
    parser.add_argument("--pretrain-source", required=True)
    parser.add_argument("--pretrain-license", required=True)
    parser.add_argument("--sft-train")
    parser.add_argument("--sft-validation")
    parser.add_argument("--sft-test")
    parser.add_argument("--sft-source")
    parser.add_argument("--sft-license")
    parser.add_argument("--output", default="data/qmini_addsub_v1")
    parser.add_argument(
        "--confirm-rights-and-human-data",
        action="store_true",
        help="Confirm you have reviewed the source terms and supplied non-synthetic data.",
    )
    parser.add_argument(
        "--confirm-english-data",
        action="store_true",
        help="Confirm that all supplied pretraining and optional SFT records are English.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = import_user_dataset(
        pretrain_train=args.pretrain_train,
        pretrain_validation=args.pretrain_validation,
        pretrain_test=args.pretrain_test,
        output_dir=args.output,
        pretrain_source=args.pretrain_source,
        pretrain_license=args.pretrain_license,
        confirm_rights_and_human_data=args.confirm_rights_and_human_data,
        confirm_english_data=args.confirm_english_data,
        sft_train=args.sft_train,
        sft_validation=args.sft_validation,
        sft_test=args.sft_test,
        sft_source=args.sft_source,
        sft_license=args.sft_license,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "manifest": str(Path(args.output) / "dataset_manifest.json"),
        "pretraining_splits": manifest["pretraining"]["splits"],
        "instruction_splits": manifest["instruction_splits"],
        "synthetic_training_records_included": 0,
        "downloads_performed": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
