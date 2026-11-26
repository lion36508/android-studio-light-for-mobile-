from __future__ import annotations

import argparse
import json
from pathlib import Path

from qmini_addsub.train import encode_text_file, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode Add/Sub v1 pretraining train/validation text into uint16 token streams."
    )
    parser.add_argument("--data-dir", default="data/qmini_addsub_v1")
    parser.add_argument("--tokenizer", default="artifacts/qmini_addsub_v1/tokenizer.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = Path(args.tokenizer)
    if not tokenizer.is_file():
        raise FileNotFoundError("Train the Add/Sub v1 tokenizer first.")
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("A dataset_manifest.json is required beside the source text splits.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "qmini-addsub-data-v1"
        or manifest.get("language") != "English"
        or manifest.get("synthetic_training_records_included") != 0
    ):
        raise ValueError("Only verified English, non-synthetic Add/Sub data can be encoded.")

    outputs = {}
    tokenizer_sha = sha256_file(tokenizer)
    for split in ("train", "validation", "test"):
        source = data_dir / f"pretrain_{split}.txt"
        target = data_dir / f"pretrain_{split}.u16"
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_source_sha = (
            manifest.get("pretraining", {}).get("splits", {}).get(split, {}).get("sha256")
        )
        if expected_source_sha != sha256_file(source):
            raise ValueError(f"{split} source text does not match the dataset manifest.")
        count, digest, source_bytes = encode_text_file(source, tokenizer, target)
        outputs[split] = {
            "tokens": count,
            "sha256": digest,
            "path": target.name,
            "source_sha256": expected_source_sha,
            "source_bytes": source_bytes,
        }

    manifest["token_streams"] = outputs
    manifest["tokenizer"] = {
        "path": tokenizer.name,
        "sha256": tokenizer_sha,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(outputs)


if __name__ == "__main__":
    main()
