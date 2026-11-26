from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import build_source_provenance, expand_data_paths, iter_training_text_records
from .text import SPECIAL_TOKENS


def train_tokenizer(
    data_paths: Iterable[str | Path],
    output_path: str | Path,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    overwrite: bool = False,
    validation_fraction: float = 0.02,
    near_duplicate_distance: int = 0,
    source_metadata_path: str | Path | None = None,
    release_mode: bool = False,
) -> Tokenizer:
    """Train byte-level BPE on unique training documents, excluding validation docs."""
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Tokenizer already exists: {target}; pass overwrite=True to replace it."
        )
    minimum_vocab = 256 + len(SPECIAL_TOKENS)
    if vocab_size < minimum_vocab:
        raise ValueError(
            f"Byte-level BPE vocab_size must be at least {minimum_vocab} "
            "to reserve the complete byte alphabet and Q-Mini control tokens."
        )
    if min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer.")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError(
            "validation_fraction must be greater than 0 and below 0.5; "
            "tokenizer fitting must exclude a non-empty validation holdout."
        )
    paths = expand_data_paths(data_paths)
    if not paths:
        raise FileNotFoundError("No supported dataset files were found.")
    source_files, source_provenance_sha256, sensitive_records_screened = (
        build_source_provenance(
            paths, metadata_path=source_metadata_path, release_mode=release_mode
        )
    )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_training_text_records(
            paths,
            valid_fraction=validation_fraction,
            near_duplicate_distance=near_duplicate_distance,
        ),
        trainer,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    manifest_path = Path(str(target) + ".qmini.json")
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        tokenizer.save(str(temporary))
        digest = hashlib.sha256()
        with temporary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        manifest = {
            "format": "qmini-tokenizer-manifest-v1",
            "tokenizer_sha256": digest.hexdigest(),
            "validation_fraction": float(validation_fraction),
            "near_duplicate_hamming_distance": int(near_duplicate_distance),
            "source_provenance_sha256": source_provenance_sha256,
            "source_provenance_enforced": bool(release_mode or source_metadata_path is not None),
            "source_files": source_files,
            "release_mode": bool(release_mode),
            "sensitive_records_screened": sensitive_records_screened,
        }
        with manifest_temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.replace(manifest_temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {target}")
    return Tokenizer.from_file(str(target))


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer lacks required token {token!r}.")
    return int(value)
