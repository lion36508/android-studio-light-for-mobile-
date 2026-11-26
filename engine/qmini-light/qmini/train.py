from __future__ import annotations

import hashlib
import json
import math
import os
import re
import random
import shutil
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import QMiniConfig
from .model import QMiniMoE


def _sha256_file(path: str | Path) -> dict[str, int | str]:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": target.stat().st_size}


def _load_tokens(path: str | Path, name: str) -> np.memmap:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{name} token stream not found: {target}")
    if target.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"{name} token stream is not aligned to uint16: {target}")
    values = np.memmap(target, mode="r", dtype=np.uint16)
    if values.size < 4:
        raise ValueError(f"{name} token stream is too short: {target}")
    return values


def _validate_token_ids(tokens: np.memmap, vocab_size: int, name: str) -> None:
    """Scan a memmap in bounded chunks; fail before training if IDs exceed vocab."""
    chunk_size = 1 << 20
    observed_max = 0
    for start in range(0, len(tokens), chunk_size):
        chunk = np.asarray(tokens[start:start + chunk_size])
        if chunk.size:
            observed_max = max(observed_max, int(chunk.max()))
    if observed_max >= vocab_size:
        raise ValueError(
            f"{name} stream contains token ID {observed_max}, which is outside the configured vocabulary "
            f"[0, {vocab_size - 1}]. "
            "Rebuild token streams with the tokenizer paired to this model config."
        )


def _load_masks(path: str | Path | None, tokens: np.memmap, name: str) -> np.memmap | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{name} loss-mask file not found: {target}")
    if target.stat().st_size != len(tokens):
        raise ValueError(
            f"{name} loss mask has {target.stat().st_size} bytes, but token stream has {len(tokens)} tokens."
        )
    masks = np.memmap(target, mode="r", dtype=np.uint8)
    chunk_size = 1 << 20
    target_count = 0
    for start in range(0, len(masks), chunk_size):
        chunk = np.asarray(masks[start:start + chunk_size])
        if not np.isin(chunk, (0, 1)).all():
            raise ValueError(f"{name} loss mask must contain only 0 and 1.")
        target_count += int(chunk.sum())
    if target_count == 0:
        raise ValueError(f"{name} loss mask contains no supervised target tokens.")
    return masks


class SupervisedWindowIndex:
    """Disk-backed, bounded-memory index of windows containing target tokens.

    The index stores only one count per start-position block. During sampling,
    a selected block is scanned once to choose uniformly among its valid
    windows. This preserves uniform sampling over all valid windows without
    retaining every positive start position in RAM.
    """

    def __init__(
        self,
        masks: np.ndarray,
        token_count: int,
        seq_len: int,
        block_size: int = 1 << 20,
    ):
        if seq_len < 1 or token_count < seq_len + 2:
            raise ValueError("A supervised window index requires a legal sequence window.")
        if len(masks) != token_count:
            raise ValueError("Loss-mask length must match its token stream.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")

        self.masks = masks
        self.seq_len = int(seq_len)
        self.max_start = int(token_count - seq_len - 1)
        self.block_size = int(block_size)
        self.blocks: list[tuple[int, int]] = []
        self.valid_window_counts: list[int] = []

        for start in range(0, self.max_start + 1, self.block_size):
            count = min(self.block_size, self.max_start + 1 - start)
            self.blocks.append((start, count))
            self.valid_window_counts.append(
                int(self._valid_offsets(start, count).size)
            )

        self.total_valid_windows = sum(self.valid_window_counts)
        if self.total_valid_windows == 0:
            raise ValueError(
                "Loss mask has no supervised target inside any legal training window. "
                "Move at least one target to positions 1..N-1 or increase seq_len."
            )

        weights = np.asarray(self.valid_window_counts, dtype=np.float64)
        self.block_probabilities = weights / weights.sum()

    def _valid_offsets(self, block_start: int, start_count: int) -> np.ndarray:
        # For start s, labels are masks[s+1:s+seq_len+1]. This single
        # bounded read contains the union of all windows in the selected block.
        values = np.asarray(
            self.masks[
                block_start + 1:block_start + start_count + self.seq_len
            ],
            dtype=np.uint8,
        )
        prefix = np.empty(values.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(values, dtype=np.int64, out=prefix[1:])
        counts = prefix[self.seq_len:self.seq_len + start_count] - prefix[:start_count]
        return np.flatnonzero(counts > 0)

    def sample_start(self, rng: np.random.Generator) -> int:
        block_index = int(
            rng.choice(len(self.blocks), p=self.block_probabilities)
        )
        block_start, start_count = self.blocks[block_index]
        valid_offsets = self._valid_offsets(block_start, start_count)
        if valid_offsets.size == 0:
            raise RuntimeError(
                "Supervised-window index changed after construction; "
                "the loss mask may have been modified."
            )
        offset = int(valid_offsets[int(rng.integers(0, valid_offsets.size))])
        return block_start + offset


def _sample_batch(
    tokens: np.memmap,
    masks: np.memmap | None,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: torch.device,
    window_index: SupervisedWindowIndex | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - seq_len - 1
    if max_start < 0:
        raise ValueError(
            f"Token stream needs at least {seq_len + 2} tokens; found {len(tokens)}."
        )

    samples = []
    target_masks = []
    if masks is not None:
        window_index = window_index or SupervisedWindowIndex(
            masks, len(tokens), seq_len
        )
        if window_index.max_start != max_start or window_index.seq_len != seq_len:
            raise ValueError("Supervised-window index does not match this token stream and sequence length.")

    for _row in range(batch_size):
        start = (
            int(rng.integers(0, max_start + 1))
            if masks is None
            else window_index.sample_start(rng)
        )
        label_mask = (
            np.ones(seq_len, dtype=np.uint8)
            if masks is None
            else np.asarray(masks[start + 1:start + seq_len + 1], dtype=np.uint8)
        )
        if int(label_mask.sum()) <= 0:
            raise RuntimeError("Selected training window unexpectedly contains no supervised targets.")
        samples.append(np.asarray(tokens[start:start + seq_len + 1], dtype=np.int64))
        target_masks.append(label_mask)

    batch = np.stack(samples)
    x = torch.as_tensor(batch[:, :-1], dtype=torch.long, device=device)
    y = torch.as_tensor(batch[:, 1:], dtype=torch.long, device=device)
    loss_mask = torch.as_tensor(np.stack(target_masks), dtype=torch.float32, device=device)
    return x, y, loss_mask


def _masked_cross_entropy_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    weights = target_mask.to(dtype=per_token.dtype)
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("A sampled batch contains no supervised target tokens.")
    return (per_token * weights).sum(), denominator


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    numerator, denominator = _masked_cross_entropy_sum(logits, targets, target_mask)
    return numerator / denominator


def _evaluate(
    model: QMiniMoE,
    tokens: np.memmap,
    masks: np.memmap | None,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    seed: int,
    window_index: SupervisedWindowIndex | None = None,
    thinking_steps: int | None = None,
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    total_negative_log_likelihood = 0.0
    total_supervised_targets = 0.0
    evaluation_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for _ in range(batches):
                x, y, target_mask = _sample_batch(
                    tokens, masks, batch_size, seq_len, rng, device,
                    window_index=window_index,
                )
                result = (
                    model(x)
                    if thinking_steps is None
                    else model(x, thinking_steps=thinking_steps)
                )
                if not torch.isfinite(result["logits"]).all():
                    raise FloatingPointError(
                        f"Non-finite logits at thinking_steps={thinking_steps!r}."
                    )
                nll_sum, target_count = _masked_cross_entropy_sum(
                    result["logits"], y, target_mask
                )
                if not torch.isfinite(nll_sum):
                    raise FloatingPointError("Validation loss became non-finite.")
                total_negative_log_likelihood += float(nll_sum.item())
                total_supervised_targets += float(target_count.item())
    finally:
        model.train(was_training)

    if total_supervised_targets <= 0:
        raise ValueError("Validation run contains no supervised target tokens.")
    mean_loss = total_negative_log_likelihood / total_supervised_targets
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else None
    evaluation_elapsed = max(time.perf_counter() - evaluation_started, 1e-9)
    return {
        "eval_loss": mean_loss,
        "eval_perplexity": perplexity,
        "eval_supervised_tokens": int(total_supervised_targets),
        "eval_elapsed_seconds": float(evaluation_elapsed),
        "eval_supervised_tokens_per_second": float(
            total_supervised_targets / evaluation_elapsed
        ),
        "eval_thinking_steps": (
            int(thinking_steps) if thinking_steps is not None
            else int(model.config.thinking_steps)
        ),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_initial_model_artifact(
    path: str | Path,
    config: QMiniConfig,
    tokenizer_path: str | Path | None,
) -> dict[str, Any]:
    """Require a hash-verified Q-Mini final_model artifact for SFT initialization.

    A matching tensor shape is not evidence of model ownership. The source must
    be a Q-Mini final_model directory whose manifest, config, packed export and
    selected dense/packed artifact agree. When a tokenizer is supplied, it must
    be the exact tokenizer saved with that source checkpoint.
    """
    source_path = Path(path).expanduser().resolve()
    source_dir = source_path.parent
    manifest_path = source_dir / "model_manifest.json"
    config_path = source_dir / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError(
            "SFT initialization requires a verified Q-Mini final_model directory "
            "containing model_manifest.json and config.json."
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("Q-Mini source manifest must be a JSON object.")
        source_config = QMiniConfig.from_json(config_path)
        manifest_config = QMiniConfig.from_dict(manifest.get("config"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Q-Mini source manifest/config is invalid.") from exc

    if manifest.get("format") != "qmini-final-model-v2":
        raise ValueError("Unsupported Q-Mini final-model manifest format.")
    if source_config.to_dict() != manifest_config.to_dict():
        raise ValueError("Q-Mini source manifest config does not match config.json.")
    if source_config.to_dict() != config.to_dict():
        raise ValueError(
            "Q-Mini SFT source architecture does not match the requested model config."
        )
    expected_parameters = source_config.estimated_parameter_count()
    if manifest.get("parameters") != expected_parameters:
        raise ValueError("Q-Mini source manifest parameter count is inconsistent.")
    if (
        isinstance(manifest.get("step"), bool)
        or not isinstance(manifest.get("step"), int)
        or manifest["step"] < 1
        or isinstance(manifest.get("tokens_seen"), bool)
        or not isinstance(manifest.get("tokens_seen"), int)
        or manifest["tokens_seen"] < 1
    ):
        raise ValueError("Q-Mini SFT source must record a completed training step and tokens_seen.")

    packed_name = manifest.get("packed_export")
    if (
        not isinstance(packed_name, str)
        or not packed_name
        or Path(packed_name).name != packed_name
    ):
        raise ValueError("Q-Mini source manifest has an unsafe packed-export filename.")
    packed_path = source_dir / packed_name
    if not packed_path.is_file():
        raise ValueError("Q-Mini source packed export is missing.")
    packed_fingerprint = _sha256_file(packed_path)
    if (
        manifest.get("packed_export_sha256") != packed_fingerprint["sha256"]
        or manifest.get("packed_export_bytes") != packed_fingerprint["size_bytes"]
    ):
        raise ValueError("Q-Mini source packed export SHA-256/size does not match its manifest.")

    from .export import verify_packed_export

    packed_report = verify_packed_export(packed_path)
    if (
        packed_report.get("format") != "qmini-packed-v2"
        or packed_report.get("q2_version") != 2
        or packed_report.get("parameter_count") != expected_parameters
    ):
        raise ValueError(
            "Q-Mini SFT source must contain a verified Q2-v2 two-bit packed export "
            "with the expected parameter count."
        )

    if source_path.suffix.lower() == ".pt":
        raise ValueError(
            "Dense .pt SFT initialization is disabled: current Q-Mini final-model "
            "exports are Q2-v2 .npz-only. Use the manifest-listed model.qmini.npz."
        )
    elif source_path.suffix.lower() == ".npz":
        if source_path != packed_path.resolve():
            raise ValueError(
                "Packed SFT initialization must use the manifest-listed Q-Mini packed export."
            )
    else:
        raise ValueError(
            "SFT --init-from accepts only the manifest-listed Q2-v2 "
            "final_model/model.qmini.npz artifact."
        )

    if tokenizer_path is not None:
        target_tokenizer = Path(tokenizer_path).expanduser().resolve()
        source_tokenizer = source_dir / "tokenizer.json"
        tokenizer_hash = manifest.get("tokenizer_sha256")
        if (
            not source_tokenizer.is_file()
            or not isinstance(tokenizer_hash, str)
            or not tokenizer_hash
        ):
            raise ValueError(
                "Q-Mini SFT source has no verified tokenizer; pretrain with --tokenizer first."
            )
        source_tokenizer_hash = _sha256_file(source_tokenizer)["sha256"]
        target_tokenizer_hash = _sha256_file(target_tokenizer)["sha256"]
        if (
            source_tokenizer_hash != tokenizer_hash
            or target_tokenizer_hash != tokenizer_hash
        ):
            raise ValueError(
                "SFT tokenizer must exactly match the tokenizer saved with the Q-Mini source checkpoint."
            )

    return manifest



def _verify_initial_brain_artifact(
    path: str | Path,
    config: QMiniConfig,
) -> dict[str, Any]:
    """Verify a step-zero Q2-v2 initialization artifact for pretraining."""
    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() != ".npz":
        raise ValueError("--init-from-initial accepts only model.qmini.npz.")
    source_dir = source_path.parent
    manifest_path = source_dir / "model_manifest.json"
    config_path = source_dir / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError(
            "Initial-brain initialization requires model_manifest.json and config.json."
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        source_config = QMiniConfig.from_json(config_path)
        manifest_config = QMiniConfig.from_dict(manifest.get("config"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Q-Mini initial-brain manifest/config is invalid.") from exc

    if manifest.get("format") != "qmini-initial-brain-v1":
        raise ValueError("Unsupported Q-Mini initial-brain manifest format.")
    if manifest.get("status") != "untrained_initialization":
        raise ValueError("Initial-brain source is not marked as an untrained initialization.")
    if source_config.to_dict() != manifest_config.to_dict():
        raise ValueError("Initial-brain manifest config does not match config.json.")
    if source_config.to_dict() != config.to_dict():
        raise ValueError("Initial-brain architecture does not match the requested training config.")
    expected_parameters = source_config.estimated_parameter_count()
    if manifest.get("parameters") != expected_parameters:
        raise ValueError("Initial-brain parameter count does not match config.")
    for field in ("training_steps", "optimizer_steps", "tokens_seen"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError(f"Initial-brain manifest must record {field}=0.")
    if manifest.get("matrix_weight_format") != "qmini-packed-v2" or manifest.get("matrix_weight_bits") != 2:
        raise ValueError("Initial-brain source is not the required Q2-v2 two-bit format.")

    packed_name = manifest.get("packed_export")
    if (
        not isinstance(packed_name, str)
        or not packed_name
        or Path(packed_name).name != packed_name
    ):
        raise ValueError("Initial-brain manifest has an unsafe packed-export filename.")
    packed_path = source_dir / packed_name
    if packed_path.resolve() != source_path or not packed_path.is_file():
        raise ValueError("Initial-brain path is not the manifest-listed packed artifact.")
    fingerprint = _sha256_file(packed_path)
    if (
        manifest.get("packed_export_sha256") != fingerprint["sha256"]
        or manifest.get("packed_export_bytes") != fingerprint["size_bytes"]
    ):
        raise ValueError("Initial-brain packed artifact SHA-256/size does not match its manifest.")

    from .export import verify_packed_export
    report = verify_packed_export(packed_path)
    if (
        report.get("format") != "qmini-packed-v2"
        or report.get("q2_version") != 2
        or report.get("parameter_count") != expected_parameters
    ):
        raise ValueError("Initial-brain packed artifact failed Q2-v2 verification.")
    return manifest


def _verify_prepared_dataset_manifest(
    train_path: str | Path,
    valid_path: str | Path,
    tokenizer_path: str | Path | None,
    train_mask_path: str | Path | None,
    valid_mask_path: str | Path | None,
    vocab_size: int,
) -> Path | None:
    """Verify v3 preparation fingerprints and enforce assistant-loss masks.

    Raw uint16 streams without a dataset manifest remain supported for advanced
    callers. A v3 manifest, however, is a binding contract: every supplied file
    must match its SHA-256, tokenizer vocabulary must match, and auto/assistant
    chat datasets may not be trained without their response-only masks.
    """
    train_file = Path(train_path).resolve()
    valid_file = Path(valid_path).resolve()
    manifest_path = train_file.parent / "dataset.json"
    if not manifest_path.is_file():
        return None
    if valid_file.parent != train_file.parent:
        raise ValueError(
            "Prepared train and validation streams must come from the same directory manifest."
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prepared dataset manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "qmini-token-stream-v3":
        raise ValueError(
            "Unsupported or incomplete dataset manifest. Re-prepare into a new versioned "
            "output directory rather than modifying existing artifacts."
        )

    if int(manifest.get("vocab_size", -1)) != int(vocab_size):
        raise ValueError(
            "Dataset manifest vocabulary does not match model config. "
            "Use the active config saved alongside this tokenizer."
        )
    if tokenizer_path is None or not Path(tokenizer_path).is_file():
        raise ValueError(
            "A tokenizer file is required when training from a prepared dataset manifest."
        )
    actual_tokenizer_hash = _sha256_file(tokenizer_path)["sha256"]
    if manifest.get("tokenizer_sha256") != actual_tokenizer_hash:
        raise ValueError("Tokenizer SHA-256 does not match the prepared dataset manifest.")

    tokenizer_sidecar = Path(str(tokenizer_path) + ".qmini.json")
    if tokenizer_sidecar.is_file():
        try:
            sidecar = json.loads(tokenizer_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read tokenizer split manifest: {tokenizer_sidecar}"
            ) from exc
        if sidecar.get("format") != "qmini-tokenizer-manifest-v1":
            raise ValueError("Unsupported tokenizer split manifest format.")
        if sidecar.get("tokenizer_sha256") != actual_tokenizer_hash:
            raise ValueError("Tokenizer split manifest does not match tokenizer.json.")
        if sidecar.get("validation_fraction") != float(
            manifest.get("valid_fraction_requested", -1)
        ):
            raise ValueError(
                "Tokenizer and prepared dataset use different validation fractions."
            )
        if sidecar.get("near_duplicate_hamming_distance") != int(
            manifest.get("near_duplicate_hamming_distance", 0)
        ):
            raise ValueError(
                "Tokenizer and prepared dataset use different near-duplicate thresholds."
            )
        expected_provenance = manifest.get("source_provenance_sha256")
        sidecar_provenance = sidecar.get("source_provenance_sha256")
        if manifest.get("source_provenance_enforced") is True:
            if expected_provenance is None or sidecar_provenance != expected_provenance:
                raise ValueError(
                    "Tokenizer and prepared dataset source provenance hashes differ."
                )
    elif manifest.get("source_provenance_enforced") is True:
        raise ValueError(
            "Prepared dataset provenance cannot be verified without its tokenizer sidecar."
        )

    if manifest.get("source_provenance_sha256") is not None:
        source_files = manifest.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError("Prepared dataset is missing its source provenance file list.")
        canonical_sources = []
        source_keys = (
            "source_index", "path", "size_bytes", "sha256",
            "source_name", "source_uri", "license_spdx",
            "collected_at", "language", "transformation_version",
        )
        for source in source_files:
            if not isinstance(source, dict):
                raise ValueError("Prepared dataset source provenance entries must be objects.")
            canonical = {key: source.get(key) for key in source_keys}
            canonical["document_count"] = None
            canonical_sources.append(canonical)
        actual_provenance = hashlib.sha256(
            json.dumps(
                canonical_sources, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if actual_provenance != manifest.get("source_provenance_sha256"):
            raise ValueError("Prepared dataset source provenance hash is invalid.")

    if manifest.get("release_mode") is True:
        provenance_hash = manifest.get("source_provenance_sha256")
        source_files = manifest.get("source_files")
        if (
            manifest.get("source_provenance_enforced") is not True
            or not isinstance(provenance_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", provenance_hash)
            or not isinstance(source_files, list)
            or not source_files
        ):
            raise ValueError("Release dataset manifest is missing a valid source provenance record.")
        if not tokenizer_sidecar.is_file():
            raise ValueError("Release training requires a release-mode tokenizer manifest.")
        if sidecar.get("release_mode") is not True:
            raise ValueError("Release training requires tokenizer fitting with --release-mode.")
        for source in source_files:
            if not isinstance(source, dict):
                raise ValueError("Release dataset source provenance entries must be objects.")
            required = (
                "source_name", "source_uri", "license_spdx", "collected_at",
                "language", "transformation_version", "sha256", "size_bytes",
                "document_count",
            )
            if any(source.get(key) in (None, "", "UNKNOWN", "unknown") for key in required):
                raise ValueError("Release dataset contains incomplete or unknown source provenance.")
            if str(source["license_spdx"]).casefold() in {"unknown", "unlicensed", "none", "n/a"}:
                raise ValueError("Release dataset contains an unknown or unapproved source license.")

    artifacts = [
        ("train_tokens_sha256", train_file, "Training token stream"),
        ("valid_tokens_sha256", valid_file, "Validation token stream"),
    ]
    for field, artifact_path, label in artifacts:
        expected = manifest.get(field)
        actual = _sha256_file(artifact_path)["sha256"]
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(f"{label} SHA-256 does not match the prepared dataset manifest.")

    loss_mode = manifest.get("loss_mask_mode")
    if loss_mode not in {"auto", "all", "assistant"}:
        raise ValueError(f"Unsupported loss_mask_mode in dataset manifest: {loss_mode!r}.")
    masks_required = loss_mode in {"auto", "assistant"}
    if masks_required and (train_mask_path is None or valid_mask_path is None):
        raise ValueError(
            f"Prepared dataset uses loss_mask_mode={loss_mode!r}; both train and validation "
            "loss-mask files are mandatory. Refusing to silently train on user/system prompts."
        )

    mask_artifacts = [
        ("train_mask_sha256", train_mask_path, "Training loss mask"),
        ("valid_mask_sha256", valid_mask_path, "Validation loss mask"),
    ]
    for field, artifact_path, label in mask_artifacts:
        if artifact_path is None:
            continue
        expected = manifest.get(field)
        actual = _sha256_file(artifact_path)["sha256"]
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(f"{label} SHA-256 does not match the prepared dataset manifest.")

    return manifest_path


def _resume_contract(
    train_path: str | Path,
    valid_path: str | Path,
    tokenizer_path: str | Path | None,
    train_mask_path: str | Path | None,
    valid_mask_path: str | Path | None,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    initial_model_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    train_file = Path(train_path).resolve()
    valid_file = Path(valid_path).resolve()
    train_fingerprint = _sha256_file(train_file)
    valid_fingerprint = _sha256_file(valid_file)
    if train_file == valid_file or train_fingerprint == valid_fingerprint:
        raise ValueError("Training and validation streams must be different files with different content.")
    tokenizer_fingerprint = (
        _sha256_file(tokenizer_path) if tokenizer_path is not None and Path(tokenizer_path).is_file()
        else None
    )
    contract = {
        "train": train_fingerprint,
        "validation": valid_fingerprint,
        "train_loss_mask": _sha256_file(train_mask_path) if train_mask_path else None,
        "validation_loss_mask": _sha256_file(valid_mask_path) if valid_mask_path else None,
        "tokenizer": tokenizer_fingerprint,
        "batch_size": int(batch_size),
        "sequence_length": int(seq_len),
        "gradient_accumulation": int(grad_accum),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "initial_model": (
            _sha256_file(initial_model_path)
            if initial_model_path is not None else None
        ),
        "initial_model_path": (
            str(Path(initial_model_path).resolve())
            if initial_model_path is not None else None
        ),
    }
    if dataset_manifest_path is not None:
        contract["dataset_manifest"] = _sha256_file(dataset_manifest_path)
    return contract


def _resume_contract_for_comparison(
    contract: dict[str, Any],
    *,
    allow_batch_size_change: bool = False,
) -> dict[str, Any]:
    """Normalize microbatching and optionally authorize an effective-batch change.

    By default, a checkpoint may resume with a different (batch_size, grad_accum)
    pair only when their product is identical. A deliberate performance-tuning
    run may explicitly authorize a different effective batch; every data,
    tokenizer, sequence, seed and optimizer setting remains strictly checked.
    """
    normalized = dict(contract)
    batch_size = normalized.get("batch_size")
    grad_accum = normalized.get("gradient_accumulation")
    valid_batch = (
        isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0
    )
    valid_accum = (
        isinstance(grad_accum, int) and not isinstance(grad_accum, bool) and grad_accum > 0
    )
    if valid_batch and valid_accum:
        normalized.pop("batch_size")
        normalized.pop("gradient_accumulation")
        if not allow_batch_size_change:
            normalized["effective_batch_size"] = batch_size * grad_accum
    return normalized


def _reconcile_metrics(path: Path, checkpoint_step: int) -> None:
    """Discard log rows newer than the last durable checkpoint after an interrupted run."""
    if not path.is_file():
        return
    kept: list[str] = []
    changed = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                step = int(record["step"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                changed = True
                continue
            if step <= checkpoint_step:
                kept.append(json.dumps(record, allow_nan=False))
            else:
                changed = True
    if changed:
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for line in kept:
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def train_model(
    config: QMiniConfig,
    train_bin: str | Path,
    valid_bin: str | Path,
    output_dir: str | Path,
    steps: int,
    batch_size: int = 4,
    seq_len: int = 512,
    grad_accum: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    eval_every: int = 250,
    save_every: int = 250,
    eval_batches: int = 20,
    resume: bool = False,
    tokenizer_path: str | Path | None = None,
    seed: int | None = None,
    extend_schedule: bool = False,
    allow_legacy_resume: bool = False,
    train_mask_bin: str | Path | None = None,
    valid_mask_bin: str | Path | None = None,
    init_from: str | Path | None = None,
    init_from_initial: str | Path | None = None,
    checkpoint_mirror_dir: str | Path | None = None,
    checkpoint_session_id: str | None = None,
    allow_batch_size_change: bool = False,
    allow_dataset_change: bool = False,
) -> Path:
    """Pretrain from scratch/step-zero Q2 brain, SFT, deliberate batch tuning, or an audited dataset switch on resume."""
    config.validate()
    positive = {
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "eval_every": eval_every,
        "save_every": save_every,
        "eval_batches": eval_batches,
    }
    invalid_types = [
        name for name, value in positive.items()
        if isinstance(value, bool) or not isinstance(value, int)
    ]
    if invalid_types:
        raise TypeError(
            "These training settings must be integers: " + ", ".join(invalid_types)
        )
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("These training settings must be positive: " + ", ".join(invalid))
    if not 2 <= seq_len <= config.max_context:
        raise ValueError(f"seq_len must be between 2 and {config.max_context}.")

    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer or None.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
    for name, value in (
        ("resume", resume),
        ("extend_schedule", extend_schedule),
        ("allow_legacy_resume", allow_legacy_resume),
        ("allow_batch_size_change", allow_batch_size_change),
        ("allow_dataset_change", allow_dataset_change),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    if allow_dataset_change and not resume:
        raise ValueError(
                "--allow-dataset-change only applies when resuming a checkpoint "
            "with --resume."
        )

    if (train_mask_bin is None) != (valid_mask_bin is None):
        raise ValueError("Provide both --train-mask-bin and --valid-mask-bin, or neither.")
    if tokenizer_path is not None and not Path(tokenizer_path).is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if checkpoint_mirror_dir is not None:
        if not isinstance(checkpoint_session_id, str) or not checkpoint_session_id.strip():
            raise ValueError("checkpoint_session_id is required when checkpoint_mirror_dir is enabled.")
    elif checkpoint_session_id is not None:
        raise ValueError("checkpoint_session_id cannot be set without checkpoint_mirror_dir.")

    checkpoint_mirror_path = (
        Path(checkpoint_mirror_dir).expanduser().resolve()
        if checkpoint_mirror_dir is not None else None
    )

    if resume and (init_from is not None or init_from_initial is not None):
        raise ValueError("--resume cannot be combined with --init-from or --init-from-initial.")
    if init_from is not None and init_from_initial is not None:
        raise ValueError("Choose either --init-from (trained SFT source) or --init-from-initial (step-zero Q2 brain), not both.")
    initial_model_path = Path(init_from).expanduser().resolve() if init_from is not None else None
    initial_brain_path = (
        Path(init_from_initial).expanduser().resolve()
        if init_from_initial is not None else None
    )
    if resume and initial_model_path is None and initial_brain_path is None:
        previous_run_path = Path(output_dir).expanduser() / "run.json"
        if previous_run_path.is_file():
            try:
                with previous_run_path.open("r", encoding="utf-8") as handle:
                    previous_run = json.load(handle)
                previous_initial = previous_run.get("initial_model")
                if previous_initial and previous_initial.get("path"):
                    restored_path = Path(previous_initial["path"]).expanduser().resolve()
                    if previous_initial.get("kind") == "q2-step-zero-initial":
                        initial_brain_path = restored_path
                    else:
                        initial_model_path = restored_path
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                raise ValueError(f"Could not recover the initialization source from {previous_run_path}.")
    initial_source_path = initial_brain_path or initial_model_path
    if initial_source_path is not None and not initial_source_path.is_file():
        raise FileNotFoundError(f"Initial model weights not found: {initial_source_path}")
    if initial_model_path is not None:
        _verify_initial_model_artifact(initial_model_path, config, tokenizer_path)
    if initial_brain_path is not None:
        _verify_initial_brain_artifact(initial_brain_path, config)
    train_tokens = _load_tokens(train_bin, "Training")
    valid_tokens = _load_tokens(valid_bin, "Validation")
    dataset_manifest_path = _verify_prepared_dataset_manifest(
        train_bin, valid_bin, tokenizer_path,
        train_mask_bin, valid_mask_bin, config.vocab_size,
    )
    train_masks = _load_masks(train_mask_bin, train_tokens, "Training")
    valid_masks = _load_masks(valid_mask_bin, valid_tokens, "Validation")
    _validate_token_ids(train_tokens, config.vocab_size, "Training")
    _validate_token_ids(valid_tokens, config.vocab_size, "Validation")
    # Fail early before model allocation when a batch cannot be sampled.
    if len(train_tokens) < seq_len + 2 or len(valid_tokens) < seq_len + 2:
        raise ValueError(
            f"Both streams need at least {seq_len + 2} tokens for sequence length {seq_len}."
        )

    # Build bounded-memory indexes once, before allocating the model/optimizer.
    train_window_index = (
        SupervisedWindowIndex(train_masks, len(train_tokens), seq_len)
        if train_masks is not None else None
    )
    valid_window_index = (
        SupervisedWindowIndex(valid_masks, len(valid_tokens), seq_len)
        if valid_masks is not None else None
    )

    actual_seed = config.seed if seed is None else seed
    contract = _resume_contract(
        train_bin, valid_bin, tokenizer_path, train_mask_bin, valid_mask_bin,
        batch_size, seq_len, grad_accum, learning_rate, weight_decay, actual_seed,
        initial_source_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .preflight import check_training_memory_budget
    memory_preflight = check_training_memory_budget(config, device)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    run_manifest_path = run_dir / "run.json"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint exists: {checkpoint_path}; pass --resume or choose a new run directory."
        )
    if metrics_path.exists() and not resume and not checkpoint_path.exists():
        raise FileExistsError(
            f"Metrics log already exists: {metrics_path}; refusing to mix unrelated training runs."
        )

    random.seed(actual_seed)
    rng = np.random.default_rng(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    model = QMiniMoE(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    if initial_source_path is not None and not resume:
        if initial_brain_path is not None or initial_source_path.suffix.lower() == ".npz":
            from .packed_runtime import load_packed_state_into_trainable_model
            load_packed_state_into_trainable_model(model, initial_source_path)
        else:
            # Never fall back to unrestricted pickle loading for model files.
            initial_state = torch.load(
                initial_source_path, map_location=device, weights_only=True
            )
            if not isinstance(initial_state, dict) or "format" in initial_state or "model" in initial_state:
                raise ValueError(
                    "--init-from must point to final_model/model.pt or a verified "
                    "final_model/model.qmini.npz, not a training checkpoint."
                )
            model.load_state_dict(initial_state, strict=True)
        stage = "QAT pretraining from step-zero Q2-v2 initialization" if initial_brain_path is not None else "fine-tuning from a trained Q-Mini checkpoint"
        print(
            f"Initialized model weights from {initial_source_path}; "
            f"{stage} starts with a fresh optimizer."
        )
    use_amp = device.type == "cuda"
    # Prefer BF16 only on Ampere-class (SM 8.0+) or newer GPUs where it is
    # natively supported. PyTorch can report BF16 as available through
    # emulation on older cards such as T4; FP16 is the native fast path there.
    bf16_supported = False
    if use_amp and hasattr(torch.cuda, "is_bf16_supported"):
        compute_major, _compute_minor = torch.cuda.get_device_capability(device)
        bf16_supported = bool(
            compute_major >= 8 and torch.cuda.is_bf16_supported()
        )
    use_bf16 = bool(use_amp and bf16_supported)
    use_fp16 = bool(use_amp and not use_bf16)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    start_step = 0
    tokens_seen = 0
    supervised_tokens_seen = 0
    thinking_step_counts = {
        str(step_count): 0
        for step_count in range(1, config.max_thinking_steps + 1)
    }
    schedule_total_steps = steps
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        # Checkpoints contain tensors and primitive metadata; use the restricted
        # unpickler so resume never executes arbitrary pickle globals.
        state = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )

        if state.get("format") not in {"qmini-checkpoint-v1", "qmini-checkpoint-v2"}:
            raise ValueError("Unsupported or unrecognized Q-Mini checkpoint format.")
        saved_config = QMiniConfig.from_dict(state["config"])
        if saved_config.to_dict() != config.to_dict():
            raise ValueError("Checkpoint model config differs; refusing an unsafe resume.")

        saved_contract = state.get("resume_contract")
        if saved_contract is None:
            if not allow_legacy_resume:
                raise ValueError(
                    "This checkpoint has no dataset/training fingerprint. "
                    "Use --allow-legacy-resume only after personally verifying its source data."
                )
            warnings.warn(
                "Legacy checkpoint accepted without dataset and optimizer-contract verification.",
                RuntimeWarning,
            )
        elif not isinstance(saved_contract, dict):
            raise ValueError("Checkpoint resume contract is malformed; refusing an unsafe resume.")
        else:
            lineage_keys = {"initial_model", "initial_model_path"}
            switch_keys = {"dataset_switch", "dataset_switch_history"}
            excluded_keys = lineage_keys | switch_keys
            saved_core = _resume_contract_for_comparison({
                key: value for key, value in saved_contract.items()
                if key not in excluded_keys
            }, allow_batch_size_change=allow_batch_size_change)
            current_core = _resume_contract_for_comparison({
                key: value for key, value in contract.items()
                if key not in excluded_keys
            }, allow_batch_size_change=allow_batch_size_change)
            _reconcile_resume_contract(
                saved_contract, saved_core, current_core, contract, state,
                allow_dataset_change=allow_dataset_change,
                )
            # The initial model is loaded exactly once. On resume the caller must
            # not pass it again, but the checkpoint's original initialization
            # fingerprint must remain in every subsequent checkpoint and manifest.
            saved_effective_batch = (
                int(saved_contract.get("batch_size", batch_size))
                * int(saved_contract.get("gradient_accumulation", grad_accum))
            )
            current_effective_batch = int(batch_size) * int(grad_accum)
            if saved_effective_batch != current_effective_batch:
                if not allow_batch_size_change:
                    raise ValueError(
                        "Changing effective batch size requires allow_batch_size_change=True."
                    )
                print(
                    "WARNING: authorized effective-batch change on resume: "
                    f"{saved_effective_batch} -> {current_effective_batch}. "
                    "Optimizer state is retained, but tokens per optimizer step change."
                )
            contract["initial_model"] = saved_contract.get("initial_model")
            contract["initial_model_path"] = saved_contract.get("initial_model_path")

        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        tokens_seen = int(state.get(
            "tokens_seen", start_step * batch_size * grad_accum * seq_len
        ))
        supervised_tokens_seen = int(state.get("supervised_tokens_seen", tokens_seen))
        saved_step_counts = state.get("thinking_step_counts")
        if isinstance(saved_step_counts, dict):
            for step_key in thinking_step_counts:
                thinking_step_counts[step_key] = int(
                    saved_step_counts.get(step_key, 0)
                )
        schedule_total_steps = int(state.get("schedule_total_steps", steps))
        if "numpy_rng" in state:
            rng.bit_generator.state = state["numpy_rng"]
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        saved_cuda_rng = state.get("cuda_rng")
        if torch.cuda.is_available() and saved_cuda_rng is not None:
            if len(saved_cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([item.cpu() for item in saved_cuda_rng])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])

        if start_step > steps:
            raise ValueError(
                f"Checkpoint is already at step {start_step}; --steps cannot be lower than the saved step."
            )
        if start_step == steps:
            print(
                f"Checkpoint already reached target step {steps:,}; "
                "skipping optimizer updates and rebuilding the final deployment export."
            )
        if steps > schedule_total_steps:
            if not extend_schedule:
                raise ValueError(
                    f"Checkpoint scheduler horizon is {schedule_total_steps} steps. "
                    "Pass --extend-schedule to deliberately extend the learning-rate schedule."
                )
            schedule_total_steps = steps
        print(
            f"Resuming from step {start_step:,}; schedule horizon={schedule_total_steps:,}."
        )

    initial_model_record = None
    if initial_source_path is not None:
        fingerprint = _sha256_file(initial_source_path)
        initial_model_record = {
            "path": str(initial_source_path),
            "sha256": fingerprint["sha256"],
            "size_bytes": fingerprint["size_bytes"],
            "kind": (
                "q2-step-zero-initial"
                if initial_brain_path is not None
                else "trained-qmini-checkpoint"
            ),
        }
    elif resume:
        # A resume does not re-load initial weights, but it must preserve the
        # lineage recorded by the first training session.
        try:
            previous_run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            previous_run_manifest = {}
        previous_initial = (
            previous_run_manifest.get("initial_model")
            if isinstance(previous_run_manifest, dict) else None
        )
        if isinstance(previous_initial, dict):
            initial_model_record = previous_initial
        else:
            saved_initial = (
                state.get("resume_contract", {}).get("initial_model")
                if isinstance(state.get("resume_contract"), dict) else None
            )
            saved_initial_path = (
                state.get("resume_contract", {}).get("initial_model_path")
                if isinstance(state.get("resume_contract"), dict) else None
            )
            if isinstance(saved_initial, dict):
                initial_model_record = {
                    "path": saved_initial_path,
                    "sha256": saved_initial.get("sha256"),
                    "size_bytes": saved_initial.get("size_bytes"),
                    "kind": "preserved-from-checkpoint-lineage",
                }

    run_manifest = {
        "format": "qmini-run-manifest-v1",
        "initial_model": initial_model_record,
        "model_config": config.to_dict(),
        "resume_contract": contract,
        "schedule_total_steps": schedule_total_steps,
        "thinking_step_sampling": {
            "distribution": "uniform_discrete",
            "minimum": 1,
            "maximum": config.max_thinking_steps,
        },
        # "precision" describes autocast only; it does not describe parameter storage.
        "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
        "autocast_precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "off",
        "trainable_parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "trainable_weight_policy": "fp32-master-q2-ste",
        "deployment_matrix_weight_format": "qmini-packed-v2",
        "deployment_matrix_weight_bits": 2,
        "device": str(device),
        "memory_preflight": memory_preflight,
        "checkpoint_mirror_dir": (
            str(checkpoint_mirror_path) if checkpoint_mirror_path is not None else None
        ),
        "checkpoint_session_id": checkpoint_session_id,
    }
    _atomic_json_save(run_manifest, run_manifest_path)
    if resume:
        _reconcile_metrics(metrics_path, start_step)

    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    print(
        f"Device={device}; AMP={'BF16' if use_bf16 else 'FP16' if use_fp16 else 'off'}; "
        f"parameters={model.parameter_report()['total']:,}; "
        f"global_batch={batch_size * grad_accum}; "
        f"tokens/optimizer_step={batch_size * grad_accum * seq_len:,}"
    )

    def learning_rate_at(step: int) -> float:
        warmup = max(1, int(schedule_total_steps * 0.03))
        if step < warmup:
            return learning_rate * (step + 1) / warmup
        progress = min(
            1.0,
            max(0.0, (step - warmup) / max(1, schedule_total_steps - warmup)),
        )
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    model.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        accum = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "targets": 0.0}
        sampled_thinking_steps = random.randint(1, config.max_thinking_steps)
        thinking_step_counts[str(sampled_thinking_steps)] += 1
        router_counts_total: list[torch.Tensor] | None = None

        # Weight assistant-response loss by supervised target count over the
        # complete optimizer batch, not by an unweighted mean of microbatches.
        microbatches = [
            _sample_batch(
                train_tokens, train_masks, batch_size, seq_len, rng, device,
                window_index=train_window_index,
            )
            for _ in range(grad_accum)
        ]
        total_supervised_targets = sum(
            float(target_mask.sum().item())
            for _inputs, _targets, target_mask in microbatches
        )
        if total_supervised_targets <= 0:
            raise ValueError("An optimizer batch contains no supervised target tokens.")

        for x, y, target_mask in microbatches:
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp else nullcontext()
            )
            with amp_context:
                result = model(x, thinking_steps=sampled_thinking_steps)
                cross_entropy_sum, _ = _masked_cross_entropy_sum(
                    result["logits"], y, target_mask
                )
                cross_entropy = cross_entropy_sum / total_supervised_targets
                auxiliary = result["aux_loss"].float()
                loss = cross_entropy + config.router_aux_loss_coef * auxiliary / grad_accum

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at optimizer step {step + 1}; checkpoint is unchanged."
                )
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum["loss"] += float(loss.detach())
            accum["ce"] += float(cross_entropy.detach())
            accum["aux"] += float(auxiliary.detach()) / grad_accum
            accum["targets"] += float(target_mask.sum().detach())
            batch_counts = [
                item.detach().to(device="cpu", dtype=torch.int64)
                for item in result["router_counts"]
            ]
            if router_counts_total is None:
                router_counts_total = batch_counts
            else:
                for index, count in enumerate(batch_counts):
                    router_counts_total[index] += count

        if use_fp16:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        lr = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        completed = step + 1
        tokens_seen += batch_size * grad_accum * seq_len
        supervised_tokens_seen += int(accum["targets"])
        record: dict[str, Any] = {
            "step": completed,
            "tokens_seen": tokens_seen,
            "supervised_tokens_seen": supervised_tokens_seen,
            "train_loss": accum["loss"],
            "train_cross_entropy": accum["ce"],
            "router_aux_loss": accum["aux"],
            "gradient_norm": float(grad_norm.detach()),
            "learning_rate": lr,
            "thinking_steps": sampled_thinking_steps,
            "thinking_step_sampling_counts": dict(thinking_step_counts),
            "router_token_counts": (
                [counts.tolist() for counts in router_counts_total]
                if router_counts_total is not None else []
            ),
        }

        if completed % eval_every == 0 or completed == steps:
            per_step_evaluation: dict[str, dict[str, Any]] = {}
            for evaluation_steps in range(1, config.max_thinking_steps + 1):
                per_step_evaluation[str(evaluation_steps)] = _evaluate(
                    model, valid_tokens, valid_masks, batch_size, seq_len,
                    eval_batches, device, actual_seed + completed,
                    window_index=valid_window_index,
                    thinking_steps=evaluation_steps,
                )
            record["eval_by_thinking_steps"] = per_step_evaluation
            record["eval_primary_thinking_steps"] = config.thinking_steps
            record.update(
                per_step_evaluation[str(config.thinking_steps)]
            )
        if use_amp:
            record["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

        checkpoint_saved = False
        if completed % save_every == 0 or completed == steps:
            checkpoint_payload = {
                "format": "qmini-checkpoint-v2",
                "step": completed,
                "tokens_seen": tokens_seen,
                "supervised_tokens_seen": supervised_tokens_seen,
                "thinking_step_counts": dict(thinking_step_counts),
                "schedule_total_steps": schedule_total_steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_dict(),
                "resume_contract": contract,
                "numpy_rng": rng.bit_generator.state,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "scaler": scaler.state_dict(),
            }
            _atomic_torch_save(checkpoint_payload, checkpoint_path)
            checkpoint_saved = True

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        if checkpoint_saved and checkpoint_mirror_path is not None:
            from .checkpointing import publish_checkpoint
            progress = publish_checkpoint(
                checkpoint_path,
                checkpoint_mirror_path,
                metrics_path,
                run_manifest_path,
                step=completed,
                target_steps=steps,
                tokens_seen=tokens_seen,
                session_id=checkpoint_session_id,
            )
            print(
                "Drive checkpoint verified: "
                f"step={progress['current_step']:,}/{progress['target_steps']:,} "
                f"({progress['progress_percent']:.2f}%)",
                flush=True,
            )

        if completed == 1 or completed % 10 == 0 or "eval_loss" in record:
            message = (
                f"step={completed:,}/{steps:,} tokens={tokens_seen:,} "
                f"loss={accum['loss']:.4f} ce={accum['ce']:.4f} "
                f"aux={accum['aux']:.4f} grad={float(grad_norm):.3f} lr={lr:.2e}"
            )
            if "eval_loss" in record:
                perplexity_value = record["eval_perplexity"]
                perplexity_text = (
                    f"{perplexity_value:.2f}"
                    if perplexity_value is not None else "overflow"
                )
                message += (
                    f" val_loss={record['eval_loss']:.4f}"
                    f" ppl={perplexity_text}"
                )
            print(message, flush=True)

    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    config.save(final_dir / "config.json")
    if tokenizer_path is not None and Path(tokenizer_path).is_file():
        shutil.copy2(tokenizer_path, final_dir / "tokenizer.json")

    from .export import export_packed_state
    packed_path = export_packed_state(
        model.state_dict(), config, final_dir / "model.qmini.npz"
    )

    # New deployment folders contain only the verified Q2-v2 matrix export.
    # Never write a dense FP32 model.pt here. If an older run already has one,
    # preserve it untouched; resumable training state remains in latest.pt.
    from .export import verify_packed_export
    packed_report = verify_packed_export(packed_path)
    if packed_report.get("format") != "qmini-packed-v2" or packed_report.get("q2_version") != 2:
        raise RuntimeError("Final deployment export is not verified Q2-v2.")
    # Older model.pt files, if present, are left untouched for manual recovery
    # but are not part of this final-model manifest or accepted as SFT sources.
    packed_fingerprint = _sha256_file(packed_path)
    dense_snapshot_fingerprint = None
    trainable_parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    optimizer_state_dtypes = sorted({
        str(value.dtype)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    })
    tokenizer_fingerprint = (
        _sha256_file(tokenizer_path)
        if tokenizer_path is not None and Path(tokenizer_path).is_file()
        else None
    )
    final_report = {
        "format": "qmini-final-model-v2",
        "step": steps,
        "tokens_seen": tokens_seen,
        "supervised_tokens_seen": supervised_tokens_seen,
        "loss_masked_training": train_masks is not None,
        "tokenizer_sha256": (
            tokenizer_fingerprint["sha256"] if tokenizer_fingerprint else None
        ),
        "tokenizer_size_bytes": (
            tokenizer_fingerprint["size_bytes"] if tokenizer_fingerprint else None
        ),
        "parameters": model.parameter_report()["total"],
        "matrix_weight_format": "qmini-packed-v2",
        "matrix_weight_bits": 2,
        "q2_codebook": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
        "packed_export_verification": packed_report,
        "packed_export": packed_path.name,
        "packed_export_bytes": packed_fingerprint["size_bytes"],
        "packed_export_sha256": packed_fingerprint["sha256"],
        "trainable_parameter_dtypes": trainable_parameter_dtypes,
        "trainable_weight_policy": "fp32-master-q2-ste",
        "optimizer_state_dtypes": optimizer_state_dtypes,
        "dense_snapshot_sha256": (
            dense_snapshot_fingerprint["sha256"]
            if dense_snapshot_fingerprint else None
        ),
        "dense_snapshot_bytes": (
            dense_snapshot_fingerprint["size_bytes"]
            if dense_snapshot_fingerprint else None
        ),
        "estimated_q2_fp16_raw_bytes": config.estimated_packed_export_bytes(),
        "deployment_weight_budget_bytes": 1_500_000_000,
        "within_1_5gb_weight_artifact_budget": (
            config.estimated_packed_export_bytes() <= 1_500_000_000
        ),
        "dense_training_snapshot_included": False,
        "resume_checkpoint": str(checkpoint_path),
        "config": config.to_dict(),
    }
    _atomic_json_save(final_report, final_dir / "model_manifest.json")
    print(
        f"Training complete: {final_dir} | packed export={packed_path.stat().st_size:,} bytes"
    )
    return final_dir

def _reconcile_resume_contract(
    saved_contract,
    saved_core,
    current_core,
    contract,
    state,
    *,
    allow_dataset_change,
):
    """Reconcile a saved resume contract with the current run's contract.

    When the data fingerprints match, any recorded dataset-switch audit trail
    is carried into the new contract. When only the data fingerprints differ
    and the caller explicitly allowed a dataset change, the switch is recorded
    (previous fingerprints plus the step it happened at) and the new contract
    is re-baselined to the new data. Anything else is still a hard mismatch:
    config, seed, learning rate, weight decay, effective batch and tokenizer
    changes are never allowed through this path.
    """
    if saved_core == current_core:
        if "dataset_switch" in saved_contract:
            contract["dataset_switch"] = saved_contract["dataset_switch"]
        if "dataset_switch_history" in saved_contract:
            contract["dataset_switch_history"] = saved_contract["dataset_switch_history"]
        return
    differing = {
        key
        for key in saved_core.keys() | current_core.keys()
        if saved_core.get(key) != current_core.get(key)
    }
    dataset_keys = {
        "train",
        "validation",
        "train_loss_mask",
        "validation_loss_mask",
        "dataset_manifest",
    }
    if allow_dataset_change and differing and differing <= dataset_keys:
        switch_record = {
            "switch_step": int(state["step"]),
            "previous_train": saved_contract.get("train"),
            "previous_validation": saved_contract.get("validation"),
            "previous_train_loss_mask": saved_contract.get("train_loss_mask"),
            "previous_validation_loss_mask": saved_contract.get("validation_loss_mask"),
            "previous_dataset_manifest": saved_contract.get("dataset_manifest"),
            "previous_tokenizer": saved_contract.get("tokenizer"),
        }
        history = list(saved_contract.get("dataset_switch_history") or [])
        history.append(switch_record)
        contract["dataset_switch"] = switch_record
        contract["dataset_switch_history"] = history
        print(
            "WARNING: authorized dataset switch on resume at step "
            f"{int(state['step']):,}: the resume contract is re-baselined to the "
            "new data. Optimizer state, model weights, config, seed, learning "
            "rate, weight decay and effective batch are unchanged; only the data "
            "fingerprints were replaced."
        )
        return
    raise ValueError(
        "Resume contract mismatch (dataset/tokenizer or optimizer settings). "
        "Use a new run directory for changed data or settings."
    )
