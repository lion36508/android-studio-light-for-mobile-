from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from .config import QMiniConfig
from .export import export_packed_state_stream, verify_packed_export


def _module_items(prefix: str, module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in module.state_dict().items():
        yield prefix + name, tensor


def iter_initial_state_tensors(config: QMiniConfig) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield deterministic initial parameters in exactly QMiniMoE constructor order.

    Only one model component is materialized at a time. CPU RNG state is
    forked and restored when iteration finishes. No dataset, optimizer,
    backward pass, training step, CUDA device or full QMiniMoE instance is used.
    """
    config.validate()
    from .model import QEmbedding, QMiniBlock, RMSNorm

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)

        module = QEmbedding(config.vocab_size, config.d_model)
        yield from _module_items("embedding.", module)
        del module

        for index, dilation in enumerate(config.dilations):
            module = QMiniBlock(config, dilation)
            yield from _module_items(f"blocks.{index}.", module)
            del module

        module = QMiniBlock(config, dilation=1)
        yield from _module_items("refinement.", module)
        del module

        module = RMSNorm(config.d_model, config.rms_norm_eps)
        yield from _module_items("final_norm.", module)
        del module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
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


def build_initial_brain(
    config: QMiniConfig,
    output_dir: str | Path,
    source_revision: str | None = None,
) -> dict:
    """Build a new untrained Q2-v2 initial artifact without dense full-model allocation.

    The output directory must not already exist. The artifact is a random
    initialization packed into Q2-v2; it is not a trained assistant.
    """
    config.validate()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=False)

    config_path = target_dir / "config.json"
    packed_path = target_dir / "model.qmini.npz"
    config.save(config_path)

    export_packed_state_stream(
        iter_initial_state_tensors(config),
        config,
        packed_path,
    )
    packed_report = verify_packed_export(packed_path)
    if (
        packed_report.get("format") != "qmini-packed-v2"
        or packed_report.get("q2_version") != 2
        or packed_report.get("parameter_count") != config.estimated_parameter_count()
    ):
        raise RuntimeError("New initial artifact failed Q2-v2 format/parameter verification.")

    # This validates exact module names and shapes using the packed/meta loader;
    # it never materializes the complete dense FP32 parameter set.
    from .packed_runtime import load_packed_model
    verification_model = load_packed_model(packed_path, expected_config=config)
    del verification_model

    manifest = {
        "format": "qmini-initial-brain-v1",
        "status": "untrained_initialization",
        "training_steps": 0,
        "optimizer_steps": 0,
        "tokens_seen": 0,
        "parameters": config.estimated_parameter_count(),
        "seed": config.seed,
        "source_revision": source_revision,
        "config": config.to_dict(),
        "packed_export": packed_path.name,
        "packed_export_bytes": packed_path.stat().st_size,
        "packed_export_sha256": _sha256_file(packed_path),
        "packed_export_report": packed_report,
        "matrix_weight_format": "qmini-packed-v2",
        "matrix_weight_bits": 2,
        "q2_codebook": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
        "row_scales_dtype": "float32",
        "one_dimensional_parameters_dtype": "float16",
        "activations_format": "signed-int8-per-token-absmax",
        "tokenizer_included": False,
        "training_started": False,
    }
    _atomic_json(target_dir / "model_manifest.json", manifest)
    return manifest
