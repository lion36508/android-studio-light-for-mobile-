from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AddSubConfig
from .model import AddSubLanguageModel, AddSubLinear
from .quantization import (
    BITS_PER_WEIGHT,
    FORMAT_NAME,
    WEIGHT_SYMBOLS,
    codes_to_values,
    pack_3bit_codes,
    quantize_to_codes,
    unpack_3bit_codes,
    weight_scale,
    LEGACY_FORMAT_NAME,
)
from .sft import SFT_FORMAT
from .train import CHECKPOINT_FORMAT, sha256_file


EXPORT_FORMAT_V1 = "qmini-addsub-packed-weights-v1"
EXPORT_FORMAT = "qmini-addsub-packed-weights-v2"
SCALED_CHECKPOINT_FORMATS = {CHECKPOINT_FORMAT, SFT_FORMAT}
ACCEPTED_CHECKPOINT_FORMATS = {
    CHECKPOINT_FORMAT, SFT_FORMAT,
    "qmini-addsub-training-v4", "qmini-addsub-sft-v5",
    "qmini-addsub-training-v3", "qmini-addsub-sft-v4",
    "qmini-addsub-sft-v3", "qmini-addsub-sft-v2",
}
ARCHIVE_MEMBERS = {"manifest.json", "weights.bin", "auxiliary.npz"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _quantized_parameter_names(model: AddSubLanguageModel) -> set[str]:
    names = {"token_embedding.weight"}
    for module_name, module in model.named_modules():
        if isinstance(module, AddSubLinear):
            names.add(f"{module_name}.weight")
        if module_name.endswith(".conv") and hasattr(module, "depthwise_weight"):
            names.add(f"{module_name}.depthwise_weight")
    return names


def export_addsub_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Pack every Add/Sub model weight and preserve other tensors in an integrity-checked sidecar.

    This is an offline weight-container export. It is not the complete integer-only
    inference runtime: auxiliary normalization, bias and gate tensors remain float32.
    """
    source = Path(checkpoint_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    run_contract = checkpoint.get("run_contract", {})
    if isinstance(run_contract, dict) and run_contract.get("wavebit_hsra") is not None:
        from .wavebit_export import export_wavebit_checkpoint
        return export_wavebit_checkpoint(source, destination, checkpoint)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in ACCEPTED_CHECKPOINT_FORMATS:
        raise ValueError("Only known Q-Mini Add/Sub v1 checkpoints can be exported.")
    config = AddSubConfig.from_dict(checkpoint["config"])
    if int(checkpoint.get("step", 0)) < 1:
        raise ValueError("Export refuses a random-only step-zero checkpoint.")

    model = AddSubLanguageModel(config, seed=config.seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    state = model.state_dict()
    quantized_names = _quantized_parameter_names(model)
    missing = sorted(quantized_names - set(state))
    if missing:
        raise ValueError(f"Model quantized parameters are missing from state_dict: {missing}")

    use_scaled_format = checkpoint_format in SCALED_CHECKPOINT_FORMATS
    archive_format = EXPORT_FORMAT if use_scaled_format else EXPORT_FORMAT_V1
    weight_format = FORMAT_NAME if use_scaled_format else LEGACY_FORMAT_NAME

    packed_weights = bytearray()
    quantized_entries = []
    auxiliary_arrays: dict[str, np.ndarray] = {}
    auxiliary_entries = []

    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Model tensor {name!r} is not finite floating-point data.")
        values = tensor.numpy().astype(np.float32, copy=False)
        if name in quantized_names:
            if use_scaled_format:
                scale = float(weight_scale(tensor).item())
                codes = quantize_to_codes(values, scale=scale)
            else:
                scale = 1.0
                codes = quantize_to_codes(values)
            packed = pack_3bit_codes(codes)
            entry = {
                "name": name, "shape": list(values.shape),
                "count": int(values.size), "offset": len(packed_weights),
                "nbytes": len(packed), "sha256": _sha256_bytes(packed),
            }
            if use_scaled_format:
                entry["scale"] = scale
            quantized_entries.append(entry)
            packed_weights.extend(packed)
        else:
            auxiliary_arrays[name] = values.copy()
            auxiliary_entries.append({
                "name": name,
                "shape": list(values.shape),
                "dtype": "float32",
            })

    if not quantized_entries:
        raise ValueError("No quantized model tensors were found for export.")
    auxiliary_buffer = io.BytesIO()
    np.savez_compressed(auxiliary_buffer, **auxiliary_arrays)
    auxiliary_payload = auxiliary_buffer.getvalue()
    weights_payload = bytes(packed_weights)

    config_dict = config.to_dict()
    canonical_config = json.dumps(
        config_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    tokenizer_sha = checkpoint.get("tokenizer_sha256")
    dataset_manifest_sha = checkpoint.get("data_manifest_sha256")
    if not isinstance(tokenizer_sha, str) or len(tokenizer_sha) != 64:
        raise ValueError("Checkpoint is missing a valid tokenizer SHA-256 fingerprint.")
    if not isinstance(dataset_manifest_sha, str) or len(dataset_manifest_sha) != 64:
        raise ValueError("Checkpoint is missing a valid dataset-manifest SHA-256 fingerprint.")
    run_contract = checkpoint.get("run_contract")
    if run_contract is not None and not isinstance(run_contract, dict):
        raise ValueError("Checkpoint training run contract is invalid.")

    manifest = {
        "format": archive_format,
        "weight_format": weight_format,
        "checkpoint_format": checkpoint_format,
        "checkpoint_step": int(checkpoint["step"]),
        "training_run_contract": run_contract,
        "config": config_dict,
        "config_sha256": _sha256_bytes(canonical_config),
        "parameter_count": model.parameter_count(),
        "tokenizer_vocab_size": config.vocab_size,
        "tokenizer_sha256": tokenizer_sha,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "base_checkpoint_sha256": checkpoint.get("base_sha256"),
        "weight_symbols": list(WEIGHT_SYMBOLS),
        "weight_code_map": [
            {"code": index, "value": value}
            for index, value in enumerate(WEIGHT_SYMBOLS)
        ],
        "reserved_codewords": [5, 6, 7],
        "bits_per_weight": BITS_PER_WEIGHT,
        "per_tensor_scale": use_scaled_format,
        "scale_multiplier": 1.4 if use_scaled_format else 1.0,
        "packing": "little-endian continuous 3-bit codes; 0..4 valid, 5..7 reserved",
        "zero_symbol_code": 2,
        "source_checkpoint_sha256": sha256_file(source),
        "quantized_tensors": quantized_entries,
        "auxiliary_tensors": auxiliary_entries,
        "weights_sha256": _sha256_bytes(weights_payload),
        "auxiliary_sha256": _sha256_bytes(auxiliary_payload),
        "inference_status": (
            "packed weight container only; complete integer-only model inference is not implemented"
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            archive.writestr("weights.bin", weights_payload)
            archive.writestr("auxiliary.npz", auxiliary_payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "format": archive_format,
        "output": str(destination),
        "parameter_count": model.parameter_count(),
        "quantized_tensors": len(quantized_entries),
        "auxiliary_tensors": len(auxiliary_entries),
        "weights_bytes": len(weights_payload),
        "auxiliary_bytes": len(auxiliary_payload),
        "sha256": sha256_file(destination),
    }


def load_addsub_export(
    export_path: str | Path,
    device_name: str = "cpu",
    tokenizer_path: str | Path | None = None,
) -> tuple[AddSubLanguageModel, dict[str, Any]]:
    """Verify and reconstruct the PyTorch reference model from a packed Add/Sub container."""
    source = Path(export_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    with zipfile.ZipFile(source, "r") as archive:
        member_names = archive.namelist()
        names = set(member_names)
        wavebit_members = {
            "manifest.json", "weights.bin", "auxiliary.npz", "group_metadata.npz"
        }
        if len(member_names) != len(names):
            raise ValueError("Add/Sub export contains duplicate archive members.")
        if names == wavebit_members:
            from .wavebit_export import load_wavebit_export
            return load_wavebit_export(
                source, device_name=device_name, tokenizer_path=tokenizer_path
            )
        if names != ARCHIVE_MEMBERS:
            raise ValueError(
                "Add/Sub export contains missing, unexpected or incompatible archive members."
            )
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        weights_payload = archive.read("weights.bin")
        auxiliary_payload = archive.read("auxiliary.npz")

    archive_format = manifest.get("format")
    if archive_format not in {EXPORT_FORMAT_V1, EXPORT_FORMAT}:
        raise ValueError("Unsupported Add/Sub export format.")
    scaled_archive = archive_format == EXPORT_FORMAT
    expected_weight_format = FORMAT_NAME if scaled_archive else LEGACY_FORMAT_NAME
    if manifest.get("weight_format") != expected_weight_format:
        raise ValueError("Export weight quantization format is not supported.")
    if scaled_archive and manifest.get("per_tensor_scale") is not True:
        raise ValueError("Scaled export is missing its scale declaration.")
    if not scaled_archive and manifest.get("per_tensor_scale") not in (None, False):
        raise ValueError("Legacy export scale declaration is invalid.")
    if manifest.get("weight_symbols") != list(WEIGHT_SYMBOLS):
        raise ValueError("Export weight alphabet does not match the five-symbol codebook.")
    expected_code_map = [
        {"code": index, "value": value}
        for index, value in enumerate(WEIGHT_SYMBOLS)
    ]
    if manifest.get("weight_code_map") != expected_code_map:
        raise ValueError("Export code map does not match the five-symbol codebook.")
    if manifest.get("zero_symbol_code") != 2:
        raise ValueError("Export zero symbol must use code 2.")
    if manifest.get("bits_per_weight") != BITS_PER_WEIGHT:
        raise ValueError("Export bit width is incompatible with the five-symbol codebook.")
    if manifest.get("packing") != (
        "little-endian continuous 3-bit codes; 0..4 valid, 5..7 reserved"
    ):
        raise ValueError("Export bit packing declaration is not supported.")
    if _sha256_bytes(weights_payload) != manifest.get("weights_sha256"):
        raise ValueError("Packed weight payload failed SHA-256 verification.")
    if _sha256_bytes(auxiliary_payload) != manifest.get("auxiliary_sha256"):
        raise ValueError("Auxiliary tensor payload failed SHA-256 verification.")

    config_dict = manifest["config"]
    canonical_config = json.dumps(
        config_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if _sha256_bytes(canonical_config) != manifest.get("config_sha256"):
        raise ValueError("Export config failed SHA-256 verification.")
    config = AddSubConfig.from_dict(config_dict)
    if manifest.get("tokenizer_vocab_size") != config.vocab_size:
        raise ValueError("Export tokenizer vocabulary does not match the model config.")
    if manifest.get("checkpoint_format") not in ACCEPTED_CHECKPOINT_FORMATS:
        raise ValueError("Export source checkpoint format is not accepted.")
    if manifest.get("reserved_codewords") != [5, 6, 7]:
        raise ValueError("Export reserved codeword declaration is invalid.")
    tokenizer_sha = manifest.get("tokenizer_sha256")
    dataset_sha = manifest.get("dataset_manifest_sha256")
    if not isinstance(tokenizer_sha, str) or len(tokenizer_sha) != 64:
        raise ValueError("Export tokenizer fingerprint is missing or invalid.")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        raise ValueError("Export dataset-manifest fingerprint is missing or invalid.")
    if tokenizer_path is not None:
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.is_file():
            raise FileNotFoundError(tokenizer_file)
        if sha256_file(tokenizer_file) != tokenizer_sha:
            raise ValueError("Tokenizer fingerprint does not match the packed model export.")
    model = AddSubLanguageModel(config, seed=config.seed)
    expected_state = model.state_dict()
    quantized_names = _quantized_parameter_names(model)
    entries = manifest.get("quantized_tensors")
    auxiliary_entries = manifest.get("auxiliary_tensors")
    if not isinstance(entries, list) or not isinstance(auxiliary_entries, list):
        raise ValueError("Export tensor index is invalid.")

    quantized_entry_names = [entry.get("name") for entry in entries]
    auxiliary_entry_names = [entry.get("name") for entry in auxiliary_entries]
    if (
        len(set(quantized_entry_names)) != len(quantized_entry_names)
        or set(quantized_entry_names) != quantized_names
    ):
        raise ValueError("Quantized tensor index does not exactly match the model architecture.")
    expected_auxiliary_names = set(expected_state) - quantized_names
    if (
        len(set(auxiliary_entry_names)) != len(auxiliary_entry_names)
        or set(auxiliary_entry_names) != expected_auxiliary_names
    ):
        raise ValueError("Auxiliary tensor index does not exactly match the model architecture.")

    restored: dict[str, torch.Tensor] = {}
    expected_offset = 0
    for entry in entries:
        name = entry["name"]
        shape = tuple(entry["shape"])
        expected_shape = tuple(expected_state[name].shape)
        if shape != expected_shape:
            raise ValueError(f"Packed tensor shape mismatch for {name!r}.")
        if int(entry["count"]) != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"Packed tensor element count mismatch for {name!r}.")
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        if offset != expected_offset or nbytes < 0 or offset + nbytes > len(weights_payload):
            raise ValueError("Packed tensor offsets are overlapping, missing or out of bounds.")
        packed = weights_payload[offset : offset + nbytes]
        if _sha256_bytes(packed) != entry.get("sha256"):
            raise ValueError(f"Packed tensor {name!r} failed SHA-256 verification.")
        codes = unpack_3bit_codes(packed, shape)
        values = codes_to_values(codes)
        if scaled_archive:
            scale = entry.get("scale")
            if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                    or not np.isfinite(scale) or scale <= 0.0):
                raise ValueError(f"Packed tensor {name!r} has an invalid scale.")
            values = values * np.float32(scale)
        restored[name] = torch.from_numpy(values.copy())
        expected_offset += nbytes

    if expected_offset != len(weights_payload):
        raise ValueError("Packed weight payload has unindexed trailing bytes.")

    with np.load(io.BytesIO(auxiliary_payload), allow_pickle=False) as auxiliary:
        if set(auxiliary.files) != expected_auxiliary_names:
            raise ValueError("Auxiliary archive tensors do not match the manifest.")
        shape_index = {
            entry["name"]: tuple(entry["shape"])
            for entry in auxiliary_entries
        }
        for name in sorted(expected_auxiliary_names):
            array = auxiliary[name]
            if array.dtype != np.float32 or tuple(array.shape) != shape_index[name]:
                raise ValueError(f"Auxiliary tensor shape or dtype mismatch for {name!r}.")
            if not np.isfinite(array).all():
                raise ValueError(f"Auxiliary tensor {name!r} contains non-finite values.")
            restored[name] = torch.from_numpy(np.array(array, copy=True))

    if set(restored) != set(expected_state):
        raise ValueError("Export does not reconstruct the complete model state.")
    model.load_state_dict(restored, strict=True)
    if model.parameter_count() != manifest.get("parameter_count"):
        raise ValueError("Export parameter count does not match the reconstructed model.")
    model.to(torch.device(device_name))
    # Freeze the exact forward scale after device placement. Legacy v1 archives
    # intentionally use scale 1.0; v2 archives carry one scale per tensor.
    for entry in entries:
        fixed_scale = float(entry.get("scale", 1.0))
        if not np.isfinite(fixed_scale) or fixed_scale <= 0.0:
            raise ValueError(f"Packed tensor {entry['name']!r} has an invalid scale.")
        model.get_parameter(entry["name"])._qmini_fixed_scale = fixed_scale
    model.eval()
    return model, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack all five-symbol Add/Sub model weights from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(
        export_addsub_checkpoint(args.checkpoint, args.output),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
