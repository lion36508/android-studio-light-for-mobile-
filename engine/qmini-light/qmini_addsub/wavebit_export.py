from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AddSubConfig
from .model import AddSubLanguageModel
from .wavebit_hsra import (
    OCTAVE_BASE_LEVELS,
    octave_shift_quantize,
    pack_octave_stream,
    unpack_octave_stream,
    validate_wavebit_options,
)


WAVEBIT_EXPORT_FORMAT = "qmini-addsub-wavebit-packed-v1"
WAVEBIT_WEIGHT_FORMAT = "qmini-octave-shift-groupwise-v1"
WAVEBIT_ARCHIVE_MEMBERS = {
    "manifest.json",
    "weights.bin",
    "auxiliary.npz",
    "group_metadata.npz",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _extract_group_scales(scale: torch.Tensor, group_size: int) -> np.ndarray:
    rows = scale.shape[0]
    flat = scale.detach().cpu().reshape(rows, -1).numpy().astype(np.float32, copy=False)
    result = np.ascontiguousarray(flat[:, ::group_size], dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("WaveBit group scales must be finite and positive.")
    return result


def export_wavebit_checkpoint(
    source: Path,
    destination: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    from .export import ACCEPTED_CHECKPOINT_FORMATS, _quantized_parameter_names
    from .train import sha256_file

    if checkpoint.get("format") not in ACCEPTED_CHECKPOINT_FORMATS:
        raise ValueError("Only known Q-Mini Add/Sub checkpoints can be exported.")
    if int(checkpoint.get("step", 0)) < 1:
        raise ValueError("Export refuses a random-only step-zero checkpoint.")
    run_contract = checkpoint.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("WaveBit checkpoint is missing its training run contract.")
    raw_options = run_contract.get("wavebit_hsra")
    if not isinstance(raw_options, dict):
        raise ValueError("WaveBit checkpoint run contract has invalid options.")
    options = validate_wavebit_options(raw_options)
    assert options is not None
    config = AddSubConfig.from_dict(checkpoint["config"])
    model = AddSubLanguageModel(config, seed=config.seed, wavebit_options=options)
    model.load_state_dict(checkpoint["model"], strict=True)
    state = model.state_dict()
    quantized_names = _quantized_parameter_names(model)
    if quantized_names - set(state):
        raise ValueError("WaveBit model is missing quantized tensors from its state_dict.")

    weights_payload = bytearray()
    quantized_entries: list[dict[str, Any]] = []
    metadata_arrays: dict[str, np.ndarray] = {}
    auxiliary_arrays: dict[str, np.ndarray] = {}
    auxiliary_entries: list[dict[str, Any]] = []
    group_size = options["group_size"]
    boost_factor = options["octave_boost_factor"]

    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Model tensor {name!r} is not finite floating-point data.")
        values = tensor.numpy().astype(np.float32, copy=False)
        if name not in quantized_names:
            auxiliary_arrays[name] = values.copy()
            auxiliary_entries.append({
                "name": name,
                "shape": list(values.shape),
                "dtype": "float32",
            })
            continue

        _, symbol_codes, boost_modes, scale = octave_shift_quantize(
            tensor,
            group_size=group_size,
            ewgs_strength=options["ewgs_strength"],
            octave_temperature=options["octave_temperature"],
            boost_factor=boost_factor,
        )
        rows = tensor.shape[0]
        features = tensor.numel() // rows
        codes = np.ascontiguousarray(
            symbol_codes.detach().cpu().reshape(rows, features).numpy(),
            dtype=np.uint8,
        )
        modes = np.ascontiguousarray(
            boost_modes.detach().cpu().numpy(), dtype=np.bool_
        )
        packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=True)
        if packed.size > np.iinfo(np.uint32).max:
            raise ValueError(f"Packed WaveBit tensor {name!r} exceeds uint32 offsets.")
        offsets = np.ascontiguousarray(offsets, dtype=np.uint32)
        group_scales = _extract_group_scales(scale, group_size)
        expected_groups = (features + group_size - 1) // group_size
        if group_scales.shape != (rows, expected_groups):
            raise ValueError(f"WaveBit group-scale shape mismatch for {name!r}.")

        index = len(quantized_entries)
        scale_key = f"group_scales_{index:05d}"
        offsets_key = f"row_offsets_{index:05d}"
        metadata_arrays[scale_key] = group_scales
        metadata_arrays[offsets_key] = offsets
        quantized_entries.append({
            "name": name,
            "shape": list(values.shape),
            "count": int(values.size),
            "offset": len(weights_payload),
            "nbytes": int(packed.size),
            "sha256": _sha256_bytes(packed.tobytes()),
            "group_size": group_size,
            "group_scale_key": scale_key,
            "group_scale_shape": list(group_scales.shape),
            "row_offsets_key": offsets_key,
            "row_offsets_shape": list(offsets.shape),
        })
        weights_payload.extend(packed.tobytes())

    if not quantized_entries:
        raise ValueError("No WaveBit-quantized model tensors were found.")
    tokenizer_sha = checkpoint.get("tokenizer_sha256")
    dataset_sha = checkpoint.get("data_manifest_sha256")
    if not _valid_sha(tokenizer_sha):
        raise ValueError("Checkpoint is missing a valid tokenizer SHA-256 fingerprint.")
    if not _valid_sha(dataset_sha):
        raise ValueError("Checkpoint is missing a valid dataset-manifest SHA-256 fingerprint.")

    auxiliary_buffer = io.BytesIO()
    np.savez_compressed(auxiliary_buffer, **auxiliary_arrays)
    auxiliary_payload = auxiliary_buffer.getvalue()
    metadata_buffer = io.BytesIO()
    np.savez_compressed(metadata_buffer, **metadata_arrays)
    metadata_payload = metadata_buffer.getvalue()
    weights_bytes = bytes(weights_payload)

    canonical_contract = dict(run_contract)
    canonical_contract["wavebit_hsra"] = options
    config_dict = config.to_dict()
    canonical_config = json.dumps(
        config_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = {
        "format": WAVEBIT_EXPORT_FORMAT,
        "weight_format": WAVEBIT_WEIGHT_FORMAT,
        "checkpoint_format": checkpoint["format"],
        "checkpoint_step": int(checkpoint["step"]),
        "training_run_contract": canonical_contract,
        "wavebit_options": options,
        "config": config_dict,
        "config_sha256": _sha256_bytes(canonical_config),
        "parameter_count": model.parameter_count(),
        "tokenizer_vocab_size": config.vocab_size,
        "tokenizer_sha256": tokenizer_sha,
        "dataset_manifest_sha256": dataset_sha,
        "base_checkpoint_sha256": checkpoint.get("base_sha256"),
        "weight_symbols": list(OCTAVE_BASE_LEVELS),
        "zero_symbol_code": 2,
        "boost_factor": boost_factor,
        "boost_scope": "one shared x2 or x4 mode per three-weight triplet",
        "group_scale": {
            "method": "1.4 * mean(abs(original weights)) per output row and group",
            "group_size": group_size,
            "storage": "float32, one value per output row and input group",
        },
        "packing": {
            "payload": "one byte per triplet plus reserved zero-run skip tokens",
            "symbols_per_triplet": 3,
            "symbol_radix": 5,
            "mode_multiplier": boost_factor,
            "skip_codes": {"250": 4, "251": 8, "252": 16, "253": 32, "254": 64},
            "escape_code": 255,
            "escape_payload": "one byte containing a skip count from 1 through 255",
        },
        "source_checkpoint_sha256": sha256_file(source),
        "quantized_tensors": quantized_entries,
        "auxiliary_tensors": auxiliary_entries,
        "weights_sha256": _sha256_bytes(weights_bytes),
        "auxiliary_sha256": _sha256_bytes(auxiliary_payload),
        "group_metadata_sha256": _sha256_bytes(metadata_payload),
        "inference_status": (
            "group-wise packed weights with standalone integer projection primitives; "
            "complete model inference and output requantization are not implemented"
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
            archive.writestr("weights.bin", weights_bytes)
            archive.writestr("auxiliary.npz", auxiliary_payload)
            archive.writestr("group_metadata.npz", metadata_payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "format": WAVEBIT_EXPORT_FORMAT,
        "output": str(destination),
        "parameter_count": model.parameter_count(),
        "quantized_tensors": len(quantized_entries),
        "auxiliary_tensors": len(auxiliary_entries),
        "weights_bytes": len(weights_bytes),
        "group_metadata_bytes": len(metadata_payload),
        "sha256": sha256_file(destination),
    }


def load_wavebit_export(
    source: Path,
    device_name: str = "cpu",
    tokenizer_path: str | Path | None = None,
) -> tuple[AddSubLanguageModel, dict[str, Any]]:
    from .export import ACCEPTED_CHECKPOINT_FORMATS, _quantized_parameter_names
    from .train import sha256_file

    with zipfile.ZipFile(source, "r") as archive:
        member_names = archive.namelist()
        if (
            len(member_names) != len(set(member_names))
            or set(member_names) != WAVEBIT_ARCHIVE_MEMBERS
        ):
            raise ValueError("WaveBit archive has duplicate, missing or unexpected members.")
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            weights_payload = archive.read("weights.bin")
            auxiliary_payload = archive.read("auxiliary.npz")
            metadata_payload = archive.read("group_metadata.npz")
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("WaveBit archive metadata is malformed.") from exc

    if manifest.get("format") != WAVEBIT_EXPORT_FORMAT:
        raise ValueError("Unsupported WaveBit archive format.")
    if manifest.get("weight_format") != WAVEBIT_WEIGHT_FORMAT:
        raise ValueError("WaveBit packed-weight format is unsupported.")
    if manifest.get("weight_symbols") != list(OCTAVE_BASE_LEVELS):
        raise ValueError("WaveBit base-symbol alphabet does not match.")
    if manifest.get("zero_symbol_code") != 2:
        raise ValueError("WaveBit zero symbol must use code 2.")
    if manifest.get("boost_factor") not in (2, 4):
        raise ValueError("WaveBit archive boost factor must be 2 or 4.")
    if manifest.get("boost_scope") != "one shared x2 or x4 mode per three-weight triplet":
        raise ValueError("WaveBit boost-mode declaration is unsupported.")
    if _sha256_bytes(weights_payload) != manifest.get("weights_sha256"):
        raise ValueError("WaveBit packed weights failed SHA-256 verification.")
    if _sha256_bytes(auxiliary_payload) != manifest.get("auxiliary_sha256"):
        raise ValueError("WaveBit auxiliary tensors failed SHA-256 verification.")
    if _sha256_bytes(metadata_payload) != manifest.get("group_metadata_sha256"):
        raise ValueError("WaveBit group metadata failed SHA-256 verification.")

    try:
        config_dict = manifest["config"]
        canonical_config = json.dumps(
            config_dict, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if _sha256_bytes(canonical_config) != manifest.get("config_sha256"):
            raise ValueError("WaveBit config failed SHA-256 verification.")
        config = AddSubConfig.from_dict(config_dict)
        if manifest.get("tokenizer_vocab_size") != config.vocab_size:
            raise ValueError("WaveBit tokenizer vocabulary does not match the model config.")
        if manifest.get("checkpoint_format") not in ACCEPTED_CHECKPOINT_FORMATS:
            raise ValueError("WaveBit source checkpoint format is not accepted.")
        if not _valid_sha(manifest.get("tokenizer_sha256")):
            raise ValueError("WaveBit tokenizer fingerprint is missing or invalid.")
        if not _valid_sha(manifest.get("dataset_manifest_sha256")):
            raise ValueError("WaveBit dataset-manifest fingerprint is missing or invalid.")
        options = validate_wavebit_options(manifest.get("wavebit_options"))
        if options is None:
            raise ValueError("WaveBit archive is missing its quantization options.")
        if options["octave_boost_factor"] != manifest["boost_factor"]:
            raise ValueError("WaveBit boost factor disagrees with quantization options.")
        if manifest.get("group_scale", {}).get("group_size") != options["group_size"]:
            raise ValueError("WaveBit group-size metadata disagrees with model options.")
        run_contract = manifest.get("training_run_contract")
        if not isinstance(run_contract, dict):
            raise ValueError("WaveBit training run contract is invalid.")
        contract_options = validate_wavebit_options(run_contract.get("wavebit_hsra"))
        if contract_options != options:
            raise ValueError("WaveBit run contract and archive options disagree.")
    except (KeyError, TypeError) as exc:
        raise ValueError("WaveBit archive manifest is incomplete or malformed.") from exc

    if tokenizer_path is not None:
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.is_file():
            raise FileNotFoundError(tokenizer_file)
        if sha256_file(tokenizer_file) != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer fingerprint does not match the WaveBit archive.")

    model = AddSubLanguageModel(config, seed=config.seed, wavebit_options=options)
    expected_state = model.state_dict()
    quantized_names = _quantized_parameter_names(model)
    quantized_entries = manifest.get("quantized_tensors")
    auxiliary_entries = manifest.get("auxiliary_tensors")
    if not isinstance(quantized_entries, list) or not isinstance(auxiliary_entries, list):
        raise ValueError("WaveBit tensor index is invalid.")
    quantized_entry_names = [entry.get("name") for entry in quantized_entries]
    auxiliary_entry_names = [entry.get("name") for entry in auxiliary_entries]
    if (
        len(set(quantized_entry_names)) != len(quantized_entry_names)
        or set(quantized_entry_names) != quantized_names
    ):
        raise ValueError("WaveBit quantized tensor index does not match the model architecture.")
    expected_auxiliary = set(expected_state) - quantized_names
    if (
        len(set(auxiliary_entry_names)) != len(auxiliary_entry_names)
        or set(auxiliary_entry_names) != expected_auxiliary
    ):
        raise ValueError("WaveBit auxiliary tensor index does not match the model architecture.")

    restored: dict[str, torch.Tensor] = {}
    frozen_group_scales: dict[str, np.ndarray] = {}
    metadata_keys: set[str] = set()
    expected_offset = 0
    group_size = options["group_size"]
    boost_factor = options["octave_boost_factor"]
    base_levels = np.asarray(OCTAVE_BASE_LEVELS, dtype=np.float32)

    try:
        metadata = np.load(io.BytesIO(metadata_payload), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise ValueError("WaveBit group metadata archive is malformed.") from exc
    with metadata:
        for entry in quantized_entries:
            if not isinstance(entry, dict):
                raise ValueError("WaveBit tensor index entry is malformed.")
            name = entry.get("name")
            if name not in quantized_names:
                raise ValueError("WaveBit tensor index contains an unknown parameter.")
            shape = tuple(entry.get("shape", ()))
            if shape != tuple(expected_state[name].shape):
                raise ValueError(f"WaveBit packed tensor shape mismatch for {name!r}.")
            count = int(np.prod(shape, dtype=np.int64))
            if entry.get("count") != count:
                raise ValueError(f"WaveBit tensor element count mismatch for {name!r}.")
            if entry.get("group_size") != group_size:
                raise ValueError(f"WaveBit tensor {name!r} has a mismatched group size.")
            offset, nbytes = entry.get("offset"), entry.get("nbytes")
            if (
                isinstance(offset, bool) or not isinstance(offset, int)
                or isinstance(nbytes, bool) or not isinstance(nbytes, int)
                or offset != expected_offset or nbytes < 0
                or offset + nbytes > len(weights_payload)
            ):
                raise ValueError("WaveBit packed tensor offsets are overlapping or out of bounds.")
            payload = weights_payload[offset : offset + nbytes]
            if _sha256_bytes(payload) != entry.get("sha256"):
                raise ValueError(f"WaveBit packed tensor {name!r} failed SHA-256 verification.")

            rows = shape[0]
            features = count // rows
            expected_group_count = (features + group_size - 1) // group_size
            expected_triplets = (features + 2) // 3
            scale_key = entry.get("group_scale_key")
            offsets_key = entry.get("row_offsets_key")
            if not isinstance(scale_key, str) or not isinstance(offsets_key, str):
                raise ValueError("WaveBit tensor metadata keys are invalid.")
            if scale_key in metadata_keys or offsets_key in metadata_keys:
                raise ValueError("WaveBit metadata keys are duplicated.")
            metadata_keys.update((scale_key, offsets_key))
            if scale_key not in metadata.files or offsets_key not in metadata.files:
                raise ValueError("WaveBit group metadata is missing a tensor index.")
            group_scales = metadata[scale_key]
            row_offsets = metadata[offsets_key]
            if (
                group_scales.dtype != np.float32
                or group_scales.shape != (rows, expected_group_count)
                or not np.isfinite(group_scales).all()
                or np.any(group_scales <= 0.0)
                or list(group_scales.shape) != entry.get("group_scale_shape")
            ):
                raise ValueError(f"WaveBit group scales for {name!r} are invalid.")
            if (
                row_offsets.dtype != np.uint32
                or row_offsets.shape != (rows + 1,)
                or list(row_offsets.shape) != entry.get("row_offsets_shape")
                or row_offsets[0] != 0
                or row_offsets[-1] != nbytes
                or np.any(row_offsets[1:] < row_offsets[:-1])
            ):
                raise ValueError(f"WaveBit row offsets for {name!r} are invalid.")

            codes, modes = unpack_octave_stream(
                np.frombuffer(payload, dtype=np.uint8),
                row_offsets,
                (rows, features),
            )
            if modes.shape != (rows, expected_triplets):
                raise ValueError(f"WaveBit mode shape mismatch for {name!r}.")
            boosted = np.repeat(modes, 3, axis=1)[:, :features]
            factors = np.where(boosted, boost_factor, 1).astype(np.float32)
            expanded_scales = np.repeat(group_scales, group_size, axis=1)[:, :features]
            values = base_levels[codes] * factors * expanded_scales
            restored[name] = torch.from_numpy(
                np.ascontiguousarray(values.reshape(shape), dtype=np.float32)
            )
            frozen_group_scales[name] = np.ascontiguousarray(
                expanded_scales.reshape(shape), dtype=np.float32
            )
            expected_offset += nbytes

        if set(metadata.files) != metadata_keys:
            raise ValueError("WaveBit group metadata contains unindexed arrays.")
    if expected_offset != len(weights_payload):
        raise ValueError("WaveBit packed payload has unindexed trailing bytes.")

    try:
        with np.load(io.BytesIO(auxiliary_payload), allow_pickle=False) as auxiliary:
            if set(auxiliary.files) != expected_auxiliary:
                raise ValueError("WaveBit auxiliary tensors do not match the manifest.")
            shape_index = {
                entry["name"]: tuple(entry["shape"])
                for entry in auxiliary_entries
            }
            for name in sorted(expected_auxiliary):
                array = auxiliary[name]
                if (
                    array.dtype != np.float32
                    or tuple(array.shape) != shape_index[name]
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"WaveBit auxiliary tensor {name!r} has invalid shape/data.")
                restored[name] = torch.from_numpy(np.array(array, copy=True))
    except (ValueError, OSError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("WaveBit auxiliary"):
            raise
        raise ValueError("WaveBit auxiliary archive is malformed.") from exc

    if set(restored) != set(expected_state):
        raise ValueError("WaveBit export does not reconstruct the complete model state.")
    model.load_state_dict(restored, strict=True)
    if model.parameter_count() != manifest.get("parameter_count"):
        raise ValueError("WaveBit archive parameter count does not match the model.")
    model.to(torch.device(device_name))
    for name, scales in frozen_group_scales.items():
        parameter = model.get_parameter(name)
        parameter._qmini_fixed_group_scale = torch.as_tensor(
            scales, device=parameter.device, dtype=parameter.dtype
        )
    model.eval()
    return model, manifest
