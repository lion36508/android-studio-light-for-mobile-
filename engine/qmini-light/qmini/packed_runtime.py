from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QMiniConfig
from .quantization import (
    fake_quantize_int8,
    quantize_int8_per_token,
    unpack_quaternary,
)


def _buffer_array(module: nn.Module, name: str) -> np.ndarray:
    value = getattr(module, name, None)
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise RuntimeError(f"Packed runtime buffer {name} is missing or is not on CPU.")
    return value.detach().contiguous().numpy()


def _unpack_rows(
    packed: np.ndarray,
    scales: np.ndarray,
    row_ids: np.ndarray,
    in_features: int,
    format_version: int = 2,
) -> np.ndarray:
    selected = np.asarray(packed[row_ids], dtype=np.uint8)
    selected_scales = np.asarray(scales[row_ids], dtype=np.float32).reshape(-1, 1)
    codes = np.stack(
        [((selected >> shift) & 3) for shift in (0, 2, 4, 6)],
        axis=-1,
    ).reshape(selected.shape[0], -1)[:, :in_features]
    if format_version == 1:
        codebook = (-2.0, -1.0, 1.0, 2.0)
    elif format_version == 2:
        codebook = (-1.0, -0.5, 0.5, 1.0)
    else:
        raise ValueError(f"Unsupported Q2 format version: {format_version!r}.")
    values = np.asarray(codebook, dtype=np.float32)[codes]
    return values * selected_scales


def packed_embedding_lookup(module: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    if token_ids.device.type != "cpu":
        raise RuntimeError("Q-Mini packed 2-bit runtime currently supports CPU inference only.")
    if module.training:
        raise RuntimeError("Packed 2-bit weights are inference-only; call model.eval().")
    packed = _buffer_array(module, "_qmini_packed_weight")
    scales = _buffer_array(module, "_qmini_weight_scales")
    ids = token_ids.detach().to(dtype=torch.long).numpy().reshape(-1)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= packed.shape[0]):
        raise ValueError("Token IDs are outside the packed embedding vocabulary.")
    rows = _unpack_rows(
        packed, scales, ids, int(module._qmini_in_features),
        int(getattr(module, "_qmini_q2_version", 2)),
    )
    result = torch.from_numpy(rows).reshape(
        *token_ids.shape, int(module._qmini_in_features)
    )
    return fake_quantize_int8(result)


def _fallback_packed_linear(
    module: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    packed = _buffer_array(module, "_qmini_packed_weight")
    scales = _buffer_array(module, "_qmini_weight_scales")
    in_features = int(module._qmini_in_features)
    out_features = int(module._qmini_out_features)
    quantized_input = fake_quantize_int8(x)

    # The tied vocabulary projection can be large. Decode it in bounded row
    # chunks so CPU fallback never materializes the complete FP32 vocabulary.
    chunk_rows = 256 if module.__class__.__name__ == "QEmbedding" else out_features
    outputs: list[torch.Tensor] = []
    for start in range(0, out_features, chunk_rows):
        end = min(out_features, start + chunk_rows)
        weight = torch.from_numpy(
            unpack_quaternary(
                packed[start:end], scales[start:end], in_features,
                format_version=int(getattr(module, "_qmini_q2_version", 2)),
            )
        )
        chunk_bias = (
            bias[start:end].float()
            if bias is not None else None
        )
        outputs.append(F.linear(quantized_input, weight, chunk_bias))
    return torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]


def packed_linear_forward(
    module: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if module.training:
        raise RuntimeError("Packed 2-bit weights are inference-only; call model.eval().")
    if x.device.type != "cpu":
        raise RuntimeError("Q-Mini packed 2-bit runtime currently supports CPU inference only.")
    if x.ndim < 1 or x.shape[-1] != int(module._qmini_in_features):
        raise ValueError("Input feature dimension does not match packed Q-Mini weight.")

    from .native import get_native_kernel, native_kernel_should_be_used

    kernel = get_native_kernel()
    if native_kernel_should_be_used(kernel) and x.dtype == torch.float32:
        activations_i8, activation_scales = quantize_int8_per_token(x)
        bias_array = (
            bias.detach().float().contiguous().numpy()
            if bias is not None else None
        )
        output = kernel.linear(
            activations_i8.contiguous().numpy(),
            _buffer_array(module, "_qmini_packed_weight"),
            _buffer_array(module, "_qmini_weight_scales"),
            activation_scales.contiguous().numpy(),
            bias_array,
            q2_version=int(getattr(module, "_qmini_q2_version", 2)),
        )
        return torch.from_numpy(output).reshape(
            *x.shape[:-1], int(module._qmini_out_features)
        )
    return _fallback_packed_linear(module, x, bias)


def _resolve_parent(model: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    if "." in full_name:
        parent_name, leaf = full_name.rsplit(".", 1)
        return model.get_submodule(parent_name), leaf
    return model, full_name


def load_packed_model(
    path: str | Path,
    expected_config: QMiniConfig | None = None,
) -> nn.Module:
    """Load a packed Q2 archive without ever allocating the full FP32 model."""
    from .model import CausalGatedDepthwiseConv, QEmbedding, QLinear, QMiniMoE

    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Packed Q-Mini model not found: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError("Packed Q-Mini archive is missing metadata_json.")
        metadata = json.loads(str(archive["metadata_json"].item()))
        archive_format = metadata.get("format")
        if archive_format not in {"qmini-packed-v1", "qmini-packed-v2"}:
            raise ValueError("Unsupported Q-Mini packed export format.")
        q2_version = 1 if archive_format == "qmini-packed-v1" else 2
        if q2_version == 2:
            expected_contract = {
                "version": 2,
                "codebook": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
                "row_scale": "2 * mean(abs(original_output_row))",
                "all_zero_row_scale": 0.0,
            }
            if (
                metadata.get("q2_contract") != expected_contract
                or metadata.get("quaternary_codes") != expected_contract["codebook"]
            ):
                raise ValueError("Q2-v2 archive contract/codebook metadata is missing or invalid.")
        config = QMiniConfig.from_dict(metadata["config"])
        if expected_config is not None and config.to_dict() != expected_config.to_dict():
            raise ValueError("Packed model config does not match config.json.")

        # Meta tensors preserve the exact Python module graph and tensor shapes
        # while allocating no dense FP32 parameter storage.
        with torch.device("meta"):
            model = QMiniMoE(config)

        expected_parameters = dict(model.named_parameters())
        specs = metadata.get("tensors")
        if not isinstance(specs, list) or len(specs) != len(expected_parameters):
            raise ValueError("Packed archive tensor manifest does not match model parameter count.")

        observed_names: set[str] = set()
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError("Packed archive contains an invalid tensor specification.")
            name = str(spec.get("name", ""))
            if not name or name in observed_names or name not in expected_parameters:
                raise ValueError(f"Unexpected or duplicate packed tensor name: {name!r}")
            observed_names.add(name)
            parameter = expected_parameters[name]
            shape = tuple(int(value) for value in spec.get("shape", []))
            if shape != tuple(parameter.shape):
                raise ValueError(f"Packed tensor shape mismatch for {name}: {shape} != {tuple(parameter.shape)}")
            parent, leaf = _resolve_parent(model, name)
            kind = spec.get("kind")

            if kind == "q2":
                if parameter.ndim < 2:
                    raise ValueError(f"Q2 tensor {name} must have rank 2 or greater.")
                packed = np.asarray(archive[spec["packed_key"]])
                scales = np.asarray(archive[spec["scale_key"]])
                rows = shape[0]
                in_features = math.prod(shape[1:])
                if packed.dtype != np.uint8 or packed.shape != (rows, (in_features + 3) // 4):
                    raise ValueError(f"Invalid packed matrix dimensions for {name}.")
                if scales.dtype != np.float32 or scales.shape != (rows,):
                    raise ValueError(f"Invalid Q2 row scales for {name}.")
                if not np.isfinite(scales).all() or (
                    np.any(scales <= 0) if q2_version == 1 else np.any(scales < 0)
                ):
                    raise ValueError(f"Non-finite or invalid Q2 scale in {name}.")
                if not isinstance(parent, (QLinear, QEmbedding, CausalGatedDepthwiseConv)):
                    raise ValueError(f"Unsupported Q2 parameter owner for {name}.")
                if isinstance(parent, CausalGatedDepthwiseConv) and leaf != "depthwise_weight":
                    raise ValueError(f"Unexpected convolution Q2 parameter: {name}.")
                if isinstance(parent, CausalGatedDepthwiseConv) and in_features != parent.kernel_size:
                    raise ValueError(f"Convolution kernel width mismatch for {name}.")

                parent.register_buffer(
                    "_qmini_packed_weight", torch.from_numpy(packed), persistent=False
                )
                parent.register_buffer(
                    "_qmini_weight_scales", torch.from_numpy(scales), persistent=False
                )
                parent._qmini_q2_version = q2_version
                parent._qmini_weight_shape = shape
                parent._qmini_in_features = in_features
                parent._qmini_out_features = rows
                parent._parameters[leaf] = None
            elif kind == "fp16":
                if parameter.ndim >= 2:
                    raise ValueError(f"Non-Q2 matrix found in packed archive: {name}.")
                value = np.asarray(archive[spec["value_key"]])
                if value.dtype != np.float16 or value.shape != shape:
                    raise ValueError(f"Invalid FP16 vector shape or dtype for {name}.")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite FP16 parameter found in {name}.")
                parent._parameters[leaf] = nn.Parameter(
                    torch.from_numpy(value), requires_grad=False
                )
            else:
                raise ValueError(f"Unknown packed tensor encoding for {name}: {kind!r}")

        if observed_names != set(expected_parameters):
            missing = sorted(set(expected_parameters) - observed_names)
            raise ValueError("Packed archive is missing model parameters: " + ", ".join(missing[:8]))

    model.eval()
    model._qmini_packed_runtime = True
    model._qmini_packed_path = str(archive_path.resolve())
    model._qmini_packed_file_bytes = archive_path.stat().st_size
    model._qmini_original_parameter_count = config.estimated_parameter_count()
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise RuntimeError("Packed loader left an unmaterialized meta parameter.")
    if any(parameter.ndim >= 2 for parameter in model.parameters()):
        raise RuntimeError("Packed loader unexpectedly materialized a dense matrix parameter.")
    return model


def load_packed_state_into_trainable_model(
    model: nn.Module,
    path: str | Path,
) -> None:
    """Stream-dequantize an archive into an already allocated training model."""
    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        archive_format = metadata.get("format")
        if archive_format not in {"qmini-packed-v1", "qmini-packed-v2"}:
            raise ValueError("Unsupported Q-Mini packed export format.")
        q2_version = 1 if archive_format == "qmini-packed-v1" else 2
        if q2_version == 2:
            expected_contract = {
                "version": 2,
                "codebook": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
                "row_scale": "2 * mean(abs(original_output_row))",
                "all_zero_row_scale": 0.0,
            }
            if (
                metadata.get("q2_contract") != expected_contract
                or metadata.get("quaternary_codes") != expected_contract["codebook"]
            ):
                raise ValueError("Q2-v2 archive contract/codebook metadata is missing or invalid.")
        config = QMiniConfig.from_dict(metadata["config"])
        if config.to_dict() != model.config.to_dict():
            raise ValueError("Packed initialization config differs from the trainable model.")
        parameters = dict(model.named_parameters())
        if len(metadata.get("tensors", [])) != len(parameters):
            raise ValueError("Packed initialization manifest has an unexpected tensor count.")

        with torch.no_grad():
            seen: set[str] = set()
            for spec in metadata["tensors"]:
                name = str(spec["name"])
                if name in seen or name not in parameters:
                    raise ValueError(f"Unexpected or duplicate packed initialization tensor: {name!r}")
                seen.add(name)
                parameter = parameters[name]
                shape = tuple(int(value) for value in spec["shape"])
                if shape != tuple(parameter.shape):
                    raise ValueError(f"Packed initialization shape mismatch for {name}.")
                if spec["kind"] == "q2":
                    packed = np.asarray(archive[spec["packed_key"]], dtype=np.uint8)
                    scales = np.asarray(archive[spec["scale_key"]], dtype=np.float32)
                    if not np.isfinite(scales).all() or (
                        np.any(scales <= 0) if q2_version == 1 else np.any(scales < 0)
                    ):
                        raise ValueError(f"Invalid Q2 row scales in packed initialization tensor: {name}.")
                    value = unpack_quaternary(
                        packed, scales, math.prod(shape[1:]),
                        format_version=q2_version,
                    ).reshape(shape)
                elif spec["kind"] == "fp16":
                    value = np.asarray(archive[spec["value_key"]], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown packed initialization encoding: {spec['kind']!r}")
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid packed initialization tensor: {name}.")
                parameter.copy_(torch.from_numpy(np.asarray(value, dtype=np.float32)).to(
                    device=parameter.device, dtype=parameter.dtype
                ))
                del value
        if seen != set(parameters):
            raise ValueError("Packed initialization archive is missing trainable model tensors.")
