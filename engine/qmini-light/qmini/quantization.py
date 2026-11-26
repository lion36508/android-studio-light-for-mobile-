from __future__ import annotations

import numpy as np
import torch


def quantize_quaternary(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Q2-v2 fake quantization with {-1,-0.5,+0.5,+1} normalized symbols.

    Each row scale is 2 * mean(abs(original row)). An all-zero row retains
    scale zero and reconstructs to exact zeros. Non-zero rows use +0.5 for
    exact-zero entries because the production codebook has no zero symbol.
    """
    if weight.ndim < 2 or any(size < 1 for size in weight.shape):
        raise ValueError("Weight tensor must have at least two non-empty dimensions.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if not torch.isfinite(weight).all():
        raise ValueError("Q2 quantization rejects non-finite weights.")
    flat = weight.reshape(weight.shape[0], -1)
    scale = 2.0 * flat.detach().abs().mean(dim=-1, keepdim=True)
    safe_scale = scale.clamp_min(eps)
    magnitude = torch.where(
        flat.detach().abs() / safe_scale >= 0.75, 1.0, 0.5
    )
    sign = torch.where(flat.detach() < 0, -1.0, 1.0)
    q = (sign * magnitude * scale).reshape_as(weight)
    return weight + (q - weight).detach()


def quantize_int8_rowwise(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric signed-INT8 weight quantization with one AbsMax scale per row."""
    if weight.ndim < 2:
        raise ValueError("Weight tensor must have at least two dimensions.")
    if not torch.isfinite(weight).all():
        raise ValueError("INT8 quantization rejects non-finite weights.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    flat = weight.detach().float().reshape(weight.shape[0], -1)
    scales = (flat.abs().amax(dim=-1) / 127.0).clamp_min(eps)
    quantized = torch.round(flat / scales[:, None]).clamp(-127, 127)
    return quantized.reshape_as(weight).to(torch.int8), scales.to(torch.float32)


def dequantize_int8_rowwise(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize row-scaled INT8 weights while preserving the original tensor shape."""
    if quantized.ndim < 2 or quantized.dtype != torch.int8:
        raise TypeError("INT8 weights must be an int8 tensor with at least two dimensions.")
    if scales.dtype != torch.float32 or scales.shape != (quantized.shape[0],):
        raise TypeError("INT8 row scales must be a float32 vector, one scale per row.")
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise ValueError("INT8 row scales must be finite and positive.")
    broadcast_shape = (scales.shape[0],) + (1,) * (quantized.ndim - 1)
    return quantized.float() * scales.reshape(broadcast_shape)


def quantize_int8_per_token(
    activation: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed INT8 values and FP32 per-token AbsMax scales for native inference."""
    if activation.ndim < 1:
        raise ValueError("Activation tensor must have a feature dimension.")
    flat = activation.detach().reshape(-1, activation.shape[-1]).float()
    scale = flat.abs().amax(dim=-1).clamp_min(eps) / 127.0
    quantized = torch.round(flat / scale.unsqueeze(1)).clamp(-127, 127)
    return quantized.to(torch.int8), scale.to(torch.float32)


def fake_quantize_int8(activation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-token AbsMax signed INT8 fake quantization with STE."""
    if activation.ndim < 1:
        raise ValueError("Activation tensor must have a feature dimension.")
    scale = activation.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(activation / scale).clamp(-127, 127) * scale
    return activation + (q - activation).detach()


def pack_quaternary(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Pack Q2-v2 normalized symbols: 00=-1, 01=-0.5, 10=+0.5, 11=+1.

    Row scales are exactly 2 * mean(abs(original row)). All-zero rows use
    scale 0; their symbol bytes are ignored during reconstruction.
    """
    if weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] < 1:
        raise ValueError("Expected a non-empty [out_features, in_features] matrix.")
    values = weight.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Q2 packing rejects non-finite weights.")
    scales = (2.0 * np.mean(np.abs(values), axis=1)).astype(np.float32)
    # Match the QAT quantizer's clamp_min(1e-8) decision threshold,
    # while retaining the exact (possibly zero) scale in the packed artifact.
    safe_scales = np.maximum(scales, 1e-8)
    magnitude = np.where(
        np.abs(values) / safe_scales[:, None] >= 0.75, 1.0, 0.5
    )
    normalized = np.where(values < 0.0, -magnitude, magnitude)
    codes = np.where(
        normalized == -1.0, 0,
        np.where(normalized == -0.5, 1,
                 np.where(normalized == 0.5, 2, 3)),
    ).astype(np.uint8)
    rows, columns = codes.shape
    padded = np.full((rows, ((columns + 3) // 4) * 4), 2, dtype=np.uint8)
    padded[:, :columns] = codes
    packed = (
        padded[:, 0::4]
        | (padded[:, 1::4] << 2)
        | (padded[:, 2::4] << 4)
        | (padded[:, 3::4] << 6)
    )
    return packed, scales


def unpack_quaternary(
    packed: np.ndarray,
    scales: np.ndarray,
    in_features: int,
    format_version: int = 2,
) -> np.ndarray:
    """Decode Q2-v1 legacy or Q2-v2 normalized four-symbol row-scaled matrices."""
    packed = np.asarray(packed)
    scales = np.asarray(scales)
    if packed.dtype != np.uint8 or scales.dtype != np.float32:
        raise TypeError("Q2 payload requires uint8 packed values and float32 row scales.")
    if packed.ndim != 2 or scales.ndim != 1 or packed.shape[0] != scales.shape[0]:
        raise ValueError("Packed matrix and scale dimensions do not match.")
    if in_features < 1 or packed.shape[1] * 4 < in_features:
        raise ValueError("Invalid original input dimension for packed weights.")
    if not np.isfinite(scales).all():
        raise ValueError("Q2 row scales must be finite.")
    if format_version == 1:
        if np.any(scales <= 0):
            raise ValueError("Q2-v1 row scales must be positive.")
        codebook = (-2.0, -1.0, 1.0, 2.0)
    elif format_version == 2:
        if np.any(scales < 0):
            raise ValueError("Q2-v2 row scales must be non-negative.")
        codebook = (-1.0, -0.5, 0.5, 1.0)
    else:
        raise ValueError(f"Unsupported Q2 format version: {format_version!r}.")

    codes = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
    for offset in range(4):
        codes[:, offset::4] = (packed >> (2 * offset)) & 3
    values = np.asarray(codebook, dtype=np.float32)[codes]
    return values[:, :in_features] * scales[:, None]


def quantize_ternary_rowwise(
    weight: torch.Tensor,
    threshold_ratio: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-preserving row-wise ternary quantization for 2D+ weight tensors.

    Returns int8 values in {-1, 0, +1} and one FP32 absolute-mean scale per
    output row. Exact zero weights always remain zero.
    """
    if weight.ndim < 2:
        raise ValueError("Weight tensor must have at least two dimensions.")
    if not torch.isfinite(weight).all():
        raise ValueError("Ternary quantization rejects non-finite weights.")
    if not np.isfinite(threshold_ratio) or not 0.0 <= threshold_ratio < 1.0:
        raise ValueError("threshold_ratio must be finite and in [0, 1).")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")

    flat = weight.detach().float().reshape(weight.shape[0], -1)
    scales = flat.abs().mean(dim=-1).clamp_min(eps)
    threshold = scales.unsqueeze(1) * float(threshold_ratio)
    ternary = torch.where(
        flat.abs() < threshold,
        torch.zeros_like(flat),
        torch.sign(flat),
    )
    return ternary.reshape_as(weight).to(torch.int8), scales.to(torch.float32)


def pack_ternary(
    weight: torch.Tensor,
    threshold_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack four {-1,0,+1} weights per byte using two bits per value.

    Codes are -1 -> 0, 0 -> 1, +1 -> 2; code 3 is reserved and rejected
    by the decoder. Padding values are encoded as zero.
    """
    if weight.ndim != 2:
        raise ValueError("Expected a [out_features, in_features] matrix.")
    ternary, scales = quantize_ternary_rowwise(weight, threshold_ratio)
    values = ternary.detach().cpu().numpy()
    rows, columns = values.shape
    codes = (values.astype(np.int16) + 1).astype(np.uint8)
    padded = np.ones((rows, ((columns + 3) // 4) * 4), dtype=np.uint8)
    padded[:, :columns] = codes
    packed = (
        padded[:, 0::4]
        | (padded[:, 1::4] << 2)
        | (padded[:, 2::4] << 4)
        | (padded[:, 3::4] << 6)
    )
    return packed, scales.cpu().numpy().astype(np.float32)


def unpack_ternary(
    packed: np.ndarray,
    scales: np.ndarray,
    in_features: int,
) -> np.ndarray:
    packed = np.asarray(packed)
    scales = np.asarray(scales)
    if packed.dtype != np.uint8 or scales.dtype != np.float32:
        raise TypeError("Ternary archive requires uint8 packed values and float32 scales.")
    if packed.ndim != 2 or scales.ndim != 1 or packed.shape[0] != scales.shape[0]:
        raise ValueError("Packed matrix and scale dimensions do not match.")
    if in_features < 1 or packed.shape[1] * 4 < in_features:
        raise ValueError("Invalid original input dimension for packed ternary weights.")
    codes = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
    for offset in range(4):
        codes[:, offset::4] = (packed >> (2 * offset)) & 3
    if np.any(codes[:, :in_features] == 3):
        raise ValueError("Ternary archive contains reserved code 3.")
    values = codes.astype(np.int8) - 1
    return values[:, :in_features].astype(np.float32) * scales[:, None]


def pack_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 64,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack signed symmetric INT4 weights with one AbsMax scale per row group."""
    if weight.ndim != 2:
        raise ValueError("Expected a [out_features, in_features] matrix.")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer.")
    if not torch.isfinite(weight).all():
        raise ValueError("INT4 quantization rejects non-finite weights.")
    values = weight.detach().float().cpu().numpy()
    rows, columns = values.shape
    groups = (columns + group_size - 1) // group_size
    padded_columns = groups * group_size
    padded = np.zeros((rows, padded_columns), dtype=np.float32)
    padded[:, :columns] = values
    grouped = padded.reshape(rows, groups, group_size)
    scales = np.maximum(
        np.max(np.abs(grouped), axis=-1, keepdims=True) / 7.0, eps
    ).astype(np.float32)
    quantized = np.rint(grouped / scales).clip(-7, 7).astype(np.int8)
    codes = (quantized.astype(np.int16) & 0x0F).astype(np.uint8)
    flattened = codes.reshape(rows, padded_columns)
    if padded_columns % 2:
        flattened = np.pad(flattened, ((0, 0), (0, 1)), constant_values=0)
    packed = flattened[:, 0::2] | (flattened[:, 1::2] << 4)
    return packed.astype(np.uint8), scales.squeeze(-1).astype(np.float32)


def unpack_int4_groupwise(
    packed: np.ndarray,
    scales: np.ndarray,
    in_features: int,
    group_size: int = 64,
) -> np.ndarray:
    packed = np.asarray(packed)
    scales = np.asarray(scales)
    if packed.dtype != np.uint8 or scales.dtype != np.float32:
        raise TypeError("INT4 archive requires uint8 packed values and float32 scales.")
    if packed.ndim != 2 or scales.ndim != 2 or packed.shape[0] != scales.shape[0]:
        raise ValueError("INT4 packed matrix and scale dimensions do not match.")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer.")
    groups = scales.shape[1]
    padded_columns = groups * group_size
    if in_features < 1 or in_features > padded_columns:
        raise ValueError("Invalid original input dimension for packed INT4 weights.")
    expected_packed_columns = (padded_columns + 1) // 2
    if packed.shape[1] != expected_packed_columns:
        raise ValueError("INT4 packed width does not match its group scales.")
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F
    signed = codes[:, :padded_columns].astype(np.int8)
    signed = np.where(signed >= 8, signed - 16, signed).astype(np.float32)
    grouped = signed.reshape(packed.shape[0], groups, group_size)
    dequantized = grouped * scales[:, :, None]
    return dequantized.reshape(packed.shape[0], padded_columns)[:, :in_features]


def quantization_error_metrics(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Report deterministic tensor-level error; this is not a quality metric."""
    if reference.shape != approximation.shape:
        raise ValueError("Reference and approximation tensor shapes must match.")
    if not torch.isfinite(reference).all() or not torch.isfinite(approximation).all():
        raise ValueError("Quantization error metrics require finite tensors.")
    ref = reference.detach().float().reshape(-1)
    approx = approximation.detach().float().reshape(-1)
    difference = approx - ref
    relative_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(ref).clamp_min(eps)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(ref.unsqueeze(0), approx.unsqueeze(0), dim=-1)[0]
    )
    return {
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "max_absolute_error": float(difference.abs().max()),
    }
