"""WaveBit-HSRA components for the single evolving Q-Mini Add/Sub model.

The PyTorch path is a training surrogate, not a claim of float-free execution.
The integer routines are reference kernels for isolated primitives only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


OCTAVE_BASE_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
OCTAVE_SKIP_TRIPLES = {250: 4, 251: 8, 252: 16, 253: 32, 254: 64}
OCTAVE_ESCAPE_CODE = 255

# Compatibility constants for the earlier isolated WaveBit integer reference.
POLARITY_BY_CODE = (-1, -1, 0, 1, 1)
PHASE_BY_CODE = (1, 0, 0, 0, 1)

DEFAULT_WAVEBIT_OPTIONS = {
    "group_size": 30,
    "rotation": 1,
    "gamma": 4,
    "ewgs_strength": 0.5,
    "octave_temperature": 0.25,
    "context_threshold": 0.5,
    "context_temperature": 0.75,
    "state_limit": 16.0,
    # 4 preserves existing checkpoints; x2 is an explicit research ablation.
    "octave_boost_factor": 4,
    "hadamard": False,
    "hadamard_block_size": 8,
    "activation_bits": 4,
}


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, not a boolean or string.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validated_octave_shape(original_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(original_shape, (tuple, list)) or len(original_shape) != 2:
        raise ValueError("original_shape must contain exactly two positive integer dimensions.")
    dimensions = []
    for dimension in original_shape:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(
            dimension, (int, np.integer)
        ):
            raise TypeError("original_shape dimensions must be integers; booleans and floats are invalid.")
        if dimension < 1:
            raise ValueError("original_shape dimensions must be positive.")
        dimensions.append(int(dimension))
    return dimensions[0], dimensions[1]


def validate_wavebit_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TypeError("wavebit_hsra options must be a dictionary.")
    unknown = set(options) - set(DEFAULT_WAVEBIT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown WaveBit-HSRA options: {sorted(unknown)}")
    result = {**DEFAULT_WAVEBIT_OPTIONS, **options}
    boost_factor = result["octave_boost_factor"]
    if isinstance(boost_factor, bool) or not isinstance(boost_factor, int) or boost_factor not in (2, 4):
        raise ValueError("Octave-Shift boost factor must be 2 or 4.")
    if not isinstance(result["hadamard"], bool):
        raise TypeError("WaveBit hadamard must be a boolean.")
    block_size = result["hadamard_block_size"]
    if (
        isinstance(block_size, bool) or not isinstance(block_size, int)
        or block_size < 1 or block_size > 64 or block_size & (block_size - 1)
    ):
        raise ValueError("Hadamard block size must be a power of two in [1, 64].")
    activation_bits = result["activation_bits"]
    if (
        isinstance(activation_bits, bool) or not isinstance(activation_bits, int)
        or activation_bits < 2 or activation_bits > 8
    ):
        raise ValueError("Activation quantization bits must be an integer in [2, 8].")
    if isinstance(result["group_size"], bool) or not isinstance(result["group_size"], int) or result["group_size"] < 3 or result["group_size"] % 3:
        raise ValueError("Octave-Shift group_size must be a positive multiple of three.")
    if isinstance(result["gamma"], bool) or not isinstance(result["gamma"], int) or not 1 <= result["gamma"] <= 15:
        raise ValueError("WaveBit gamma must be an integer in [1, 15].")
    if isinstance(result["rotation"], bool) or not isinstance(result["rotation"], int):
        raise ValueError("WaveBit rotation must be an integer.")
    ewgs_strength = _finite_real(result["ewgs_strength"], "WaveBit ewgs_strength")
    if not 0.0 <= ewgs_strength <= 2.0:
        raise ValueError("WaveBit ewgs_strength must be in [0, 2].")
    octave_temperature = _finite_real(result["octave_temperature"], "Octave-Shift temperature")
    if octave_temperature <= 0.0:
        raise ValueError("Octave-Shift temperature must be positive.")
    result["octave_temperature"] = octave_temperature
    for key in ("context_threshold", "context_temperature", "state_limit"):
        value = _finite_real(result[key], f"WaveBit {key}")
        if value <= 0.0:
            raise ValueError(f"WaveBit {key} must be positive.")
        result[key] = value
    result["ewgs_strength"] = ewgs_strength
    return result


def groupwise_scale(weight: torch.Tensor, group_size: int = 32, multiplier: float = 1.4) -> torch.Tensor:
    """Detached per-output-row scale, independently estimated for each input group."""
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("WaveBit weights must be floating-point tensors.")
    if weight.ndim < 2 or weight.numel() == 0:
        raise ValueError("WaveBit weights must have at least two non-empty dimensions.")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer.")
    multiplier = _finite_real(multiplier, "scale multiplier")
    if multiplier <= 0.0:
        raise ValueError("scale multiplier must be positive.")
    fixed_scale = getattr(weight, "_qmini_fixed_group_scale", None)
    if fixed_scale is not None:
        value = torch.as_tensor(fixed_scale, device=weight.device, dtype=weight.dtype)
        if value.shape != weight.shape or not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("Frozen group-wise quantization scale is invalid.")
        return value.detach()
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("WaveBit weights contain non-finite values.")
    flat = weight.detach().reshape(weight.shape[0], -1)
    scaled = torch.empty_like(flat)
    for start in range(0, flat.shape[1], group_size):
        stop = min(start + group_size, flat.shape[1])
        block = flat[:, start:stop]
        scale = (block.abs().mean(dim=1, keepdim=True) * multiplier).clamp_min(1e-8)
        scaled[:, start:stop] = scale.expand_as(block)
    return scaled.reshape_as(weight).detach()


def _hard_octave_symbol(normalized: torch.Tensor) -> torch.Tensor:
    """Five base levels with symmetric nearest-neighbour decision boundaries."""
    x = normalized.clamp(-1.0, 1.0)
    return torch.where(
        x <= -0.75, -1.0,
        torch.where(x < -0.25, -0.5,
                    torch.where(x <= 0.25, 0.0,
                                torch.where(x < 0.75, 0.5, 1.0))),
    )


def octave_shift_quantize(
    weight: torch.Tensor,
    group_size: int = 30,
    ewgs_strength: float = 0.5,
    octave_temperature: float = 0.25,
    boost_factor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five base symbols plus one shared x2 or x4 mode bit per triplet."""
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("Octave-Shift weights must be floating-point tensors.")
    if weight.ndim < 2 or weight.numel() == 0:
        raise ValueError("Octave-Shift weights must have at least two non-empty dimensions.")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 3 or group_size % 3:
        raise ValueError("Octave-Shift group_size must be a positive multiple of three.")
    ewgs_strength = _finite_real(ewgs_strength, "ewgs_strength")
    octave_temperature = _finite_real(octave_temperature, "octave_temperature")
    if not 0.0 <= ewgs_strength <= 2.0:
        raise ValueError("ewgs_strength must be in [0, 2].")
    if octave_temperature <= 0.0:
        raise ValueError("octave_temperature must be positive.")
    if isinstance(boost_factor, bool) or not isinstance(boost_factor, int) or boost_factor not in (2, 4):
        raise ValueError("boost_factor must be the integer 2 or 4.")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Octave-Shift weights contain non-finite values.")

    scale = groupwise_scale(weight, group_size=group_size)
    rows = weight.shape[0]
    flat_weight = weight.reshape(rows, -1)
    flat_scale = scale.reshape(rows, -1)
    features = flat_weight.shape[1]
    padding = (-features) % 3
    if padding:
        flat_weight = torch.cat(
            (flat_weight, flat_weight.new_zeros((rows, padding))), dim=1
        )
        flat_scale = torch.cat(
            (flat_scale, flat_scale[:, -1:].expand(-1, padding)), dim=1
        )

    normalized = flat_weight / flat_scale
    grouped = normalized.reshape(rows, -1, 3)
    normal_hard = _hard_octave_symbol(grouped)
    boost_base_hard = _hard_octave_symbol(grouped / float(boost_factor))
    boost_hard = boost_base_hard * float(boost_factor)

    normal_error = (grouped - normal_hard).square().sum(dim=-1)
    boost_error = (grouped - boost_hard).square().sum(dim=-1)
    boost_hard_mode = boost_error < normal_error
    boost_soft_mode = torch.sigmoid(
        (normal_error - boost_error) / float(octave_temperature)
    )
    boost_mode_ste = (
        boost_hard_mode.to(weight.dtype).detach()
        + boost_soft_mode
        - boost_soft_mode.detach()
    )

    normal_q = _EWGSQuantize.apply(
        flat_weight,
        normal_hard.reshape(rows, -1) * flat_scale,
        normalized.detach(),
        normal_hard.reshape(rows, -1).detach(),
        ewgs_strength,
    )
    boost_q = _EWGSQuantize.apply(
        flat_weight,
        boost_hard.reshape(rows, -1) * flat_scale,
        normalized.detach(),
        boost_hard.reshape(rows, -1).detach(),
        ewgs_strength,
    )
    normal_q = normal_q.reshape(rows, -1, 3)
    boost_q = boost_q.reshape(rows, -1, 3)
    selected = normal_q + boost_mode_ste.unsqueeze(-1) * (boost_q - normal_q)
    quantized = selected.reshape(rows, -1)[:, :features].reshape_as(weight)

    selected_base = torch.where(
        boost_hard_mode.unsqueeze(-1), boost_base_hard, normal_hard
    )
    levels = weight.new_tensor(OCTAVE_BASE_LEVELS)
    symbol_codes = (selected_base.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    symbol_codes = symbol_codes.reshape(rows, -1)[:, :features].to(torch.uint8)
    return quantized, symbol_codes, boost_hard_mode, scale


def wavebit_quantize(
    weight: torch.Tensor,
    group_size: int = 30,
    ewgs_strength: float = 0.5,
    octave_temperature: float = 0.25,
    boost_factor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility entry point; the active weight quantizer is Octave-Shift."""
    return octave_shift_quantize(
        weight, group_size=group_size,
        ewgs_strength=ewgs_strength,
        octave_temperature=octave_temperature,
        boost_factor=boost_factor,
    )


def pack_octave_triplets(
    symbol_codes: np.ndarray,
    boost_modes: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Three base-5 symbol codes plus one shared mode bit fit in one byte."""
    codes = np.asarray(symbol_codes)
    modes = np.asarray(boost_modes)
    if codes.dtype.kind not in "iu" or codes.ndim != 2 or min(codes.shape) < 1:
        raise TypeError("Octave symbol_codes must be a non-empty 2-D integer array.")
    if modes.dtype.kind not in "biu" or modes.ndim != 2:
        raise TypeError("Octave boost_modes must be a 2-D boolean/integer array.")
    if np.any(codes < 0) or np.any(codes > 4) or np.any(modes < 0) or np.any(modes > 1):
        raise ValueError("Octave symbols must be [0,4] and boost modes must be [0,1].")
    rows, features = codes.shape
    triples = (features + 2) // 3
    if modes.shape != (rows, triples):
        raise ValueError("Octave boost_modes must contain one flag per three-weight group.")
    padded = np.full((rows, triples * 3), 2, dtype=np.uint8)
    padded[:, :features] = codes.astype(np.uint8)
    packed = (
        padded[:, 0::3].astype(np.uint16)
        + 5 * padded[:, 1::3].astype(np.uint16)
        + 25 * padded[:, 2::3].astype(np.uint16)
        + 125 * modes.astype(np.uint16)
    )
    if np.any(packed > 249):
        raise AssertionError("Octave payload collided with a reserved control code.")
    return packed.astype(np.uint8), (rows, features)


def unpack_octave_triplets(
    packed: np.ndarray,
    original_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode fixed-width triplets; reserved control bytes need the stream decoder."""
    payload = np.asarray(packed)
    if payload.dtype.kind not in "iu" or payload.ndim != 2:
        raise TypeError("Packed Octave payload must be a 2-D integer array.")
    rows, features = _validated_octave_shape(original_shape)
    triples = (features + 2) // 3
    if payload.shape != (rows, triples) or np.any(payload < 0) or np.any(payload > 249):
        raise ValueError("Packed payload shape is wrong or contains reserved control bytes.")
    base = payload.astype(np.uint16) % 125
    modes = (payload >= 125).astype(np.bool_)
    codes = np.stack(
        (base % 5, (base // 5) % 5, base // 25), axis=-1
    ).astype(np.uint8).reshape(rows, triples * 3)
    return codes[:, :features], modes


def pack_octave_stream(
    symbol_codes: np.ndarray,
    boost_modes: np.ndarray,
    *,
    use_zero_skips: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Skip tokens count whole all-zero triplet-groups, preserving pack alignment.

    250..254 skip 4/8/16/32/64 triplet-groups. 255 is followed by one byte
    containing an extended count (1..255 triplet-groups).
    """
    fixed, shape = pack_octave_triplets(symbol_codes, boost_modes)
    rows, _ = shape
    row_streams = []
    for row in range(rows):
        values = fixed[row]
        codes_row = np.asarray(symbol_codes[row], dtype=np.uint8)
        mode_row = np.asarray(boost_modes[row], dtype=np.bool_)
        groups = len(values)
        zero_group = np.zeros(groups, dtype=np.bool_)
        for group in range(groups):
            start = group * 3
            active = codes_row[start:min(start + 3, len(codes_row))]
            zero_group[group] = bool(np.all(active == 2) and not mode_row[group])
        stream_row = []
        index = 0
        while index < groups:
            if use_zero_skips and zero_group[index]:
                end = index
                while end < groups and zero_group[end]:
                    end += 1
                run = end - index
                if run >= 4:
                    while run >= 4:
                        skip = 64 if run >= 64 else 32 if run >= 32 else 16 if run >= 16 else 8 if run >= 8 else 4
                        stream_row.append({4: 250, 8: 251, 16: 252, 32: 253, 64: 254}[skip])
                        index += skip
                        run -= skip
                    for _ in range(run):
                        stream_row.append(int(values[index]))
                        index += 1
                    continue
            stream_row.append(int(values[index]))
            index += 1
        row_streams.append(stream_row)
    offsets = np.zeros(rows + 1, dtype=np.int64)
    for index, values in enumerate(row_streams):
        offsets[index + 1] = offsets[index] + len(values)
    packed_stream = np.asarray(
        [value for values in row_streams for value in values], dtype=np.uint8
    )
    return packed_stream, offsets


def unpack_octave_stream(
    packed_stream: np.ndarray,
    row_offsets: np.ndarray,
    original_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Restore triplets and all-zero groups from the variable-length stream."""
    stream = np.asarray(packed_stream)
    offsets = np.asarray(row_offsets)
    if stream.dtype.kind not in "iu" or stream.ndim != 1 or np.any(stream < 0) or np.any(stream > 255):
        raise ValueError("Octave packed_stream must be a 1-D byte array.")
    if offsets.dtype.kind not in "iu" or offsets.ndim != 1:
        raise TypeError("Octave row_offsets must be a 1-D integer array.")
    rows, features = _validated_octave_shape(original_shape)
    groups = (features + 2) // 3
    if offsets.shape != (rows + 1,) or offsets[0] != 0 or offsets[-1] != len(stream):
        raise ValueError("Octave row_offsets do not match stream length/row count.")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Octave row_offsets must be monotonically increasing.")

    decoded_codes = np.full((rows, groups * 3), 2, dtype=np.uint8)
    decoded_modes = np.zeros((rows, groups), dtype=np.bool_)
    for row in range(rows):
        cursor = int(offsets[row])
        end = int(offsets[row + 1])
        group = 0
        while cursor < end and group < groups:
            token = int(stream[cursor])
            cursor += 1
            if token < 250:
                base = token % 125
                decoded_codes[row, group * 3 : group * 3 + 3] = (
                    base % 5, (base // 5) % 5, base // 25
                )
                decoded_modes[row, group] = token >= 125
                group += 1
                continue
            if token == OCTAVE_ESCAPE_CODE:
                if cursor >= end:
                    raise ValueError("Octave escape byte is missing its skip-count payload.")
                skip = int(stream[cursor])
                cursor += 1
                if skip < 1:
                    raise ValueError("Octave extended skip count must be in [1,255].")
            else:
                skip = OCTAVE_SKIP_TRIPLES[token]
            if group + skip > groups:
                raise ValueError("Octave zero-run skip overruns the declared row shape.")
            group += skip
        if group != groups or cursor != end:
            raise ValueError("Octave packed row did not decode to its declared group count.")
    return decoded_codes[:, :features], decoded_modes


def build_octave_decode_lut(boost_factor: int = 4) -> np.ndarray:
    """256-entry decode table; boost_factor is recorded alongside packed weights."""
    if isinstance(boost_factor, bool) or not isinstance(boost_factor, int) or boost_factor not in (2, 4):
        raise ValueError("boost_factor must be the integer 2 or 4.")
    table = np.full((256, 4), -128, dtype=np.int8)
    half_units = (-2, -1, 0, 1, 2)
    for token in range(250):
        base = token % 125
        boost = token >= 125
        symbol_codes = (base % 5, (base // 5) % 5, base // 25)
        factor = boost_factor if boost else 1
        table[token, :3] = [half_units[code] * factor for code in symbol_codes]
        table[token, 3] = int(boost)
    return table


def integer_octave_projection(
    activation_codes: np.ndarray,
    packed_stream: np.ndarray,
    row_offsets: np.ndarray,
    input_features: int,
    bias_half_units: np.ndarray | None = None,
    boost_factor: int = 4,
) -> np.ndarray:
    """Integer reference: LUT decode plus add/sub/left-shift, result in half-units."""
    if isinstance(boost_factor, bool) or not isinstance(boost_factor, int) or boost_factor not in (2, 4):
        raise ValueError("boost_factor must be the integer 2 or 4.")
    activations = np.asarray(activation_codes)
    stream = np.asarray(packed_stream)
    offsets = np.asarray(row_offsets)
    if activations.dtype.kind not in "iu" or activations.ndim != 2:
        raise TypeError("Octave activations must be a 2-D integer array.")
    if isinstance(input_features, bool) or not isinstance(input_features, int) or input_features < 1:
        raise ValueError("input_features must be a positive integer.")
    if activations.shape[1] != input_features:
        raise ValueError("input_features does not match activation width.")
    if stream.dtype.kind not in "iu" or stream.ndim != 1 or np.any(stream < 0) or np.any(stream > 255):
        raise ValueError("Octave packed stream must be a 1-D byte array.")
    if offsets.dtype.kind not in "iu" or offsets.ndim != 1:
        raise TypeError("Octave row_offsets must be a 1-D integer array.")
    out_features = len(offsets) - 1
    if out_features < 1 or offsets[0] != 0 or offsets[-1] != len(stream):
        raise ValueError("Octave row offsets do not match packed stream.")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Octave row offsets must be monotonically increasing.")

    required_groups = (input_features + 2) // 3
    lut = build_octave_decode_lut(boost_factor=boost_factor)
    output = np.zeros((activations.shape[0], out_features), dtype=np.int64)
    for batch in range(activations.shape[0]):
        for out_index in range(out_features):
            cursor = int(offsets[out_index])
            end = int(offsets[out_index + 1])
            group = 0
            accumulator = 0
            while cursor < end and group < required_groups:
                token = int(stream[cursor])
                cursor += 1
                if token == OCTAVE_ESCAPE_CODE:
                    if cursor >= end:
                        raise ValueError("Octave escape byte has no skip-count payload.")
                    skip = int(stream[cursor])
                    cursor += 1
                    if skip < 1:
                        raise ValueError("Octave extended skip count must be in [1,255].")
                    group += skip
                    continue
                if token in OCTAVE_SKIP_TRIPLES:
                    group += OCTAVE_SKIP_TRIPLES[token]
                    continue
                coefficients = lut[token]
                if int(coefficients[0]) == -128:
                    raise ValueError("Reserved Octave byte used outside the skip grammar.")
                for lane in range(3):
                    feature = group * 3 + lane
                    if feature >= input_features:
                        break
                    coefficient = int(coefficients[lane])
                    if coefficient == 0:
                        continue
                    shift = {1: 0, 2: 1, 4: 2, 8: 3}.get(abs(coefficient))
                    if shift is None:
                        raise ValueError("Invalid Octave half-unit coefficient.")
                    term = int(activations[batch, feature]) << shift
                    if coefficient > 0:
                        accumulator += term
                    else:
                        accumulator -= term
                group += 1
            if group != required_groups or cursor != end:
                raise ValueError("Octave stream row does not match declared input width.")
            int64_info = np.iinfo(np.int64)
            if accumulator < int64_info.min or accumulator > int64_info.max:
                raise OverflowError("Octave projection accumulator exceeds int64 range.")
            output[batch, out_index] = accumulator
    if bias_half_units is not None:
        bias = np.asarray(bias_half_units)
        if bias.dtype.kind not in "iu" or bias.shape != (out_features,):
            raise ValueError("bias_half_units must be an integer output vector.")
        int64_info = np.iinfo(np.int64)
        if any(int(value) < int64_info.min or int(value) > int64_info.max for value in bias):
            raise ValueError("bias_half_units values must fit in int64 without narrowing.")
        combined = output.astype(object) + bias.astype(object)[None, :]
        if any(int(value) < int64_info.min or int(value) > int64_info.max for value in combined.flat):
            raise OverflowError("Octave projection plus bias exceeds int64 range.")
        output = np.asarray(combined, dtype=np.int64)
    return output


class _EWGSQuantize(torch.autograd.Function):
    """Hard five-symbol forward with a bounded EWGS-style straight-through gradient."""
    @staticmethod
    def forward(ctx, weight, hard_value, normalized, hard_normalized, strength):
        ctx.save_for_backward(normalized, hard_normalized)
        ctx.strength = float(strength)
        return hard_value

    @staticmethod
    def backward(ctx, grad_output):
        normalized, hard_normalized = ctx.saved_tensors
        correction = ctx.strength * torch.sign(grad_output) * (normalized - hard_normalized)
        factor = (1.0 + correction).clamp(0.5, 1.5)
        return grad_output * factor, None, None, None, None


def wavebit_context_gain(context: torch.Tensor) -> torch.Tensor:
    """Hard forward gains {0.5, 1, 2} with a smooth differentiable surrogate."""
    hard = torch.where(
        context < -0.5, torch.full_like(context, 0.5),
        torch.where(context > 0.5, torch.full_like(context, 2.0), torch.ones_like(context)),
    )
    soft = torch.exp2(context)
    return hard.detach() + soft - soft.detach()


def resize_context(context: torch.Tensor, width: int) -> torch.Tensor:
    """Deterministic phase-slot remapping for an expert's hidden width."""
    if context.shape[-1] == width:
        return context
    indices = torch.arange(width, device=context.device) % context.shape[-1]
    return context.index_select(-1, indices)




def _hadamard_power_of_two(inputs: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard butterfly for a power-of-two final dimension."""
    width = inputs.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard transform width must be a positive power of two.")
    transformed = inputs
    offset = 1
    while offset < width:
        shape = (*transformed.shape[:-1], width // (2 * offset), 2, offset)
        pairs = transformed.reshape(shape)
        left = pairs.select(-2, 0)
        right = pairs.select(-2, 1)
        transformed = torch.stack((left + right, left - right), dim=-2).reshape_as(transformed)
        offset *= 2
    return transformed / math.sqrt(width)


def blockwise_hadamard(inputs: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """Apply orthonormal Hadamard blocks and transform any power-of-two tail."""
    if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or inputs.shape[-1] < 1:
        raise ValueError("Hadamard inputs must have a non-empty final feature dimension.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1 or block_size > 64 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a power of two in [1, 64].")
    width = inputs.shape[-1]
    pieces = []
    cursor = 0
    while cursor < width:
        remaining = width - cursor
        current = min(block_size, remaining)
        current = 1 << (current.bit_length() - 1)
        pieces.append(_hadamard_power_of_two(inputs[..., cursor : cursor + current]))
        cursor += current
    return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]





def blockwise_hadamard_quantize_i8(
    activations_i8: np.ndarray,
    input_scales: np.ndarray,
    block_size: int = 8,
    activation_bits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy integer reference for native blockwise Hadamard + low-bit activation quantization."""
    activations = np.asarray(activations_i8)
    scales = np.asarray(input_scales)
    if activations.dtype != np.int8 or activations.ndim != 2 or min(activations.shape) < 1:
        raise TypeError("activations_i8 must be a non-empty 2-D int8 array.")
    if scales.dtype != np.float32 or scales.shape != (activations.shape[0],):
        raise TypeError("input_scales must be a float32 vector with one scale per row.")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("input_scales must be finite and positive.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1 or block_size > 64 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two in [1, 64].")
    if isinstance(activation_bits, bool) or not isinstance(activation_bits, int) or not 2 <= activation_bits <= 8:
        raise ValueError("activation_bits must be an integer in [2, 8].")

    rows, features = activations.shape
    max_code = (1 << (activation_bits - 1)) - 1
    output = np.zeros_like(activations)
    output_scales = np.empty(rows, dtype=np.float32)
    for row in range(rows):
        transformed = np.empty(features, dtype=np.float32)
        cursor = 0
        max_abs = 1.0e-8
        while cursor < features:
            width = min(block_size, features - cursor)
            block_width = 1 << (width.bit_length() - 1)
            values = activations[row, cursor : cursor + block_width].astype(np.int32)
            stride = 1
            while stride < block_width:
                for base in range(0, block_width, stride * 2):
                    left = values[base : base + stride].copy()
                    right = values[base + stride : base + stride * 2].copy()
                    values[base : base + stride] = left + right
                    values[base + stride : base + stride * 2] = left - right
                stride *= 2
            normalized = values.astype(np.float32) / np.float32(math.sqrt(block_width))
            transformed[cursor : cursor + block_width] = normalized
            if normalized.size:
                max_abs = max(max_abs, float(np.max(np.abs(normalized))))
            cursor += block_width
        codes = np.rint(transformed * (float(max_code) / max_abs))
        output[row] = np.clip(codes, -max_code, max_code).astype(np.int8)
        output_scales[row] = np.float32(scales[row] * (max_abs / max_code))
    return output, output_scales


def quantize_activation_ste(inputs: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-token symmetric activation fake-quantization with an identity STE."""
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise TypeError("Activation quantization expects a floating-point tensor.")
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("Activation bits must be an integer in [2, 8].")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("Activation tensor contains non-finite values.")
    maximum_code = (1 << (bits - 1)) - 1
    scale = inputs.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / maximum_code
    codes = torch.round(inputs / scale).clamp(-maximum_code, maximum_code)
    dequantized = codes * scale
    return inputs + (dequantized - inputs).detach()


def wavebit_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    context: torch.Tensor,
    *,
    group_size: int = 30,
    ewgs_strength: float = 0.5,
    octave_temperature: float = 0.25,
    boost_factor: int = 4,
    hadamard: bool = False,
    hadamard_block_size: int = 8,
    activation_bits: int = 4,
) -> torch.Tensor:
    """Octave-Shift weights plus optional Hadamard-rotated low-bit activations."""
    if inputs.shape != context.shape:
        raise ValueError("WaveBit linear context must match the input tensor shape.")
    quantized, _, _, _ = wavebit_quantize(
        weight, group_size=group_size, ewgs_strength=ewgs_strength,
        octave_temperature=octave_temperature,
        boost_factor=boost_factor,
    )
    projected_inputs = inputs * wavebit_context_gain(context)
    if hadamard:
        projected_inputs = blockwise_hadamard(
            projected_inputs, block_size=hadamard_block_size
        )
        projected_inputs = quantize_activation_ste(
            projected_inputs, bits=activation_bits
        )
    return F.linear(projected_inputs, quantized, bias)


def wavebit_depthwise_conv1d(
    padded_inputs: torch.Tensor,
    padded_context: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    dilation: int,
    group_size: int,
    ewgs_strength: float,
    octave_temperature: float = 0.25,
    boost_factor: int = 4,
) -> torch.Tensor:
    """Depthwise Octave-Shift projection with context-modulated inputs."""
    if padded_inputs.shape != padded_context.shape:
        raise ValueError("Padded WaveBit inputs and contexts must have identical shapes.")
    quantized, _, _, _ = wavebit_quantize(
        weight, group_size=group_size, ewgs_strength=ewgs_strength,
        octave_temperature=octave_temperature, boost_factor=boost_factor,
    )
    return F.conv1d(
        padded_inputs * wavebit_context_gain(padded_context),
        quantized,
        bias,
        dilation=dilation,
        groups=padded_inputs.shape[1],
    )


class HolographicRotationMemory(torch.nn.Module):
    """Parameter-free fixed-width recurrent context memory, O(width) per stream."""
    def __init__(
        self, width: int, rotation: int = 1, gamma: int = 4,
        threshold: float = 0.5, temperature: float = 0.75, state_limit: float = 16.0,
    ):
        super().__init__()
        if width < 1 or gamma < 1 or threshold <= 0 or temperature <= 0 or state_limit <= 0:
            raise ValueError("invalid WaveBit holographic memory settings")
        self.width = int(width)
        self.rotation = int(rotation) % self.width
        self.gamma = int(gamma)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.state_limit = float(state_limit)

    def step(self, inputs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.width or state.shape != inputs.shape:
            raise ValueError("WaveBit memory expects matching [batch, width] input and state")
        hard = torch.where(
            state > self.threshold, torch.ones_like(state),
            torch.where(state < -self.threshold, -torch.ones_like(state), torch.zeros_like(state)),
        )
        soft = torch.tanh(state / self.temperature)
        context = hard.detach() + soft - soft.detach()
        rotated = torch.roll(state, shifts=self.rotation, dims=-1)
        retention = 1.0 - (2.0 ** -self.gamma)
        updated = (rotated * retention + torch.tanh(inputs)).clamp(
            -self.state_limit, self.state_limit
        )
        return context, updated

    def forward(
        self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.width or inputs.shape[1] < 1:
            raise ValueError("WaveBit memory expects [batch, sequence, width]")
        state = inputs.new_zeros((inputs.shape[0], self.width)) if initial_state is None else initial_state
        if state.shape != (inputs.shape[0], self.width):
            raise ValueError("WaveBit memory initial state has incompatible shape")
        contexts = []
        for index in range(inputs.shape[1]):
            context, state = self.step(inputs[:, index, :], state)
            contexts.append(context)
        return torch.stack(contexts, dim=1), state


# Keep the historical public API while centralizing overflow-safe arithmetic.
from .integer_safety import (
    CapacitorCharge,
    holographic_state_step_int,
    integer_wavebit_projection,
)
