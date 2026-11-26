from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_INT64 = np.iinfo(np.int64)


def checked_int64_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer values.")
    if array.size and (
        int(array.min()) < _INT64.min or int(array.max()) > _INT64.max
    ):
        raise OverflowError(f"{name} contains values outside the signed int64 range.")
    return array.astype(np.int64, copy=False)


def checked_int64_add(left: int, right: int, name: str) -> int:
    result = int(left) + int(right)
    if result < _INT64.min or result > _INT64.max:
        raise OverflowError(f"{name} exceeds the signed int64 range.")
    return result


def integer_wavebit_projection(
    activation_codes: np.ndarray,
    weight_codes: np.ndarray,
    context_codes: np.ndarray,
    bias_half_units: np.ndarray | None = None,
) -> np.ndarray:
    """Integer WaveBit reference projection with checked signed arithmetic."""
    raw_x = np.asarray(activation_codes)
    codes = np.asarray(weight_codes)
    context = np.asarray(context_codes)
    if raw_x.dtype.kind not in "iu" or raw_x.ndim < 1:
        raise TypeError("activation codes must be an integer tensor")
    if codes.dtype.kind not in "iu" or codes.ndim != 2 or np.any(codes < 0) or np.any(codes > 4):
        raise ValueError("weight codes must be [out, in] integers in [0, 4]")
    if context.dtype.kind not in "iu" or context.shape != raw_x.shape:
        raise ValueError("context codes must match activation shape")
    if raw_x.shape[-1] != codes.shape[-1] or np.any((context < -1) | (context > 1)):
        raise ValueError("WaveBit input/context dimensions or context values are invalid")

    x = checked_int64_array(raw_x, "activation codes").reshape(-1, raw_x.shape[-1])
    flat_context = context.reshape(-1, context.shape[-1]).astype(np.int8, copy=False)
    polarity = np.asarray((-1, -1, 0, 1, 1), dtype=np.int8)[codes]
    phase = np.asarray((1, 0, 0, 0, 1), dtype=np.int8)[codes]
    output = np.zeros((x.shape[0], codes.shape[0]), dtype=np.int64)

    for row in range(x.shape[0]):
        for out_index in range(codes.shape[0]):
            accumulator = 0
            for feature in range(x.shape[1]):
                sign = int(polarity[out_index, feature])
                if sign == 0:
                    continue
                term = int(x[row, feature])
                if sign < 0:
                    if term == _INT64.min:
                        raise OverflowError("WaveBit sign inversion exceeds int64 range.")
                    term = -term
                if int(phase[out_index, feature]) != 0:
                    context_code = int(flat_context[row, feature])
                    if context_code >= 0:
                        term <<= 1 if context_code == 0 else 2
                        if term < _INT64.min or term > _INT64.max:
                            raise OverflowError("WaveBit shifted contribution exceeds int64 range.")
                accumulator = checked_int64_add(
                    accumulator, term, "WaveBit projection accumulator"
                )
            output[row, out_index] = accumulator

    if bias_half_units is not None:
        bias = checked_int64_array(bias_half_units, "bias_half_units")
        if bias.shape != (codes.shape[0],):
            raise ValueError("bias_half_units must be an integer output vector")
        for row in range(output.shape[0]):
            for out_index in range(output.shape[1]):
                output[row, out_index] = checked_int64_add(
                    int(output[row, out_index]), int(bias[out_index]),
                    "WaveBit projection plus bias",
                )
    return output.reshape(raw_x.shape[:-1] + (codes.shape[0],))


def holographic_state_step_int(
    state: np.ndarray,
    input_codes: np.ndarray,
    rotation: int = 1,
    gamma: int = 4,
    threshold: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    raw_state = np.asarray(state)
    raw_input = np.asarray(input_codes)
    if raw_state.dtype.kind not in "iu" or raw_input.dtype.kind not in "iu" or raw_state.shape != raw_input.shape or raw_state.ndim < 1:
        raise ValueError("integer WaveBit state and input must have matching integer shapes")
    if isinstance(rotation, bool) or not isinstance(rotation, (int, np.integer)):
        raise TypeError("rotation must be an integer")
    if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma < 1:
        raise ValueError("gamma must be a positive integer")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)) or threshold < 1 or int(threshold) > _INT64.max:
        raise ValueError("threshold must be a positive int64-compatible integer")

    h = checked_int64_array(raw_state, "WaveBit state")
    u = checked_int64_array(raw_input, "WaveBit input codes")
    rotated = np.roll(h, shift=int(rotation), axis=-1)
    updated = np.empty_like(rotated)
    for index, raw_value in enumerate(rotated.flat):
        value = int(raw_value)
        delta = abs(value) >> gamma
        retained = value - delta if value >= 0 else value + delta
        combined = retained + int(u.flat[index])
        if combined < _INT64.min or combined > _INT64.max:
            raise OverflowError("WaveBit recurrent state update exceeds int64 range.")
        updated.flat[index] = combined
    context = np.where(h < -int(threshold), -1, np.where(h > int(threshold), 1, 0)).astype(np.int8)
    return context, updated


@dataclass
class CapacitorCharge:
    codes: np.ndarray
    threshold: int = 1024

    def __post_init__(self) -> None:
        raw_codes = np.asarray(self.codes)
        if raw_codes.dtype.kind not in "iu":
            raise TypeError("Capacitor charge codes must be integers.")
        if raw_codes.size and (int(raw_codes.min()) < 0 or int(raw_codes.max()) > 4):
            raise ValueError("Capacitor charge codes must be in [0, 4]")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, np.integer)) or self.threshold < 1 or int(self.threshold) > _INT64.max:
            raise ValueError("Capacitor charge threshold must be a positive int64-compatible integer.")
        self.threshold = int(self.threshold)
        self.codes = raw_codes.astype(np.uint8, copy=True)
        self.charge = np.zeros(self.codes.shape, dtype=np.int64)

    def step(self, signed_impulse: np.ndarray) -> np.ndarray:
        raw = np.asarray(signed_impulse)
        if raw.dtype.kind not in "iu" or raw.shape != self.codes.shape:
            raise ValueError("Capacitor impulses must be an integer array matching codes")
        impulse = checked_int64_array(raw, "Capacitor impulses")
        combined = self.charge.astype(object) + impulse.astype(object)
        if any(int(value) < _INT64.min or int(value) > _INT64.max for value in combined.flat):
            raise OverflowError("Capacitor charge accumulator exceeds int64 range.")
        charge = np.asarray(combined, dtype=np.int64)
        positive = charge >= self.threshold
        negative = charge <= -self.threshold
        current = self.codes.astype(np.int16)
        current = np.where(positive, np.minimum(current + 1, 4), current)
        current = np.where(negative, np.maximum(current - 1, 0), current)
        charge = np.where(positive, charge - self.threshold, charge)
        charge = np.where(negative, charge + self.threshold, charge)
        self.codes = current.astype(np.uint8)
        self.charge = charge.astype(np.int64, copy=False)
        return self.codes.copy()
