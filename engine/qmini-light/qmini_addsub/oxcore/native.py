"""Small ctypes bridge to the standalone OxCore C ABI.

No ML framework, tokenizer, NumPy, or runtime compilation is required here.
Build the C++20 library with scripts/build_oxcore.sh and set
QMINI_OXCORE_LIBRARY to its path.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Sequence


_OX_OK = 0
_OX_SKIP = 1
_OX_EINVAL = 2
_OX_ECAPACITY = 3
_DTYPE_CODES = {"f32": 1, "i8": 2, "i32": 3, "i64": 4, "u8": 5}
_DTYPE_CTYPES = {
    "f32": ctypes.c_float,
    "i8": ctypes.c_int8,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "u8": ctypes.c_uint8,
}


class OxCoreError(RuntimeError):
    """Raised when a native OxCore operation rejects its input."""


class _Arena(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p),
        ("capacity", ctypes.c_size_t),
        ("used", ctypes.c_size_t),
        ("high_water", ctypes.c_size_t),
    ]


class _TensorView(ctypes.Structure):
    _fields_ = [
        ("storage", ctypes.c_void_p),
        ("storage_bytes", ctypes.c_size_t),
        ("offset_bytes", ctypes.c_size_t),
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 8),
        ("strides_bytes", ctypes.c_int64 * 8),
    ]


def _as_int8(values: Sequence[int]) -> ctypes.Array:
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Activations must be integers.")
        if not -128 <= value <= 127:
            raise ValueError("Activations must be in the int8 range [-128, 127].")
        checked.append(value)
    return (ctypes.c_int8 * len(checked))(*checked)


def _as_uint8(values: Sequence[int]) -> ctypes.Array:
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Packed bytes must be integers.")
        if not 0 <= value <= 255:
            raise ValueError("Packed bytes must be in the uint8 range [0, 255].")
        checked.append(value)
    return (ctypes.c_uint8 * len(checked))(*checked)


def _as_int32(values: Sequence[int]) -> ctypes.Array:
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("SSM states and injections must be integers.")
        if not -(2**31) <= value <= 2**31 - 1:
            raise ValueError("SSM states and injections must fit signed int32.")
        checked.append(value)
    return (ctypes.c_int32 * len(checked))(*checked)


def _as_uint16(values: Sequence[int]) -> ctypes.Array:
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Q15 decay coefficients must be integers.")
        if not 0 <= value <= 65535:
            raise ValueError("Q15 decay coefficients must fit uint16.")
        checked.append(value)
    return (ctypes.c_uint16 * len(checked))(*checked)


def _as_int16(values: Sequence[int]) -> ctypes.Array:
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("LUT values and router scores must be integers.")
        if not -(2**15) <= value <= 2**15 - 1:
            raise ValueError("LUT values and router scores must fit signed int16.")
        checked.append(value)
    return (ctypes.c_int16 * len(checked))(*checked)


class OxCoreNative:
    """Explicit handle to one compiled OxCore shared library."""

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OxCore library does not exist: {path}")
        self.path = path
        self._lib = ctypes.CDLL(str(path))
        self._configure_signatures()
        version = self._lib.oxcore_version()
        self.version = version.decode("utf-8") if version else "unknown"

    @classmethod
    def load(cls, library_path: str | os.PathLike[str] | None = None) -> "OxCoreNative":
        selected = library_path or os.environ.get("QMINI_OXCORE_LIBRARY")
        if selected is None:
            package_dir = Path(__file__).resolve().parent
            for filename in ("liboxcore.so", "liboxcore.dylib", "oxcore.dll"):
                candidate = package_dir / filename
                if candidate.is_file():
                    selected = candidate
                    break
        if selected is None:
            raise FileNotFoundError(
                "OxCore is not built. Run scripts/build_oxcore.sh and set "
                "QMINI_OXCORE_LIBRARY to the resulting shared library."
            )
        return cls(selected)

    def _configure_signatures(self) -> None:
        lib = self._lib
        lib.oxcore_abi_version.argtypes = []
        lib.oxcore_abi_version.restype = ctypes.c_uint32
        lib.oxcore_version.argtypes = []
        lib.oxcore_version.restype = ctypes.c_char_p
        lib.ox_status_name.argtypes = [ctypes.c_int]
        lib.ox_status_name.restype = ctypes.c_char_p

        lib.ox_arena_init.argtypes = [
            ctypes.POINTER(_Arena), ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.ox_arena_init.restype = ctypes.c_int
        lib.ox_arena_reset.argtypes = [ctypes.POINTER(_Arena)]
        lib.ox_arena_reset.restype = None
        lib.ox_arena_mark.argtypes = [ctypes.POINTER(_Arena)]
        lib.ox_arena_mark.restype = ctypes.c_size_t
        lib.ox_arena_rewind.argtypes = [ctypes.POINTER(_Arena), ctypes.c_size_t]
        lib.ox_arena_rewind.restype = ctypes.c_int
        lib.ox_arena_alloc.argtypes = [
            ctypes.POINTER(_Arena), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        lib.ox_arena_alloc.restype = ctypes.c_int

        lib.ox_mirror63_pack_triplet.argtypes = [
            ctypes.c_int8, ctypes.c_int8, ctypes.c_int8,
            ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint8)
        ]
        lib.ox_mirror63_pack_triplet.restype = ctypes.c_int
        lib.ox_mirror63_unpack_triplet.argtypes = [
            ctypes.c_uint8, ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_uint8)
        ]
        lib.ox_mirror63_unpack_triplet.restype = ctypes.c_int
        lib.ox_mirror63_dot_i8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int8),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64)
        ]
        lib.ox_mirror63_dot_i8.restype = ctypes.c_int
        lib.ox_linear_mirror63_acc_i8.argtypes = [
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64)
        ]
        lib.ox_linear_mirror63_acc_i8.restype = ctypes.c_int
        lib.ox_causal_depthwise_conv_mirror63_acc_i8.argtypes = [
            ctypes.POINTER(ctypes.c_int8), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64)
        ]
        lib.ox_causal_depthwise_conv_mirror63_acc_i8.restype = ctypes.c_int
        lib.ox_ssm_scan_q15_i32.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32)
        ]
        lib.ox_ssm_scan_q15_i32.restype = ctypes.c_int
        lib.ox_lut_u8_i16.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int16),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16)
        ]
        lib.ox_lut_u8_i16.restype = ctypes.c_int
        lib.ox_top1_i16.argtypes = [
            ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_int16)
        ]
        lib.ox_top1_i16.restype = ctypes.c_int

        lib.ox_fwht_i64.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t
        ]
        lib.ox_fwht_i64.restype = ctypes.c_int
        lib.ox_sinkhorn4.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint32
        ]
        lib.ox_sinkhorn4.restype = ctypes.c_int
        lib.ox_normalize_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_float
        ]
        lib.ox_normalize_f32.restype = ctypes.c_int
        lib.ox_tensor_view_validate.argtypes = [ctypes.POINTER(_TensorView)]
        lib.ox_tensor_view_validate.restype = ctypes.c_int

    def _check(self, status: int, operation: str, allow_skip: bool = False) -> None:
        if status == _OX_OK or (allow_skip and status == _OX_SKIP):
            return
        raw = self._lib.ox_status_name(status)
        detail = raw.decode("utf-8") if raw else f"status {status}"
        raise OxCoreError(f"{operation}: {detail}")

    @property
    def abi_version(self) -> int:
        return int(self._lib.oxcore_abi_version())

    def mirror63_pack(self, q: Sequence[int], octave: int = 0) -> int:
        if isinstance(octave, bool) or not isinstance(octave, int):
            raise TypeError("Mirror-63 octave must be an integer.")
        if len(q) != 3 or octave not in (0, 1):
            raise ValueError("Mirror-63 needs three balanced digits and octave 0 or 1.")
        output = ctypes.c_uint8()
        values = []
        for value in q:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Mirror-63 digits must be integers.")
            if not -2 <= value <= 2:
                raise ValueError("Mirror-63 digits must be in [-2, 2].")
            values.append(value)
        status = self._lib.ox_mirror63_pack_triplet(
            values[0], values[1], values[2], octave, ctypes.byref(output)
        )
        self._check(status, "Mirror-63 pack")
        return int(output.value)

    def mirror63_unpack(self, code: int) -> tuple[tuple[int, int, int], int] | None:
        if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255:
            raise ValueError("Mirror-63 code must be an integer byte.")
        q = (ctypes.c_int8 * 3)()
        octave = ctypes.c_uint8()
        status = self._lib.ox_mirror63_unpack_triplet(
            code, q, ctypes.byref(octave)
        )
        if status == _OX_SKIP:
            return None
        self._check(status, "Mirror-63 unpack")
        return (int(q[0]), int(q[1]), int(q[2])), int(octave.value)

    def mirror63_dot_i8(
        self, packed_blocks: Sequence[int], activations: Sequence[int]
    ) -> int:
        blocks = _as_uint8(packed_blocks)
        if len(activations) != 3 * len(packed_blocks):
            raise ValueError("Mirror-63 dot needs exactly three int8 activations per block.")
        inputs = _as_int8(activations)
        output = ctypes.c_int64()
        status = self._lib.ox_mirror63_dot_i8(
            blocks, inputs, len(packed_blocks), ctypes.byref(output)
        )
        self._check(status, "Mirror-63 dot")
        return int(output.value)

    def linear_mirror63_acc_i8(
        self,
        inputs: Sequence[Sequence[int]],
        packed_weights: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Compute [batch, out] integer numerators from Mirror-63 packed rows.

        Inputs are signed int8. Each output row stores ceil(in_features/3)
        Mirror-63 bytes; the final byte's unused activation lanes are zero.
        """
        if not inputs or not packed_weights:
            raise ValueError("Linear projection needs a non-empty batch and output matrix.")
        in_features = len(inputs[0])
        if in_features < 1 or any(len(row) != in_features for row in inputs):
            raise ValueError("Linear inputs must be a non-empty rectangular matrix.")
        blocks_per_row = (in_features + 2) // 3
        if any(len(row) != blocks_per_row for row in packed_weights):
            raise ValueError(
                "Each packed linear weight row must contain ceil(in_features/3) bytes."
            )
        batch = len(inputs)
        out_features = len(packed_weights)
        source = _as_int8([value for row in inputs for value in row])
        weights = _as_uint8([value for row in packed_weights for value in row])
        output = (ctypes.c_int64 * (batch * out_features))()
        status = self._lib.ox_linear_mirror63_acc_i8(
            source, weights, batch, in_features, out_features, output
        )
        self._check(status, "Mirror-63 linear projection")
        return [
            [int(output[row * out_features + column]) for column in range(out_features)]
            for row in range(batch)
        ]

    def causal_depthwise_conv_mirror63_acc_i8(
        self,
        inputs: Sequence[Sequence[int]],
        packed_weights: Sequence[Sequence[int]],
        kernel_size: int,
    ) -> list[list[int]]:
        """Causal depthwise convolution numerators for [time, channels] int8 input.

        Weight rows are per channel and contain ceil(kernel_size/3) Mirror-63
        bytes. Tap zero is oldest and the final tap uses the current token.
        Missing left context is zero padded.
        """
        if not inputs or not packed_weights:
            raise ValueError("Causal convolution needs input timesteps and channels.")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
            raise TypeError("Causal convolution kernel_size must be an integer.")
        if kernel_size < 1:
            raise ValueError("Causal convolution kernel_size must be positive.")
        time_steps = len(inputs)
        channels = len(inputs[0])
        if channels < 1 or any(len(row) != channels for row in inputs):
            raise ValueError("Causal convolution inputs must be a non-empty rectangular matrix.")
        if len(packed_weights) != channels:
            raise ValueError("Causal convolution needs exactly one packed weight row per channel.")
        blocks_per_channel = (kernel_size + 2) // 3
        if any(len(row) != blocks_per_channel for row in packed_weights):
            raise ValueError(
                "Each channel weight row must contain ceil(kernel_size/3) bytes."
            )
        source = _as_int8([value for row in inputs for value in row])
        weights = _as_uint8([value for row in packed_weights for value in row])
        output = (ctypes.c_int64 * (time_steps * channels))()
        status = self._lib.ox_causal_depthwise_conv_mirror63_acc_i8(
            source, time_steps, channels, weights, kernel_size, output
        )
        self._check(status, "Mirror-63 causal depthwise convolution")
        return [
            [int(output[time * channels + channel]) for channel in range(channels)]
            for time in range(time_steps)
        ]

    def ssm_scan_q15_i32(
        self,
        initial_state: Sequence[Sequence[int]],
        injection: Sequence[Sequence[Sequence[int]]],
        decay_q15: Sequence[Sequence[Sequence[int]]],
    ) -> tuple[list[list[list[int]]], list[list[int]]]:
        """Run a multiply-free Q15 diagonal SSM recurrence.

        initial_state: [batch, channels]
        injection/decay_q15: [time, batch, channels]
        decay is unsigned Q15 in [0, 32768]. Output contains every state and
        the final cache, both int32. Q15 scaling uses shift-add and
        round-to-nearest-even; conservative overflow is rejected before writes.
        """
        if not initial_state or not initial_state[0]:
            raise ValueError("SSM initial state must have non-empty batch and channels.")
        batch = len(initial_state)
        channels = len(initial_state[0])
        if any(len(row) != channels for row in initial_state):
            raise ValueError("SSM initial state must be rectangular.")
        if not injection or len(injection) != len(decay_q15):
            raise ValueError("SSM injection and decay need matching non-empty time axes.")
        time_steps = len(injection)
        for name, values in (("injection", injection), ("decay_q15", decay_q15)):
            if any(len(step) != batch for step in values):
                raise ValueError(f"SSM {name} batch dimension does not match initial state.")
            if any(len(row) != channels for step in values for row in step):
                raise ValueError(f"SSM {name} channel dimension does not match initial state.")

        initial = _as_int32([value for row in initial_state for value in row])
        injected = _as_int32(
            [value for step in injection for row in step for value in row]
        )
        decay = _as_uint16(
            [value for step in decay_q15 for row in step for value in row]
        )
        state_count = batch * channels
        output = (ctypes.c_int32 * (time_steps * state_count))()
        final_state = (ctypes.c_int32 * state_count)()
        status = self._lib.ox_ssm_scan_q15_i32(
            initial, injected, decay, time_steps, batch, channels,
            output, final_state
        )
        self._check(status, "Q15 SSM recurrence")
        all_states = [
            [
                [
                    int(output[(time * batch + row) * channels + channel])
                    for channel in range(channels)
                ]
                for row in range(batch)
            ]
            for time in range(time_steps)
        ]
        final = [
            [
                int(final_state[row * channels + channel])
                for channel in range(channels)
            ]
            for row in range(batch)
        ]
        return all_states, final

    def lut_u8_i16(self, indices: Sequence[int], table: Sequence[int]) -> list[int]:
        """Map uint8 activation/gate indices through a caller-provided 256-entry LUT."""
        if len(table) != 256:
            raise ValueError("OxCore activation LUT must contain exactly 256 int16 values.")
        input_values = _as_uint8(indices)
        table_values = _as_int16(table)
        if not indices:
            raise ValueError("OxCore LUT input must be non-empty.")
        output = (ctypes.c_int16 * len(indices))()
        status = self._lib.ox_lut_u8_i16(
            input_values, table_values, len(indices), output
        )
        self._check(status, "uint8 activation LUT")
        return [int(value) for value in output]

    def top1_i16(
        self, scores: Sequence[Sequence[int]]
    ) -> tuple[list[int], list[int]]:
        """Stable per-row Top-1 selection; equal scores choose lowest expert id."""
        if not scores or not scores[0]:
            raise ValueError("Top-1 scores must have non-empty batch and expert axes.")
        batch = len(scores)
        experts = len(scores[0])
        if any(len(row) != experts for row in scores):
            raise ValueError("Top-1 score matrix must be rectangular.")
        if experts > 65536:
            raise ValueError("Top-1 supports at most 65536 experts.")
        flattened = _as_int16([score for row in scores for score in row])
        selected = (ctypes.c_uint16 * batch)()
        selected_scores = (ctypes.c_int16 * batch)()
        status = self._lib.ox_top1_i16(
            flattened, batch, experts, selected, selected_scores
        )
        self._check(status, "stable Top-1 router selection")
        return (
            [int(value) for value in selected],
            [int(value) for value in selected_scores],
        )

    def fwht_i64(self, values: Sequence[int]) -> list[int]:
        if not values:
            raise ValueError("FWHT input must be non-empty.")
        checked = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("FWHT values must be integers.")
            if not -(2**63) <= value <= 2**63 - 1:
                raise ValueError("FWHT values must fit in signed int64.")
            checked.append(value)
        buffer = (ctypes.c_int64 * len(checked))(*checked)
        status = self._lib.ox_fwht_i64(buffer, len(values))
        self._check(status, "FWHT")
        return [int(value) for value in buffer]

    def sinkhorn4(
        self, logits: Sequence[Sequence[float]], iterations: int = 20
    ) -> list[list[float]]:
        if len(logits) != 4 or any(len(row) != 4 for row in logits):
            raise ValueError("Sinkhorn input must be a 4 by 4 matrix.")
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            raise TypeError("Sinkhorn iterations must be an integer.")
        try:
            flat = [float(value) for row in logits for value in row]
            source = (ctypes.c_double * 16)(*flat)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Sinkhorn logits must be finite real values.") from exc
        output = (ctypes.c_double * 16)()
        status = self._lib.ox_sinkhorn4(source, output, iterations)
        self._check(status, "Sinkhorn-4")
        return [
            [float(output[row * 4 + col]) for col in range(4)]
            for row in range(4)
        ]

    def normalize_f32(
        self, values: Sequence[float], epsilon: float = 1e-6
    ) -> list[float]:
        if not values:
            raise ValueError("Normalization input must be non-empty.")
        try:
            source = (ctypes.c_float * len(values))(*values)
            output = (ctypes.c_float * len(values))()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Normalization values must be float32-compatible.") from exc
        status = self._lib.ox_normalize_f32(
            source, output, len(values), ctypes.c_float(epsilon)
        )
        self._check(status, "nGPT L2 normalization")
        return [float(value) for value in output]

    def validate_tensor_view(
        self,
        values: Sequence[int | float],
        dtype: str,
        shape: Sequence[int],
        strides_bytes: Sequence[int] | None = None,
        offset_bytes: int = 0,
    ) -> bool:
        if dtype not in _DTYPE_CODES:
            raise ValueError(f"Unsupported OxCore dtype: {dtype}")
        if len(shape) > 8 or (strides_bytes is not None and len(strides_bytes) != len(shape)):
            raise ValueError("Tensor rank and stride metadata must fit the 8-axis ABI.")
        if isinstance(offset_bytes, bool) or not isinstance(offset_bytes, int) or offset_bytes < 0:
            raise ValueError("Tensor storage offset must be a non-negative integer.")
        scalar_type = _DTYPE_CTYPES[dtype]
        storage = (scalar_type * len(values))(*values)
        view = _TensorView()
        view.storage = ctypes.cast(storage, ctypes.c_void_p) if values else None
        view.storage_bytes = ctypes.sizeof(storage)
        view.offset_bytes = offset_bytes
        view.dtype = _DTYPE_CODES[dtype]
        view.rank = len(shape)
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise ValueError("Tensor dimensions must be non-negative integers.")
            view.shape[axis] = dimension
        if strides_bytes is None:
            stride = ctypes.sizeof(scalar_type)
            reversed_strides = []
            for dimension in reversed(shape):
                reversed_strides.append(stride)
                stride *= max(1, int(dimension))
            strides = list(reversed(reversed_strides))
        else:
            strides = list(strides_bytes)
        for axis, stride_value in enumerate(strides):
            if isinstance(stride_value, bool) or not isinstance(stride_value, int):
                raise ValueError("Tensor byte strides must be integers.")
            view.strides_bytes[axis] = stride_value
        status = self._lib.ox_tensor_view_validate(ctypes.byref(view))
        if status == _OX_OK:
            return True
        # A tensor-view predicate reports ordinary invalid/out-of-bounds
        # metadata as False. Numeric overflow remains an exceptional failure.
        if status in (_OX_EINVAL, _OX_ECAPACITY):
            return False
        self._check(status, "Tensor view validation")
        return False


class StaticArena:
    """Caller-owned, fixed-capacity arena backed by one aligned Python buffer."""

    def __init__(self, core: OxCoreNative, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Arena capacity must be a positive integer.")
        self._core = core
        self._storage = ctypes.create_string_buffer(capacity + 63)
        raw_address = ctypes.addressof(self._storage)
        aligned_address = (raw_address + 63) & ~63
        self._arena = _Arena()
        self._core._check(
            self._core._lib.ox_arena_init(
                ctypes.byref(self._arena),
                ctypes.c_void_p(aligned_address),
                capacity,
            ),
            "Arena initialization",
        )

    @property
    def capacity(self) -> int:
        return int(self._arena.capacity)

    @property
    def used(self) -> int:
        return int(self._arena.used)

    @property
    def high_water(self) -> int:
        return int(self._arena.high_water)

    def mark(self) -> int:
        return int(self._core._lib.ox_arena_mark(ctypes.byref(self._arena)))

    def rewind(self, mark: int) -> None:
        self._core._check(
            self._core._lib.ox_arena_rewind(ctypes.byref(self._arena), mark),
            "Arena rewind",
        )

    def reset(self) -> None:
        self._core._lib.ox_arena_reset(ctypes.byref(self._arena))

    def alloc(self, size: int, alignment: int = 64) -> int:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("Arena allocation size must be an integer.")
        if isinstance(alignment, bool) or not isinstance(alignment, int):
            raise TypeError("Arena alignment must be an integer.")
        output = ctypes.c_void_p()
        self._core._check(
            self._core._lib.ox_arena_alloc(
                ctypes.byref(self._arena), size, alignment, ctypes.byref(output)
            ),
            "Arena allocation",
        )
        if output.value is None:
            raise OxCoreError("Arena allocator returned a null pointer after success.")
        return int(output.value)
