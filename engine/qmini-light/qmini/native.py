from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def build_kernel(output=None):
    source = Path(__file__).resolve().parent / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("Install clang++ or g++ to build the native kernel.")
    target = Path(output) if output else source.with_name("libqmini_kernel.so")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared",
         str(source), "-o", str(temporary)],
        check=True,
    )
    temporary.replace(target)
    return target


class NativeKernel:
    def __init__(self, library_path):
        self.path = Path(library_path).resolve()
        self.library = ctypes.CDLL(str(self.path))
        try:
            self.function = self.library.qmini_linear_i8_v2
            self.neon_function = self.library.qmini_kernel_has_neon
        except AttributeError as exc:
            raise RuntimeError(
                "Native library is outdated (Q2-v2 ABI required). Rebuild it with: python -m qmini.native --build"
            ) from exc
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.function.argtypes = [
            i8, u8, f32, f32, f32, f32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.function.restype = ctypes.c_int
        self.neon_function.argtypes = []
        self.neon_function.restype = ctypes.c_int
        self.uses_neon = bool(self.neon_function())
        self.hadamard_function = getattr(
            self.library, "qmini_hadamard_quantize_i8_v1", None
        )
        if self.hadamard_function is not None:
            f32_vector = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
            self.hadamard_function.argtypes = [
                i8, f32_vector, i8, f32_vector,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.hadamard_function.restype = ctypes.c_int
        self.octave_function = getattr(
            self.library, "qmini_octave_linear_i8_v1", None
        )
        self.octave_function_v2 = getattr(
            self.library, "qmini_octave_linear_i8_v2", None
        )
        u32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        octave_argtypes = [
            i8, u8, u32, i64, i64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        if self.octave_function is not None:
            self.octave_function.argtypes = octave_argtypes
            self.octave_function.restype = ctypes.c_int
        if self.octave_function_v2 is not None:
            self.octave_function_v2.argtypes = octave_argtypes + [ctypes.c_int]
            self.octave_function_v2.restype = ctypes.c_int

        self.groupwise_octave_function = getattr(
            self.library, "qmini_octave_groupwise_linear_i8_v1", None
        )
        if self.groupwise_octave_function is not None:
            self.groupwise_octave_function.argtypes = [
                i8, u8, u32, f32, f32, f32, f32,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.groupwise_octave_function.restype = ctypes.c_int



    def hadamard_quantize_i8(
        self, activations_i8, input_scales, block_size: int = 8,
        activation_bits: int = 4,
    ):
        """Apply the native integer Hadamard butterfly and symmetric low-bit quantizer."""
        if self.hadamard_function is None:
            raise RuntimeError(
                "Native library does not include the Hadamard ABI; rebuild it."
            )
        raw = np.asarray(activations_i8)
        scales_raw = np.asarray(input_scales)
        if raw.dtype != np.int8 or raw.ndim != 2 or min(raw.shape) < 1:
            raise TypeError("activations_i8 must be a non-empty 2-D int8 array.")
        if scales_raw.dtype != np.float32 or scales_raw.shape != (raw.shape[0],):
            raise TypeError("input_scales must be a float32 vector with one scale per row.")
        if not np.isfinite(scales_raw).all() or np.any(scales_raw <= 0.0):
            raise ValueError("input_scales must be finite and positive.")
        if (
            isinstance(block_size, bool) or not isinstance(block_size, int)
            or block_size < 1 or block_size > 64 or block_size & (block_size - 1)
        ):
            raise ValueError("block_size must be a power of two in [1, 64].")
        if (
            isinstance(activation_bits, bool) or not isinstance(activation_bits, int)
            or activation_bits < 2 or activation_bits > 8
        ):
            raise ValueError("activation_bits must be an integer in [2, 8].")
        inputs = np.ascontiguousarray(raw)
        scales = np.ascontiguousarray(scales_raw)
        output = np.empty_like(inputs)
        output_scales = np.empty(inputs.shape[0], dtype=np.float32)
        status = self.hadamard_function(
            inputs, scales, output, output_scales,
            inputs.shape[0], inputs.shape[1], block_size, activation_bits,
        )
        if status:
            raise RuntimeError(f"Native Hadamard kernel returned error {status}.")
        return output, output_scales

    def octave_linear(
        self, activations_i8, packed_stream, row_offsets, bias_half_units=None,
        boost_factor: int = 4,
    ):
        """Run the Octave-Shift native kernel; x2 requires the versioned v2 ABI."""
        if isinstance(boost_factor, bool) or not isinstance(boost_factor, int) or boost_factor not in (2, 4):
            raise ValueError("boost_factor must be the integer 2 or 4.")
        function = self.octave_function
        use_v2 = boost_factor == 2 or function is None
        if use_v2:
            function = self.octave_function_v2
        if function is None:
            raise RuntimeError(
                "Native library does not include the required Octave-Shift ABI; rebuild it."
            )
        raw_x = np.asarray(activations_i8)
        raw_stream = np.asarray(packed_stream)
        raw_offsets = np.asarray(row_offsets)
        if raw_x.dtype != np.int8 or raw_x.ndim != 2 or min(raw_x.shape) < 1:
            raise TypeError("activations_i8 must be a non-empty 2-D int8 array.")
        if raw_stream.dtype != np.uint8 or raw_stream.ndim != 1:
            raise TypeError("packed_stream must already be a 1-D uint8 array.")
        if raw_stream.size > np.iinfo(np.int32).max:
            raise ValueError("packed_stream exceeds the native ABI size limit.")
        if raw_offsets.dtype.kind not in "iu" or raw_offsets.ndim != 1:
            raise TypeError("row_offsets must be a 1-D integer array.")
        if raw_offsets.size < 2:
            raise ValueError("row_offsets must include at least one output row.")
        if np.any(raw_offsets < 0) or np.any(raw_offsets > np.iinfo(np.uint32).max):
            raise ValueError("row_offsets must fit in uint32.")
        if raw_offsets[0] != 0 or raw_offsets[-1] != raw_stream.size:
            raise ValueError("row_offsets must span the packed stream exactly.")
        if np.any(raw_offsets[1:] < raw_offsets[:-1]):
            raise ValueError("row_offsets must be monotonically increasing.")
        x = np.ascontiguousarray(raw_x)
        stream = np.ascontiguousarray(raw_stream)
        offsets = np.ascontiguousarray(raw_offsets, dtype=np.uint32)
        out_features = offsets.size - 1
        if bias_half_units is None:
            bias = np.zeros(out_features, dtype=np.int64)
        else:
            raw_bias = np.asarray(bias_half_units)
            if raw_bias.dtype.kind not in "iu" or raw_bias.shape != (out_features,):
                raise ValueError("bias_half_units must be an integer output vector.")
            int64_info = np.iinfo(np.int64)
            if any(int(value) < int64_info.min or int(value) > int64_info.max for value in raw_bias):
                raise ValueError("bias_half_units values must fit in int64 without narrowing.")
            bias = np.ascontiguousarray(raw_bias, dtype=np.int64)
        output = np.empty((x.shape[0], out_features), dtype=np.int64)
        arguments = (
            x, stream, offsets, bias, output,
            x.shape[0], x.shape[1], out_features, stream.size,
        )
        status = function(*(arguments + (boost_factor,)) if use_v2 else arguments)
        if status:
            raise RuntimeError(f"Octave-Shift native kernel returned error {status}.")
        return output

    def octave_groupwise_linear(
        self, activations_i8, packed_stream, row_offsets, group_scales,
        input_scales, bias=None, group_size: int = 30, boost_factor: int = 4,
    ):
        """Run group-scaled Octave projection; integer add/sub is followed by FP32 scaling."""
        if self.groupwise_octave_function is None:
            raise RuntimeError(
                "Native library does not include the groupwise Octave ABI; rebuild it."
            )
        if (
            isinstance(boost_factor, bool) or not isinstance(boost_factor, int)
            or boost_factor not in (2, 4)
        ):
            raise ValueError("boost_factor must be the integer 2 or 4.")
        if (
            isinstance(group_size, bool) or not isinstance(group_size, int)
            or group_size < 1 or group_size > np.iinfo(np.int32).max
        ):
            raise ValueError("group_size must be a positive int32-compatible integer.")

        raw_x = np.asarray(activations_i8)
        raw_stream = np.asarray(packed_stream)
        raw_offsets = np.asarray(row_offsets)
        raw_group_scales = np.asarray(group_scales)
        raw_input_scales = np.asarray(input_scales)
        if raw_x.dtype != np.int8 or raw_x.ndim != 2 or min(raw_x.shape) < 1:
            raise TypeError("activations_i8 must be a non-empty 2-D int8 array.")
        if raw_stream.dtype != np.uint8 or raw_stream.ndim != 1:
            raise TypeError("packed_stream must already be a 1-D uint8 array.")
        if raw_stream.size > np.iinfo(np.int32).max:
            raise ValueError("packed_stream exceeds the native ABI size limit.")
        if raw_offsets.dtype.kind not in "iu" or raw_offsets.ndim != 1:
            raise TypeError("row_offsets must be a 1-D integer array.")
        if raw_offsets.size < 2:
            raise ValueError("row_offsets must include at least one output row.")
        if np.any(raw_offsets < 0) or np.any(raw_offsets > np.iinfo(np.uint32).max):
            raise ValueError("row_offsets must fit in uint32.")
        if raw_offsets[0] != 0 or raw_offsets[-1] != raw_stream.size:
            raise ValueError("row_offsets must span the packed stream exactly.")
        if np.any(raw_offsets[1:] < raw_offsets[:-1]):
            raise ValueError("row_offsets must be monotonically increasing.")

        rows, in_features = raw_x.shape
        out_features = raw_offsets.size - 1
        int32_max = np.iinfo(np.int32).max
        if max(rows, in_features, out_features) > int32_max:
            raise ValueError("Native projection dimensions exceed the int32 ABI limit.")
        group_count = (in_features + group_size - 1) // group_size
        if (
            raw_group_scales.dtype != np.float32
            or raw_group_scales.shape != (out_features, group_count)
        ):
            raise TypeError(
                "group_scales must be a float32 matrix shaped "
                "[out_features, ceil(in_features/group_size)]."
            )
        if not np.isfinite(raw_group_scales).all() or np.any(raw_group_scales <= 0.0):
            raise ValueError("group_scales must contain finite positive values.")
        if (
            raw_input_scales.dtype != np.float32
            or raw_input_scales.shape != (rows,)
        ):
            raise TypeError("input_scales must be a float32 vector with one scale per row.")
        if not np.isfinite(raw_input_scales).all() or np.any(raw_input_scales <= 0.0):
            raise ValueError("input_scales must contain finite positive values.")

        if bias is None:
            bias_array = np.zeros(out_features, dtype=np.float32)
        else:
            raw_bias = np.asarray(bias)
            if raw_bias.dtype != np.float32 or raw_bias.shape != (out_features,):
                raise TypeError("bias must be a float32 vector with one value per output.")
            if not np.isfinite(raw_bias).all():
                raise ValueError("bias values must be finite.")
            bias_array = np.ascontiguousarray(raw_bias)

        x = np.ascontiguousarray(raw_x)
        stream = np.ascontiguousarray(raw_stream)
        offsets = np.ascontiguousarray(raw_offsets, dtype=np.uint32)
        scales = np.ascontiguousarray(raw_group_scales)
        activation_scales = np.ascontiguousarray(raw_input_scales)
        output = np.empty((rows, out_features), dtype=np.float32)
        status = self.groupwise_octave_function(
            x, stream, offsets, scales, activation_scales, bias_array, output,
            rows, in_features, out_features, stream.size,
            group_size, group_count, boost_factor,
        )
        if status:
            raise RuntimeError(
                f"Groupwise Octave native kernel returned error {status}."
            )
        return output

    def linear(
        self, activations_i8, packed_weights, weight_scales, activation_scales,
        bias=None, q2_version: int = 2,
    ):
        raw_x = np.asarray(activations_i8)
        raw_packed = np.asarray(packed_weights)
        if raw_x.dtype != np.int8:
            raise TypeError("activations_i8 must already be int8; silent narrowing is refused.")
        if raw_packed.dtype != np.uint8:
            raise TypeError("packed_weights must already be uint8.")
        x = np.ascontiguousarray(raw_x)
        packed = np.ascontiguousarray(raw_packed)
        ws = np.ascontiguousarray(weight_scales, dtype=np.float32)
        xs = np.ascontiguousarray(activation_scales, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("activations_i8 must have non-empty shape [rows, in_features].")
        if packed.ndim != 2 or ws.ndim != 1 or xs.ndim != 1:
            raise ValueError("Packed weights and scales have invalid ranks.")
        expected_bytes = (x.shape[1] + 3) // 4
        if packed.shape[0] != ws.size or packed.shape[1] != expected_bytes:
            raise ValueError("Packed weight shape does not exactly match activation dimensions.")
        if xs.shape != (x.shape[0],):
            raise ValueError("activation_scales must contain one scale per input row.")
        if (
            isinstance(q2_version, bool)
            or not isinstance(q2_version, int)
            or q2_version not in (1, 2)
        ):
            raise ValueError("q2_version must be the integer 1 or 2.")
        if not np.isfinite(ws).all() or (
            np.any(ws <= 0) if q2_version == 1 else np.any(ws < 0)
        ):
            raise ValueError(
                "weight_scales must be finite and positive for Q2-v1, "
                "or finite and non-negative for Q2-v2."
            )
        if not np.isfinite(xs).all() or np.any(xs <= 0):
            raise ValueError("activation_scales must be finite and positive.")
        if bias is None:
            b = np.zeros(ws.size, dtype=np.float32)
        else:
            b = np.ascontiguousarray(bias, dtype=np.float32)
            if b.shape != (ws.size,):
                raise ValueError("bias must contain one value per output feature.")
            if not np.isfinite(b).all():
                raise ValueError("bias values must be finite.")
        output = np.empty((x.shape[0], ws.size), dtype=np.float32)
        status = self.function(
            x, packed, ws, xs, b, output,
            x.shape[0], x.shape[1], ws.size, int(q2_version)
        )
        if status:
            raise RuntimeError(f"Native kernel returned error {status}.")
        return output


_NATIVE_KERNEL: NativeKernel | None = None
_NATIVE_CACHE_KEY: tuple[str, int, int] | None = None


def configured_library_path() -> Path:
    override = os.environ.get("QMINI_NATIVE_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().with_name("libqmini_kernel.so")


def get_native_kernel() -> NativeKernel | None:
    """Load/reload the optional native CPU projection when its artifact changes."""
    global _NATIVE_KERNEL, _NATIVE_CACHE_KEY
    path = configured_library_path()
    if not path.is_file():
        _NATIVE_KERNEL = None
        _NATIVE_CACHE_KEY = (str(path), -1, -1)
        return None

    stat = path.stat()
    cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
    if _NATIVE_CACHE_KEY == cache_key:
        return _NATIVE_KERNEL

    try:
        loaded = NativeKernel(path)
    except (OSError, RuntimeError) as exc:
        _NATIVE_KERNEL = None
        _NATIVE_CACHE_KEY = None
        raise RuntimeError(
            f"Q-Mini native library exists but cannot be loaded: {path}: {exc}"
        ) from exc
    _NATIVE_KERNEL = loaded
    _NATIVE_CACHE_KEY = cache_key
    return _NATIVE_KERNEL


def native_kernel_should_be_used(kernel: NativeKernel | None = None) -> bool:
    """Use NEON by default; make the scalar C++ path an explicit opt-in.

    On common x86 CPUs, optimized PyTorch/BLAS can outperform the scalar
    unpack-and-accumulate loop. Set QMINI_NATIVE_FORCE_SCALAR=1 only when a
    target-specific benchmark shows that scalar native execution is faster.
    """
    active = kernel if kernel is not None else get_native_kernel()
    if active is None:
        return False
    if active.uses_neon:
        return True
    return os.environ.get("QMINI_NATIVE_FORCE_SCALAR", "").strip().lower() in {
        "1", "true", "yes"
    }


def native_kernel_status() -> dict[str, str | bool]:
    path = configured_library_path()
    kernel = get_native_kernel()
    use_native = native_kernel_should_be_used(kernel)
    return {
        "available": kernel is not None,
        "path": str(path),
        "backend": (
            "native-int8-quaternary-neon"
            if use_native and kernel is not None and kernel.uses_neon
            else "native-int8-quaternary-scalar"
            if use_native and kernel is not None
            else "pytorch-fallback"
        ),
        "native_kernel_available": kernel is not None,
        "native_kernel_force_scalar": (
            os.environ.get("QMINI_NATIVE_FORCE_SCALAR", "").strip().lower()
            in {"1", "true", "yes"}
        ),
        "uses_neon": bool(kernel.uses_neon) if kernel is not None else False,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the optional Q-Mini C++ projection primitive.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.build:
        print(build_kernel(args.output))
    else:
        print(native_kernel_status())


if __name__ == "__main__":
    main()
