import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini.native import NativeKernel
from qmini.quantization import pack_quaternary, unpack_quaternary
from qmini_addsub.wavebit_hsra import (
    integer_octave_projection,
    pack_octave_stream,
    unpack_octave_stream,
)


@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("clang++")), reason="C++ compiler missing")
def test_native_projection_matches_reference_on_odd_dimensions(tmp_path):
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    library = tmp_path / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )

    weight = torch.tensor([
        [-4., -1., 1., 4., 4., 1., -1.],
        [1., 4., -4., -1., -1., -4., 4.],
        [2., -3., 1., -2., 4., -1., 3.],
    ])
    packed, scales = pack_quaternary(weight)
    dequantized = unpack_quaternary(packed, scales, in_features=7)
    x = np.array([
        [.2, -.4, .7, 1., -.1, .3, .5],
        [-.9, .1, .4, -.2, .7, -.5, .3],
    ], dtype=np.float32)
    x_scale = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.
    x_q = np.rint(x / x_scale[:, None]).clip(-127, 127).astype(np.int8)
    bias = np.array([.25, -.5, .1], dtype=np.float32)

    output = NativeKernel(library).linear(x_q, packed, scales, x_scale, bias)
    expected = x_q.astype(np.float32) @ dequantized.T
    expected = expected * x_scale[:, None] + bias
    assert output.shape == (2, 3)
    assert np.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_native_wrapper_rejects_incompatible_shapes(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    native = NativeKernel(library)
    assert isinstance(native.uses_neon, bool)
    with pytest.raises(ValueError, match="one scale per input row"):
        native.linear(
            np.ones((2, 4), dtype=np.int8),
            np.zeros((3, 1), dtype=np.uint8),
            np.ones(3, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )


@pytest.fixture(scope="module")
def compiled_native_library(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path_factory.mktemp("native-runtime") / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    return library


def test_qlinear_uses_native_kernel_and_invalidates_packed_cache(
    compiled_native_library, monkeypatch
):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QLinear
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    kernel = NativeKernel(compiled_native_library)
    calls = {"count": 0}
    original = kernel.linear

    def counted_linear(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    kernel.linear = counted_linear
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    monkeypatch.setenv("QMINI_NATIVE_FORCE_SCALAR", "1")

    layer = QLinear(7, 5).eval()
    values = torch.randn(2, 3, 7, dtype=torch.float32)
    with torch.inference_mode():
        actual = layer(values)
        expected = F.linear(
            fake_quantize_int8(values),
            quantize_quaternary(layer.weight),
            layer.bias,
        )
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert calls["count"] == 1
        first_key = layer._qmini_native_packed_key

        # A changed master weight must regenerate the packed row data.
        layer.weight.add_(0.05)
        actual_after_update = layer(values)
        expected_after_update = F.linear(
            fake_quantize_int8(values),
            quantize_quaternary(layer.weight),
            layer.bias,
        )
        assert torch.allclose(
            actual_after_update, expected_after_update, atol=2e-5, rtol=2e-5
        )
        assert calls["count"] == 2
        assert layer._qmini_native_packed_key != first_key


def test_tied_embedding_output_uses_native_projection(
    compiled_native_library, monkeypatch
):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QEmbedding
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    kernel = NativeKernel(compiled_native_library)
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    monkeypatch.setenv("QMINI_NATIVE_FORCE_SCALAR", "1")
    embedding = QEmbedding(37, 9).eval()
    hidden = torch.randn(2, 4, 9)
    with torch.inference_mode():
        actual = embedding.project(hidden)
        expected = F.linear(
            fake_quantize_int8(hidden),
            quantize_quaternary(embedding.weight),
        )
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert hasattr(embedding, "_qmini_native_packed_weight")


def test_pytorch_fallback_is_used_without_native_library(monkeypatch):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QLinear
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    monkeypatch.setattr(native_module, "get_native_kernel", lambda: None)
    layer = QLinear(4, 3).eval()
    values = torch.randn(2, 4)
    with torch.inference_mode():
        actual = layer(values)
        expected = F.linear(
            fake_quantize_int8(values), quantize_quaternary(layer.weight), layer.bias
        )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not hasattr(layer, "_qmini_native_packed_weight")


def test_native_loader_retries_after_library_is_built(
    tmp_path, monkeypatch
):
    import qmini.native as native_module

    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path / "late-built.so"

    monkeypatch.setenv("QMINI_NATIVE_LIBRARY", str(library))
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setattr(native_module, "_NATIVE_CACHE_KEY", None)
    assert native_module.get_native_kernel() is None

    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    assert native_module.get_native_kernel() is not None
    assert native_module.native_kernel_status()["available"] is True


def test_native_wrapper_rejects_silent_dtype_narrowing(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    packed = np.zeros((2, 1), dtype=np.uint8)
    scales = np.ones(2, dtype=np.float32)
    activation_scales = np.ones(1, dtype=np.float32)

    with pytest.raises(TypeError, match="already be int8"):
        native.linear(
            np.ones((1, 4), dtype=np.int64),
            packed,
            scales,
            activation_scales,
        )
    with pytest.raises(TypeError, match="already be uint8"):
        native.linear(
            np.ones((1, 4), dtype=np.int8),
            np.zeros((2, 1), dtype=np.int16),
            scales,
            activation_scales,
        )


def test_outdated_native_library_fails_with_rebuild_instruction(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    old_source = tmp_path / "old_native.cpp"
    old_library = tmp_path / "old_native.so"
    old_source.write_text(
        'extern "C" int qmini_linear_i8() { return 0; }',
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(old_source), "-o", str(old_library)],
        check=True,
    )
    with pytest.raises(RuntimeError, match="Native library is outdated"):
        NativeKernel(old_library)


def test_scalar_native_kernel_is_opt_in_on_non_neon_platform(
    compiled_native_library, monkeypatch
):
    import qmini.native as native_module
    from qmini.model import QLinear

    kernel = NativeKernel(compiled_native_library)
    if kernel.uses_neon:
        pytest.skip("NEON is a default-supported native backend.")
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    monkeypatch.delenv("QMINI_NATIVE_FORCE_SCALAR", raising=False)

    layer = QLinear(7, 5).eval()
    values = torch.randn(2, 3, 7, dtype=torch.float32)
    with torch.inference_mode():
        output = layer(values)
    assert torch.isfinite(output).all()
    assert not hasattr(layer, "_qmini_native_packed_weight")
    assert native_module.native_kernel_should_be_used(kernel) is False

    monkeypatch.setenv("QMINI_NATIVE_FORCE_SCALAR", "1")
    with torch.inference_mode():
        output_forced = layer(values)
    assert torch.allclose(output_forced, output, atol=2e-5, rtol=2e-5)
    assert hasattr(layer, "_qmini_native_packed_weight")
    assert native_module.native_kernel_should_be_used(kernel) is True


def test_neon_native_kernel_is_selected_without_force_flag(
    compiled_native_library, monkeypatch
):
    import qmini.native as native_module

    kernel = NativeKernel(compiled_native_library)
    if not kernel.uses_neon:
        pytest.skip("This runner has no NEON backend.")
    monkeypatch.delenv("QMINI_NATIVE_FORCE_SCALAR", raising=False)
    assert native_module.native_kernel_should_be_used(kernel) is True



def test_native_q2_v2_zero_scale_row_matches_python_reference(compiled_native_library):
    from qmini.quantization import pack_quaternary, unpack_quaternary

    native = NativeKernel(compiled_native_library)
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, -0.5, 0.0, 0.5, 2.0],
    ])
    packed, scales = pack_quaternary(weight)
    assert scales[0] == 0.0
    assert scales[1] > 0.0
    x = np.array([[0.25, -0.5, 1.0, 0.75, -1.0]], dtype=np.float32)
    x_scale = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.0
    x_q = np.rint(x / x_scale[:, None]).clip(-127, 127).astype(np.int8)

    actual = native.linear(
        x_q, packed, scales, x_scale.astype(np.float32), q2_version=2
    )
    dequantized = unpack_quaternary(packed, scales, 5, format_version=2)
    expected = (x_q.astype(np.float32) @ dequantized.T) * x_scale[:, None]
    assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert actual[0, 0] == 0.0


def test_native_q2_v1_legacy_projection_uses_legacy_codebook(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    # Four legacy symbols [-2,-1,+1,+2] with a single positive v1 scale.
    packed = np.array([[0b11100100]], dtype=np.uint8)
    scales = np.array([0.25], dtype=np.float32)
    x_q = np.array([[1, 2, 3, 4]], dtype=np.int8)
    x_scale = np.array([0.1], dtype=np.float32)
    actual = native.linear(
        x_q, packed, scales, x_scale, q2_version=1
    )
    legacy_weight = np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float32) * 0.25
    expected = (x_q.astype(np.float32) @ legacy_weight.T) * 0.1
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_native_q2_v2_rejects_negative_scale_but_accepts_zero(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    x_q = np.ones((1, 4), dtype=np.int8)
    packed = np.zeros((2, 1), dtype=np.uint8)
    x_scale = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        native.linear(
            x_q, packed, np.array([0.0, -1.0], dtype=np.float32),
            x_scale, q2_version=2,
        )
    out = native.linear(
        x_q, packed, np.array([0.0, 0.0], dtype=np.float32),
        x_scale, q2_version=2,
    )
    assert np.array_equal(out, np.zeros((1, 2), dtype=np.float32))


def test_native_octave_projection_matches_python_reference_with_zero_skips(
    compiled_native_library,
):
    rng = np.random.default_rng(20261125)
    symbol_codes = rng.integers(0, 5, size=(5, 225), dtype=np.uint8)
    boost_modes = rng.integers(0, 2, size=(5, 75), dtype=np.uint8).astype(bool)
    # Add long, all-zero sections so the byte stream includes skip commands.
    symbol_codes[:, 30:30 + 3 * 18] = 2
    boost_modes[:, 10:28] = False
    packed, offsets = pack_octave_stream(symbol_codes, boost_modes)
    activations = rng.integers(-127, 128, size=(4, 225), dtype=np.int16).astype(np.int8)

    native = NativeKernel(compiled_native_library)
    actual = native.octave_linear(activations, packed, offsets)
    expected = integer_octave_projection(
        activations, packed, offsets, input_features=225
    )
    np.testing.assert_array_equal(actual, expected)


def test_native_octave_projection_accepts_integer_bias_and_rejects_bad_types(
    compiled_native_library,
):
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 5, size=(2, 12), dtype=np.uint8)
    modes = rng.integers(0, 2, size=(2, 4), dtype=np.uint8).astype(bool)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    activations = rng.integers(-100, 101, size=(3, 12), dtype=np.int16).astype(np.int8)
    bias = np.asarray([17, -9], dtype=np.int64)
    native = NativeKernel(compiled_native_library)
    actual = native.octave_linear(activations, packed, offsets, bias)
    expected = integer_octave_projection(
        activations, packed, offsets, input_features=12, bias_half_units=bias
    )
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(TypeError, match="already be a 1-D uint8"):
        native.octave_linear(activations, packed.astype(np.int16), offsets)


def test_native_octave_projection_detects_missing_escape_payload(
    compiled_native_library,
):
    native = NativeKernel(compiled_native_library)
    activations = np.ones((1, 12), dtype=np.int8)
    stream = np.asarray([255], dtype=np.uint8)
    offsets = np.asarray([0, 1], dtype=np.uint32)
    with pytest.raises(RuntimeError, match="returned error"):
        native.octave_linear(activations, stream, offsets)



def test_native_octave_x2_projection_matches_python_reference_with_zero_skips(
    compiled_native_library,
):
    rng = np.random.default_rng(20261126)
    symbol_codes = rng.integers(0, 5, size=(4, 225), dtype=np.uint8)
    boost_modes = rng.integers(0, 2, size=(4, 75), dtype=np.uint8).astype(bool)
    symbol_codes[:, 30:30 + 3 * 18] = 2
    boost_modes[:, 10:28] = False
    packed, offsets = pack_octave_stream(symbol_codes, boost_modes)
    activations = rng.integers(-127, 128, size=(3, 225), dtype=np.int16).astype(np.int8)

    native = NativeKernel(compiled_native_library)
    actual = native.octave_linear(
        activations, packed, offsets, boost_factor=2
    )
    expected = integer_octave_projection(
        activations, packed, offsets, input_features=225, boost_factor=2
    )
    np.testing.assert_array_equal(actual, expected)


def test_native_octave_x2_abi_validates_boost_factor(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    activations = np.ones((1, 3), dtype=np.int8)
    packed, offsets = pack_octave_stream(
        np.asarray([[4, 3, 2]], dtype=np.uint8),
        np.asarray([[True]], dtype=np.bool_),
        use_zero_skips=False,
    )
    with pytest.raises(ValueError, match="2 or 4"):
        native.octave_linear(
            activations, packed, offsets, boost_factor=3
        )



def test_native_hadamard_quantization_matches_integer_reference(compiled_native_library):
    from qmini.native import NativeKernel
    from qmini_addsub.wavebit_hsra import blockwise_hadamard_quantize_i8

    rng = np.random.default_rng(20261127)
    inputs = rng.integers(-127, 128, size=(5, 170), dtype=np.int16).astype(np.int8)
    inputs[0] = 0
    input_scales = np.asarray([0.125, 0.03125, 0.0625, 0.2, 0.5], dtype=np.float32)
    expected_codes, expected_scales = blockwise_hadamard_quantize_i8(
        inputs, input_scales, block_size=8, activation_bits=4
    )
    native_codes, native_scales = NativeKernel(compiled_native_library).hadamard_quantize_i8(
        inputs, input_scales, block_size=8, activation_bits=4
    )
    np.testing.assert_array_equal(native_codes, expected_codes)
    np.testing.assert_allclose(native_scales, expected_scales, rtol=2e-6, atol=1e-8)
    assert int(np.max(np.abs(native_codes))) <= 7
    assert np.all(native_codes[0] == 0)
    assert native_scales[0] > 0.0


def test_native_hadamard_quantization_supports_nondefault_bits_and_tail_blocks(
    compiled_native_library,
):
    from qmini.native import NativeKernel
    from qmini_addsub.wavebit_hsra import blockwise_hadamard_quantize_i8

    rng = np.random.default_rng(91)
    inputs = rng.integers(-100, 101, size=(2, 19), dtype=np.int16).astype(np.int8)
    scales = np.asarray([0.25, 0.5], dtype=np.float32)
    expected, expected_scales = blockwise_hadamard_quantize_i8(
        inputs, scales, block_size=4, activation_bits=6
    )
    actual, actual_scales = NativeKernel(compiled_native_library).hadamard_quantize_i8(
        inputs, scales, block_size=4, activation_bits=6
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual_scales, expected_scales, rtol=2e-6, atol=1e-8)
    assert int(np.max(np.abs(actual))) <= 31


def test_native_hadamard_quantization_rejects_invalid_parameters(compiled_native_library):
    from qmini.native import NativeKernel

    native = NativeKernel(compiled_native_library)
    activations = np.ones((2, 8), dtype=np.int8)
    scales = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="power of two"):
        native.hadamard_quantize_i8(activations, scales, block_size=6)
    with pytest.raises(ValueError, match="bits"):
        native.hadamard_quantize_i8(activations, scales, activation_bits=9)
    with pytest.raises(TypeError, match="float32 vector"):
        native.hadamard_quantize_i8(activations, scales.astype(np.float64))


@pytest.mark.parametrize("group_size", [17, 30])
@pytest.mark.parametrize("boost_factor", [2, 4])
def test_native_octave_groupwise_projection_matches_scaled_reference(
    compiled_native_library, group_size, boost_factor,
):
    from qmini_addsub.wavebit_hsra import unpack_octave_stream

    rng = np.random.default_rng(20261128 + group_size + boost_factor)
    rows, out_features, in_features = 4, 3, 101
    triplets = (in_features + 2) // 3
    codes = rng.integers(0, 5, size=(out_features, in_features), dtype=np.uint8)
    modes = rng.integers(0, 2, size=(out_features, triplets), dtype=np.uint8).astype(bool)
    # Force a long all-zero run to exercise skip tokens in the group-wise parser.
    codes[:, 24:84] = 2
    modes[:, 8:28] = False
    packed, offsets = pack_octave_stream(codes, modes)

    activations = rng.integers(-127, 128, size=(rows, in_features), dtype=np.int16).astype(np.int8)
    input_scales = rng.uniform(0.005, 0.2, size=rows).astype(np.float32)
    group_count = (in_features + group_size - 1) // group_size
    group_scales = rng.uniform(
        0.01, 0.25, size=(out_features, group_count)
    ).astype(np.float32)
    bias = rng.normal(0.0, 0.15, size=out_features).astype(np.float32)

    actual = NativeKernel(compiled_native_library).octave_groupwise_linear(
        activations, packed, offsets, group_scales, input_scales,
        bias=bias, group_size=group_size, boost_factor=boost_factor,
    )

    decoded_codes, decoded_modes = unpack_octave_stream(
        packed, offsets, (out_features, in_features)
    )
    base_half_units = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)[decoded_codes]
    boosted = np.repeat(decoded_modes, 3, axis=1)[:, :in_features]
    mode_factor = np.where(boosted, boost_factor, 1).astype(np.float32)
    group_index = np.arange(in_features) // group_size
    dequantized_weights = (
        base_half_units * mode_factor * 0.5 * group_scales[:, group_index]
    )
    expected = (
        (activations.astype(np.float32) * input_scales[:, None])
        @ dequantized_weights.T
    ) + bias[None, :]

    assert actual.shape == (rows, out_features)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_native_octave_groupwise_projection_validates_group_metadata(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    activations = np.ones((2, 11), dtype=np.int8)
    codes = np.full((2, 11), 2, dtype=np.uint8)
    modes = np.zeros((2, 4), dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes)
    group_scales = np.ones((2, 2), dtype=np.float32)
    input_scales = np.ones(2, dtype=np.float32)

    with pytest.raises(TypeError, match="group_scales must be a float32 matrix"):
        native.octave_groupwise_linear(
            activations, packed, offsets, group_scales.astype(np.float64),
            input_scales, group_size=8,
        )
    with pytest.raises(ValueError, match="group_scales must contain finite positive"):
        native.octave_groupwise_linear(
            activations, packed, offsets,
            np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            input_scales, group_size=8,
        )
    with pytest.raises(ValueError, match="group_size must be a positive"):
        native.octave_groupwise_linear(
            activations, packed, offsets, group_scales, input_scales,
            group_size=0,
        )
    with pytest.raises(ValueError, match="boost_factor"):
        native.octave_groupwise_linear(
            activations, packed, offsets, group_scales, input_scales,
            group_size=8, boost_factor=3,
        )


def test_native_octave_groupwise_projection_handles_all_zero_groups(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    activations = np.asarray([[127, -127, 64, 32, -16, 8, 4]], dtype=np.int8)
    codes = np.full((2, 7), 2, dtype=np.uint8)
    modes = np.zeros((2, 3), dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=True)
    group_scales = np.ones((2, 2), dtype=np.float32)
    input_scales = np.asarray([0.125], dtype=np.float32)

    actual = native.octave_groupwise_linear(
        activations, packed, offsets, group_scales, input_scales,
        group_size=6, boost_factor=4,
    )
    np.testing.assert_array_equal(actual, np.zeros((1, 2), dtype=np.float32))


def test_native_octave_groupwise_projection_supports_missing_bias(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    activations = np.asarray([[4, -3, 7, 1, 2]], dtype=np.int8)
    codes = np.asarray([[4, 1, 2, 3, 0]], dtype=np.uint8)
    modes = np.asarray([[False, True]], dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    group_size = 3
    group_scales = np.asarray([[0.125, 0.25]], dtype=np.float32)
    input_scales = np.asarray([0.0625], dtype=np.float32)

    actual = native.octave_groupwise_linear(
        activations, packed, offsets, group_scales, input_scales,
        bias=None, group_size=group_size, boost_factor=4,
    )
    decoded_codes, decoded_modes = unpack_octave_stream(packed, offsets, (1, 5))
    levels = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
    mode = np.repeat(decoded_modes, 3, axis=1)[:, :5]
    weights = levels[decoded_codes] * np.where(mode, 4.0, 1.0) * 0.5
    weights *= np.repeat(group_scales, group_size, axis=1)[:, :5]
    expected = (activations.astype(np.float32) * input_scales[:, None]) @ weights.T
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_native_octave_bias_uint64_overflow_is_rejected_before_cast(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    activations = np.asarray([[1]], dtype=np.int8)
    codes = np.asarray([[4]], dtype=np.uint8)
    modes = np.asarray([[False]], dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    bias = np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(ValueError, match="fit in int64"):
        native.octave_linear(activations, packed, offsets, bias)


def test_native_octave_bias_accumulator_overflow_is_rejected(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    activations = np.asarray([[1]], dtype=np.int8)
    codes = np.asarray([[4]], dtype=np.uint8)
    modes = np.asarray([[False]], dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    bias = np.asarray([np.iinfo(np.int64).max], dtype=np.int64)

    with pytest.raises(RuntimeError, match="returned error 11"):
        native.octave_linear(activations, packed, offsets, bias)


def test_integer_octave_reference_preserves_wide_int_inputs_and_guards_overflow():
    wide = integer_octave_projection(
        np.asarray([[200]], dtype=np.int16),
        np.asarray([0], dtype=np.uint8),
        np.asarray([0, 1], dtype=np.int64),
        input_features=1,
    )
    assert wide.shape == (1, 1)
    with pytest.raises(OverflowError, match="accumulator exceeds int64"):
        integer_octave_projection(
            np.asarray([[np.iinfo(np.int64).max]], dtype=np.int64),
            np.asarray([0], dtype=np.uint8),
            np.asarray([0, 1], dtype=np.int64),
            input_features=1,
        )

    codes = np.asarray([[4]], dtype=np.uint8)
    modes = np.asarray([[False]], dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    with pytest.raises(OverflowError, match="exceeds int64 range"):
        integer_octave_projection(
            np.asarray([[1]], dtype=np.int8),
            packed,
            offsets,
            input_features=1,
            bias_half_units=np.asarray([np.iinfo(np.int64).max], dtype=np.int64),
        )
