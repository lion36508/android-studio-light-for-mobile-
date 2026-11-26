from __future__ import annotations

import ast
import itertools
import math
import os
import subprocess
from pathlib import Path

import pytest

from qmini_addsub.oxcore import OxCoreError, OxCoreNative, StaticArena


@pytest.fixture(scope="session")
def oxcore(tmp_path_factory):
    configured = os.environ.get("QMINI_OXCORE_LIBRARY")
    if configured:
        library = Path(configured)
        assert library.is_file(), library
    else:
        root = Path(__file__).resolve().parents[1]
        library = tmp_path_factory.mktemp("oxcore-native") / "libqmini_oxcore.so"
        subprocess.run(
            ["bash", str(root / "scripts" / "build_oxcore.sh"), str(library)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    return OxCoreNative(library)


def test_oxcore_loads_as_standalone_c_abi(oxcore):
    assert oxcore.abi_version == 1
    assert oxcore.version == "OxCore 0.1.0"


def test_mirror63_all_250_codes_round_trip_and_exactly_six_escapes(oxcore):
    codes = set()
    for q in itertools.product(range(-2, 3), repeat=3):
        for octave in (0, 1):
            code = oxcore.mirror63_pack(q, octave)
            assert code not in codes
            codes.add(code)
            assert oxcore.mirror63_unpack(code) == (q, octave)

    assert len(codes) == 250
    assert set(range(256)) - codes == {
        0x3F, 0x40, 0x7F, 0xBF, 0xC0, 0xFF
    }
    assert oxcore.mirror63_pack((0, 0, 0), 0) == 0x00
    assert oxcore.mirror63_pack((0, 0, 0), 1) == 0x80
    assert oxcore.mirror63_unpack(0x3F) is None


def test_mirror63_rejects_all_reserved_non_skip_codes(oxcore):
    for code in (0x40, 0x7F, 0xBF, 0xC0, 0xFF):
        with pytest.raises(OxCoreError, match="reserved"):
            oxcore.mirror63_unpack(code)

    with pytest.raises(ValueError):
        oxcore.mirror63_pack((3, 0, 0), 0)
    with pytest.raises(ValueError):
        oxcore.mirror63_pack((0, 0, 0), 2)


def test_mirror63_dot_returns_exact_deferred_half_scale_numerator(oxcore):
    first = oxcore.mirror63_pack((2, -1, 0), octave=1)
    second = oxcore.mirror63_pack((-2, 2, -1), octave=0)
    accumulator = oxcore.mirror63_dot_i8(
        [first, second, 0x3F],
        [3, -4, 9, -128, 127, -3, 100, 100, 100],
    )
    # The represented real dot is accumulator / 2 = 266.5.
    assert accumulator == 533


def test_mirror63_dot_fails_closed_on_reserved_byte(oxcore):
    with pytest.raises(OxCoreError, match="reserved"):
        oxcore.mirror63_dot_i8([0x7F], [1, 2, 3])


def test_mirror63_python_bridge_rejects_implicit_integer_wrapping(oxcore):
    code = oxcore.mirror63_pack((1, 0, -1))
    with pytest.raises(ValueError, match="int8 range"):
        oxcore.mirror63_dot_i8([code], [128, 0, 0])
    with pytest.raises(TypeError, match="integers"):
        oxcore.mirror63_dot_i8([code], [1.5, 0, 0])
    with pytest.raises(ValueError, match="uint8 range"):
        oxcore.mirror63_dot_i8([256], [0, 0, 0])


def test_oxcore_python_bridge_rejects_float_and_boolean_octave_or_fwht_values(oxcore):
    with pytest.raises(TypeError, match="octave"):
        oxcore.mirror63_pack((0, 0, 0), octave=1.0)
    with pytest.raises(TypeError, match="octave"):
        oxcore.mirror63_pack((0, 0, 0), octave=True)
    with pytest.raises(TypeError, match="FWHT values must be integers"):
        oxcore.fwht_i64([1, 2.0])
    with pytest.raises(TypeError, match="FWHT values must be integers"):
        oxcore.fwht_i64([1, True])


def test_mirror63_linear_projection_matches_integer_reference_and_handles_partial_block(oxcore):
    weight_rows = [
        [
            oxcore.mirror63_pack((1, -1, 0), octave=0),
            oxcore.mirror63_pack((2, 2, -2), octave=1),
        ],
        [
            oxcore.mirror63_pack((-2, 2, 1), octave=0),
            oxcore.mirror63_pack((0, 0, 0), octave=0),
        ],
    ]
    # Only the first lane of the second packed block is live (in_features=4).
    actual = oxcore.linear_mirror63_acc_i8(
        [[1, 2, 3, 4], [-1, 5, -2, 3]],
        weight_rows,
    )
    assert actual == [[15, 5], [6, 10]]


def test_mirror63_linear_projection_rejects_bad_shapes_and_reserved_codes(oxcore):
    with pytest.raises(ValueError, match="rectangular"):
        oxcore.linear_mirror63_acc_i8([[1, 2], [3]], [[0]])
    with pytest.raises(ValueError, match="ceil"):
        oxcore.linear_mirror63_acc_i8([[1, 2, 3, 4]], [[0]])
    with pytest.raises(OxCoreError, match="reserved"):
        oxcore.linear_mirror63_acc_i8([[1, 2, 3]], [[0x7F]])


def test_mirror63_causal_depthwise_convolution_matches_reference(oxcore):
    packed_weights = [
        [oxcore.mirror63_pack((1, -1, 2), octave=1)],
        [oxcore.mirror63_pack((2, 0, -1), octave=0)],
    ]
    actual = oxcore.causal_depthwise_conv_mirror63_acc_i8(
        [[1, 10], [2, 20], [3, 30], [4, 40]],
        packed_weights,
        kernel_size=3,
    )
    assert actual == [[4, -10], [6, -20], [10, -10], [14, 0]]


def test_mirror63_linear_and_causal_convolution_match_reference_over_multiple_shapes(oxcore):
    import random

    rng = random.Random(20261126)

    batch, in_features, out_features = 3, 7, 5
    inputs = [
        [rng.randint(-128, 127) for _ in range(in_features)]
        for _ in range(batch)
    ]
    q_rows = []
    packed_rows = []
    for _ in range(out_features):
        q_blocks = []
        packed_blocks = []
        for _ in range((in_features + 2) // 3):
            q = tuple(rng.randint(-2, 2) for _ in range(3))
            octave = rng.randint(0, 1)
            q_blocks.append((q, octave))
            packed_blocks.append(oxcore.mirror63_pack(q, octave))
        q_rows.append(q_blocks)
        packed_rows.append(packed_blocks)

    expected_linear = []
    for input_row in inputs:
        expected_row = []
        for q_blocks in q_rows:
            total = 0
            for block_index, (q, octave) in enumerate(q_blocks):
                for lane in range(3):
                    feature = block_index * 3 + lane
                    activation = input_row[feature] if feature < in_features else 0
                    total += q[lane] * activation * (2 if octave else 1)
            expected_row.append(total)
        expected_linear.append(expected_row)
    assert oxcore.linear_mirror63_acc_i8(inputs, packed_rows) == expected_linear

    time_steps, channels, kernel_size = 4, 3, 5
    conv_inputs = [
        [rng.randint(-128, 127) for _ in range(channels)]
        for _ in range(time_steps)
    ]
    conv_q = []
    conv_packed = []
    for _ in range(channels):
        q_blocks = []
        packed_blocks = []
        for _ in range((kernel_size + 2) // 3):
            q = tuple(rng.randint(-2, 2) for _ in range(3))
            octave = rng.randint(0, 1)
            q_blocks.append((q, octave))
            packed_blocks.append(oxcore.mirror63_pack(q, octave))
        conv_q.append(q_blocks)
        conv_packed.append(packed_blocks)

    expected_conv = []
    for time in range(time_steps):
        row = []
        for channel in range(channels):
            total = 0
            for block_index, (q, octave) in enumerate(conv_q[channel]):
                for lane in range(3):
                    tap = block_index * 3 + lane
                    if tap >= kernel_size:
                        continue
                    source_time = time - (kernel_size - 1 - tap)
                    activation = (
                        conv_inputs[source_time][channel]
                        if source_time >= 0
                        else 0
                    )
                    total += q[lane] * activation * (2 if octave else 1)
            row.append(total)
        expected_conv.append(row)
    assert oxcore.causal_depthwise_conv_mirror63_acc_i8(
        conv_inputs, conv_packed, kernel_size
    ) == expected_conv


def test_mirror63_causal_depthwise_convolution_rejects_bad_weights_and_shapes(oxcore):
    with pytest.raises(ValueError, match="one packed weight row"):
        oxcore.causal_depthwise_conv_mirror63_acc_i8(
            [[1, 2]], [[0]], kernel_size=1
        )
    with pytest.raises(ValueError, match="kernel_size"):
        oxcore.causal_depthwise_conv_mirror63_acc_i8(
            [[1]], [[0]], kernel_size=0
        )
    with pytest.raises(OxCoreError, match="reserved"):
        oxcore.causal_depthwise_conv_mirror63_acc_i8(
            [[1]], [[0x7F]], kernel_size=1
        )


def test_q15_ssm_scan_matches_sequential_reference_and_returns_final_cache(oxcore):
    initial = [[100, -100]]
    injection = [[[10, 5]], [[-5, 10]], [[20, -10]]]
    decay = [[[16384, 32768]], [[32768, 16384]], [[0, 32768]]]
    states, final = oxcore.ssm_scan_q15_i32(initial, injection, decay)
    assert states == [[[60, -95]], [[55, -38]], [[20, -48]]]
    assert final == [[20, -48]]


def test_q15_ssm_scan_uses_round_to_nearest_even_for_positive_and_negative_ties(oxcore):
    initial = [[1, 3, -3, 5, -5]]
    injection = [[[0, 0, 0, 0, 0]]]
    decay = [[[16384, 16384, 16384, 16384, 16384]]]
    states, _ = oxcore.ssm_scan_q15_i32(initial, injection, decay)
    assert states == [[[0, 2, -2, 2, -2]]]


def test_q15_ssm_scan_rejects_bad_shape_or_decay_before_execution(oxcore):
    with pytest.raises(ValueError, match="batch dimension"):
        oxcore.ssm_scan_q15_i32([[0]], [[[1], [2]]], [[[32768], [32768]]])
    with pytest.raises(OxCoreError, match="invalid argument"):
        oxcore.ssm_scan_q15_i32([[0]], [[[1]]], [[[32769]]])


def test_q15_ssm_scan_fails_closed_on_conservative_int32_overflow(oxcore):
    with pytest.raises(OxCoreError, match="overflow"):
        oxcore.ssm_scan_q15_i32(
            [[2**31 - 1]],
            [[[1]]],
            [[[32768]]],
        )


def test_uint8_activation_lut_maps_exact_table_entries(oxcore):
    table = [index - 128 for index in range(256)]
    assert oxcore.lut_u8_i16([0, 1, 128, 254, 255], table) == [
        -128, -127, 0, 126, 127
    ]


def test_uint8_activation_lut_rejects_invalid_shape_and_values(oxcore):
    with pytest.raises(ValueError, match="256"):
        oxcore.lut_u8_i16([0], [0, 1])
    with pytest.raises(ValueError, match="uint8 range"):
        oxcore.lut_u8_i16([256], [0] * 256)
    with pytest.raises(ValueError, match="int16"):
        oxcore.lut_u8_i16([0], [32768] + [0] * 255)


def test_top1_i16_is_stable_on_ties_and_returns_scores(oxcore):
    indices, values = oxcore.top1_i16([
        [2, 5, 5, -1],
        [7, 7, 1, 7],
        [-32768, -1, -2, -1],
    ])
    assert indices == [1, 0, 1]
    assert values == [5, 7, -1]


def test_top1_i16_rejects_invalid_shapes_and_scores(oxcore):
    with pytest.raises(ValueError, match="rectangular"):
        oxcore.top1_i16([[1, 2], [3]])
    with pytest.raises(ValueError, match="int16"):
        oxcore.top1_i16([[32768]])


def test_fwht_is_unnormalized_and_invertible_up_to_length(oxcore):
    values = [3, -1, 4, 2, 0, 7, -5, 9]
    transformed = oxcore.fwht_i64(values)
    restored = oxcore.fwht_i64(transformed)
    assert restored == [len(values) * value for value in values]


def test_fwht_rejects_non_power_of_two_and_preflights_overflow(oxcore):
    with pytest.raises(OxCoreError, match="invalid argument"):
        oxcore.fwht_i64([1, 2, 3])
    with pytest.raises(OxCoreError, match="overflow"):
        oxcore.fwht_i64([2**63 - 1, 0])


def test_sinkhorn4_returns_nonnegative_doubly_stochastic_matrix(oxcore):
    logits = [
        [0.0, 0.2, -0.1, 0.4],
        [0.3, -0.2, 0.1, 0.0],
        [-0.1, 0.4, 0.0, 0.2],
        [0.2, 0.0, 0.3, -0.2],
    ]
    matrix = oxcore.sinkhorn4(logits, iterations=40)
    assert all(math.isfinite(value) and value >= 0.0 for row in matrix for value in row)
    for row in matrix:
        assert sum(row) == pytest.approx(1.0, abs=1e-9)
    for column in range(4):
        assert sum(matrix[row][column] for row in range(4)) == pytest.approx(
            1.0, abs=1e-9
        )


def test_sinkhorn4_rejects_invalid_shape_and_nonfinite_values(oxcore):
    with pytest.raises(ValueError, match="4 by 4"):
        oxcore.sinkhorn4([[0.0]])
    with pytest.raises(OxCoreError, match="numeric"):
        oxcore.sinkhorn4([[float("nan")] * 4 for _ in range(4)])


def test_ngpt_l2_normalization_and_zero_vector(oxcore):
    normalized = oxcore.normalize_f32([3.0, 4.0])
    assert math.sqrt(sum(value * value for value in normalized)) == pytest.approx(
        1.0, abs=1e-6
    )
    assert oxcore.normalize_f32([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_ngpt_normalization_rejects_nonfinite_input(oxcore):
    with pytest.raises(OxCoreError, match="numeric"):
        oxcore.normalize_f32([1.0, float("inf")])


def test_static_arena_alignment_marks_rewind_and_high_water(oxcore):
    arena = StaticArena(oxcore, 512)
    first = arena.alloc(13, 64)
    assert first % 64 == 0
    mark = arena.mark()
    second = arena.alloc(31, 128)
    assert second % 128 == 0
    high_water = arena.high_water
    assert arena.used == high_water
    arena.rewind(mark)
    assert arena.used == mark
    assert arena.high_water == high_water
    arena.reset()
    assert arena.used == 0
    assert arena.high_water == high_water


def test_static_arena_rejects_capacity_and_alignment_errors(oxcore):
    arena = StaticArena(oxcore, 64)
    with pytest.raises(OxCoreError, match="capacity"):
        arena.alloc(1024)
    with pytest.raises(OxCoreError, match="invalid argument"):
        arena.alloc(4, 3)
    with pytest.raises(OxCoreError, match="invalid argument"):
        arena.rewind(arena.capacity + 1)


def test_tensor_view_checks_dtype_rank_strides_and_storage_bounds(oxcore):
    values = [1, 2, 3, 4, 5, 6]
    assert oxcore.validate_tensor_view(values, "i32", (2, 3))
    # A reversed length-3 view starts at element 2 and walks backwards.
    assert oxcore.validate_tensor_view(
        values, "i32", (3,), strides_bytes=(-4,), offset_bytes=8
    )
    assert not oxcore.validate_tensor_view(values, "i32", (7,))
    assert not oxcore.validate_tensor_view(
        values, "i32", (2,), strides_bytes=(2,)
    )
    assert oxcore.validate_tensor_view([], "i32", (0,))


def test_tensor_view_rejects_shape_product_overflow(oxcore):
    with pytest.raises(OxCoreError, match="overflow"):
        oxcore.validate_tensor_view([1], "i64", (2**63, 2))


def test_oxcore_python_package_has_no_ml_framework_imports():
    root = Path(__file__).resolve().parents[1] / "qmini_addsub" / "oxcore"
    forbidden = {"torch", "tensorflow", "transformers", "jax", "triton"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert not (names & forbidden), (path, names & forbidden)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, (path, node.module)
