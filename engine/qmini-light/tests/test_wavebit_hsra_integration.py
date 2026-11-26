from __future__ import annotations

import numpy as np
import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.wavebit_hsra import (
    CapacitorCharge, HolographicRotationMemory, groupwise_scale,
    holographic_state_step_int, integer_wavebit_projection, wavebit_quantize,
    pack_octave_stream, unpack_octave_stream, pack_octave_triplets,
    unpack_octave_triplets, integer_octave_projection, build_octave_decode_lut,
    blockwise_hadamard, quantize_activation_ste, octave_shift_quantize,
    validate_wavebit_options,
)


def tiny_config() -> AddSubConfig:
    return AddSubConfig(
        vocab_size=264, d_model=16, d_hidden=24, n_layers=2, n_experts=2,
        kernel_size=3, max_context=32, dropout=0.0, seed=17,
    )


def options() -> dict:
    return {"group_size": 30, "rotation": 1, "gamma": 4, "ewgs_strength": 0.5}


def test_wavebit_has_no_additional_model_parameters_or_state_keys():
    config = tiny_config()
    baseline = AddSubLanguageModel(config, seed=17)
    wavebit = AddSubLanguageModel(config, seed=17, wavebit_options=options())
    assert baseline.parameter_count() == wavebit.parameter_count()
    assert set(baseline.state_dict()) == set(wavebit.state_dict())


def test_wavebit_full_forward_backward_is_finite():
    model = AddSubLanguageModel(tiny_config(), seed=17, wavebit_options=options()).train()
    tokens = torch.randint(0, 264, (2, 9))
    result = model(tokens)
    loss = result["logits"].float().square().mean() + result["aux_loss"] * 0.01
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert model.parameter_count() > 0


def test_wavebit_incremental_cache_matches_full_sequence():
    torch.manual_seed(29)
    model = AddSubLanguageModel(tiny_config(), seed=29, wavebit_options=options()).eval()
    tokens = torch.randint(0, 264, (2, 11))
    with torch.inference_mode():
        expected = model(tokens)["logits"]
        caches = model.init_cache(batch_size=tokens.shape[0])
        pieces = []
        for index in range(tokens.shape[1]):
            logits, caches = model.forward_step(tokens[:, index], caches)
            pieces.append(logits.unsqueeze(1))
        actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_groupwise_scale_is_independent_for_each_block():
    weight = torch.cat([torch.ones(2, 32), torch.full((2, 32), 4.0)], dim=1)
    scale = groupwise_scale(weight, group_size=32)
    assert scale.shape == weight.shape
    assert torch.all(scale[:, 32:] > scale[:, :32])


def test_wavebit_ewgs_preserves_nonzero_gradient_at_zero():
    weight = torch.zeros(3, 8, requires_grad=True)
    quantized, symbol_codes, boost_modes, scale = wavebit_quantize(
        weight, group_size=6, ewgs_strength=0.5
    )
    quantized.sum().backward()
    assert torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(weight.grad).item() == weight.numel()
    assert torch.count_nonzero(quantized).item() == 0
    assert symbol_codes.shape == weight.shape
    assert boost_modes.shape == (3, 3)
    assert torch.all(scale > 0)


def test_holographic_memory_is_parameter_free_and_fixed_width():
    memory = HolographicRotationMemory(width=16, rotation=3, gamma=4)
    inputs = torch.randn(2, 13, 16)
    contexts, state = memory(inputs)
    assert contexts.shape == inputs.shape
    assert state.shape == (2, 16)
    assert len(list(memory.parameters())) == 0
    assert torch.count_nonzero(contexts[:, 0]) == 0


def test_integer_projection_matches_seven_effective_states():
    activations = np.asarray([[7, -5, 3, 2]], dtype=np.int16)
    codes = np.asarray([[0, 1, 2, 4], [4, 3, 1, 0]], dtype=np.uint8)
    context = np.asarray([[-1, 0, 1, 1]], dtype=np.int8)
    actual = integer_wavebit_projection(activations, codes, context)
    polarity = (-1, -1, 0, 1, 1)
    phase = (1, 0, 0, 0, 1)
    expected = np.zeros((1, 2), dtype=np.int64)
    for out_index in range(codes.shape[0]):
        for feature in range(codes.shape[1]):
            p = polarity[int(codes[out_index, feature])]
            g = phase[int(codes[out_index, feature])]
            c = int(context[0, feature])
            x = int(activations[0, feature])
            if p == 0:
                continue
            term = -x if p < 0 else x
            copies = 1 if not g else (1 if c < 0 else (2 if c == 0 else 4))
            for _ in range(copies):
                expected[0, out_index] += term
    np.testing.assert_array_equal(actual, expected)


def test_integer_rotation_uses_shift_decay_and_previous_state_context():
    state = np.asarray([[16, -16, 32, -32]], dtype=np.int32)
    context, updated = holographic_state_step_int(
        state, np.zeros_like(state), rotation=0, gamma=4, threshold=1
    )
    np.testing.assert_array_equal(context, [[1, -1, 1, -1]])
    np.testing.assert_array_equal(updated, [[15, -15, 30, -30]])


def test_capacitor_charge_changes_code_only_after_threshold_and_saturates():
    controller = CapacitorCharge(np.asarray([0, 4, 2], dtype=np.uint8), threshold=10)
    np.testing.assert_array_equal(
        controller.step(np.asarray([-15, 15, 6], dtype=np.int32)),
        np.asarray([0, 4, 2], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        controller.step(np.asarray([0, 0, 5], dtype=np.int32)),
        np.asarray([0, 4, 3], dtype=np.uint8),
    )


def test_invalid_wavebit_configuration_is_rejected():
    with pytest.raises(ValueError):
        AddSubLanguageModel(tiny_config(), wavebit_options={"gamma": 0})


def test_octave_triplet_packing_roundtrips_shared_boost_modes():
    rng = np.random.default_rng(31)
    codes = rng.integers(0, 5, size=(4, 11), dtype=np.uint8)
    modes = rng.integers(0, 2, size=(4, 4), dtype=np.uint8).astype(bool)
    packed, shape = pack_octave_triplets(codes, modes)
    decoded_codes, decoded_modes = unpack_octave_triplets(packed, shape)
    np.testing.assert_array_equal(decoded_codes, codes)
    np.testing.assert_array_equal(decoded_modes, modes)
    assert int(packed.max()) <= 249


def test_octave_zero_skip_codes_compress_and_restore_long_runs():
    codes = np.full((2, 70 * 3), 2, dtype=np.uint8)
    modes = np.zeros((2, 70), dtype=np.bool_)
    codes[1, 3 * 65] = 4
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=True)
    decoded_codes, decoded_modes = unpack_octave_stream(packed, offsets, codes.shape)
    np.testing.assert_array_equal(decoded_codes, codes)
    np.testing.assert_array_equal(decoded_modes, modes)
    assert len(packed) < codes.shape[0] * modes.shape[1]
    assert any(int(value) in (250, 251, 252, 253, 254) for value in packed)


def test_octave_integer_projection_uses_lookup_add_sub_and_shifts():
    codes = np.asarray(
        [[4, 3, 2, 0, 1, 2], [2, 2, 2, 1, 4, 3]], dtype=np.uint8
    )
    modes = np.asarray([[False, True], [False, True]], dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    activations = np.asarray([[7, -5, 3, 2, -4, 6]], dtype=np.int16)
    actual = integer_octave_projection(activations, packed, offsets, input_features=6)
    half_units = (-2, -1, 0, 1, 2)
    expected = np.zeros((1, 2), dtype=np.int64)
    for out_index in range(2):
        for feature in range(6):
            coefficient = half_units[int(codes[out_index, feature])]
            if modes[out_index, feature // 3]:
                coefficient *= 4
            expected[0, out_index] += int(activations[0, feature]) * coefficient
    np.testing.assert_array_equal(actual, expected)


def test_octave_zero_skip_projection_matches_dense_zero_contribution():
    codes = np.full((1, 70 * 3), 2, dtype=np.uint8)
    modes = np.zeros((1, 70), dtype=np.bool_)
    codes[0, 3 * 66 : 3 * 66 + 3] = np.asarray([4, 0, 3], dtype=np.uint8)
    modes[0, 66] = True
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=True)
    activations = np.arange(1, 70 * 3 + 1, dtype=np.int16).reshape(1, -1)
    actual = integer_octave_projection(
        activations, packed, offsets, input_features=70 * 3
    )
    half_units = (-2, -1, 0, 1, 2)
    expected = 0
    for feature in range(codes.shape[1]):
        coefficient = half_units[int(codes[0, feature])]
        if modes[0, feature // 3]:
            coefficient *= 4
        expected += int(activations[0, feature]) * coefficient
    np.testing.assert_array_equal(actual, np.asarray([[expected]], dtype=np.int64))


def test_octave_decode_lut_has_256_entries_and_marks_reserved_codes():
    lut = build_octave_decode_lut()
    assert lut.shape == (256, 4)
    assert np.all(lut[250:, :] == -128)
    expected = {-8, -4, -2, -1, 0, 1, 2, 4, 8}
    assert set(np.unique(lut[:250, :3])).issubset(expected)


def test_octave_group_size_must_align_to_triplet_packing():
    with pytest.raises(ValueError):
        AddSubLanguageModel(tiny_config(), wavebit_options={"group_size": 32})



def test_octave_x2_quantizer_uses_shared_two_times_boost_without_x4_values():
    weight = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    quantized, codes, modes, scale = octave_shift_quantize(
        weight, group_size=3, ewgs_strength=0.5, boost_factor=2
    )
    assert modes.shape == (1, 1)
    assert bool(modes[0, 0])
    assert torch.all(quantized.abs() <= 2.0 * scale + 1e-6)
    assert codes.shape == weight.shape
    assert torch.isfinite(quantized).all()


def test_octave_x2_integer_decoder_matches_signed_half_unit_reference():
    codes = np.asarray([[4, 2, 0, 1, 3, 2]], dtype=np.uint8)
    modes = np.asarray([[True, False]], dtype=np.bool_)
    packed, offsets = pack_octave_stream(codes, modes, use_zero_skips=False)
    activations = np.asarray([[7, -5, 3, 2, -4, 6]], dtype=np.int16)
    actual = integer_octave_projection(
        activations, packed, offsets, input_features=6, boost_factor=2
    )
    base_levels = (-2, -1, 0, 1, 2)
    expected = 0
    for feature, code in enumerate(codes[0]):
        factor = 2 if modes[0, feature // 3] else 1
        expected += int(activations[0, feature]) * base_levels[int(code)] * factor
    np.testing.assert_array_equal(actual, np.asarray([[expected]], dtype=np.int64))
    assert build_octave_decode_lut(boost_factor=2).shape == (256, 4)
    with pytest.raises(ValueError, match="2 or 4"):
        build_octave_decode_lut(boost_factor=3)


def test_blockwise_hadamard_is_energy_preserving_for_odd_model_widths():
    inputs = torch.randn(3, 170, dtype=torch.float32)
    transformed = blockwise_hadamard(inputs, block_size=8)
    assert transformed.shape == inputs.shape
    torch.testing.assert_close(
        transformed.square().sum(dim=-1),
        inputs.square().sum(dim=-1),
        rtol=2e-5,
        atol=2e-5,
    )


def test_activation_fake_quant_is_four_bit_symmetric_and_has_ste_gradient():
    inputs = torch.randn(4, 170, dtype=torch.float32, requires_grad=True)
    quantized = quantize_activation_ste(inputs, bits=4)
    quantized.sum().backward()
    assert torch.isfinite(quantized).all()
    assert torch.equal(inputs.grad, torch.ones_like(inputs))
    scale = inputs.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7
    codes = torch.round(quantized.detach() / scale)
    assert int(codes.abs().max()) <= 7
    torch.testing.assert_close(codes, torch.round(codes))


def test_x2_hadamard_addsub_model_preserves_state_keys_and_incremental_parity():
    torch.manual_seed(43)
    config = tiny_config()
    experimental = options() | {
        "octave_boost_factor": 2,
        "hadamard": True,
        "hadamard_block_size": 8,
        "activation_bits": 4,
    }
    baseline = AddSubLanguageModel(config, seed=43)
    model = AddSubLanguageModel(config, seed=43, wavebit_options=experimental).eval()
    assert model.parameter_count() == baseline.parameter_count()
    assert set(model.state_dict()) == set(baseline.state_dict())
    tokens = torch.randint(0, 264, (2, 9))
    with torch.inference_mode():
        expected = model(tokens)["logits"]
        caches = model.init_cache(batch_size=tokens.shape[0])
        pieces = []
        for index in range(tokens.shape[1]):
            logits, caches = model.forward_step(tokens[:, index], caches)
            pieces.append(logits.unsqueeze(1))
        actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def test_invalid_hadamard_options_are_rejected():
    with pytest.raises(ValueError, match="power of two"):
        AddSubLanguageModel(tiny_config(), wavebit_options=options() | {
            "hadamard": True, "hadamard_block_size": 6,
        })
    with pytest.raises(ValueError, match="bits"):
        AddSubLanguageModel(tiny_config(), wavebit_options=options() | {
            "activation_bits": 9,
        })


@pytest.mark.parametrize("bad_shape", [(2.5, 3), (2, True), (0, 3), (2,), [2, 3.0]])
def test_octave_decoders_reject_non_integer_or_invalid_shapes(bad_shape):
    packed = np.zeros((2, 1), dtype=np.uint8)
    with pytest.raises((TypeError, ValueError)):
        unpack_octave_triplets(packed, bad_shape)
    stream = np.zeros(2, dtype=np.uint8)
    offsets = np.asarray([0, 1, 2], dtype=np.int64)
    with pytest.raises((TypeError, ValueError)):
        unpack_octave_stream(stream, offsets, bad_shape)


@pytest.mark.parametrize("bad_value", [True, "0.5"])
def test_wavebit_configuration_rejects_boolean_and_string_hyperparameters(bad_value):
    with pytest.raises((TypeError, ValueError)):
        validate_wavebit_options({"ewgs_strength": bad_value})


def test_wavebit_configuration_rejects_boolean_temperature():
    with pytest.raises((TypeError, ValueError)):
        validate_wavebit_options({"octave_temperature": True})
