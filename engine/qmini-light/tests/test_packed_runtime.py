import pytest
import json
import numpy as np
import torch

from qmini.export import export_packed_state
from qmini.model import QMiniMoE
from qmini.packed_runtime import load_packed_model


def _packed_config(tiny_config, state_space=False, vocab_size=None):
    values = tiny_config.to_dict()
    values["use_state_space"] = state_space
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return type(tiny_config).from_dict(values)


def test_packed_runtime_keeps_matrices_packed_and_matches_reference(tiny_config, tmp_path):
    torch.manual_seed(901)
    config = _packed_config(tiny_config)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "model.qmini.npz"
    )
    packed = load_packed_model(path, expected_config=config)

    assert packed._qmini_packed_runtime is True
    assert packed.parameter_report()["total"] == config.estimated_parameter_count()
    assert packed.parameter_report()["trainable"] == 0
    assert all(parameter.device.type == "cpu" for parameter in packed.parameters())
    assert all(parameter.ndim < 2 for parameter in packed.parameters())

    ids = torch.randint(0, config.vocab_size, (2, 13))
    with torch.inference_mode():
        expected = reference(ids, thinking_steps=2)["logits"]
        actual = packed(ids, thinking_steps=2)["logits"]

        state = None
        streamed = []
        for position in range(ids.shape[1]):
            result, state = packed.forward_step(
                ids[:, position], state=state, thinking_steps=2
            )
            streamed.append(result["logits"])
        stream_logits = torch.stack(streamed, dim=1)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.allclose(stream_logits, actual, atol=3e-5, rtol=3e-5)


def test_packed_runtime_supports_adaptive_state_space(tiny_config, tmp_path):
    config = _packed_config(tiny_config, state_space=True)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "hybrid.qmini.npz"
    )
    packed = load_packed_model(path, expected_config=config)
    ids = torch.randint(0, config.vocab_size, (1, 9))

    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_packed_runtime_has_bounded_pytorch_fallback(
    tiny_config, tmp_path, monkeypatch
):
    import qmini.native as native_module

    config = _packed_config(tiny_config, vocab_size=300)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "fallback.qmini.npz"
    )
    packed = load_packed_model(path, expected_config=config)
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: None)
    ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_dense_single_expert_archive_loads_without_router(tiny_config, tmp_path):
    from qmini.export import verify_packed_export
    from qmini.model import QMiniMoE

    values = tiny_config.to_dict()
    values["n_experts"] = 1
    config = type(tiny_config).from_dict(values)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "dense.qmini.npz"
    )
    packed_report = verify_packed_export(path)
    assert packed_report["parameter_count"] == config.estimated_parameter_count()
    assert packed_report["decoded_payload_bytes"] + 65536 == config.estimated_packed_export_bytes()

    packed = load_packed_model(path, expected_config=config)
    assert not any(
        "router" in name
        for name, _ in packed.named_parameters()
    )
    ids = torch.randint(0, config.vocab_size, (1, 7))
    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)



def _write_legacy_q2_v1_archive(state_dict, config, path):
    arrays = {}
    specs = []
    for index, (name, tensor) in enumerate(state_dict.items()):
        value = tensor.detach().float().cpu().contiguous().numpy()
        if value.ndim >= 2:
            rows = value.shape[0]
            flat = value.reshape(rows, -1)
            scales = np.maximum(np.mean(np.abs(flat), axis=1), 1e-8).astype(np.float32)
            magnitude = np.where(
                np.abs(flat) / scales[:, None] >= 1.5, 2, 1
            )
            quantized = np.where(flat < 0, -magnitude, magnitude).astype(np.int8)
            codes = np.where(
                quantized == -2, 0,
                np.where(quantized == -1, 1, np.where(quantized == 1, 2, 3)),
            ).astype(np.uint8)
            columns = flat.shape[1]
            padded = np.zeros((rows, ((columns + 3) // 4) * 4), dtype=np.uint8)
            padded[:, :columns] = codes
            packed = (
                padded[:, 0::4]
                | (padded[:, 1::4] << 2)
                | (padded[:, 2::4] << 4)
                | (padded[:, 3::4] << 6)
            )
            packed_key, scale_key = f"p_{index}", f"s_{index}"
            arrays[packed_key] = packed
            arrays[scale_key] = scales
            specs.append({
                "name": name, "kind": "q2", "shape": list(value.shape),
                "packed_key": packed_key, "scale_key": scale_key,
                "in_features": int(columns),
            })
        else:
            value_key = f"f_{index}"
            arrays[value_key] = value.astype(np.float16)
            specs.append({
                "name": name, "kind": "fp16", "shape": list(value.shape),
                "value_key": value_key,
            })
    metadata = {
        "format": "qmini-packed-v1",
        "config": config.to_dict(),
        "tensor_count": len(specs),
        "quaternary_tensor_count": sum(item["kind"] == "q2" for item in specs),
        "fp16_tensor_count": sum(item["kind"] == "fp16" for item in specs),
        "tensors": specs,
        "quaternary_codes": {"00": -2, "01": -1, "10": 1, "11": 2},
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
    np.savez_compressed(path, **arrays)
    return path



def test_legacy_q2_v1_export_reconstruct_and_runtime_compatibility(
    tiny_config, tmp_path, monkeypatch
):
    from qmini.export import reconstruct_packed_state, verify_packed_export

    torch.manual_seed(992)
    config = _packed_config(tiny_config, state_space=True)
    reference = QMiniMoE(config).eval()
    path = _write_legacy_q2_v1_archive(
        reference.state_dict(), config, tmp_path / "legacy-v1.qmini.npz"
    )

    report = verify_packed_export(path)
    assert report["format"] == "qmini-packed-v1"
    assert report["q2_version"] == 1
    legacy_config, reconstructed = reconstruct_packed_state(path)
    assert legacy_config.to_dict() == config.to_dict()
    assert set(reconstructed) == set(reference.state_dict())

    packed = load_packed_model(path, expected_config=config)
    assert packed._qmini_packed_runtime is True
    for module in packed.modules():
        if getattr(module, "_qmini_packed_weight", None) is not None:
            assert module._qmini_q2_version == 1

    ids = torch.randint(0, config.vocab_size, (1, 11))

    # The source model's current forward path uses Q2-v2 STE. To compare a
    # legacy v1 archive fairly, reproduce the historical Q2-v1 fake quantizer
    # only for the reference forward pass.
    import qmini.model as model_module

    def legacy_v1_fake_quantizer(weight):
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        magnitude = torch.where(
            flat.detach().abs() / scale >= 1.5, 2.0, 1.0
        )
        sign = torch.where(flat.detach() < 0, -1.0, 1.0)
        q = (sign * magnitude * scale).reshape_as(weight)
        return weight + (q - weight).detach()

    monkeypatch.setattr(
        model_module, "quantize_quaternary", legacy_v1_fake_quantizer
    )
    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)



def test_q2_v2_all_zero_depthwise_row_survives_export_and_packed_inference(
    tiny_config, tmp_path
):
    from qmini.export import verify_packed_export

    torch.manual_seed(1001)
    config = _packed_config(tiny_config)
    reference = QMiniMoE(config).eval()
    with torch.no_grad():
        reference.blocks[0].mixer.depthwise_weight[0].zero_()

    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "q2-v2-zero-row.npz"
    )
    report = verify_packed_export(path)
    assert report["format"] == "qmini-packed-v2"

    packed = load_packed_model(path, expected_config=config)
    convolution = packed.blocks[0].mixer
    assert convolution._qmini_q2_version == 2
    assert convolution._qmini_weight_scales[0].item() == 0.0
    kernel = convolution._kernel()
    assert torch.equal(kernel[0], torch.zeros_like(kernel[0]))

    ids = torch.randint(0, config.vocab_size, (1, 9))
    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)



def test_packed_runtime_rejects_q2_v2_codebook_metadata_mismatch(
    tiny_config, tmp_path
):
    from qmini.export import export_packed_state

    model = QMiniMoE(_packed_config(tiny_config))
    source = export_packed_state(
        model.state_dict(), model.config, tmp_path / "valid-v2.npz"
    )
    corrupted = tmp_path / "mismatched-v2.npz"
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["quaternary_codes"]["01"] = -1.0
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, separators=(",", ":"))
    )
    np.savez_compressed(corrupted, **arrays)

    with pytest.raises(ValueError, match="contract/codebook metadata"):
        load_packed_model(corrupted, expected_config=model.config)
