import json
import pytest
import numpy as np
from qmini.export import export_packed_state, reconstruct_packed_state, verify_packed_export
from qmini.model import QMiniMoE
from qmini.quantization import quantize_quaternary


def test_packed_export_roundtrip_matches_quaternary_projection(tiny_config, tmp_path):
    model = QMiniMoE(tiny_config)
    target = export_packed_state(
        model.state_dict(), tiny_config, tmp_path / "model.qmini.npz"
    )
    config, restored = reconstruct_packed_state(target)
    assert config.to_dict() == tiny_config.to_dict()
    with np.load(target, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["format"] == "qmini-packed-v2"
    assert metadata["q2_contract"] == {
        "version": 2,
        "codebook": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
        "row_scale": "2 * mean(abs(original_output_row))",
        "all_zero_row_scale": 0.0,
    }
    original = model.state_dict()
    assert set(restored) == set(original)

    for name, tensor in original.items():
        if tensor.ndim >= 2:
            expected = quantize_quaternary(tensor).detach().cpu().numpy()
            assert restored[name].shape == tuple(tensor.shape)
            assert np.allclose(restored[name], expected, atol=1e-6, rtol=1e-6), name
        else:
            assert np.allclose(
                restored[name], tensor.detach().cpu().numpy(), atol=1e-3, rtol=1e-3
            ), name

    report = verify_packed_export(target)
    assert report["format"] == "qmini-packed-v2"
    assert report["q2_version"] == 2
    assert report["tensor_count"] == len(original)
    assert report["parameter_count"] == sum(t.numel() for t in original.values())
    assert report["context"] == tiny_config.max_context


def test_verify_packed_export_does_not_expand_full_state(
    tiny_config, tmp_path, monkeypatch
):
    import qmini.export as export_module

    model = QMiniMoE(tiny_config)
    target = export_packed_state(
        model.state_dict(), tiny_config, tmp_path / "bounded.qmini.npz"
    )

    def forbidden_reconstruction(*args, **kwargs):
        raise AssertionError("large-export verification must not reconstruct FP32 state")

    monkeypatch.setattr(export_module, "reconstruct_packed_state", forbidden_reconstruction)
    report = verify_packed_export(target)
    assert report["parameter_count"] == tiny_config.estimated_parameter_count()
    assert report["decoded_payload_bytes"] + 65536 == report["estimated_raw_layout_bytes"]
    assert report["artifact_bytes"] == target.stat().st_size



def test_verify_export_rejects_contradictory_q2_v2_codebook_metadata(
    tiny_config, tmp_path
):
    model = QMiniMoE(tiny_config)
    source = export_packed_state(
        model.state_dict(), tiny_config, tmp_path / "source.qmini.npz"
    )
    corrupted = tmp_path / "wrong-codebook.qmini.npz"
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["quaternary_codes"]["00"] = -2.0
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, separators=(",", ":"))
    )
    np.savez_compressed(corrupted, **arrays)

    import pytest
    with pytest.raises(ValueError, match="contract/codebook metadata"):
        verify_packed_export(corrupted)


@pytest.mark.parametrize(
    "metadata_payload",
    [
        '{"format":"qmini-packed-v2","q2_contract":{},"quaternary_codes":{} }',
        'null',
        '[]',
    ],
)
def test_verify_packed_export_normalizes_malformed_metadata(
    tmp_path, metadata_payload
):
    target = tmp_path / "malformed-metadata.npz"
    np.savez(target, metadata_json=np.asarray(metadata_payload))
    import pytest

    with pytest.raises(ValueError, match="Malformed Q-Mini packed archive|invalid|Unsupported"):
        verify_packed_export(target)


def test_verify_packed_export_normalizes_missing_required_archive_member(tmp_path):
    target = tmp_path / "missing-config.npz"
    metadata = {
        "format": "qmini-packed-v2",
        "q2_contract": {
            "version": 2,
            "codebook": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
            "row_scale": "2 * mean(abs(original_output_row))",
            "all_zero_row_scale": 0.0,
        },
        "quaternary_codes": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
    }
    np.savez(target, metadata_json=np.asarray(json.dumps(metadata)))
    import pytest

    with pytest.raises(ValueError, match="Malformed Q-Mini packed archive"):
        verify_packed_export(target)
    with pytest.raises(ValueError, match="Malformed Q-Mini packed archive"):
        reconstruct_packed_state(target)
