import hashlib
import json
import zipfile

import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.wavebit_hsra import validate_wavebit_options


@pytest.mark.parametrize("boost_factor", [2, 4])
def test_wavebit_groupwise_export_roundtrip_preserves_reference_model(
    tmp_path, boost_factor
):
    from qmini_addsub.export import export_addsub_checkpoint, load_addsub_export
    from qmini_addsub.train import CHECKPOINT_FORMAT

    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=9127 + boost_factor,
    )
    options = validate_wavebit_options({
        "group_size": 6,
        "octave_boost_factor": boost_factor,
        "hadamard": True,
        "hadamard_block_size": 8,
        "activation_bits": 4,
    })
    original = AddSubLanguageModel(
        config, seed=config.seed, wavebit_options=options
    ).eval()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(json.dumps({"vocab": config.vocab_size}), encoding="utf-8")
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    checkpoint_path = tmp_path / "wavebit-checkpoint.pt"
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "step": 4,
        "config": config.to_dict(),
        "model": original.state_dict(),
        "tokenizer_sha256": tokenizer_sha,
        "data_manifest_sha256": "e" * 64,
        "run_contract": {
            "planned_steps": 8,
            "batch_size": 2,
            "sequence_length": 16,
            "wavebit_hsra": options,
        },
    }, checkpoint_path)

    export_path = tmp_path / f"wavebit-x{boost_factor}.qma"
    result = export_addsub_checkpoint(checkpoint_path, export_path)
    with zipfile.ZipFile(export_path, "r") as archive:
        assert set(archive.namelist()) == {
            "manifest.json", "weights.bin", "auxiliary.npz", "group_metadata.npz"
        }
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["format"] == "qmini-addsub-wavebit-packed-v1"
        assert manifest["boost_factor"] == boost_factor
        assert manifest["group_scale"]["group_size"] == 6
        assert len(manifest["quantized_tensors"]) > 0
        assert all(
            entry["group_scale_shape"][1] >= 1
            and entry["row_offsets_shape"][0] == entry["shape"][0] + 1
            for entry in manifest["quantized_tensors"]
        )

    restored, loaded_manifest = load_addsub_export(
        export_path, tokenizer_path=tokenizer_path
    )
    assert result["format"] == loaded_manifest["format"]
    assert restored.parameter_count() == original.parameter_count()
    assert set(restored.state_dict()) == set(original.state_dict())
    assert restored.wavebit_options == options

    torch.manual_seed(81)
    input_ids = torch.randint(0, config.vocab_size, (2, 11))
    with torch.inference_mode():
        expected = original(input_ids)["logits"]
        actual = restored(input_ids)["logits"]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_wavebit_export_rejects_tampered_group_metadata(tmp_path):
    from qmini_addsub.export import export_addsub_checkpoint, load_addsub_export
    from qmini_addsub.train import CHECKPOINT_FORMAT

    config = AddSubConfig(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=101,
    )
    options = validate_wavebit_options({
        "group_size": 6,
        "octave_boost_factor": 2,
    })
    model = AddSubLanguageModel(config, seed=config.seed, wavebit_options=options)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "step": 1,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "tokenizer_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64,
        "run_contract": {"wavebit_hsra": options},
    }, checkpoint_path)
    source = tmp_path / "source.qma"
    target = tmp_path / "tampered.qma"
    export_addsub_checkpoint(checkpoint_path, source)
    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    damaged = bytearray(members["group_metadata.npz"])
    damaged[-1] ^= 0x01
    members["group_metadata.npz"] = bytes(damaged)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="group metadata failed SHA-256"):
        load_addsub_export(target)


def test_legacy_non_wavebit_export_keeps_existing_archive_format(tmp_path):
    from qmini_addsub.export import export_addsub_checkpoint, load_addsub_export
    from qmini_addsub.train import CHECKPOINT_FORMAT

    config = AddSubConfig(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=202,
    )
    model = AddSubLanguageModel(config, seed=config.seed)
    checkpoint_path = tmp_path / "legacy-checkpoint.pt"
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "step": 1,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "tokenizer_sha256": "c" * 64,
        "data_manifest_sha256": "d" * 64,
        "run_contract": {"seed": config.seed},
    }, checkpoint_path)
    export_path = tmp_path / "legacy.qma"
    export_addsub_checkpoint(checkpoint_path, export_path)
    restored, manifest = load_addsub_export(export_path)
    assert manifest["format"] == "qmini-addsub-packed-weights-v2"
    assert restored.parameter_count() == model.parameter_count()
