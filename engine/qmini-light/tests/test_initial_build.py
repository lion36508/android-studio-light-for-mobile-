import hashlib
import json

import torch

from qmini.export import verify_packed_export
from qmini.initial_build import build_initial_brain, iter_initial_state_tensors
from qmini.model import QMiniMoE
from qmini.packed_runtime import load_packed_model


def test_streamed_initial_state_matches_full_model_initialization(tiny_config):
    torch.manual_seed(tiny_config.seed)
    full_model = QMiniMoE(tiny_config)
    expected = full_model.state_dict()
    streamed = dict(iter_initial_state_tensors(tiny_config))

    assert set(streamed) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(streamed[name], tensor), name


def test_initial_build_emits_verified_q2_v2_artifact(tiny_config, tmp_path):
    target_dir = tmp_path / "fresh-q2-initial"
    manifest = build_initial_brain(
        tiny_config, target_dir, source_revision="test-revision"
    )

    assert manifest["format"] == "qmini-initial-brain-v1"
    assert manifest["status"] == "untrained_initialization"
    assert manifest["training_steps"] == 0
    assert manifest["optimizer_steps"] == 0
    assert manifest["tokens_seen"] == 0
    assert manifest["matrix_weight_format"] == "qmini-packed-v2"
    assert manifest["matrix_weight_bits"] == 2
    assert manifest["parameters"] == tiny_config.estimated_parameter_count()
    assert not (target_dir / "model.pt").exists()

    report = verify_packed_export(target_dir / "model.qmini.npz")
    assert report["format"] == "qmini-packed-v2"
    assert report["q2_version"] == 2
    assert report["parameter_count"] == tiny_config.estimated_parameter_count()

    with (target_dir / "model_manifest.json").open(encoding="utf-8") as handle:
        saved_manifest = json.load(handle)
    from qmini.config import QMiniConfig
    assert QMiniConfig.from_dict(saved_manifest.pop("config")).to_dict() == tiny_config.to_dict()
    expected_manifest = dict(manifest)
    expected_manifest.pop("config")
    assert saved_manifest == expected_manifest

    packed_model = load_packed_model(
        target_dir / "model.qmini.npz", expected_config=tiny_config
    )
    assert getattr(packed_model, "_qmini_packed_runtime", False)
    assert all(parameter.ndim < 2 for parameter in packed_model.parameters())


def test_initial_build_refuses_to_overwrite_existing_directory(tiny_config, tmp_path):
    target_dir = tmp_path / "existing"
    target_dir.mkdir()
    sentinel = target_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    import pytest
    with pytest.raises(FileExistsError):
        build_initial_brain(tiny_config, target_dir)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_pretraining_can_start_from_step_zero_q2_initial_brain(tiny_config, tmp_path):
    import numpy as np
    from qmini.train import train_model

    initial_dir = tmp_path / "initial-brain"
    build_initial_brain(tiny_config, initial_dir, source_revision="test-source")
    train_tokens = np.random.default_rng(10).integers(
        0, tiny_config.vocab_size, size=128, dtype=np.uint16
    )
    valid_tokens = np.random.default_rng(11).integers(
        0, tiny_config.vocab_size, size=128, dtype=np.uint16
    )
    train_path = tmp_path / "train.bin"
    valid_path = tmp_path / "valid.bin"
    train_tokens.tofile(train_path)
    valid_tokens.tofile(valid_path)

    run_dir = tmp_path / "q2-pretraining-run"
    train_model(
        tiny_config,
        train_path,
        valid_path,
        run_dir,
        steps=1,
        batch_size=1,
        seq_len=8,
        grad_accum=1,
        eval_every=1,
        save_every=1,
        eval_batches=1,
        init_from_initial=initial_dir / "model.qmini.npz",
    )

    with (run_dir / "run.json").open(encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    assert run_manifest["initial_model"]["kind"] == "q2-step-zero-initial"
    assert run_manifest["initial_model"]["path"] == str(
        (initial_dir / "model.qmini.npz").resolve()
    )
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["resume_contract"]["initial_model"]["sha256"]



def test_q2_initialization_resume_preserves_original_source_lineage(tiny_config, tmp_path):
    import numpy as np
    from qmini.checkpointing import PROGRESS_FILENAME, TrainingSessionLease, restore_latest_checkpoint
    from qmini.train import train_model

    initial_dir = tmp_path / "initial-q2"
    build_initial_brain(tiny_config, initial_dir, source_revision="resume-lineage-test")
    train_tokens = np.random.default_rng(91).integers(
        0, tiny_config.vocab_size, size=160, dtype=np.uint16
    )
    valid_tokens = np.random.default_rng(92).integers(
        0, tiny_config.vocab_size, size=160, dtype=np.uint16
    )
    train_path = tmp_path / "train.bin"
    valid_path = tmp_path / "valid.bin"
    train_tokens.tofile(train_path)
    valid_tokens.tofile(valid_path)

    local_run = tmp_path / "local-run"
    drive_mirror = tmp_path / "drive-mirror"
    initial_model_path = initial_dir / "model.qmini.npz"

    with TrainingSessionLease(drive_mirror, stale_after_seconds=20, heartbeat_seconds=5) as lease:
        train_model(
            tiny_config, train_path, valid_path, local_run,
            steps=1, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            init_from_initial=initial_model_path,
            checkpoint_mirror_dir=drive_mirror,
            checkpoint_session_id=lease.session_id,
        )

    first_manifest = json.loads((local_run / "run.json").read_text(encoding="utf-8"))
    original_lineage = first_manifest["initial_model"]
    assert original_lineage["kind"] == "q2-step-zero-initial"
    expected_sha = hashlib.sha256(initial_model_path.read_bytes()).hexdigest()
    assert original_lineage["sha256"] == expected_sha

    restored_run = tmp_path / "restored-run"
    restored = restore_latest_checkpoint(drive_mirror, restored_run)
    assert restored["step"] == 1

    with TrainingSessionLease(drive_mirror, stale_after_seconds=20, heartbeat_seconds=5) as lease:
        train_model(
            tiny_config, train_path, valid_path, restored_run,
            steps=2, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            resume=True, extend_schedule=True,
            init_from_initial=None,
            checkpoint_mirror_dir=drive_mirror,
            checkpoint_session_id=lease.session_id,
        )

    resumed_manifest = json.loads((restored_run / "run.json").read_text(encoding="utf-8"))
    assert resumed_manifest["initial_model"] == original_lineage
    resumed_checkpoint = torch.load(
        restored_run / "latest.pt", map_location="cpu", weights_only=True
    )
    assert resumed_checkpoint["resume_contract"]["initial_model"]["sha256"] == expected_sha
    progress = json.loads((drive_mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert progress["current_step"] == 2
    assert progress["status"] == "checkpointed"
