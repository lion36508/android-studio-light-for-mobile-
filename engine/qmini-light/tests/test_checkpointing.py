import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from qmini.checkpointing import (
    PROGRESS_FILENAME,
    TrainingSessionLease,
    mark_training_status,
    publish_checkpoint,
    restore_latest_checkpoint,
)


def _make_generation_files(root: Path, step: int):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "latest.pt"
    torch.save(
        {
            "format": "qmini-checkpoint-v2",
            "step": step,
            "model": {"weight": torch.tensor([float(step)])},
            "optimizer": {"state": {}, "param_groups": []},
            "config": {"seed": 42},
            "resume_contract": {"train": {"sha256": "abc"}},
        },
        checkpoint,
    )
    metrics = root / "metrics.jsonl"
    metrics.write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
    run = root / "run.json"
    run.write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
    return checkpoint, metrics, run


def _publish_with_lease(checkpoint, mirror, metrics, run, **kwargs):
    with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5) as lease:
        return publish_checkpoint(
            checkpoint, mirror, metrics, run,
            session_id=lease.session_id,
            **kwargs,
        )


def test_checkpoint_generations_publish_progress_and_keep_one_recovery_generation(tmp_path):
    mirror = tmp_path / "drive"
    for step in (2, 4, 6):
        local = tmp_path / f"local-{step}"
        checkpoint, metrics, run = _make_generation_files(local, step)
        progress = _publish_with_lease(
            checkpoint, mirror, metrics, run,
            step=step, target_steps=10, tokens_seen=step * 64,
        )
        assert progress["current_step"] == step
        assert progress["target_steps"] == 10

    current = json.loads((mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert current["current_step"] == 6
    assert current["progress_percent"] == 60.0
    assert len(list(mirror.glob("checkpoint-step-*.manifest.json"))) == 2
    assert (mirror / "checkpoint-step-000000000006.pt").is_file()


def test_restore_falls_back_to_previous_verified_checkpoint_when_latest_is_corrupt(tmp_path):
    mirror = tmp_path / "drive"
    local = tmp_path / "restore"
    for step in (2, 4):
        files = _make_generation_files(tmp_path / f"local-{step}", step)
        _publish_with_lease(
            files[0], mirror, files[1], files[2],
            step=step, target_steps=10, tokens_seen=step * 32,
        )

    latest = mirror / "checkpoint-step-000000000004.pt"
    latest.write_bytes(b"interrupted/corrupt upload")
    restored = restore_latest_checkpoint(mirror, local)

    assert restored is not None
    assert restored["step"] == 2
    restored_progress = json.loads((mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert restored_progress["current_step"] == 2
    payload = torch.load(local / "latest.pt", map_location="cpu", weights_only=True)
    assert payload["step"] == 2
    assert json.loads((local / "metrics.jsonl").read_text())["step"] == 2


def test_checkpoint_publish_does_not_advance_progress_when_upload_is_incomplete(tmp_path):
    mirror = tmp_path / "drive"
    files = _make_generation_files(tmp_path / "local", 3)
    _publish_with_lease(
        files[0], mirror, files[1], files[2],
        step=3, target_steps=9, tokens_seen=192,
    )
    previous = (mirror / PROGRESS_FILENAME).read_bytes()
    (mirror / ".checkpoint-step-000000000006.pt.uploading").write_bytes(b"partial")
    assert (mirror / PROGRESS_FILENAME).read_bytes() == previous
    assert restore_latest_checkpoint(mirror, tmp_path / "restored")["step"] == 3


def test_mark_training_status_replaces_the_single_progress_record(tmp_path):
    mirror = tmp_path / "drive"
    files = _make_generation_files(tmp_path / "local", 7)
    _publish_with_lease(
        files[0], mirror, files[1], files[2],
        step=7, target_steps=7, tokens_seen=448,
    )
    progress = mark_training_status(mirror, "completed", final_model_sha256="f" * 64)
    assert progress["status"] == "completed"
    assert progress["current_step"] == 7
    assert progress["final_model_sha256"] == "f" * 64


def test_training_session_lease_prevents_concurrent_writer(tmp_path):
    mirror = tmp_path / "drive"
    with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5):
        with pytest.raises(RuntimeError, match="Another Q-Mini training session"):
            with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5):
                pass
    assert not (mirror / "training_session.lease.json").exists()


def test_training_session_lease_recovers_abandoned_stale_lease(tmp_path):
    mirror = tmp_path / "drive"
    mirror.mkdir()
    lease_path = mirror / "training_session.lease.json"
    lease_path.write_text(json.dumps({
        "format": "qmini-training-lease-v1",
        "session_id": "abandoned-session",
        "heartbeat_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
    }), encoding="utf-8")

    with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5) as recovered:
        active = json.loads(lease_path.read_text(encoding="utf-8"))
        assert active["session_id"] == recovered.session_id
        assert active["session_id"] != "abandoned-session"
    assert not lease_path.exists()


def test_train_model_mirrors_verified_checkpoint_after_each_save(tiny_config, tmp_path):
    import numpy as np
    from qmini.train import train_model

    train = tmp_path / "train.bin"
    valid = tmp_path / "valid.bin"
    np.random.default_rng(13).integers(
        0, tiny_config.vocab_size, size=128, dtype=np.uint16
    ).tofile(train)
    np.random.default_rng(14).integers(
        0, tiny_config.vocab_size, size=128, dtype=np.uint16
    ).tofile(valid)

    local_run = tmp_path / "local-run"
    mirror = tmp_path / "drive-checkpoints"
    with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5) as lease:
        train_model(
            tiny_config, train, valid, local_run,
            steps=2, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            checkpoint_mirror_dir=mirror,
            checkpoint_session_id=lease.session_id,
        )
    progress = json.loads((mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert progress["current_step"] == 2
    assert progress["target_steps"] == 2
    assert progress["status"] == "checkpointed"

    restored_dir = tmp_path / "restored-run"
    restored = restore_latest_checkpoint(mirror, restored_dir)
    assert restored["step"] == 2
    checkpoint = torch.load(
        restored_dir / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["step"] == 2

    # A fresh Colab runtime restores model, optimizer and RNG state and continues
    # the same fingerprinted training run instead of restarting from step zero.
    with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5) as lease:
        train_model(
            tiny_config, train, valid, restored_dir,
            steps=3, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            resume=True, extend_schedule=True,
            checkpoint_mirror_dir=mirror,
            checkpoint_session_id=lease.session_id,
        )
    final_progress = json.loads((mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert final_progress["current_step"] == 3
    assert final_progress["target_steps"] == 3
    resumed_checkpoint = torch.load(
        restored_dir / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert resumed_checkpoint["step"] == 3

def test_restore_ignores_generation_written_before_progress_commit(tmp_path):
    mirror = tmp_path / "drive"
    first = _make_generation_files(tmp_path / "first", 2)
    _publish_with_lease(
        first[0], mirror, first[1], first[2],
        step=2, target_steps=10, tokens_seen=64,
    )
    committed_progress = (mirror / PROGRESS_FILENAME).read_bytes()

    second = _make_generation_files(tmp_path / "second", 4)
    _publish_with_lease(
        second[0], mirror, second[1], second[2],
        step=4, target_steps=10, tokens_seen=128,
    )
    # Simulate a completed generation upload whose atomic progress commit
    # did not occur.
    (mirror / PROGRESS_FILENAME).write_bytes(committed_progress)

    restored = restore_latest_checkpoint(mirror, tmp_path / "restored")
    assert restored["step"] == 2



def test_checkpoint_publish_rejects_progress_regression(tmp_path):
    mirror = tmp_path / "drive"
    first = _make_generation_files(tmp_path / "first", 6)
    _publish_with_lease(
        first[0], mirror, first[1], first[2],
        step=6, target_steps=10, tokens_seen=384,
    )
    before = json.loads((mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))

    stale = _make_generation_files(tmp_path / "stale", 4)
    with TrainingSessionLease(mirror, stale_after_seconds=20, heartbeat_seconds=5) as lease:
        with pytest.raises(ValueError, match="backwards"):
            publish_checkpoint(
                stale[0], mirror, stale[1], stale[2],
                step=4, target_steps=10, tokens_seen=256,
                session_id=lease.session_id,
            )
    after = json.loads((mirror / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert after == before


def test_checkpoint_publish_requires_active_writer_lease(tmp_path):
    files = _make_generation_files(tmp_path / "local", 2)
    with pytest.raises(RuntimeError, match="No active Q-Mini training lease"):
        publish_checkpoint(
            files[0], tmp_path / "drive", files[1], files[2],
            step=2, target_steps=5, tokens_seen=128,
            session_id="not-leased",
        )
