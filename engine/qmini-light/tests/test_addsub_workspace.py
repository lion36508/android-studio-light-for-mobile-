from __future__ import annotations

from pathlib import Path

import pytest

from qmini_addsub.workspace import resolve_workspace_root


def test_default_workspace_is_created_without_forcing_resume(tmp_path: Path) -> None:
    root = resolve_workspace_root(tmp_path / "version")
    assert root == (tmp_path / "version").resolve()
    assert root.is_dir()


def test_existing_default_workspace_fails_closed_without_resume(tmp_path: Path) -> None:
    version_root = tmp_path / "version"
    old_data = version_root / "data" / "dataset_manifest.json"
    old_data.parent.mkdir(parents=True)
    old_data.write_text('{"old": true}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="No existing files were deleted"):
        resolve_workspace_root(version_root)

    assert old_data.read_text(encoding="utf-8") == '{"old": true}'
    assert resolve_workspace_root(version_root, resume_existing=True) == version_root.resolve()


def test_force_fresh_uses_isolated_directory_and_preserves_old_artifacts(tmp_path: Path) -> None:
    version_root = tmp_path / "version"
    old_checkpoint = version_root / "runs" / "pretrain" / "checkpoint.pt"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_bytes(b"old-checkpoint")

    fresh = resolve_workspace_root(
        version_root,
        force_fresh=True,
        run_id="retrain-2026-11-26-a",
    )

    assert fresh == (version_root / "fresh_runs" / "retrain-2026-11-26-a").resolve()
    assert fresh != version_root.resolve()
    assert old_checkpoint.read_bytes() == b"old-checkpoint"


def test_fresh_workspace_requires_explicit_resume_after_files_exist(tmp_path: Path) -> None:
    version_root = tmp_path / "version"
    fresh = resolve_workspace_root(
        version_root,
        force_fresh=True,
        run_id="new-run",
    )
    artifact = fresh / "data" / "dataset_manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"run": "new-run"}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Set RESUME_EXISTING_RUN=True"):
        resolve_workspace_root(version_root, force_fresh=True, run_id="new-run")

    assert resolve_workspace_root(
        version_root,
        force_fresh=True,
        run_id="new-run",
        resume_existing=True,
    ) == fresh


@pytest.mark.parametrize("run_id", ["", "../escape", "..", "bad/name", "-starts-with-dash", "a" * 65])
def test_fresh_workspace_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="fresh run requires run_id"):
        resolve_workspace_root(tmp_path / "version", force_fresh=True, run_id=run_id)


def test_run_id_is_rejected_when_fresh_mode_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id is valid only"):
        resolve_workspace_root(tmp_path / "version", run_id="unused")



def test_symlinked_workspace_entry_blocks_implicit_reuse(tmp_path: Path) -> None:
    version_root = tmp_path / "version"
    external = tmp_path / "external"
    external.mkdir()
    version_root.mkdir()
    (version_root / "selected_dataset").symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError, match="No existing files were deleted"):
        resolve_workspace_root(version_root)


def test_fresh_workspace_rejects_symlinked_parent(tmp_path: Path) -> None:
    version_root = tmp_path / "version"
    external = tmp_path / "external"
    version_root.mkdir()
    external.mkdir()
    (version_root / "fresh_runs").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="must not traverse symlinks"):
        resolve_workspace_root(version_root, force_fresh=True, run_id="safe-run")
