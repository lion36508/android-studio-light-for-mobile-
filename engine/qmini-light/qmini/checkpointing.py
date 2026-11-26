from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROGRESS_FILENAME = "training_progress.json"
LEASE_FILENAME = "training_session.lease.json"
CHECKPOINT_PREFIX = "checkpoint-step-"
KEEP_CHECKPOINTS = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint sidecar does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _fingerprint(source)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.uploading")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        copied = _fingerprint(temporary)
        if (
            copied["size_bytes"] != source_fingerprint["size_bytes"]
            or copied["sha256"] != source_fingerprint["sha256"]
        ):
            raise IOError(f"Drive copy verification failed for {target.name}.")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return {
            "filename": target.name,
            "size_bytes": copied["size_bytes"],
            "sha256": copied["sha256"],
        }
    finally:
        temporary.unlink(missing_ok=True)


def _verify_torch_archive(path: Path) -> None:
    """Check checkpoint container and CRC without materializing model tensors."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if not any(name.endswith("data.pkl") for name in names):
                raise ValueError("PyTorch checkpoint archive has no data.pkl record.")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"PyTorch checkpoint CRC failed for {corrupt}.")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Checkpoint is not a complete PyTorch archive: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")
    return payload


def _same_existing_file(source: Path, target: Path) -> dict[str, Any] | None:
    if not target.exists():
        return None
    current = _fingerprint(target)
    incoming = _fingerprint(source)
    if current["size_bytes"] != incoming["size_bytes"] or current["sha256"] != incoming["sha256"]:
        raise FileExistsError(
            f"Refusing to replace an existing immutable checkpoint generation: {target}"
        )
    return current


def _publish_immutable(source: Path, target: Path) -> dict[str, Any]:
    existing = _same_existing_file(source, target)
    return existing if existing is not None else _atomic_copy(source, target)


def publish_checkpoint(
    checkpoint_path: str | Path,
    mirror_dir: str | Path,
    metrics_path: str | Path,
    run_manifest_path: str | Path,
    *,
    step: int,
    target_steps: int,
    tokens_seen: int,
    session_id: str,
    keep_last: int = KEEP_CHECKPOINTS,
) -> dict[str, Any]:
    """Publish an immutable Drive generation, then advance one progress record."""
    for name, value in (("step", step), ("target_steps", target_steps), ("tokens_seen", tokens_seen), ("keep_last", keep_last)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if step < 1 or target_steps < step or tokens_seen < 0 or keep_last < 2:
        raise ValueError("Invalid checkpoint step, target, token count or retention count.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")

    source = Path(checkpoint_path)
    metrics_source = Path(metrics_path)
    run_source = Path(run_manifest_path)
    destination = Path(mirror_dir).expanduser()
    if not source.is_file() or not metrics_source.is_file() or not run_source.is_file():
        raise FileNotFoundError("Checkpoint, metrics log and run manifest must all exist before publishing.")

    stem = f"{CHECKPOINT_PREFIX}{step:012d}"
    checkpoint_target = destination / f"{stem}.pt"
    metrics_target = destination / f"{stem}.metrics.jsonl"
    run_target = destination / f"{stem}.run.json"
    manifest_target = destination / f"{stem}.manifest.json"

    destination.mkdir(parents=True, exist_ok=True)
    # A checkpoint mirror is only safe while this session owns the writer lease.
    touch_training_lease(destination, session_id)
    committed_step = 0
    progress_path = destination / PROGRESS_FILENAME
    previous_progress: dict[str, Any] | None = None
    if progress_path.is_file():
        previous_progress = _read_json(progress_path)
        previous_step = previous_progress.get("current_step", 0)
        if isinstance(previous_step, bool) or not isinstance(previous_step, int) or previous_step < 0:
            raise ValueError("Existing Drive progress record has an invalid current_step.")
        committed_step = previous_step

    if step < committed_step:
        raise ValueError(
            f"Refusing to move Drive progress backwards from step {committed_step} to {step}. "
            "Restore the latest committed checkpoint while holding the training lease."
        )
    if step == committed_step and previous_progress is not None:
        manifest_name = previous_progress.get("checkpoint_manifest_file")
        if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
            raise RuntimeError("Current progress has no safe committed checkpoint manifest.")
        committed_manifest_path = destination / manifest_name
        committed_manifest = _validate_generation(destination, committed_manifest_path)
        source_hashes = {
            "checkpoint": sha256_file(source),
            "metrics": sha256_file(metrics_source),
            "run_manifest": sha256_file(run_source),
        }
        if (
            committed_manifest.get("step") != step
            or committed_manifest.get("target_steps") != target_steps
            or committed_manifest.get("tokens_seen") != tokens_seen
            or any(
                committed_manifest.get(key, {}).get("sha256") != digest
                for key, digest in source_hashes.items()
            )
        ):
            raise ValueError(
                "This step is already committed with different checkpoint/metrics/run data; "
                "refusing to replace its immutable generation."
            )
        return previous_progress

    _discard_uncommitted_generations(destination, committed_step)
    checkpoint_copy = _publish_immutable(source, checkpoint_target)
    checkpoint_fingerprint = checkpoint_copy
    metrics_copy = _publish_immutable(metrics_source, metrics_target)
    run_copy = _publish_immutable(run_source, run_target)

    _verify_torch_archive(checkpoint_target)
    manifest = {
        "format": "qmini-checkpoint-generation-v1",
        "step": step,
        "target_steps": target_steps,
        "tokens_seen": tokens_seen,
        "session_id": session_id,
        "checkpoint": checkpoint_copy,
        "metrics": metrics_copy,
        "run_manifest": run_copy,
        "published_at": utc_now(),
    }
    if manifest_target.exists():
        existing_manifest = _read_json(manifest_target)
        identity = ("step", "target_steps", "tokens_seen", "session_id", "checkpoint", "metrics", "run_manifest")
        if any(existing_manifest.get(key) != manifest.get(key) for key in identity):
            raise FileExistsError(f"Checkpoint generation metadata conflicts: {manifest_target}")
        manifest = existing_manifest
    else:
        _atomic_json(manifest, manifest_target)

    progress = {
        "format": "qmini-training-progress-v1",
        "status": "checkpointed",
        "current_step": step,
        "target_steps": target_steps,
        "progress_percent": round(100.0 * step / target_steps, 4),
        "tokens_seen": tokens_seen,
        "session_id": session_id,
        "checkpoint_file": checkpoint_target.name,
        "checkpoint_manifest_file": manifest_target.name,
        "checkpoint_sha256": checkpoint_fingerprint["sha256"],
        "checkpoint_bytes": checkpoint_fingerprint["size_bytes"],
        "updated_at": utc_now(),
    }
    # The progress JSON is the one atomic commit record. If this write fails,
    # the previous current step remains authoritative and the new generation
    # is treated as an orphan on the next writer's start.
    touch_training_lease(destination, session_id)
    _atomic_json(progress, destination / PROGRESS_FILENAME)
    _prune_generations(destination, keep_last=keep_last)
    return progress


def _discard_uncommitted_generations(directory: Path, committed_step: int) -> None:
    """Remove only incomplete/uncommitted generations above the durable progress step."""
    for manifest_path in directory.glob(f"{CHECKPOINT_PREFIX}*.manifest.json"):
        step = _step_from_name(manifest_path.name)
        if step <= committed_step:
            continue
        try:
            manifest = _read_json(manifest_path)
        except ValueError:
            manifest = {}
        for field in ("checkpoint", "metrics", "run_manifest"):
            record = manifest.get(field)
            filename = record.get("filename") if isinstance(record, dict) else None
            if isinstance(filename, str) and Path(filename).name == filename:
                (directory / filename).unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    # Files whose upload completed but whose generation manifest/commit did not
    # are also safe to remove because the visible progress record is unchanged.
    for path in directory.glob(f"{CHECKPOINT_PREFIX}*"):
        step = _step_from_name(path.name)
        if step > committed_step and path.is_file():
            path.unlink(missing_ok=True)
    for path in directory.glob(".*.uploading"):
        path.unlink(missing_ok=True)


def _prune_generations(directory: Path, keep_last: int) -> None:
    manifests = sorted(
        directory.glob(f"{CHECKPOINT_PREFIX}*.manifest.json"),
        key=lambda path: _step_from_name(path.name),
        reverse=True,
    )
    for manifest_path in manifests[keep_last:]:
        try:
            manifest = _read_json(manifest_path)
        except ValueError:
            continue
        for item in ("checkpoint", "metrics", "run_manifest"):
            record = manifest.get(item)
            filename = record.get("filename") if isinstance(record, dict) else None
            if isinstance(filename, str) and Path(filename).name == filename:
                (directory / filename).unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)


def _step_from_name(name: str) -> int:
    try:
        return int(name[len(CHECKPOINT_PREFIX):len(CHECKPOINT_PREFIX) + 12])
    except (ValueError, IndexError):
        return -1


def _generation_candidates(directory: Path) -> list[Path]:
    progress_path = directory / PROGRESS_FILENAME
    if not progress_path.is_file():
        if list(directory.glob(f"{CHECKPOINT_PREFIX}*.manifest.json")):
            raise RuntimeError(
                "Drive contains checkpoint generations but its atomic progress record is missing. "
                "Refusing to guess which generation was committed."
            )
        return []

    progress = _read_json(progress_path)
    committed_step = progress.get("current_step", 0)
    if isinstance(committed_step, bool) or not isinstance(committed_step, int) or committed_step < 0:
        raise ValueError("Drive progress record has an invalid current_step.")
    if committed_step == 0:
        return []

    ordered: list[Path] = []
    filename = progress.get("checkpoint_manifest_file")
    if isinstance(filename, str) and Path(filename).name == filename:
        candidate = directory / filename
        if candidate.is_file() and _step_from_name(candidate.name) == committed_step:
            ordered.append(candidate)

    for path in sorted(
        directory.glob(f"{CHECKPOINT_PREFIX}*.manifest.json"),
        key=lambda item: _step_from_name(item.name),
        reverse=True,
    ):
        step = _step_from_name(path.name)
        if step > committed_step:
            continue
        if path not in ordered:
            ordered.append(path)
    return ordered

def _validate_generation(directory: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("format") != "qmini-checkpoint-generation-v1":
        raise ValueError("Unsupported checkpoint generation manifest.")
    for field in ("checkpoint", "metrics", "run_manifest"):
        record = manifest.get(field)
        if not isinstance(record, dict):
            raise ValueError(f"Checkpoint generation has no {field} fingerprint.")
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Checkpoint generation has an unsafe {field} filename.")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint generation file is missing: {filename}")
        actual = _fingerprint(path)
        if actual["size_bytes"] != record.get("size_bytes") or actual["sha256"] != record.get("sha256"):
            raise ValueError(f"Checkpoint generation hash mismatch: {filename}")
    _verify_torch_archive(directory / manifest["checkpoint"]["filename"])
    step = manifest.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("Checkpoint generation has an invalid step.")
    if _step_from_name(manifest_path.name) != step:
        raise ValueError("Checkpoint generation filename and step do not match.")
    return manifest


def restore_latest_checkpoint(
    mirror_dir: str | Path,
    local_run_dir: str | Path,
) -> dict[str, Any] | None:
    """Restore the newest complete verified generation, falling back on corruption."""
    source_dir = Path(mirror_dir).expanduser()
    local_dir = Path(local_run_dir).expanduser()
    if not source_dir.is_dir():
        return None
    errors: list[str] = []
    for manifest_path in _generation_candidates(source_dir):
        try:
            manifest = _validate_generation(source_dir, manifest_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            for field, destination_name in (
                ("checkpoint", "latest.pt"),
                ("metrics", "metrics.jsonl"),
                ("run_manifest", "run.json"),
            ):
                record = manifest[field]
                _atomic_copy(source_dir / record["filename"], local_dir / destination_name)
            restored = {
                "step": manifest["step"],
                "target_steps": manifest["target_steps"],
                "tokens_seen": manifest["tokens_seen"],
                "session_id": manifest["session_id"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "source_manifest": manifest_path.name,
            }
            progress_path = source_dir / PROGRESS_FILENAME
            progress = _read_json(progress_path)
            if restored["step"] < int(progress.get("current_step", 0)):
                progress.update({
                    "status": "checkpointed",
                    "current_step": restored["step"],
                    "target_steps": restored["target_steps"],
                    "progress_percent": round(
                        100.0 * restored["step"] / restored["target_steps"], 4
                    ),
                    "tokens_seen": restored["tokens_seen"],
                    "session_id": restored["session_id"],
                    "checkpoint_file": manifest["checkpoint"]["filename"],
                    "checkpoint_manifest_file": manifest_path.name,
                    "checkpoint_sha256": restored["checkpoint_sha256"],
                    "checkpoint_bytes": manifest["checkpoint"]["size_bytes"],
                    "updated_at": utc_now(),
                    "note": "Recovered previous verified checkpoint after a newer generation failed validation.",
                })
                progress.pop("final_model_sha256", None)
                _atomic_json(progress, progress_path)
            return restored
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{manifest_path.name}: {exc}")
    if errors:
        raise RuntimeError(
            "No complete verified Drive checkpoint generation could be restored. "
            + " | ".join(errors)
        )
    return None


def mark_training_status(
    mirror_dir: str | Path,
    status: str,
    *,
    final_model_sha256: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if status not in {"running", "paused", "completed", "interrupted", "checkpointed"}:
        raise ValueError("Unsupported training status.")
    directory = Path(mirror_dir)
    progress_path = directory / PROGRESS_FILENAME
    if not progress_path.is_file():
        raise FileNotFoundError("No verified checkpoint progress record exists yet.")
    progress = _read_json(progress_path)
    progress["status"] = status
    progress["updated_at"] = utc_now()
    if final_model_sha256 is not None:
        progress["final_model_sha256"] = final_model_sha256
    if note is not None:
        progress["note"] = note
    _atomic_json(progress, progress_path)
    return progress


class TrainingSessionLease:
    """Prevent two Colab sessions from training into the same Drive checkpoint store."""

    def __init__(
        self,
        mirror_dir: str | Path,
        *,
        stale_after_seconds: int = 180,
        heartbeat_seconds: int = 45,
    ):
        if stale_after_seconds < heartbeat_seconds * 2 or heartbeat_seconds < 5:
            raise ValueError("Lease expiry must be at least two heartbeat intervals.")
        self.directory = Path(mirror_dir).expanduser()
        self.path = self.directory / LEASE_FILENAME
        self.stale_after_seconds = stale_after_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.session_id = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "format": "qmini-training-lease-v1",
            "session_id": self.session_id,
            "pid": os.getpid(),
            "heartbeat_at": utc_now(),
        }

    def _write_lease(self, *, exclusive: bool = False) -> None:
        encoded = json.dumps(self._payload(), sort_keys=True).encode("utf-8") + b"\n"
        if exclusive:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            current = _read_json(self.path)
            if current.get("session_id") != self.session_id:
                raise RuntimeError("Training lease ownership changed; stopping this session.")
            temporary = self.path.with_name(f".{self.path.name}.{self.session_id}.tmp")
            try:
                with temporary.open("wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                _fsync_directory(self.directory)
            finally:
                temporary.unlink(missing_ok=True)

    def __enter__(self) -> "TrainingSessionLease":
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._write_lease(exclusive=True)
        except FileExistsError:
            try:
                existing = _read_json(self.path)
                heartbeat = _parse_time(str(existing["heartbeat_at"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Training lease exists but is unreadable: {self.path}. "
                    "Inspect it manually; a second trainer will not be started."
                ) from exc
            age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
            if age <= self.stale_after_seconds:
                raise RuntimeError(
                    "Another Q-Mini training session appears active "
                    f"(last heartbeat {int(age)} seconds ago). Refusing concurrent training."
                )
            self.path.unlink(missing_ok=True)
            self._write_lease(exclusive=True)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="qmini-drive-checkpoint-lease",
            daemon=True,
        )
        self._thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                self._write_lease()
            except (OSError, RuntimeError, ValueError):
                # The last verified checkpoint remains untouched during a Drive outage.
                continue

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.heartbeat_seconds + 2)
        if exc_type is not None:
            try:
                mark_training_status(
                    self.directory,
                    "interrupted",
                    note=f"Training session exited with {exc_type.__name__}; resume from the latest verified checkpoint.",
                )
            except (OSError, ValueError, RuntimeError):
                # A Drive outage must never hide the original training exception.
                pass
        try:
            current = _read_json(self.path)
            if current.get("session_id") == self.session_id:
                self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def touch_training_lease(mirror_dir: str | Path, session_id: str) -> None:
    path = Path(mirror_dir) / LEASE_FILENAME
    if not path.is_file():
        raise RuntimeError(
            "No active Q-Mini training lease exists. Acquire TrainingSessionLease "
            "before publishing a Drive checkpoint."
        )
    current = _read_json(path)
    if current.get("session_id") != session_id:
        raise RuntimeError("Checkpoint publisher does not own the active training lease.")
    payload = {
        "format": "qmini-training-lease-v1",
        "session_id": session_id,
        "pid": current.get("pid"),
        "heartbeat_at": utc_now(),
    }
    _atomic_json(payload, path)
