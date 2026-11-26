from __future__ import annotations

import re
from pathlib import Path


def _step_from_name(path: Path, prefix: str) -> int:
    match = re.fullmatch(re.escape(prefix) + r"-step-(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"malformed checkpoint filename: {path.name}")
    step = int(match.group(1))
    if step < 1:
        raise ValueError(f"checkpoint step must be positive: {path.name}")
    return step


def latest_checkpoint(
    folder: str | Path, prefix: str
) -> tuple[Path | None, int | None]:
    """Return the checkpoint with the greatest numeric step, never lexicographic order."""
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("checkpoint prefix must be a non-empty filename component")
    directory = Path(folder)
    candidates = list(directory.glob(f"{prefix}-step-*.pt"))
    if not candidates:
        return None, None

    by_step: dict[int, Path] = {}
    for path in candidates:
        if not path.is_file():
            raise ValueError(f"checkpoint candidate is not a regular file: {path}")
        step = _step_from_name(path, prefix)
        if step in by_step:
            raise ValueError(f"duplicate numeric checkpoint step {step}: {by_step[step]} and {path}")
        by_step[step] = path

    step = max(by_step)
    return by_step[step], step


def resolve_checkpoint(
    folder: str | Path,
    prefix: str,
    metrics_filename: str,
    explicit: str | Path | None = None,
) -> tuple[Path | None, int | None]:
    """Select an explicit or newest checkpoint and reject unsafe resume states."""
    directory = Path(folder)
    latest_path, latest_step = latest_checkpoint(directory, prefix)

    if explicit is not None:
        selected = Path(explicit)
        if not selected.is_file():
            raise FileNotFoundError(f"requested checkpoint does not exist: {selected}")
        selected_step = _step_from_name(selected, prefix)
        if latest_step is not None and latest_step > selected_step:
            raise ValueError(
                f"newer checkpoint step {latest_step} exists; refusing to resume from older "
                f"step {selected_step}"
            )
        return selected, selected_step

    if latest_path is None and (directory / metrics_filename).is_file():
        raise FileNotFoundError(
            f"metrics file exists but no checkpoint matching {prefix}-step-*.pt was found "
            f"in {directory}; refusing to start over a potentially incomplete run"
        )
    return latest_path, latest_step
