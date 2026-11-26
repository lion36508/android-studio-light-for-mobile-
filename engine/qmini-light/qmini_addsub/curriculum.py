from __future__ import annotations

import math
import random
from typing import Any

import torch


class SelfPacedCurriculumSampler:
    """Loss-aware curriculum with a permanent uniform replay floor.

    Examples are ranked by their EMA response loss, but only after their
    difficulty has actually been measured. Sampling gradually moves from the
    middle of the observed difficulty distribution toward moderately hard
    examples. A fractional uniform quota is carried across steps, so even
    batch-size-one training preserves the requested long-run replay share.
    This is an SFT data sampler; it does not change model weights or fabricate
    training examples.
    """

    FORMAT = "qmini-self-paced-curriculum-v2"
    UNIFORM_QUOTA_SCALE = 1_000_000_000

    def __init__(
        self,
        dataset_size: int,
        total_steps: int,
        warmup_steps: int = 50,
        uniform_fraction: float = 0.35,
        start_quantile: float = 0.55,
        end_quantile: float = 0.80,
        bandwidth: float = 0.18,
        ema_decay: float = 0.80,
    ) -> None:
        for name, value in (("dataset_size", dataset_size), ("total_steps", total_steps), ("warmup_steps", warmup_steps)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if dataset_size < 1 or total_steps < 1 or warmup_steps < 1:
            raise ValueError("dataset_size, total_steps and warmup_steps must be positive.")
        for name, value in (("uniform_fraction", uniform_fraction), ("start_quantile", start_quantile), ("end_quantile", end_quantile), ("bandwidth", bandwidth), ("ema_decay", ema_decay)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite number.")
        if not 0.35 <= uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be between 0.35 and 1.0.")
        if not 0.0 < start_quantile <= end_quantile < 1.0:
            raise ValueError("difficulty quantiles must satisfy 0 < start <= end < 1.")
        if bandwidth <= 0.0 or not 0.0 < ema_decay < 1.0:
            raise ValueError("bandwidth must be positive and ema_decay must be in (0, 1).")
        self.dataset_size = dataset_size
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.uniform_fraction = float(uniform_fraction)
        self.start_quantile = float(start_quantile)
        self.end_quantile = float(end_quantile)
        self.bandwidth = float(bandwidth)
        self.ema_decay = float(ema_decay)
        self.step = 0
        self.ema_loss: list[float | None] = [None] * dataset_size
        self.update_count = [0] * dataset_size
        self.last_focus_quantile = self.start_quantile
        self.uniform_fraction_units = int(round(self.uniform_fraction * self.UNIFORM_QUOTA_SCALE))
        self.uniform_remainder_units = 0

    def _focus_quantile(self) -> float:
        progress = (self.step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.start_quantile + (self.end_quantile - self.start_quantile) * progress

    def sample(self, batch_size: int, rng: random.Random) -> tuple[list[int], dict[str, Any]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if batch_size > self.dataset_size:
            raise ValueError("batch_size cannot exceed the number of SFT records.")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random for exact resume.")
        self.step += 1
        if self.uniform_fraction >= 1.0:
            # Keep the default baseline O(batch_size), not O(dataset_size), and
            # preserve the exact rng.sample call used by legacy SFT v6.
            self.last_focus_quantile = self.start_quantile
            selected = rng.sample(range(self.dataset_size), k=batch_size)
            return selected, {
                "mode": "uniform_control",
                "focus_quantile": self.last_focus_quantile,
                "uniform_fraction": 1.0,
                "requested_uniform_fraction": self.uniform_fraction,
                "uniform_samples": batch_size,
                "focused_samples": 0,
                "seen_fraction": 0.0,
            }
        seen = [index for index, value in enumerate(self.ema_loss) if value is not None]

        def uniform_batch(mode: str, focus_quantile: float | None = None) -> tuple[list[int], dict[str, Any]]:
            self.last_focus_quantile = self.start_quantile if focus_quantile is None else focus_quantile
            selected = rng.sample(range(self.dataset_size), k=batch_size)
            return selected, {
                "mode": mode,
                "focus_quantile": self.last_focus_quantile,
                "uniform_fraction": 1.0,
                "requested_uniform_fraction": self.uniform_fraction,
                "uniform_samples": batch_size,
                "focused_samples": 0,
                "seen_fraction": len(seen) / self.dataset_size,
            }

        if self.step <= self.warmup_steps:
            return uniform_batch("uniform_warmup")

        # Carry fractional replay quota across steps. This preserves the requested
        # long-run fraction even for batch_size=1, without stochastic rounding drift.
        replay_budget_units = (
            self.uniform_remainder_units
            + batch_size * self.uniform_fraction_units
        )
        uniform_count = min(
            batch_size,
            replay_budget_units // self.UNIFORM_QUOTA_SCALE,
        )
        self.uniform_remainder_units = (
            replay_budget_units - uniform_count * self.UNIFORM_QUOTA_SCALE
        )
        uniform_indices = rng.sample(range(self.dataset_size), k=uniform_count)
        selected_set = set(uniform_indices)
        focused_count = batch_size - uniform_count

        if focused_count == 0:
            self.last_focus_quantile = self._focus_quantile()
            return uniform_indices, {
                "mode": "self_paced_replay_only",
                "focus_quantile": self.last_focus_quantile,
                "uniform_fraction": uniform_count / batch_size,
                "requested_uniform_fraction": self.uniform_fraction,
                "uniform_samples": uniform_count,
                "focused_samples": 0,
                "seen_fraction": len(seen) / self.dataset_size,
            }

        # Never assign an artificial median difficulty to examples not evaluated yet.
        # Focus only on records with real EMA response-loss measurements.
        candidates = [index for index in seen if index not in selected_set]
        if len(candidates) < focused_count:
            remaining = [index for index in range(self.dataset_size) if index not in selected_set]
            fill = rng.sample(remaining, k=focused_count)
            selected = uniform_indices + fill
            self.last_focus_quantile = self.start_quantile
            return selected, {
                "mode": "uniform_fallback",
                "focus_quantile": self.last_focus_quantile,
                "uniform_fraction": 1.0,
                "requested_uniform_fraction": self.uniform_fraction,
                "uniform_samples": batch_size,
                "focused_samples": 0,
                "seen_fraction": len(seen) / self.dataset_size,
            }

        ordered = sorted(candidates, key=lambda index: self.ema_loss[index])
        ranks: dict[int, float] = {}
        denominator = max(1, len(ordered) - 1)
        cursor = 0
        # Give ties their average rank so record order is not a fake difficulty signal.
        while cursor < len(ordered):
            end = cursor + 1
            current_loss = self.ema_loss[ordered[cursor]]
            while end < len(ordered) and self.ema_loss[ordered[end]] == current_loss:
                end += 1
            average_rank = ((cursor + end - 1) / 2.0) / denominator
            for position in range(cursor, end):
                ranks[ordered[position]] = average_rank
            cursor = end

        target = self._focus_quantile()
        self.last_focus_quantile = target
        scores = [
            math.exp(-0.5 * ((ranks[index] - target) / self.bandwidth) ** 2)
            for index in ordered
        ]
        score_total = sum(scores)
        if not math.isfinite(score_total) or score_total <= 0.0:
            remaining = [index for index in range(self.dataset_size) if index not in selected_set]
            fill = rng.sample(remaining, k=focused_count)
            selected = uniform_indices + fill
            return selected, {
                "mode": "uniform_fallback",
                "focus_quantile": target,
                "uniform_fraction": 1.0,
                "requested_uniform_fraction": self.uniform_fraction,
                "uniform_samples": batch_size,
                "focused_samples": 0,
                "seen_fraction": len(seen) / self.dataset_size,
            }

        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng.randrange(0, 2**63 - 1))
        weights = torch.tensor(scores, dtype=torch.float64)
        chosen_positions = torch.multinomial(weights, focused_count, replacement=False, generator=generator)
        focused_indices = [ordered[int(position)] for position in chosen_positions.tolist()]
        selected = uniform_indices + focused_indices
        return selected, {
            "mode": "self_paced",
            "focus_quantile": target,
            "uniform_fraction": uniform_count / batch_size,
            "requested_uniform_fraction": self.uniform_fraction,
            "uniform_samples": uniform_count,
            "focused_samples": focused_count,
            "seen_fraction": len(seen) / self.dataset_size,
        }
    def update(self, indices: list[int], losses: list[float]) -> None:
        if len(indices) != len(losses):
            raise ValueError("indices and losses must have the same length.")
        if len(set(indices)) != len(indices):
            raise ValueError("a sampled batch must not contain duplicate record indices.")
        for index, loss in zip(indices, losses):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.dataset_size:
                raise ValueError("curriculum record index is out of range.")
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0.0:
                raise ValueError("per-record loss must be finite and non-negative.")
            previous = self.ema_loss[index]
            self.ema_loss[index] = float(loss) if previous is None else (
                self.ema_decay * previous + (1.0 - self.ema_decay) * float(loss)
            )
            self.update_count[index] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "dataset_size": self.dataset_size,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "uniform_fraction": self.uniform_fraction,
            "start_quantile": self.start_quantile,
            "end_quantile": self.end_quantile,
            "bandwidth": self.bandwidth,
            "ema_decay": self.ema_decay,
            "step": self.step,
            "ema_loss": self.ema_loss,
            "update_count": self.update_count,
            "last_focus_quantile": self.last_focus_quantile,
            "uniform_remainder_units": self.uniform_remainder_units,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("format") != self.FORMAT:
            raise ValueError("Unsupported curriculum state format.")
        expected = self.state_dict()
        for key in ("dataset_size", "warmup_steps", "uniform_fraction", "start_quantile", "end_quantile", "bandwidth", "ema_decay"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Curriculum resume contract mismatch: {key}.")
        saved_total_steps = state.get("total_steps")
        if (
            isinstance(saved_total_steps, bool)
            or not isinstance(saved_total_steps, int)
            or saved_total_steps < 1
            or saved_total_steps > self.total_steps
        ):
            raise ValueError("Curriculum schedule may only be extended, never shortened.")
        ema_loss = state.get("ema_loss")
        update_count = state.get("update_count")
        if not isinstance(ema_loss, list) or len(ema_loss) != self.dataset_size:
            raise ValueError("Curriculum EMA state has an invalid length.")
        if not isinstance(update_count, list) or len(update_count) != self.dataset_size:
            raise ValueError("Curriculum update-count state has an invalid length.")
        for value in ema_loss:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0):
                raise ValueError("Curriculum EMA state contains an invalid loss.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in update_count):
            raise ValueError("Curriculum update counts must be non-negative integers.")
        if any((loss is None) != (count == 0) for loss, count in zip(ema_loss, update_count)):
            raise ValueError("Curriculum EMA losses and update counts are inconsistent.")
        step = state.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step > saved_total_steps
        ):
            raise ValueError("Curriculum step is outside its saved schedule horizon.")
        self.step = step
        self.ema_loss = [None if value is None else float(value) for value in ema_loss]
        self.update_count = list(update_count)
        last_focus_quantile = state.get("last_focus_quantile", self.start_quantile)
        if (
            isinstance(last_focus_quantile, bool)
            or not isinstance(last_focus_quantile, (int, float))
            or not math.isfinite(last_focus_quantile)
            or not 0.0 < last_focus_quantile < 1.0
        ):
            raise ValueError("Curriculum checkpoint contains an invalid focus quantile.")
        self.last_focus_quantile = float(last_focus_quantile)
        uniform_remainder_units = state.get("uniform_remainder_units")
        if (
            isinstance(uniform_remainder_units, bool)
            or not isinstance(uniform_remainder_units, int)
            or not 0 <= uniform_remainder_units < self.UNIFORM_QUOTA_SCALE
        ):
            raise ValueError("Curriculum checkpoint contains an invalid fixed-point replay remainder.")
        self.uniform_remainder_units = uniform_remainder_units

    @property
    def uniform_remainder(self) -> float:
        """Fractional replay quota carried into the next optimizer step."""
        return self.uniform_remainder_units / self.UNIFORM_QUOTA_SCALE
