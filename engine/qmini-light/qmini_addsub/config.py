from __future__ import annotations

import math

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AddSubConfig:
    """Configuration for the independent Q-Mini Add/Sub v1 model.

    Defaults are a compact English-first model. They are not inherited from
    the legacy Q2-v2 configuration and can be changed only for a new run.
    """

    vocab_size: int = 1024
    d_model: int = 96
    d_hidden: int = 170
    n_layers: int = 4
    n_experts: int = 2
    kernel_size: int = 7
    max_context: int = 512
    dropout: float = 0.05
    # Residual branches start small and grow only when the task supports them.
    residual_init: float = 0.1
    # Explicit coefficients keep the router objectives visible in the run config.
    moe_balance_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    # Optional non-gradient MoE balancing bias, updated once after each optimizer step.
    moe_loss_free_bias: bool = False
    moe_bias_update_rate: float = 0.001
    moe_bias_max_abs: float = 2.0
    seed: int = 20261119

    def validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "kernel_size": self.kernel_size,
            "max_context": self.max_context,
        }
        invalid_types = [
            name for name, value in positive.items()
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_types:
            raise TypeError(f"Configuration dimensions must be integers: {invalid_types}")
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}")
        if not 264 <= self.vocab_size <= 65536:
            raise ValueError("vocab_size must be between 264 and 65536 (256 byte symbols + 8 special tokens; uint16 token streams).")
        if self.max_context < 8:
            raise ValueError("max_context must be at least 8 tokens for the chat control format.")
        if self.d_model % 8:
            raise ValueError("d_model must be divisible by 8 for the reference integer path.")
        for name, value in (
            ("dropout", self.dropout),
            ("residual_init", self.residual_init),
            ("moe_balance_coef", self.moe_balance_coef),
            ("router_z_loss_coef", self.router_z_loss_coef),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, not a boolean or string.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 < self.residual_init < 1.0:
            raise ValueError("residual_init must be strictly between 0 and 1.")
        for name, value in (
            ("moe_balance_coef", self.moe_balance_coef),
            ("router_z_loss_coef", self.router_z_loss_coef),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if not isinstance(self.moe_loss_free_bias, bool):
            raise TypeError("moe_loss_free_bias must be a boolean.")
        for name, value in (
            ("moe_bias_update_rate", self.moe_bias_update_rate),
            ("moe_bias_max_abs", self.moe_bias_max_abs),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.n_experts < 2:
            raise ValueError("n_experts must be at least 2 for sparse Top-1 routing.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AddSubConfig":
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a dictionary.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Add/Sub configuration fields: {unknown}")
        config = cls(**values)
        config.validate()
        return config
