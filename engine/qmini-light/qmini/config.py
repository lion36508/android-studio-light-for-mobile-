from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class QMiniConfig:
    vocab_size: int = 4096
    d_model: int = 256
    d_hidden: int = 384
    n_layers: int = 6
    n_experts: int = 4
    top_k: int = 1
    max_context: int = 512
    kernel_size: int = 10
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    max_thinking_steps: int = 4
    thinking_steps: int = 2
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    router_aux_loss_coef: float = 0.01
    use_state_space: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42

    def validate(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "max_context": self.max_context,
            "kernel_size": self.kernel_size,
            "max_thinking_steps": self.max_thinking_steps,
            "thinking_steps": self.thinking_steps,
        }
        invalid_types = [
            name for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_types:
            raise TypeError("These Q-Mini settings must be integers: " + ", ".join(invalid_types))

        non_positive = [
            name for name, value in integer_fields.items() if value <= 0
        ]
        if non_positive:
            raise ValueError("Must be positive: " + ", ".join(non_positive))
        if self.max_context < 2:
            raise ValueError("max_context must be at least 2 tokens.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not isinstance(self.use_state_space, bool):
            raise TypeError("use_state_space must be a boolean.")
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")

        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k != 1:
            raise ValueError("Q-Mini currently implements sparse Top-1 routing only.")
        if not isinstance(self.dilations, (tuple, list)):
            raise TypeError("dilations must be a tuple or list of integers.")
        if (
            len(self.dilations) != self.n_layers
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.dilations
            )
        ):
            raise ValueError("dilations must contain one positive integer per main block.")
        if self.thinking_steps > self.max_thinking_steps:
            raise ValueError("thinking_steps cannot exceed max_thinking_steps.")

        numeric_fields = {
            "dropout": self.dropout,
            "rms_norm_eps": self.rms_norm_eps,
            "router_aux_loss_coef": self.router_aux_loss_coef,
        }
        for name, value in numeric_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.rms_norm_eps <= 0 or self.router_aux_loss_coef < 0:
            raise ValueError("Invalid RMSNorm epsilon or router auxiliary loss coefficient.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "QMiniConfig":
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a dictionary.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("Unknown Q-Mini config keys: " + ", ".join(unknown))
        clean = dict(values)
        if "dilations" in clean:
            clean["dilations"] = tuple(clean["dilations"]) if isinstance(clean["dilations"], (list, tuple)) else clean["dilations"]
        config = cls(**clean)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "QMiniConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object.")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return target

    def _block_parameter_counts(self) -> tuple[int, int]:
        """Return (full block parameters, one expert's parameters)."""
        d, h, experts = self.d_model, self.d_hidden, self.n_experts
        router_parameters = d * experts + experts if experts > 1 else 0
        router_norm_parameters = d if experts > 1 else 0
        shared = (
            3 * d * d + 6 * d + d * self.kernel_size
            + router_norm_parameters + router_parameters
        )
        if self.use_state_space:
            # Four QLinear projections plus the state-space RMSNorm.
            shared += 4 * d * d + 5 * d
        one_expert = 2 * d * h + h + d
        return shared + experts * one_expert, one_expert

    def estimated_parameter_count(self) -> int:
        """Exact model-layout count without allocating model tensors."""
        self.validate()
        block_total, _ = self._block_parameter_counts()
        return self.vocab_size * self.d_model + (self.n_layers + 1) * block_total + self.d_model

    def estimated_unique_active_parameter_count(self) -> int:
        """Unique parameters touched per token under the current Top-1 routing policy."""
        self.validate()
        block_total, one_expert = self._block_parameter_counts()
        shared = block_total - self.n_experts * one_expert
        return (
            self.vocab_size * self.d_model
            + self.d_model
            + (self.n_layers + 1) * (shared + one_expert)
        )

    def estimated_adamw_training_state_bytes(self) -> int:
        """FP32 parameters, gradients and two FP32 Adam moments; excludes activations."""
        return self.estimated_parameter_count() * 16

    def reference_tokens_at_20_per_parameter(self) -> int:
        """Dense-LM planning heuristic only; not a guarantee or sparse-MoE scaling law."""
        return self.estimated_parameter_count() * 20

    def reference_tokens_at_20_per_active_parameter(self) -> int:
        """Compute-budget heuristic using active Top-1 parameters; not an MoE scaling law."""
        return self.estimated_unique_active_parameter_count() * 20

    def estimated_packed_export_bytes(self, metadata_reserve_bytes: int = 65536) -> int:
        """Conservative raw size of the Q2+FP16 deployment archive.

        All rank-2/3 projection weights use four signed quaternary values per
        byte plus one FP32 scale per output row. One-dimensional vectors use
        FP16. Row padding is counted exactly; the metadata reserve covers the
        JSON manifest and ZIP container overhead.
        """
        self.validate()
        if (
            isinstance(metadata_reserve_bytes, bool)
            or not isinstance(metadata_reserve_bytes, int)
            or metadata_reserve_bytes < 0
        ):
            raise ValueError("metadata_reserve_bytes must be a non-negative integer.")

        d, h, experts = self.d_model, self.d_hidden, self.n_experts
        blocks = self.n_layers + 1
        vocab = self.vocab_size
        ceil4 = lambda value: (value + 3) // 4

        total = self.estimated_parameter_count()
        state_projection_factor = 4 if self.use_state_space else 0
        shared_projection_factor = 3 + state_projection_factor
        matrix_parameters = (
            vocab * d
            + blocks * (
                shared_projection_factor * d * d
                + d * self.kernel_size
                + (d * experts if experts > 1 else 0)
                + 2 * experts * d * h
            )
        )
        vector_parameters = total - matrix_parameters
        if vector_parameters < 0:
            raise RuntimeError("Packed layout estimate produced a negative vector count.")

        packed_weight_bytes = (
            vocab * ceil4(d)
            + blocks * (
                shared_projection_factor * d * ceil4(d)
                + d * ceil4(self.kernel_size)
                + (experts * ceil4(d) if experts > 1 else 0)
                + experts * (
                    h * ceil4(d)
                    + d * ceil4(h)
                )
            )
        )
        # Depthwise convolution is a separate Q2 matrix with one row
        # per channel, in addition to the dense projection rows.
        q2_row_count = vocab + blocks * (
            (shared_projection_factor + 1) * d
            + experts * (h + d)
            + (experts if experts > 1 else 0)
        )
        row_scale_bytes = q2_row_count * 4
        fp16_vector_bytes = vector_parameters * 2
        return (
            packed_weight_bytes
            + row_scale_bytes
            + fp16_vector_bytes
            + metadata_reserve_bytes
        )

    def estimated_streaming_state_bytes(
        self,
        batch_size: int = 1,
        thinking_steps: int | None = None,
    ) -> int:
        """Estimate persistent CPU streaming state for the requested batch.

        Uses FP32 convolution/SSM caches. This counts model state only, not
        Python/PyTorch allocator overhead.
        """
        self.validate()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        steps = self.max_thinking_steps if thinking_steps is None else thinking_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= self.max_thinking_steps:
            raise ValueError("thinking_steps is outside the configured range.")

        convolution_values = (
            sum(self.dilations) + steps
        ) * (self.kernel_size - 1) * self.d_model
        convolution_bytes = convolution_values * 4 * batch_size
        state_space_layers = self.n_layers + steps if self.use_state_space else 0
        state_space_bytes = state_space_layers * self.d_model * 4 * batch_size
        return convolution_bytes + state_space_bytes

    def estimated_cpu_inference_working_set_bytes(
        self,
        batch_size: int = 1,
        thinking_steps: int | None = None,
    ) -> int:
        """Estimate packed weights + stream caches + largest FP32 fallback buffers.

        This is a model-level working-set estimate, not a measured process RSS.
        It excludes Python/PyTorch runtime baseline, allocator fragmentation,
        OS overhead, tokenizer memory and transient NPZ decompression overlap.
        """
        self.validate()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        activation_width = 3 * self.d_model + self.d_hidden + self.vocab_size
        activation_bytes = activation_width * 4 * batch_size
        fallback_matrix_bytes = self.d_model * self.d_hidden * 4
        return (
            self.estimated_packed_export_bytes()
            + self.estimated_streaming_state_bytes(batch_size, thinking_steps)
            + activation_bytes
            + fallback_matrix_bytes
        )

    def main_receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)
