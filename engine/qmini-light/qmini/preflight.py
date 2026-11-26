from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import torch

from .config import QMiniConfig


DEFAULT_TRAINING_MEMORY_SAFETY_FACTOR = 2.0


def available_memory_info(device: torch.device) -> dict[str, Any]:
    """Return available accelerator/host memory when it can be measured safely."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "available_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "source": "torch.cuda.mem_get_info",
        }

    if platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    available_kib = int(line.split()[1])
                    total_kib = next(
                        int(item.split()[1])
                        for item in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
                        if item.startswith("MemTotal:")
                    )
                    return {
                        "available_bytes": available_kib * 1024,
                        "total_bytes": total_kib * 1024,
                        "source": "/proc/meminfo:MemAvailable",
                    }
        except (OSError, StopIteration, ValueError, IndexError):
            pass

    return {
        "available_bytes": None,
        "total_bytes": None,
        "source": "unavailable",
    }


def training_memory_floor_bytes(
    config: QMiniConfig,
    safety_factor: float = DEFAULT_TRAINING_MEMORY_SAFETY_FACTOR,
) -> int:
    """Return a conservative free-memory floor from the FP32 AdamW state lower bound."""
    config.validate()
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or safety_factor < 1.0
    ):
        raise ValueError("safety_factor must be a finite number >= 1.")
    return math.ceil(
        config.estimated_adamw_training_state_bytes() * float(safety_factor)
    )


def check_training_memory_budget(
    config: QMiniConfig,
    device: torch.device,
    safety_factor: float = DEFAULT_TRAINING_MEMORY_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Fail before allocation if available memory is below a conservative state floor.

    The floor is a multiple of FP32 parameters + gradients + two AdamW moments.
    It is deliberately only a lower-bound guard: activations, workspaces,
    fragmentation, checkpoint staging and runtime overhead remain workload-specific.
    """
    config.validate()
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or safety_factor < 1.0
    ):
        raise ValueError("safety_factor must be a finite number >= 1.")

    state_floor = config.estimated_adamw_training_state_bytes()
    required = training_memory_floor_bytes(config, safety_factor)
    measured = available_memory_info(device)
    available = measured["available_bytes"]
    report = {
        "device": str(device),
        "parameters": config.estimated_parameter_count(),
        "fp32_params_grads_adamw_lower_bound_bytes": state_floor,
        "safety_factor": float(safety_factor),
        "required_free_memory_bytes": required,
        "available_memory_bytes": available,
        "total_memory_bytes": measured["total_bytes"],
        "memory_source": measured["source"],
        "checked": available is not None,
        "passed": available is None or int(available) >= required,
        "warning": (
            "Lower-bound preflight only; activations, workspaces, fragmentation, "
            "checkpoint staging and runtime overhead are not fully predicted."
        ),
    }
    if available is not None and int(available) < required:
        raise MemoryError(
            f"Training refused before model allocation: {int(available):,} bytes are available "
            f"on {device}, but the conservative lower-bound floor is {required:,} bytes "
            f"({safety_factor:.1f}x FP32 parameters/gradients/AdamW state). "
            "Use a larger runtime or a smaller Q-Mini configuration."
        )
    return report


def run_architecture_preflight(
    config: QMiniConfig,
    *,
    device: str | torch.device = "auto",
    sequence_length: int = 8,
    max_parameters: int = 250_000_000,
    check_backward: bool = True,
) -> dict[str, Any]:
    """Exercise a random-initialized Q-Mini graph using synthetic IDs, without training.

    No dataset is opened, no external checkpoint is loaded, and no optimizer step
    occurs. The optional backward pass verifies that the current graph produces
    finite gradients while leaving all model weights unchanged.
    """
    config.validate()
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer.")
    if not 2 <= sequence_length <= config.max_context:
        raise ValueError(
            f"sequence_length must be between 2 and {config.max_context}."
        )
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, int) or max_parameters < 1:
        raise ValueError("max_parameters must be a positive integer.")
    if not isinstance(check_backward, bool):
        raise TypeError("check_backward must be a boolean.")

    parameter_count = config.estimated_parameter_count()
    if parameter_count > max_parameters:
        raise MemoryError(
            f"Architecture preflight refused to materialize {parameter_count:,} parameters; "
            f"the configured safety limit is {max_parameters:,}. "
            "Increase --max-parameters only after checking the available hardware."
        )

    if isinstance(device, str) and device == "auto":
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this runtime has no available GPU.")

    memory_report = check_training_memory_budget(config, target)
    from .model import QMiniMoE

    cuda_devices = (
        [target.index if target.index is not None else torch.cuda.current_device()]
        if target.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats(target)

        model = QMiniMoE(config).to(target)
        actual_parameters = model.parameter_report()["total"]
        if actual_parameters != parameter_count:
            raise RuntimeError(
                f"Model shape audit mismatch: config estimates {parameter_count:,} "
                f"parameters, but the materialized graph contains {actual_parameters:,}."
            )

        model.train()
        input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(1, sequence_length),
            dtype=torch.long,
            device=target,
        )
        result = model(input_ids)
        logits = result.get("logits")
        expected_shape = (1, sequence_length, config.vocab_size)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Output shape audit failed: expected {expected_shape}, "
                f"got {None if not isinstance(logits, torch.Tensor) else tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all() or not torch.isfinite(result["aux_loss"]):
            raise FloatingPointError("Architecture preflight produced non-finite forward values.")

        gradient_count = 0
        nonzero_gradient_count = 0
        if check_backward:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, config.vocab_size),
                input_ids[:, 1:].reshape(-1),
            ) + config.router_aux_loss_coef * result["aux_loss"].float()
            if not torch.isfinite(loss):
                raise FloatingPointError("Architecture preflight produced a non-finite loss.")
            loss.backward()
            for parameter in model.parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                gradient_count += 1
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Architecture preflight found a non-finite gradient.")
                if torch.count_nonzero(gradient).item():
                    nonzero_gradient_count += 1
            if gradient_count == 0 or nonzero_gradient_count == 0:
                raise RuntimeError("Architecture preflight did not produce usable gradients.")
            model.zero_grad(set_to_none=True)

        report = {
            "format": "qmini-architecture-preflight-v1",
            "config": config.to_dict(),
            "device": str(target),
            "parameters_expected": parameter_count,
            "parameters_materialized": actual_parameters,
            "input_shape": [1, sequence_length],
            "logits_shape": list(logits.shape),
            "forward_finite": True,
            "backward_checked": check_backward,
            "gradient_tensors_checked": gradient_count,
            "gradient_tensors_nonzero": nonzero_gradient_count,
            "random_initialization_only": True,
            "pretrained_weights_loaded": False,
            "dataset_opened": False,
            "optimizer_step_performed": False,
            "weights_updated": False,
            "memory_preflight": memory_report,
        }
        if target.type == "cuda":
            report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(target))
        del result, logits, input_ids, model
        if target.type == "cuda":
            torch.cuda.empty_cache()
        return report
