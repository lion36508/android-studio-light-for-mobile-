from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qmini_addsub.config import AddSubConfig
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.optim import build_adamw


def _cycle_batch(
    phases: tuple[int, ...],
    *,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    symbols = (260, 261, 262, 263)
    inputs = []
    targets = []
    for phase in phases:
        sequence = [
            symbols[(position + phase) % len(symbols)]
            for position in range(sequence_length + 1)
        ]
        inputs.append(sequence[:-1])
        targets.append(sequence[1:])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def _validation_nll(
    model: AddSubLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.eval()
    with torch.inference_mode():
        result = model(inputs)
        loss = F.cross_entropy(
            result["logits"].float().reshape(-1, model.config.vocab_size),
            targets.reshape(-1),
        )
    return float(loss)


def run_learning_proof(
    *,
    output: Path,
    steps: int = 100,
    minimum_nll_improvement: float = 0.10,
    seed: int = 20261125,
) -> dict[str, Any]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 40:
        raise ValueError("steps must be an integer of at least 40.")
    if (
        isinstance(minimum_nll_improvement, bool)
        or not isinstance(minimum_nll_improvement, (int, float))
        or not math.isfinite(minimum_nll_improvement)
        or minimum_nll_improvement <= 0.0
    ):
        raise ValueError("minimum_nll_improvement must be finite and positive.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = AddSubConfig(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=seed,
    )
    config.validate()
    model = AddSubLanguageModel(config).to(device)
    optimizer = build_adamw(model, learning_rate=1e-2, weight_decay=0.0)

    # Train on two phases and evaluate on the two offset phases of the same
    # deterministic four-symbol next-token rule. No external or synthetic
    # records are added to the production dataset by this diagnostic.
    train_inputs, train_targets = _cycle_batch(
        (0, 2), sequence_length=32, device=device
    )
    heldout_inputs, heldout_targets = _cycle_batch(
        (1, 3), sequence_length=32, device=device
    )
    initial_nll = _validation_nll(model, heldout_inputs, heldout_targets)

    model.train()
    final_training_nll = float("nan")
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result = model(train_inputs)
        token_nll = F.cross_entropy(
            result["logits"].float().reshape(-1, config.vocab_size),
            train_targets.reshape(-1),
        )
        objective = (
            token_nll
            + config.moe_balance_coef * result["aux_loss"]
            + config.router_z_loss_coef * result["router_z_loss"]
        )
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError(f"Non-finite Add/Sub learning objective at step {step}.")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(
                    f"Non-finite gradient in Add/Sub parameter {name!r} at step {step}."
                )
        optimizer.step()
        final_training_nll = float(token_nll.detach())

    final_nll = _validation_nll(model, heldout_inputs, heldout_targets)
    improvement = initial_nll - final_nll
    passed = (
        math.isfinite(initial_nll)
        and math.isfinite(final_nll)
        and math.isfinite(final_training_nll)
        and improvement >= minimum_nll_improvement
    )
    report = {
        "format": "qmini-addsub-learning-proof-v1",
        "model": "qmini_addsub",
        "parameters": model.parameter_count(),
        "config": config.to_dict(),
        "device": str(device),
        "seed": seed,
        "optimizer": "AdamW",
        "optimizer_steps": steps,
        "train_batch_phases": [0, 2],
        "heldout_batch_phases": [1, 3],
        "task": "four-symbol periodic next-token prediction",
        "initial_heldout_token_nll": initial_nll,
        "final_heldout_token_nll": final_nll,
        "final_training_token_nll": final_training_nll,
        "heldout_token_nll_improvement": improvement,
        "minimum_required_improvement": float(minimum_nll_improvement),
        "all_metrics_finite": all(
            math.isfinite(value)
            for value in (initial_nll, final_nll, final_training_nll, improvement)
        ),
        "passed": passed,
        "limitations": [
            "This controlled periodic-token experiment checks that the Add/Sub training graph can optimize from random initialization.",
            "It is not an English-language, factuality, reasoning, safety or general assistant-quality benchmark.",
            "The model still requires licensed real-English pretraining, human conversation fine-tuning and independent held-out evaluation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(
            "Add/Sub learning proof failed: held-out token NLL improved by "
            f"{improvement:.4f}; minimum required improvement is "
            f"{minimum_nll_improvement:.4f}. Report: {output}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that random-initialized Add/Sub can learn a controlled next-token task."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--minimum-nll-improvement", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    report = run_learning_proof(
        output=args.output,
        steps=args.steps,
        minimum_nll_improvement=args.minimum_nll_improvement,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
