from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qmini.config import QMiniConfig
from qmini.model import QMiniMoE
from qmini.train import _evaluate, _load_tokens, train_model


def _write_alternating_stream(path: Path, start: int, pairs: int) -> None:
    values = np.resize(
        np.asarray([start, 37 if start == 11 else 11], dtype=np.uint16),
        pairs * 2,
    )
    values.tofile(path)


def run_learning_proof(
    config_path: Path,
    output_dir: Path,
    *,
    steps: int = 80,
    minimum_nll_improvement: float = 0.15,
) -> dict[str, Any]:
    if steps < 20:
        raise ValueError("The learning proof requires at least 20 optimizer steps.")
    if not np.isfinite(minimum_nll_improvement) or minimum_nll_improvement <= 0:
        raise ValueError("minimum_nll_improvement must be finite and positive.")

    config = QMiniConfig.from_json(config_path)
    values = config.to_dict()
    values.update({
        "use_state_space": True,
        "gradient_checkpointing": True,
        "dropout": 0.0,
        "max_thinking_steps": 2,
        "thinking_steps": 1,
        "seed": 20261111,
    })
    config = QMiniConfig.from_dict(values)

    seed = config.seed
    device = torch.device("cpu")
    torch.set_num_threads(1)

    with tempfile.TemporaryDirectory(prefix="qmini-learning-proof-") as temp:
        root = Path(temp)
        train_path = root / "train.bin"
        valid_path = root / "valid.bin"
        run_path = root / "run"
        _write_alternating_stream(train_path, 11, 1024)
        _write_alternating_stream(valid_path, 37, 128)

        train_tokens = _load_tokens(train_path, "Training")
        valid_tokens = _load_tokens(valid_path, "Validation")

        # Recreate the exact same random initialization used by train_model.
        torch.manual_seed(seed)
        baseline_model = QMiniMoE(config).to(device)
        baseline = {
            str(thinking_steps): _evaluate(
                baseline_model,
                valid_tokens,
                None,
                batch_size=4,
                seq_len=16,
                batches=16,
                device=device,
                seed=99173,
                thinking_steps=thinking_steps,
            )["eval_loss"]
            for thinking_steps in (1, 2)
        }
        del baseline_model

        train_model(
            config,
            train_path,
            valid_path,
            run_path,
            steps=steps,
            batch_size=4,
            seq_len=16,
            grad_accum=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            eval_every=steps,
            save_every=steps,
            eval_batches=8,
            seed=seed,
        )

        checkpoint = torch.load(
            run_path / "latest.pt",
            map_location="cpu",
            weights_only=True,
        )
        trained_model = QMiniMoE(config).to(device)
        trained_model.load_state_dict(checkpoint["model"], strict=True)
        trained_model.eval()
        after = {
            str(thinking_steps): _evaluate(
                trained_model,
                valid_tokens,
                None,
                batch_size=4,
                seq_len=16,
                batches=16,
                device=device,
                seed=99173,
                thinking_steps=thinking_steps,
            )["eval_loss"]
            for thinking_steps in (1, 2)
        }

        improvements = {
            key: float(baseline[key] - after[key]) for key in baseline
        }
        mean_improvement = sum(improvements.values()) / len(improvements)
        finite = all(
            np.isfinite(value)
            for value in [*baseline.values(), *after.values(), *improvements.values()]
        )
        report = {
            "format": "qmini-learning-proof-v1",
            "config": config.to_dict(),
            "parameters": config.estimated_parameter_count(),
            "seed": seed,
            "device": str(device),
            "optimizer_steps": int(checkpoint["step"]),
            "tokens_seen": int(checkpoint["tokens_seen"]),
            "training_stream": "deterministic alternating-token sequence",
            "heldout_stream": "same transition rule with opposite starting phase",
            "state_space_enabled": config.use_state_space,
            "gradient_checkpointing": config.gradient_checkpointing,
            "initial_validation_nll_by_thinking_steps": baseline,
            "trained_validation_nll_by_thinking_steps": after,
            "validation_nll_improvement_by_thinking_steps": improvements,
            "mean_validation_nll_improvement": mean_improvement,
            "minimum_required_mean_nll_improvement": minimum_nll_improvement,
            "all_metrics_finite": finite,
            "passed": bool(
                finite
                and checkpoint["step"] == steps
                and mean_improvement >= minimum_nll_improvement
            ),
            "limitations": [
                "This is a controlled algorithmic learnability test, not a natural-language benchmark.",
                "Passing does not demonstrate factuality, reasoning, multilingual ability, safety or general assistant quality.",
                "Real capability requires licensed representative data, a long training run and independent held-out evaluations.",
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "learning-proof.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not report["passed"]:
            raise RuntimeError(
                "Q-Mini learning proof failed: mean held-out NLL improvement "
                f"{mean_improvement:.4f} is below {minimum_nll_improvement:.4f}. "
                f"See {report_path}."
            )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove that random-initialized Q-Mini can learn a controlled next-token task."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/qmini_smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--minimum-nll-improvement", type=float, default=0.15)
    args = parser.parse_args()

    report = run_learning_proof(
        args.config,
        args.output_dir,
        steps=args.steps,
        minimum_nll_improvement=args.minimum_nll_improvement,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
