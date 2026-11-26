from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import AddSubConfig
from .model import AddSubLanguageModel
from .data import escape_special_token_literals
from .optim import (
    assert_finite_training_state, build_adamw, cosine_warmup_lr,
    reconcile_metrics_file, resolve_schedule_total_steps,
)
from .wavebit_hsra import validate_wavebit_options


CHECKPOINT_FORMAT = "qmini-addsub-training-v5"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_text_file(
    text_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> tuple[int, str, int]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain the <eos> token.")
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("Tokenizer vocabulary exceeds the uint16 token-stream capacity.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    source_byte_count = 0
    try:
        with Path(text_path).open("r", encoding="utf-8") as source, temporary.open("wb") as target:
            for line in source:
                text = line.strip()
                if not text:
                    continue
                safe_text = escape_special_token_literals(text)
                source_byte_count += len(safe_text.encode('utf-8'))
                ids = tokenizer.encode(safe_text, add_special_tokens=False).ids
                if not ids:
                    continue
                if any(token_id < 0 or token_id >= 65536 for token_id in ids):
                    raise ValueError("Tokenizer emitted a token ID outside uint16 range.")
                values = np.asarray(ids + [eos_id], dtype=np.uint16)
                values.tofile(target)
                count += int(values.size)
        if count < 2:
            raise ValueError("Encoded token stream is empty or too short.")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count, sha256_file(output), source_byte_count


def _sample_batch(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    last_start = len(tokens) - sequence_length - 1
    if last_start < 0:
        raise ValueError("Token stream is shorter than one training sequence.")
    starts = [rng.randrange(last_start + 1) for _ in range(batch_size)]
    inputs = np.stack([tokens[start : start + sequence_length] for start in starts])
    targets = np.stack(
        [tokens[start + 1 : start + sequence_length + 1] for start in starts]
    )
    return (
        torch.as_tensor(inputs.astype(np.int64), device=device),
        torch.as_tensor(targets.astype(np.int64), device=device),
    )


def train_from_scratch(
    train_bin: str | Path,
    valid_bin: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    config: AddSubConfig | None = None,
    steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    sequence_length: int = 256,
    save_every: int = 100,
    validate_every: int = 100,
    seed: int = 20261119,
    device_name: str = "cpu",
    resume: str | Path | None = None,
    extend_schedule: bool = False,
    num_threads: int = 0,
    wavebit_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train only from a new random initialization or this package's own checkpoint."""
    config = config or AddSubConfig()
    config.validate()
    wavebit_options = validate_wavebit_options(wavebit_options)
    if not isinstance(extend_schedule, bool):
        raise TypeError("extend_schedule must be a boolean.")
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 0:
        raise ValueError("num_threads must be a non-negative integer; 0 preserves the PyTorch default.")
    if num_threads > 0:
        # Explicit opt-in: PyTorch intra-op threads are process-global.
        torch.set_num_threads(num_threads)
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("save_every", save_every),
        ("validate_every", validate_every),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if steps < 1 or batch_size < 1 or sequence_length < 2:
        raise ValueError("steps, batch_size and sequence_length must be positive.")
    if save_every < 1 or validate_every < 1:
        raise ValueError("save_every and validate_every must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if sequence_length > config.max_context:
        raise ValueError("sequence_length exceeds the configured context.")
    device = torch.device(device_name)
    train_path, valid_path = Path(train_bin), Path(valid_bin)
    tokenizer_path, output = Path(tokenizer_path), Path(output_dir)
    if not train_path.is_file() or not valid_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Train stream, validation stream and tokenizer are required.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha = sha256_file(tokenizer_path)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab {tokenizer.get_vocab_size()} does not match "
            f"Add/Sub config vocab {config.vocab_size}."
        )
    manifest_path = train_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "A verified dataset_manifest.json is required beside the Add/Sub token streams."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "qmini-addsub-data-v1" or manifest.get("language") != "English":
        raise ValueError("Unsupported or non-English Add/Sub dataset manifest.")
    if manifest.get("synthetic_training_records_included") != 0:
        raise ValueError("Add/Sub v1 refuses a corpus containing synthetic training records.")
    pretraining = manifest.get("pretraining", {}).get("splits", {})
    for split, source_name in (
        ("train", "pretrain_train.txt"),
        ("validation", "pretrain_validation.txt"),
        ("test", "pretrain_test.txt"),
    ):
        source_path = train_path.parent / source_name
        expected_sha = pretraining.get(split, {}).get("sha256")
        if not source_path.is_file() or not expected_sha or sha256_file(source_path) != expected_sha:
            raise ValueError(f"{split} pretraining source does not match the manifest.")
    token_streams = manifest.get("token_streams", {})
    test_stream = token_streams.get("test", {})
    test_path = train_path.parent / str(test_stream.get("path", ""))
    declared_streams = (
        ("train", train_path, token_streams.get("train", {})),
        ("validation", valid_path, token_streams.get("validation", {})),
        ("test", test_path, test_stream),
    )
    for split, stream_path, entry in declared_streams:
        expected_source_sha = pretraining.get(split, {}).get("sha256")
        if entry.get("path") != stream_path.name:
            raise ValueError(f"{split} token stream path does not match the dataset manifest.")
        if not stream_path.is_file() or entry.get("sha256") != sha256_file(stream_path):
            raise ValueError(f"{split} token stream does not match the dataset manifest.")
        if entry.get("source_sha256") != expected_source_sha:
            raise ValueError(f"{split} token stream was not encoded from the manifested source text.")
        if not isinstance(entry.get("source_bytes"), int) or entry["source_bytes"] < 1:
            raise ValueError(f"{split} token stream is missing its exact encoded source byte count.")
    if len({train_path.resolve(), valid_path.resolve(), test_path.resolve()}) != 3:
        raise ValueError("Train, validation and test token streams must be separate files.")
    tokenizer_meta = manifest.get("tokenizer", {})
    if tokenizer_meta.get("sha256") != tokenizer_sha:
        raise ValueError("Tokenizer fingerprint does not match the dataset manifest.")

    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid_tokens = np.memmap(valid_path, dtype=np.uint16, mode="r")
    test_tokens = np.memmap(test_path, dtype=np.uint16, mode="r")
    if not len(train_tokens) or not len(valid_tokens) or not len(test_tokens):
        raise ValueError("Train, validation and test token streams must all be non-empty.")
    if (
        int(train_tokens.max()) >= config.vocab_size
        or int(valid_tokens.max()) >= config.vocab_size
        or int(test_tokens.max()) >= config.vocab_size
    ):
        raise ValueError("A token stream contains IDs outside the configured vocabulary.")
    train_sha, valid_sha = sha256_file(train_path), sha256_file(valid_path)
    test_sha = sha256_file(test_path)
    manifest_sha = sha256_file(manifest_path)
    validation_stream_bytes = int(
        token_streams.get("validation", {}).get("source_bytes", 0)
    )
    test_stream_bytes = int(test_stream.get("source_bytes", 0))
    validation_source_bytes = validation_stream_bytes
    test_source_bytes = test_stream_bytes
    if validation_source_bytes < 1 or test_source_bytes < 1:
        raise ValueError(
            "Token-stream manifest must include positive source_bytes for validation and test."
        )

    def bits_per_byte_estimate(token_loss: float, token_count: int, byte_count: int) -> float:
        """Convert sampled token NLL to an approximate, tokenizer-comparable BPB."""
        return token_loss * max(1, token_count - 1) / byte_count / math.log(2.0)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_contract = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "matrix_weight_decay": 0.01,
        "quantization": "scaled-five-symbol-v2",
        "save_every": save_every,
        "validate_every": validate_every,
        "seed": seed,
    }
    if num_threads > 0:
        run_contract["cpu_intraop_threads"] = num_threads
    if wavebit_options is not None:
        run_contract["wavebit_hsra"] = wavebit_options
    model = AddSubLanguageModel(
        config, seed=seed, wavebit_options=wavebit_options
    ).to(device)
    optimizer = build_adamw(model, learning_rate, weight_decay=0.01)
    start_step = 0
    schedule_total_steps = steps
    rng = random.Random(seed)

    if resume is not None:
        checkpoint_path = Path(resume)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        expected = {
            "format": CHECKPOINT_FORMAT,
            "config": config.to_dict(),
            "train_sha256": train_sha,
            "valid_sha256": valid_sha,
            "test_sha256": test_sha,
            "tokenizer_sha256": tokenizer_sha,
            "data_manifest_sha256": manifest_sha,
        }
        mismatches = [
            key for key, value in expected.items() if checkpoint.get(key) != value
        ]
        saved_contract = checkpoint.get("run_contract")
        if not isinstance(saved_contract, dict):
            mismatches.append("run_contract")
            saved_contract = {}
        saved_static_contract = {
            key: value for key, value in saved_contract.items()
            if key != "planned_steps"
        }
        if saved_static_contract != run_contract:
            mismatches.append("run_contract")
        if mismatches:
            raise ValueError(
                "Resume checkpoint is not compatible with this Add/Sub v1 run: "
                + ", ".join(mismatches)
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        saved_horizon = checkpoint.get(
            "schedule_total_steps",
            saved_contract.get("planned_steps", checkpoint.get("planned_steps")),
        )
        if isinstance(saved_horizon, bool) or not isinstance(saved_horizon, int):
            raise ValueError("Pretraining checkpoint has no valid saved schedule horizon.")
        schedule_total_steps = resolve_schedule_total_steps(
            steps, start_step, saved_horizon, extend_schedule, "Pretraining"
        )
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            saved_cuda_rng = checkpoint.get("cuda_rng_states")
            if not saved_cuda_rng or len(saved_cuda_rng) != torch.cuda.device_count():
                raise ValueError(
                    "CUDA RNG state is missing or device count changed; exact resume is unsafe."
                )
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_rng])
        rng.setstate(checkpoint["python_rng"])
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    existing_checkpoints = list(output.glob("checkpoint-step-*.pt"))
    if resume is None and (metrics_path.exists() or existing_checkpoints):
        raise FileExistsError(
            "Pretraining output already contains a run. Use a fresh output folder or "
            "resume from a compatible checkpoint; old metrics will not be mixed."
        )
    if resume is not None:
        newer_checkpoints = []
        for checkpoint_file in existing_checkpoints:
            try:
                checkpoint_step = int(checkpoint_file.stem.rsplit("-", 1)[-1])
            except ValueError as exc:
                raise ValueError(f"Malformed checkpoint filename: {checkpoint_file.name}") from exc
            if checkpoint_step > start_step:
                newer_checkpoints.append(checkpoint_file.name)
        if newer_checkpoints:
            raise ValueError(
                "Pretraining output contains checkpoints newer than the requested resume "
                "point: " + ", ".join(newer_checkpoints)
            )
        expected_metric_identity = {
            "train_sha256": train_sha,
            "valid_sha256": valid_sha,
            "test_sha256": test_sha,
            "tokenizer_sha256": tokenizer_sha,
            "data_manifest_sha256": manifest_sha,
            "run_contract": run_contract,
        }
        reconcile_metrics_file(
            str(metrics_path), expected_metric_identity, start_step, "Pretraining"
        )

    def evaluate(tokens: np.ndarray) -> float:
        """Evaluate on fixed evenly spaced windows without advancing train RNG."""
        model.eval()
        max_start = len(tokens) - sequence_length - 1
        if max_start < 0:
            raise ValueError("Evaluation token stream is shorter than one sequence.")
        number_of_windows = min(64, max_start + 1)
        if number_of_windows == 1:
            starts = [0]
        else:
            stride = max(1, max_start // (number_of_windows - 1))
            starts = list(range(0, max_start + 1, stride))[:number_of_windows]

        total_loss = 0.0
        target_count = 0
        with torch.inference_mode():
            for offset in range(0, len(starts), batch_size):
                current = starts[offset : offset + batch_size]
                x = np.stack([tokens[start : start + sequence_length] for start in current])
                y = np.stack(
                    [tokens[start + 1 : start + sequence_length + 1] for start in current]
                )
                inputs = torch.as_tensor(x.astype(np.int64), device=device)
                targets = torch.as_tensor(y.astype(np.int64), device=device)
                result = model(inputs)
                total_loss += float(F.cross_entropy(
                    result["logits"].float().reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ))
                target_count += int(targets.numel())
        model.train()
        if not target_count:
            raise ValueError("Evaluation did not contain any target tokens.")
        return total_loss / target_count

    def save_checkpoint(step: int, valid_loss: float | None) -> Path:
        path = output / f"checkpoint-step-{step:09d}.pt"
        payload = {
            "format": CHECKPOINT_FORMAT,
            "step": step,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_sha256": train_sha,
            "valid_sha256": valid_sha,
            "test_sha256": test_sha,
            "tokenizer_sha256": tokenizer_sha,
            "data_manifest_sha256": manifest_sha,
            "run_contract": run_contract,
            "planned_steps": steps,
            "schedule_total_steps": schedule_total_steps,
            "valid_loss": valid_loss,
            "torch_rng": torch.get_rng_state(),
            "python_rng": rng.getstate(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    model.train()
    last_loss = None
    session_started_at = time.perf_counter()
    window_train_seconds = 0.0
    window_train_steps = 0
    window_target_tokens = 0
    session_train_seconds = 0.0
    session_train_steps = 0
    session_target_tokens = 0
    for step in range(start_step + 1, steps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started_at = time.perf_counter()
        current_lr = cosine_warmup_lr(step, schedule_total_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        inputs, targets = _sample_batch(
            train_tokens, batch_size, sequence_length, rng, device
        )
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs)
        token_loss = F.cross_entropy(
            result["logits"].float().reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        loss = (
            token_loss
            + config.moe_balance_coef * result["aux_loss"]
            + config.router_z_loss_coef * result["router_z_loss"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        model.update_loss_free_router_bias(result["router_load"])
        assert_finite_training_state(model, optimizer, step)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started_at
        target_tokens = int(targets.numel())
        window_train_seconds += step_seconds
        window_train_steps += 1
        window_target_tokens += target_tokens
        session_train_seconds += step_seconds
        session_train_steps += 1
        session_target_tokens += target_tokens
        last_loss = float(token_loss.detach())

        if step % validate_every == 0 or step == steps:
            valid_loss = evaluate(valid_tokens)
            record = {
                "step": step,
                "learning_rate": current_lr,
                "train_token_loss": last_loss,
                "valid_token_loss": valid_loss,
                "router_balance_loss": float(result["aux_loss"].detach()),
                "router_z_loss": float(result["router_z_loss"].detach()),
                "router_load_by_layer": result["router_load"].detach().cpu().tolist(),
                "valid_bits_per_byte_estimate": bits_per_byte_estimate(
                    valid_loss, len(valid_tokens), validation_source_bytes
                ),
                "train_sha256": train_sha,
                "valid_sha256": valid_sha,
                "test_sha256": test_sha,
                "tokenizer_sha256": tokenizer_sha,
                "data_manifest_sha256": manifest_sha,
                "run_contract": run_contract,
                "schedule_total_steps": schedule_total_steps,
                "train_steps_since_last_validation": window_train_steps,
                "train_target_tokens_since_last_validation": window_target_tokens,
                "train_step_seconds_since_last_validation": window_train_seconds,
                "train_steps_per_second_excluding_validation": (
                    window_train_steps / window_train_seconds if window_train_seconds > 0 else 0.0
                ),
                "train_tokens_per_second_excluding_validation": (
                    window_target_tokens / window_train_seconds if window_train_seconds > 0 else 0.0
                ),
                "torch_num_threads": torch.get_num_threads(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
            window_train_seconds = 0.0
            window_train_steps = 0
            window_target_tokens = 0
        else:
            valid_loss = None

        if step % save_every == 0 or step == steps:
            save_checkpoint(step, valid_loss)

    heldout_test_loss = evaluate(test_tokens)
    final_validation_loss = evaluate(valid_tokens)
    return {
        "format": CHECKPOINT_FORMAT,
        "steps": steps,
        "schedule_total_steps": schedule_total_steps,
        "parameter_count": model.parameter_count(),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "final_train_token_loss": last_loss,
        "final_validation_token_loss": final_validation_loss,
        "final_validation_bits_per_byte_estimate": bits_per_byte_estimate(
            final_validation_loss, len(valid_tokens), validation_source_bytes
        ),
        "heldout_test_token_loss": heldout_test_loss,
        "heldout_test_bits_per_byte_estimate": bits_per_byte_estimate(
            heldout_test_loss, len(test_tokens), test_source_bytes
        ),
        "heldout_test_tokens_per_byte": len(test_tokens) / test_source_bytes,
        "heldout_test_sha256": sha256_file(test_path),
        "session_train_steps": session_train_steps,
        "session_target_tokens": session_target_tokens,
        "session_train_step_seconds": session_train_seconds,
        "session_train_steps_per_second_excluding_validation": (
            session_train_steps / session_train_seconds if session_train_seconds > 0 else 0.0
        ),
        "session_train_tokens_per_second_excluding_validation": (
            session_target_tokens / session_train_seconds if session_train_seconds > 0 else 0.0
        ),
        "session_wall_clock_seconds": time.perf_counter() - session_started_at,
        "torch_num_threads": torch.get_num_threads(),
        "requested_num_threads": num_threads,
        "output_dir": str(output),
        "data_manifest_sha256": manifest_sha,
        "note": "Throughput excludes validation time and is hardware/run-specific; training graph only, complete add/sub-only inference export is a separate gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-Mini Add/Sub v1 from random initialization.")
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default=None, help="Optional model_config.json written beside the trained tokenizer.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--extend-schedule", action="store_true", help="Explicitly allow increasing the saved cosine schedule horizon on resume.")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="Explicit PyTorch intra-op CPU thread count; 0 keeps the current PyTorch default.",
    )
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--wavebit-hsra", action="store_true",
                        help="Enable WaveBit-HSRA in the existing Add/Sub model (experimental).")
    parser.add_argument("--wavebit-group-size", type=int, default=30)
    parser.add_argument("--wavebit-rotation", type=int, default=1)
    parser.add_argument("--wavebit-gamma", type=int, default=4)
    parser.add_argument("--wavebit-ewgs-strength", type=float, default=0.5)
    parser.add_argument("--wavebit-octave-temperature", type=float, default=0.25)
    parser.add_argument(
        "--wavebit-octave-boost-factor", type=int, choices=(2, 4), default=4,
        help="Shared triplet boost; 4 keeps old behavior, 2 selects the x2 ablation.",
    )
    parser.add_argument(
        "--wavebit-hadamard", action="store_true",
        help="Enable blockwise Hadamard + per-token activation fake-quantization on expert down projections.",
    )
    parser.add_argument("--wavebit-hadamard-block-size", type=int, default=8)
    parser.add_argument("--wavebit-activation-bits", type=int, default=4)
    args = parser.parse_args()
    config = None
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = AddSubConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    result = train_from_scratch(
        train_bin=args.train_bin,
        valid_bin=args.valid_bin,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        config=config,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        validate_every=args.validate_every,
        seed=args.seed,
        device_name=args.device,
        resume=args.resume,
        extend_schedule=args.extend_schedule,
        num_threads=args.threads,
        wavebit_options=(
            {
                "group_size": args.wavebit_group_size,
                "rotation": args.wavebit_rotation,
                "gamma": args.wavebit_gamma,
                "ewgs_strength": args.wavebit_ewgs_strength,
                "octave_temperature": args.wavebit_octave_temperature,
                "octave_boost_factor": args.wavebit_octave_boost_factor,
                "hadamard": args.wavebit_hadamard,
                "hadamard_block_size": args.wavebit_hadamard_block_size,
                "activation_bits": args.wavebit_activation_bits,
            }
            if args.wavebit_hsra else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
