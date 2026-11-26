from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from .config import AddSubConfig
from .curriculum import SelfPacedCurriculumSampler
from .data import escape_special_token_literals, normalize_training_text
from .model import AddSubLanguageModel
from .optim import (
    assert_finite_training_state, build_adamw, cosine_warmup_lr,
    reconcile_metrics_file, resolve_schedule_total_steps,
)
from .wavebit_hsra import validate_wavebit_options
from .train import CHECKPOINT_FORMAT as PRETRAIN_FORMAT
from .train import sha256_file


SFT_FORMAT = "qmini-addsub-sft-v7"
LEGACY_SFT_FORMAT = "qmini-addsub-sft-v6"
SYSTEM_TEXT = "You are Q-Mini."

CHAT_SPECIAL_TOKENS = (
    "<unk>", "<bos>", "<eos>", "<pad>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
)


def _chat_special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Require every structural token and reject aliases before SFT."""
    values = {
        token: tokenizer.token_to_id(token)
        for token in CHAT_SPECIAL_TOKENS
    }
    missing = [token for token, token_id in values.items() if token_id is None]
    if missing:
        raise ValueError(f"Tokenizer is missing required Add/Sub chat tokens: {missing}")
    normalized = {token: int(token_id) for token, token_id in values.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Tokenizer chat control-token IDs must be unique.")
    return normalized


def _validate_base_tokenizer_fingerprint(
    checkpoint: dict[str, Any], tokenizer_path: str | Path
) -> str:
    """Prevent SFT from changing token meanings under an existing embedding table."""
    expected = checkpoint.get("tokenizer_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("Pretraining checkpoint is missing a valid tokenizer SHA-256 fingerprint.")
    actual = sha256_file(tokenizer_path)
    if actual != expected:
        raise ValueError(
            "SFT tokenizer fingerprint does not match the pretraining checkpoint; "
            "token IDs would refer to different embeddings."
        )
    return actual


def _load_conversations(
    path: Path, approved_sources: set[str] | None = None
) -> list[dict[str, Any]]:
    approved_sources = approved_sources or {
        "OpenAssistant/oasst1",
        "databricks/databricks-dolly-15k",
    }
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object.")
            if record.get("source") not in approved_sources:
                raise ValueError(f"{path}:{line_number}: unapproved SFT source.")
            if record.get("human_authored") is not True or record.get("synthetic") is not False:
                raise ValueError(f"{path}:{line_number}: synthetic/unverified SFT record rejected.")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            if any(not isinstance(item, dict) for item in messages):
                raise ValueError(f"{path}:{line_number}: every message must be an object.")
            roles = [item.get("role") for item in messages]
            if roles[0] != "user" or roles[-1] != "assistant":
                continue
            if any(role not in {"user", "assistant"} for role in roles):
                continue
            if any(roles[index] == roles[index - 1] for index in range(1, len(roles))):
                continue
            cleaned_messages = []
            for item in messages:
                cleaned, reason = normalize_training_text(item.get("content"))
                if reason != "accepted":
                    raise ValueError(
                        f"{path}:{line_number}: message rejected by text cleaner ({reason})."
                    )
                cleaned_item = dict(item)
                cleaned_item["content"] = cleaned
                cleaned_messages.append(cleaned_item)
            cleaned_record = dict(record)
            cleaned_record["messages"] = cleaned_messages
            records.append(cleaned_record)
    if not records:
        raise ValueError(f"No valid conversations found in {path}.")
    return records


class ConversationDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: Tokenizer, max_context: int):
        self.samples = []
        special = _chat_special_token_ids(tokenizer)

        for record in records:
            messages = [{"role": "system", "content": SYSTEM_TEXT}]
            messages.extend(record["messages"])
            encoded = []
            for message in messages:
                role = message["role"]
                role_token = "<|system|>" if role == "system" else (
                    "<|user|>" if role == "user" else "<|assistant|>"
                )
                content = str(message.get("content") or "").strip()
                content_ids = tokenizer.encode(escape_special_token_literals(content)).ids
                if not content_ids:
                    continue
                encoded.append((role, special[role_token], content_ids))

            # Retain complete assistant targets. Drop older exchanges, shorten the
            # system prompt to its compact suffix, then trim only the current user input.
            def required_length(items):
                return 1 + sum(1 + len(ids) + 1 for _, _, ids in items)

            while len(encoded) > 3 and required_length(encoded) > max_context:
                encoded.pop(1)
                if len(encoded) > 2:
                    encoded.pop(1)

            if required_length(encoded) > max_context:
                if len(encoded) < 3 or encoded[-1][0] != "assistant":
                    continue
                system_role, system_token, system_ids = encoded[0]
                user_role, user_token, user_ids = encoded[-2]
                assistant_role, assistant_token, assistant_ids = encoded[-1]
                if user_role != "user":
                    continue

                # Training and generation share the same policy: retain only the
                # newest two system tokens when the complete prompt cannot fit.
                if len(system_ids) > 2:
                    encoded[0] = (system_role, system_token, system_ids[-2:])

                room_for_answer = max_context - (required_length(encoded[:-1]) + 2)
                overflow = max(0, len(assistant_ids) - room_for_answer)
                if overflow:
                    available_user_tokens = len(user_ids) - overflow
                    if available_user_tokens < 1:
                        continue
                    encoded[-2] = (
                        user_role,
                        user_token,
                        user_ids[-available_user_tokens:],
                    )

                room_for_answer = max_context - (required_length(encoded[:-1]) + 2)
                if len(assistant_ids) > room_for_answer:
                    continue

            token_ids = [special["<bos>"]]
            label_ids = [-100]
            for role, role_token, content_ids in encoded:
                token_ids.append(role_token)
                label_ids.append(-100)
                token_ids.extend(content_ids)
                label_ids.extend(content_ids if role == "assistant" else [-100] * len(content_ids))
                token_ids.append(special["<|end|>"])
                label_ids.append(special["<|end|>"] if role == "assistant" else -100)
            if len(token_ids) > max_context or token_ids[-1] != special["<|end|>"]:
                continue
            if not any(value != -100 for value in label_ids[1:]):
                continue
            self.samples.append((token_ids, label_ids))

        if not self.samples:
            raise ValueError("No conversations fit the configured context.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    maximum = max(len(tokens) for tokens, _ in batch)
    inputs = torch.full((len(batch), maximum - 1), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maximum - 1), -100, dtype=torch.long)
    for index, (tokens, targets) in enumerate(batch):
        source = torch.as_tensor(tokens[:-1], dtype=torch.long)
        target = torch.as_tensor(targets[1:], dtype=torch.long)
        inputs[index, : source.numel()] = source
        labels[index, : target.numel()] = target
    return inputs, labels


def _require_response_targets(labels: torch.Tensor, step: int) -> int:
    """Reject a batch with no supervised assistant targets; never burn an optimizer step."""
    if not isinstance(labels, torch.Tensor):
        raise TypeError("SFT labels must be a torch.Tensor.")
    if labels.numel() == 0:
        raise ValueError(f"SFT batch at step {step} contains no labels.")
    count = int(labels.ne(-100).sum().item())
    if count == 0:
        raise ValueError(
            f"SFT batch at step {step} contains no assistant response targets; "
            "refusing to advance the optimizer schedule."
        )
    return count


def fine_tune_from_addsub_checkpoint(
    base_checkpoint: str | Path,
    tokenizer_path: str | Path,
    train_jsonl: str | Path,
    valid_jsonl: str | Path,
    test_jsonl: str | Path,
    output_dir: str | Path,
    steps: int = 800,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    save_every: int = 100,
    seed: int = 20261119,
    device_name: str = "cpu",
    resume: str | Path | None = None,
    extend_schedule: bool = False,
    curriculum_uniform_fraction: float = 1.0,
    curriculum_warmup_steps: int = 50,
) -> dict[str, Any]:
    if not isinstance(extend_schedule, bool):
        raise TypeError("extend_schedule must be a boolean.")
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("save_every", save_every),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if steps < 1 or batch_size < 1 or save_every < 1:
        raise ValueError("steps, batch_size and save_every must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if (
        isinstance(curriculum_uniform_fraction, bool)
        or not isinstance(curriculum_uniform_fraction, (int, float))
        or not math.isfinite(curriculum_uniform_fraction)
        or not 0.35 <= curriculum_uniform_fraction <= 1.0
    ):
        raise ValueError("curriculum_uniform_fraction must be finite and between 0.35 and 1.0.")
    if (
        isinstance(curriculum_warmup_steps, bool)
        or not isinstance(curriculum_warmup_steps, int)
        or curriculum_warmup_steps < 1
    ):
        raise ValueError("curriculum_warmup_steps must be a positive integer.")
    curriculum_enabled = float(curriculum_uniform_fraction) < 1.0
    device = torch.device(device_name)
    base_path, tokenizer_path = Path(base_checkpoint), Path(tokenizer_path)
    train_path, valid_path, test_path, output = (
        Path(train_jsonl), Path(valid_jsonl), Path(test_jsonl), Path(output_dir)
    )
    for path in (base_path, tokenizer_path, train_path, valid_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = torch.load(base_path, map_location=device, weights_only=True)
    if base.get("format") != PRETRAIN_FORMAT:
        raise ValueError("SFT accepts only a Q-Mini Add/Sub v1 pretraining checkpoint.")
    config = AddSubConfig.from_dict(base["config"])
    base_contract = base.get("run_contract", {})
    wavebit_options = (
        base_contract.get("wavebit_hsra") if isinstance(base_contract, dict) else None
    )
    wavebit_options = validate_wavebit_options(wavebit_options)
    if int(base.get("step", 0)) < 1:
        raise ValueError("SFT refuses a step-zero or random-only base checkpoint.")
    tokenizer_sha = _validate_base_tokenizer_fingerprint(base, tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the Add/Sub base config.")
    _chat_special_token_ids(tokenizer)

    manifest_path = train_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Verified Add/Sub dataset_manifest.json is required for SFT.")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("format") != "qmini-addsub-data-v1":
        raise ValueError("Unsupported Add/Sub dataset manifest.")
    if data_manifest.get("synthetic_training_records_included") != 0:
        raise ValueError("SFT refuses a dataset manifest containing synthetic training records.")
    sft_manifest = data_manifest.get("instruction_splits", {})
    tokenizer_meta = data_manifest.get("tokenizer", {})
    if tokenizer_meta.get("sha256") != tokenizer_sha:
        raise ValueError("SFT tokenizer does not match the curated dataset/tokenizer manifest.")
    for split_name, source_path in (
        ("train", train_path),
        ("validation", valid_path),
        ("test", test_path),
    ):
        entry = sft_manifest.get(split_name, {})
        if entry.get("path") != source_path.name or entry.get("sha256") != sha256_file(source_path):
            raise ValueError(f"{split_name} SFT file does not match the curated data manifest.")
    if len({train_path.resolve(), valid_path.resolve(), test_path.resolve()}) != 3:
        raise ValueError("SFT train, validation and test files must be distinct.")
    data_manifest_sha = sha256_file(manifest_path)

    approved_source_values = data_manifest.get("approved_instruction_sources")
    if approved_source_values is None:
        approved_source_values = [
            "OpenAssistant/oasst1",
            "databricks/databricks-dolly-15k",
        ]
    if (
        not isinstance(approved_source_values, list)
        or not approved_source_values
        or any(not isinstance(value, str) or not value.strip() for value in approved_source_values)
    ):
        raise ValueError("Manifest must list approved_instruction_sources for SFT.")
    approved_sources = set(approved_source_values)
    train_records = _load_conversations(train_path, approved_sources)
    valid_records = _load_conversations(valid_path, approved_sources)
    test_records = _load_conversations(test_path, approved_sources)
    train_data = ConversationDataset(train_records, tokenizer, config.max_context)
    valid_data = ConversationDataset(valid_records, tokenizer, config.max_context)
    test_data = ConversationDataset(test_records, tokenizer, config.max_context)
    pad_id = int(tokenizer.token_to_id("<pad>"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    sampler = SelfPacedCurriculumSampler(
        dataset_size=len(train_data),
        total_steps=steps,
        warmup_steps=curriculum_warmup_steps,
        uniform_fraction=float(curriculum_uniform_fraction),
        start_quantile=0.55,
        end_quantile=0.80,
        bandwidth=0.18,
        ema_decay=0.80,
    )
    run_contract = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "matrix_weight_decay": 0.01,
        "quantization": "scaled-five-symbol-v2",
        "system_prompt": SYSTEM_TEXT,
        "save_every": save_every,
        "seed": seed,
    }
    if curriculum_enabled:
        run_contract.update({
            "curriculum": SelfPacedCurriculumSampler.FORMAT,
            "curriculum_uniform_fraction": float(curriculum_uniform_fraction),
            "curriculum_warmup_steps": curriculum_warmup_steps,
            "curriculum_start_quantile": 0.55,
            "curriculum_end_quantile": 0.80,
            "curriculum_bandwidth": 0.18,
            "curriculum_ema_decay": 0.80,
        })
    if wavebit_options is not None:
        run_contract["wavebit_hsra"] = wavebit_options

    model = AddSubLanguageModel(
        config, seed=seed, wavebit_options=wavebit_options
    ).to(device)
    model.load_state_dict(base["model"], strict=True)
    optimizer = build_adamw(model, learning_rate, weight_decay=0.01)
    hashes = {
        "base_sha256": sha256_file(base_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_sha256": sha256_file(train_path),
        "valid_sha256": sha256_file(valid_path),
        "test_sha256": sha256_file(test_path),
        "data_manifest_sha256": data_manifest_sha,
    }
    start_step = 0
    schedule_total_steps = steps
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        checkpoint_format = checkpoint.get("format")
        if checkpoint_format not in {SFT_FORMAT, LEGACY_SFT_FORMAT}:
            raise ValueError("Unsupported Add/Sub SFT checkpoint format.")
        if checkpoint_format == LEGACY_SFT_FORMAT and curriculum_enabled:
            raise ValueError(
                "Legacy v6 SFT checkpoints do not contain per-record curriculum losses; "
                "resume them with uniform sampling or start a fresh curriculum run."
            )
        expected = {
            "config": config.to_dict(),
            **hashes,
        }
        mismatch = [key for key, value in expected.items() if checkpoint.get(key) != value]
        saved_contract = checkpoint.get("run_contract")
        if not isinstance(saved_contract, dict):
            mismatch.append("run_contract")
            saved_contract = {}
        saved_static_contract = {
            key: value for key, value in saved_contract.items()
            if key != "planned_steps"
        }
        if saved_static_contract != run_contract:
            mismatch.append("run_contract")
        if mismatch:
            raise ValueError("Incompatible Add/Sub SFT resume: " + ", ".join(mismatch))
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        saved_horizon = checkpoint.get(
            "schedule_total_steps",
            saved_contract.get("planned_steps", checkpoint.get("planned_steps")),
        )
        if isinstance(saved_horizon, bool) or not isinstance(saved_horizon, int):
            raise ValueError("SFT checkpoint has no valid saved schedule horizon.")
        schedule_total_steps = resolve_schedule_total_steps(
            steps, start_step, saved_horizon, extend_schedule, "SFT"
        )
        if "torch_rng" not in checkpoint or "python_rng" not in checkpoint:
            raise ValueError(
                "This SFT checkpoint predates exact RNG resume; start a new v4 SFT run."
            )
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            saved_cuda_rng = checkpoint.get("cuda_rng_states")
            if not saved_cuda_rng or len(saved_cuda_rng) != torch.cuda.device_count():
                raise ValueError("CUDA RNG state is missing or device count changed.")
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_rng])
        rng.setstate(checkpoint["python_rng"])
        if checkpoint_format == SFT_FORMAT:
            curriculum_state = checkpoint.get("curriculum")
            if not isinstance(curriculum_state, dict):
                raise ValueError("SFT v7 checkpoint is missing exact curriculum state.")
            sampler.load_state_dict(curriculum_state)
        else:
            # Legacy v6 used rng.sample(range(N), k) for uniform batches. The
            # sampler's uniform-control path makes the exact same RNG call.
            sampler.step = start_step
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "sft_metrics.jsonl"
    if resume is None and (
        metrics_path.exists() or any(output.glob("sft-checkpoint-step-*.pt"))
    ):
        raise FileExistsError(
            "SFT output already contains a run. Use a fresh output folder or resume "
            "from a compatible checkpoint; old metrics will not be mixed."
        )
    if resume is not None:
        existing_checkpoints = list(output.glob("sft-checkpoint-step-*.pt"))
        newer_checkpoints = []
        for checkpoint_file in existing_checkpoints:
            try:
                checkpoint_step = int(checkpoint_file.stem.rsplit("-", 1)[-1])
            except ValueError as exc:
                raise ValueError(f"Malformed SFT checkpoint filename: {checkpoint_file.name}") from exc
            if checkpoint_step > start_step:
                newer_checkpoints.append(checkpoint_file.name)
        if newer_checkpoints:
            raise ValueError(
                "SFT output contains checkpoints newer than the requested resume point: "
                + ", ".join(newer_checkpoints)
            )
        expected_metric_identity = {
            "train_sha256": hashes["train_sha256"],
            "valid_sha256": hashes["valid_sha256"],
            "test_sha256": hashes["test_sha256"],
            "tokenizer_sha256": hashes["tokenizer_sha256"],
            "data_manifest_sha256": hashes["data_manifest_sha256"],
            "run_contract": run_contract,
        }
        reconcile_metrics_file(
            str(metrics_path), expected_metric_identity, start_step, "SFT"
        )

    def evaluate(dataset: ConversationDataset, split_name: str) -> float:
        was_training = model.training
        model.eval()
        loss_sum = 0.0
        target_count = 0
        with torch.inference_mode():
            for offset in range(0, len(dataset), batch_size):
                batch = [
                    dataset[index]
                    for index in range(offset, min(offset + batch_size, len(dataset)))
                ]
                inputs, labels = _collate(batch, pad_id)
                inputs = inputs.to(device)
                labels = labels.to(device)
                result = model(inputs, attention_mask=inputs.ne(pad_id))
                count = int((labels != -100).sum())
                if count:
                    loss_sum += float(F.cross_entropy(
                        result["logits"].float().reshape(-1, config.vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    ))
                    target_count += count
        model.train(was_training)
        if not target_count:
            raise ValueError(f"{split_name} set contains no assistant response targets.")
        return loss_sum / target_count

    model.train()
    last_loss = None
    for step in range(start_step + 1, steps + 1):
        current_lr = cosine_warmup_lr(step, schedule_total_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        sample_count = min(batch_size, len(train_data))
        sampled_indices, curriculum_info = sampler.sample(sample_count, rng)
        batch = [train_data[index] for index in sampled_indices]
        inputs, labels = _collate(batch, pad_id)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs, attention_mask=inputs.ne(pad_id))
        target_count = _require_response_targets(labels, step)
        if curriculum_enabled:
            token_losses = F.cross_entropy(
                result["logits"].float().reshape(-1, config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(labels)
            per_record_counts = labels.ne(-100).sum(dim=1)
            per_record_losses = token_losses.sum(dim=1) / per_record_counts.clamp_min(1)
            response_loss = token_losses.sum() / target_count
        else:
            response_loss = F.cross_entropy(
                result["logits"].float().reshape(-1, config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            per_record_losses = None
        loss = response_loss + (
            config.moe_balance_coef * result["aux_loss"]
            + config.router_z_loss_coef * result["router_z_loss"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite SFT loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        model.update_loss_free_router_bias(result["router_load"])
        if curriculum_enabled:
            sampler.update(
                sampled_indices,
                [float(value) for value in per_record_losses.detach().cpu().tolist()],
            )
        assert_finite_training_state(model, optimizer, step)
        last_loss = float(loss.detach())

        if step % 50 == 0 or step == steps:
            valid_loss = evaluate(valid_data, "validation")
            row = {
                "step": step,
                "learning_rate": current_lr,
                "train_loss": last_loss,
                "valid_response_loss": valid_loss,
                "router_balance_loss": float(result["aux_loss"].detach()),
                "router_z_loss": float(result["router_z_loss"].detach()),
                "router_load_by_layer": result["router_load"].detach().cpu().tolist(),
                "train_sha256": hashes["train_sha256"],
                "valid_sha256": hashes["valid_sha256"],
                "test_sha256": hashes["test_sha256"],
                "tokenizer_sha256": hashes["tokenizer_sha256"],
                "data_manifest_sha256": hashes["data_manifest_sha256"],
                "run_contract": run_contract,
                "schedule_total_steps": schedule_total_steps,
            }
            if curriculum_enabled:
                row.update({
                    "curriculum_mode": curriculum_info["mode"],
                    "curriculum_focus_quantile": curriculum_info["focus_quantile"],
                    "curriculum_uniform_fraction": curriculum_info["uniform_fraction"],
                    "curriculum_requested_uniform_fraction": curriculum_info["requested_uniform_fraction"],
                    "curriculum_uniform_samples": curriculum_info["uniform_samples"],
                    "curriculum_focused_samples": curriculum_info["focused_samples"],
                    "curriculum_seen_fraction": curriculum_info["seen_fraction"],
                    "curriculum_sampled_mean_response_loss": float(per_record_losses.detach().mean()),
                })
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True))
        if step % save_every == 0 or step == steps:
            payload = {
                "format": SFT_FORMAT,
                "step": step,
                "planned_steps": steps,
                "schedule_total_steps": schedule_total_steps,
                "run_contract": run_contract,
                "config": config.to_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "curriculum": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": rng.getstate(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                **hashes,
            }
            destination = output / f"sft-checkpoint-step-{step:09d}.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)

    return {
        "format": SFT_FORMAT,
        "steps": steps,
        "schedule_total_steps": schedule_total_steps,
        "train_records": len(train_data),
        "validation_records": len(valid_data),
        "test_records": len(test_data),
        "parameter_count": model.parameter_count(),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "final_train_loss": last_loss,
        "final_validation_response_loss": evaluate(valid_data, "validation"),
        "heldout_test_response_loss": evaluate(test_data, "test"),
        "heldout_test_sha256": hashes["test_sha256"],
        "curriculum": {
            "format": sampler.FORMAT,
            "enabled": curriculum_enabled,
            "requested_uniform_replay_fraction": sampler.uniform_fraction,
            "warmup_steps": sampler.warmup_steps,
            "seen_records": sum(count > 0 for count in sampler.update_count),
            "total_records": sampler.dataset_size,
            "uniform_replay_quota_remainder": sampler.uniform_remainder,
        },
        "output_dir": str(output),
        "note": "Self-paced SFT sampling is optional and experimental; no quality improvement is claimed until held-out comparisons pass.",

    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-fine-tune an Add/Sub v1 pretraining checkpoint.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--extend-schedule", action="store_true", help="Explicitly allow increasing the saved cosine schedule horizon on resume.")
    parser.add_argument(
        "--curriculum-uniform-fraction",
        type=float,
        default=1.0,
        help="Uniform replay fraction; 1.0 keeps the baseline sampler, 0.35-0.99 enables loss-aware self-paced sampling.",
    )
    parser.add_argument(
        "--curriculum-warmup-steps",
        type=int,
        default=50,
        help="Uniform warmup steps before difficulty-based sampling (only used when the replay fraction is below 1.0).",
    )
    args = parser.parse_args()
    result = fine_tune_from_addsub_checkpoint(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        train_jsonl=args.train_jsonl,
        valid_jsonl=args.valid_jsonl,
        test_jsonl=args.test_jsonl,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        device_name=args.device,
        resume=args.resume,
        extend_schedule=args.extend_schedule,
        curriculum_uniform_fraction=args.curriculum_uniform_fraction,
        curriculum_warmup_steps=args.curriculum_warmup_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
